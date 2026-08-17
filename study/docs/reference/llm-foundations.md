<!-- Generated copy. Source: study/notes/01-llm-foundations.md
     Edit the source file; this copy is overwritten on every extraction run. -->

# Weeks 1–3 — LLM foundations: transformers, tokenization, context, MoE

**How this was built:** written from primary sources — arXiv papers, ACL Anthology, official Anthropic/OpenAI/Google/HuggingFace docs — not from lecture transcripts. Where the source material contradicts what a 2024/2025-era course teaches, that is flagged explicitly. Sources are linked inline; claims that could not be confirmed against a primary source are marked `UNVERIFIED`.

**Read the corrections table (§7) first.** Four things commonly taught in this part of the course are wrong or badly out of date, and one of them — the Italian token-cost figure — will make you over-budget your own projects by roughly 2×.

---

## 0. The one-paragraph version

Almost nothing you need from transformer internals is about the maths; it is about **which numbers you pay for at inference**. Attention's quadratic term is *compute during prefill*, not memory — FlashAttention made the memory claim obsolete in 2022 and it is still repeated in course material. The KV cache, not the parameter count, is what makes long context expensive, which is why the entire MHA→MQA→GQA→MLA lineage exists. Token counts are a property of a specific tokenizer and are not portable between models, which matters directly to you because **Italian costs roughly +36% tokens vs English on current models, not the ~2× that the widely-cited 2023 paper says** — and the spread across models is larger than the spread across languages, so you have to measure it yourself. Advertised context is a capacity claim: [NoLiMa](https://arxiv.org/pdf/2502.05167) measured Llama 3.1 70B's *effective* context at ~2K against a 128K advertised window, and a [March 2026 theory paper](https://arxiv.org/html/2603.10123v1) shows the "lost in the middle" U-shape exists at random initialization, before any training and with RoPE removed — so position bias is architectural and no future model release will fix it. Put the question last. Everything else below is detail in service of those five facts.

---

## 1. Transformer essentials — only what changes an engineering decision

### 1.1 Attention and multi-head attention

Self-attention computes, for each token, a weighted sum over every other token's value vector; weights are the softmax of query·key scores scaled by `1/√d_head`. The engineering-relevant consequence is not the algebra but this: **every token's representation is a mixture over every other token in the window**, so the number of pairwise interactions grows as n². That n² is the origin of every long-context cost and quality story in §3.

Multi-head attention runs `h` independent heads in parallel on lower-dimensional projections (`d_model/h` each), concatenated and projected back. Heads demonstrably specialise (syntactic, positional, retrieval-like "induction" heads), but **nothing in serving depends on head semantics**. What serving depends on is that **head count and head dimension determine KV cache size**, and the KV cache is what you actually pay for at inference.

### 1.2 MHA → MQA → GQA → MLA is a KV-cache axis, not a quality axis

This is the single most useful way to read a model config. All four are the same attention operation with a different number of distinct K/V projections.

| Variant | `n_kv_heads` | KV cache | Where you see it |
|---|---|---|---|
| **MHA** | = `n_q_heads` | largest | original Transformer, GPT-2, older Llama |
| **MQA** | 1 | smallest | some quality loss; used where memory dominates |
| **GQA** | small group count | middle — the modern default | Llama 3, Qwen, Mistral |
| **MLA** (Multi-head Latent Attention) | low-rank latent, decompressed on the fly | large reduction | DeepSeek |

Cost-optimality analysis of GQA: [arXiv:2503.09579](https://arxiv.org/html/2503.09579v3). MLA/GQA walkthrough: [PyImageSearch, Oct 2025](https://pyimagesearch.com/2025/10/13/kv-cache-optimization-via-multi-head-latent-attention/).

Read it as: someone chose how much inference memory to spend per token of context. Quality differences between GQA configurations are second-order; cache size differences are 4–8×.

### 1.3 FlashAttention — why "attention is O(n²) memory" is now wrong

[FlashAttention](https://arxiv.org/abs/2205.14135) computes **exact** attention with tiling, so the n×n attention matrix is never materialised in HBM. Memory becomes **linear in n**; compute stays quadratic.

> ⚠️ **"Attention needs O(n²) memory" is false on any modern stack.** It is one of the most persistent stale claims in LLM teaching material. The quadratic that survives is **compute during prefill** — which is why a 500K-token prompt is slow to *start* but not impossible to hold.

Practical consequence: when someone says "we can't do long context because of the quadratic memory," they are describing 2021. When they say "prefill on a 1M-token prompt is expensive and slow," they are describing 2026 correctly.

### 1.4 Positional encoding — four generations, one that matters

Transformers are permutation-equivariant without position information: shuffle the tokens and the attention output is the same set of vectors. Position has to be injected.

| Scheme | Mechanism | Where you'll see it |
|---|---|---|
| Sinusoidal (absolute) | fixed sin/cos added to embeddings | original 2017 Transformer |
| Learned absolute | trainable position-embedding table | BERT, GPT-2 |
| **RoPE** | rotates Q and K by a position-dependent angle → attention score depends on the *relative* offset | Llama, Qwen, Mistral, Gemma, DeepSeek — the de facto standard |
| **ALiBi** | adds a linear, head-specific distance penalty directly to attention logits; no positional embedding at all | BLOOM, MPT, some encoder work |

Overview with worked derivations: [ICLR 2025 blogpost on positional embeddings](https://iclr-blogposts.github.io/2025/blog/positional-embedding/).

RoPE's operative properties: it is injected into Q/K **at every layer**, it encodes *relative* position, and it is **the thing you manipulate to extend context**.

**Context extension is RoPE surgery.** The lineage: Position Interpolation → NTK-aware scaling → **[YaRN](https://arxiv.org/pdf/2309.00071)** (frequency-band-wise interpolation plus an attention temperature correction), which extends context with far less continued pretraining than naive interpolation. Accessible write-up: [EleutherAI](https://blog.eleuther.ai/yarn/).

Concrete real-world example: **ChatQA-2 extended Llama-3 from 8K to 128K by continued pretraining with the RoPE base frequency raised to 150M** — [arXiv:2407.14482](https://www.arxiv.org/pdf/2407.14482v3). When you see `rope_theta` in a config, that is this knob.

ALiBi's selling point was zero-shot length extrapolation. RoPE + YaRN/scaling won in practice; ALiBi is now mostly of architectural/historical interest.

> ⚠️ **"ALiBi extrapolates, so it solves long context" is wrong.** Extrapolating *perplexity* to longer sequences is not the same as *retrieving accurately* at length. See §3.4 — the two came apart badly under measurement.

### 1.5 Residual stream and normalisation placement

- **Post-norm** (original 2017): `x → Sublayer → Add → LayerNorm`. Better final quality in some settings, but gradient norms blow up with depth; needs warmup, learning-rate babysitting, and often diverges past a few dozen layers.
- **Pre-norm** (universal default since GPT-2/GPT-3): `x → LayerNorm → Sublayer → Add`. Leaves a clean identity path down the residual stream → stable gradients, trains deep models without heroics. Cost: mildly worse final loss than a *successfully trained* post-norm model.

This is a genuine live tradeoff, not a solved question. **HybridNorm** (NeurIPS 2025, [arXiv:2503.04598](https://arxiv.org/abs/2503.04598)) uses QKV-norm inside attention plus post-norm in the FFN, and reports beating both pre- and post-norm on **dense and MoE** models.

Also standard now, and worth recognising in configs:

- **RMSNorm** instead of LayerNorm — drops mean-centering and the bias term; cheaper, no measured quality loss. If you see `rms_norm_eps`, that is it.
- **QK-norm** — normalises queries and keys to suppress attention-logit explosions.

**Engineer's takeaway:** if you fine-tune, do not touch norm placement. Reading `rms_norm_eps` + pre-norm blocks in a config tells you only "this is a modern model."

### 1.6 Decoder-only vs encoder-decoder — and the 2025 result that complicates it

| Family | Attention | Generates? | Right for |
|---|---|---|---|
| **Encoder-only** (BERT and *all* the embedding models in [file 05](rag-and-vector-search.md)) | bidirectional | no | embeddings, classification, reranking |
| **Encoder-decoder** (T5, mT5, NMT) | bidirectional encoder + causal decoder with cross-attention | yes | fixed input → transformed output |
| **Decoder-only** (all frontier LLMs) | one causal stack; input and output are the same sequence | yes | everything the course covers |

Why decoder-only won: (a) every token is a training signal, (b) one stack to scale, (c) KV cache and prompt caching are trivially natural, (d) in-context learning emerges strongly.

**But the "simply better" framing is too strong.** A 2025 Google DeepMind re-examination ("RedLLM", 150M–8B, [arXiv:2510.26622](https://arxiv.org/html/2510.26622v1)) found decoder-only dominates the **compute-optimal pretraining frontier** and in-context learning — but **after instruction tuning, encoder-decoder matches or beats decoder-only on downstream tasks with substantially better inference efficiency**, and extrapolates beyond training length more gracefully. So decoder-only is better *for the pretraining-scaling + ICL regime the industry optimises for*, which is not the same claim.

---

## 2. Tokenization

### 2.1 The four algorithms (and the library everyone miscounts as a fifth)

| Algorithm | Merge / selection rule | Base units | Used by |
|---|---|---|---|
| **BPE** | greedily merge the most **frequent** adjacent pair | characters | GPT-1, Llama, Gemma, Qwen2 |
| **Byte-level BPE** | same rule, but base units are the **256 byte values** → `<unk>` is impossible | 256 bytes | GPT-2 → GPT-4/5, most modern LLMs |
| **WordPiece** | merge the pair maximising training-data likelihood: `score = freq(pair) / (freq(a)·freq(b))` — rewards pairs co-occurring *more than chance*, not merely often | characters | BERT, DistilBERT, ELECTRA |
| **Unigram** | start from a large candidate vocab and iteratively **prune** the tokens whose removal least increases loss; probabilistic, so a word has multiple valid segmentations | large seed set | T5, ALBERT, Pegasus |

Reference: [HuggingFace tokenizer summary](https://huggingface.co/docs/transformers/en/tokenizer_summary).

> **SentencePiece is not a fifth algorithm.** It is a *library* that runs BPE or Unigram directly on raw text, treats whitespace as a real character (`▁`), and needs no language-specific pre-tokenizer. That is exactly why it became the default for multilingual models and for languages without spaces (zh/ja/th). If a model card says "SentencePiece," look for whether it is SentencePiece-BPE or SentencePiece-Unigram — those are different.

Vocab sizes, illustratively: GPT-2 is **50,257** (256 bytes + 50,000 merges + `<|endoftext|>`); modern models run **32K–256K**. The 2026 *Tokenizer Tax* study surveyed 10 frontier models spanning that whole range — [arXiv:2605.24718](https://arxiv.org/html/2605.24718).

### 2.2 Why token counts are not portable across models

Five independent causes, in rough order of impact:

1. **Vocab size.** A 256K-vocab tokenizer stores more whole words and long subwords → fewer tokens for the same string. **The single biggest driver.**
2. **Training-corpus mix.** Merges are learned by frequency. A tokenizer trained on 90% English learns English words whole and shatters Italian morphology into pieces.
3. **Pre-tokenization regex.** How digits, whitespace runs, punctuation and code indentation get split before merging. `cl100k_base` (GPT-4 era) grouped digits in threes and added multi-space tokens; earlier ones did not.
4. **Byte-level vs character-level base.** Byte-level guarantees coverage, but any character outside the learned merges costs 1–4 tokens — one per UTF-8 byte.
5. **Special and chat tokens** differ per family and are counted.

> **Consequence: never size a context budget, a chunk size, or a cost estimate with another model's tokenizer.** Use the target model's own: `tiktoken` for OpenAI, `AutoTokenizer` for HF models, `/v1/messages/count_tokens` for Claude. A chunk-size sweep done with `cl100k_base` and deployed against Qwen is measuring the wrong thing.

### 2.3 Italian tokenization — the section that matters most to you

Fertility = **average tokens per word**. It is the number to quote, because it is corpus-normalised.

**Current (2026) cross-language measurements.** Parallel text across 25 European languages, 10 frontier models (Llama family, Mistral Large 3, Nemotron, Nova Pro, Qwen3, GPT-4o, Gemma 2, DeepSeek V3); API measurements validated against local tokenizers to within 1.8% — [arXiv:2605.24718](https://arxiv.org/html/2605.24718):

| Language | tok/word | vs English |
|---|---|---|
| English | **1.23** | baseline |
| **Italian** | **1.67** | **≈ +36%** |
| Ukrainian | 2.66 | ≈ 2.2× |
| Greek, Maltese | ≈ 3.1 | ≈ 2.5× |

Note the Ukrainian finding, because it isolates the mechanism: Ukrainian pays **15–18% more than cognate Slavic languages**, attributed to **under-representation in training data** rather than to morphology. Tokenizer cost is a data-mix artefact, not a linguistic law.

**Per-model Italian fertility**, measured on CulturaX-IT and Wikipedia-IT (NAACL 2025 Findings, [arXiv:2504.17025](https://arxiv.org/pdf/2504.17025)):

| Model | tok/word (CulturaX-IT / Wikipedia-IT) |
|---|---|
| Mistral-7B-v0.1 | **1.88 / 2.05** |
| Llama-3.1-8B | **1.67 / 1.80** |
| Minerva (Italian-native tokenizer) | **1.39 / 1.66** |

Same paper, on vocabulary adaptation (SAVA): cut fertility **25% for Mistral-7B** and **16% for Llama-3.1-8B**; for Llama the latter came alongside **75% vocab shrinkage → 8.03B→7.25B params (−10% model size)** with competitive quality. Relevant if Gewiss ever self-hosts an Italian-heavy model: adapting the tokenizer is a real, published lever, not folklore.

> ⚠️ **The claim you will hear in the course and in every blog post — "Italian costs about 2× English tokens" — is outdated.** It comes from *Language Model Tokenizers Introduce Unfairness Between Languages* (2023, [arXiv:2305.15425](https://arxiv.org/pdf/2305.15425)), which measured **2.01× on GPT-2** and **2.18× on cl100k/GPT-4**, and which also popularised "up to 15× difference between languages." Both figures are now **materially pessimistic for Italian**. Vocabularies went ~50K → 100K → 200K+ and training mixes got far more multilingual. The 2026 parallel-corpus figure for Italian is **≈1.36×**.

**Budget Italian at roughly +30–40% token overhead vs English — and measure it on your actual model.** The spread across models (1.67 for Llama-3.1 vs 1.88–2.05 for Mistral-7B on the same corpus) is larger than the correction you are making, so a single global multiplier is a worse estimate than five minutes with a tokenizer.

**Practical Italian-specific cost levers:**

- **Accented characters** (à è é ì ò ù) are where byte-level BPE bleeds: a rare accented form can cost 2 bytes and therefore extra tokens. Business Italian is saturated with them — *università, qualità, responsabilità, perché, può, più*.
- **Elisions** (`l'azienda`, `dell'esercizio`, `quell'infrastruttura`) fragment badly on English-trained merges.
- **Keep system prompts, schemas, tool definitions and instructions in English; keep only user text and document content in Italian.** Two reasons compound: the system prompt is the repeated, *cacheable* part (§4), *and* it is the part where you get English tokenization for free. This is the highest-value, lowest-effort lever in the whole section.
- **The overhead compounds twice** — prompt tokens *and* completion tokens. Output-heavy Italian workloads pay the premium at the (typically 4–5× higher) output rate, so an Italian-output chatbot is meaningfully more expensive per turn than the input-side arithmetic suggests.
- **Long-context Italian corpora hit the context ceiling ~35% sooner** than the equivalent English corpus. This matters when you size chunk counts and `k` in [file 05](rag-and-vector-search.md).

### 2.4 Code and JSON

- **Code is not uniformly expensive — it is bimodal.** Keywords, common identifiers and (on `cl100k`/`o200k`) **runs of whitespace** are single tokens. But long `snake_case`/`camelCase` identifiers, minified JS, base64 blobs and deeply-indented code fragment heavily.
- Tokenizers with explicit multi-whitespace tokens (GPT-3.5-turbo onward) are dramatically cheaper on Python than GPT-2-era ones, where **indentation cost one token per space**.
- **JSON is token-expensive relative to its information content** — quotes, braces, repeated keys on every object. At volume this is a real cost line. Consider terse key names, or YAML/TSV where strict schema validation is not required.

### 2.5 The measurement script — run this on your own documents

`UNVERIFIED`: no first-party Italian tokenization measurements were produced for these notes — tokenizer BPE-file downloads were blocked in the research environment. The published figures above are the source of truth here. Take ten minutes and produce your own:

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")   # GPT-4o/5-family
for label, txt in [("EN", en_text), ("IT", it_text)]:
    print(label, len(enc.encode(txt)), len(txt)/len(enc.encode(txt)))
```

And the HuggingFace equivalent with `AutoTokenizer.from_pretrained(...)` for Llama / Qwen / Mistral. For Claude, use `/v1/messages/count_tokens` rather than guessing from a public tokenizer.

> **Do this on *your own* documents — Gewiss product catalogues, technical datasheets, installation manuals, normative references — not on sample sentences.** Domain vocabulary dominates the ratio. Part numbers, IEC/CEI standard references and measurement units tokenize nothing like Italian Wikipedia prose, and those are precisely the strings your users will query on.

---

## 3. Context window, KV cache, and effective vs advertised context

### 3.1 What the context window physically is

It is the maximum sequence length for which the model's **positional scheme and attention are trained/calibrated**, plus a hard allocation limit for the KV cache. It is **not** a memory buffer the model "reads." It is the length of sequence over which attention can be computed at all.

Everything lives in the window: system prompt, **tool/function definitions** (routinely underestimated — schemas are expensive), the full conversation history, retrieved chunks, and reserved space for the output. Output tokens consume window as they are generated.

2026 advertised sizes, from official docs ([Claude models overview](https://platform.claude.com/docs/en/about-claude/models/overview)):

| Model | Context | Max output |
|---|---|---|
| Claude Opus 5 / Sonnet 5 / Fable 5 | **1M** | 128K |
| Claude Haiku 4.5 | 200K | 64K |

The Batch API supports up to **300K output tokens** behind a beta header.

`UNVERIFIED` — **Gemini per-model limits.** The Gemini models doc page fetched during research listed the current lineup (Gemini 3.7/3.6/3.5 Flash, 3.5/3.1 Flash-Lite, 2.5 Pro/Flash) but **did not expose per-model token limits**. Gemini has advertised 1M+ since 2.5 Pro; since your only closed-model key is Google's, check the current numbers directly at [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) before you design around a figure.

### 3.2 KV cache — why long context costs money at inference

During autoregressive decoding, each new token attends to all previous tokens. Rather than recompute K and V for the whole prefix at every step, they are cached. **That cache *is* the memory cost of context.**

The sizing arithmetic — state it, do not re-derive it:

```
KV bytes ≈ 2 (K and V) × n_layers × n_kv_heads × d_head × seq_len × batch × bytes_per_element
```

**Linear in sequence length and linear in batch size.** This one formula explains: why GQA/MQA/MLA exist (they shrink `n_kv_heads`, or replace it with a latent — §1.2), why KV-cache quantization to fp8/int8 is now routine, and why a serving system's max concurrency drops as your prompts get longer.

### 3.3 Prefill vs decode — the single most useful inference mental model

| Phase | What it does | Bottleneck | Cost shape |
|---|---|---|---|
| **Prefill** | processes the whole prompt at once | **compute-bound**, high GPU utilisation | attention cost **quadratic in prompt length** |
| **Decode** | generates token by token | **memory-bandwidth-bound**, terrible arithmetic intensity | ~O(current_length) work per token, serial |

Prefill is what makes a 500K-token prompt slow to *start*. Decode is what makes a long answer slow to *finish*, and its latency is governed by how fast weights + KV cache stream out of HBM. Total attention work to generate m tokens after an n-token prompt is still ~O((n+m)²). Sources: [Towards Data Science](https://towardsdatascience.com/prefill-is-compute-bound-decode-is-memory-bound-why-your-gpu-shouldnt-do-both/), [BentoML](https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation).

Production systems now **disaggregate** prefill and decode onto separate GPU pools, because the two phases want different hardware — [OpenMetal](https://openmetal.io/resources/blog/prefill-decode-two-inference-pools/).

**Pricing implication, and this is why the whole industry prices the way it does:** prompt tokens are cheap per token because prefill is parallel and compute-efficient; output tokens are expensive because decode is serial and bandwidth-starved. That is the entire reason input:output price ratios sit at **1:4 to 1:10**. It also means the Italian output-token premium (§2.3) lands on the expensive side of the ledger.

### 3.4 Advertised vs effective context — where most engineers hold stale beliefs

Five converging lines of evidence. Read all five; any one of them alone is dismissible, together they are not.

**1. "Lost in the middle" (Liu et al., 2023) — U-shaped positional accuracy.** Information at the start and end of context is used well; the middle is neglected. Still directionally true.

The **2026 theoretical result is the important update**: the U-shape is present **at initialization, before any training, and independently of positional encoding**. Causal masking produces logarithmic gradient concentration at the start (primacy); residual connections create an isolated anchor at the final token (recency); between them sits a "factorial dead zone" scaling roughly as `1/(H−1)!` in depth H. Verified on **untrained Qwen2 and GPT-2 with RoPE removed**, and pretraining does not overcome this topological baseline. *"Lost in the Middle at Birth: An Exact Theory of Transformer Position Bias"*, Mar 2026 — [arXiv:2603.10123](https://arxiv.org/html/2603.10123v1).

> **Position bias is architectural, not a training bug.** Put the most important retrieved chunk and the actual question **near the end of the prompt**. Do not wait for a better model to fix it — the paper's whole point is that no amount of training does.

**2. Needle-in-a-haystack (NIAH) is too easy and must not be your acceptance test.**

Origin: Greg Kamradt's Nov 2023 pressure-test of GPT-4 and Claude 2.1 — insert a sentence at varying depths in a long document, ask for it back ([thread](https://x.com/GregKamradt/status/1727018183608193393)).

**NoLiMa** (2025, [arXiv:2502.05167](https://arxiv.org/pdf/2502.05167)) removed literal lexical overlap between question and needle, forcing latent associative reasoning:

| Result | Number |
|---|---|
| GPT-4o | 99.3% short-context → **69.7% at 32K** → **56% of baseline at 128K** |
| Models scoring ≤50% of short-context baseline at 32K | **11 of 13** |
| Gemini 2.0 Flash at 128K | **16.4% of baseline** |
| **Effective context** (≥85% of baseline): Llama 3.1 70B | **~2K, against a 128K advertised window** |

The ablation is the proof: **re-adding literal keyword overlap restores >98% accuracy at 32K**, and adding visible multiple-choice options gives 87–96% at all lengths. So NIAH-style tests measure lexical matching, not comprehension.

> **A green NIAH heatmap tells you almost nothing.** Vendors publish them because they are easy to pass.

**3. Chroma's "Context Rot" (2025)** — 18 models across Anthropic / OpenAI / Google / Alibaba, holding task complexity constant while varying input length ([trychroma.com/research/context-rot](https://www.trychroma.com/research/context-rot)):

- Degradation with length is **universal, even on trivial tasks**.
- **Needle–question semantic similarity is a first-order variable.** High-similarity pairs (cosine ~0.7+) hold up; low-similarity pairs (0.4–0.5) degrade sharply with length. **Directly relevant to RAG**: paraphrased or vocabulary-mismatched queries fail at length long before literal ones do. For an Italian corpus queried in colloquial Italian against formal/normative document language, this is your exact situation.
- **A single distractor measurably degrades accuracy**, and specific distractors recur in hallucinations across different models.
- Repeated-Words replication (pure copying, 25→10,000 words): all models degrade — under-generating, misplacing the unique word, sometimes emitting words **absent from the input**. Unique words placed early are recalled better.
- Haystack **structural coherence** matters: shuffled/incoherent haystacks behave differently from coherent prose.

**4. Length itself hurts, isolated from retrieval difficulty and distractors** (EMNLP 2025 Findings, [aclanthology 2025.findings-emnlp.1264](https://aclanthology.org/2025.findings-emnlp.1264.pdf)). Models: Llama-3.1-8B-Instruct, Mistral-v0.3-7B-Instruct, GPT-4o, Claude-3.5-Sonnet, Gemini-2.0. Method: pad the context with **whitespace** or **masked tokens** — no distracting content whatsoever — and confirm retrieval succeeded.

| Measurement | Result |
|---|---|
| Degradation despite perfect retrieval, across tasks | **13.9% – 85%** |
| MMLU on Llama at 30K tokens, 97% retrieval success | **−24.2%** |
| Variable-summation on Llama, 30K, whitespace-only padding | **−48%** |
| HumanEval on Llama, 30K masked tokens | **−50%** |
| "Retrieve-then-reason" mitigation on GPT-4o | recovered up to ~4% |

This is the cleanest result in the set, because whitespace cannot distract. **Length alone is the harm.**

**5. Weaker evidence — know it, do not cite it as fact.** `UNVERIFIED` / low confidence: a Jan 2026 preprint reports a sharp "intelligence degradation" cliff for Qwen2.5-7B at **~43% of max context (~55K tokens)**, F1 0.556→0.302 (−45.5%) within a narrow 40–50% band — [arXiv:2601.15300](https://arxiv.org/html/2601.15300v1). Single 7B model, single task family, self-defined threshold, no evident peer review. **Treat the 43% figure as anecdote, not a law.**

**Synthesis:**

- **Advertised context = the API will accept it. Effective context = where quality is still acceptable for *your* task and *your* query phrasing** — and is often 5–50× smaller.
- Degradation is a **gradient, not a cliff**. Anthropic's framing is useful: an "attention budget" spread over n² pairwise relations, plus less training mass at extreme lengths — [effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).

> ⚠️ **"1M context means you can stop chunking and just paste everything" is contradicted by all four solid lines above.** It is the most consequential wrong belief in this part of the course.

**Actionable list:** measure effective context on your own corpus with your own query distribution; **put the question last**; minimise distractors (fewer, better chunks beats more chunks); use retrieve-then-reason scaffolding; and prefer compaction / note-taking / sub-agents over an ever-growing window.

---

## 4. Prompt caching — the economics that actually changed

All numbers verified from [Anthropic's prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching). Gewiss uses Anthropic models internally, so these are the mechanics that apply to you.

### 4.1 Mechanics

- The KV cache for a **prompt prefix** up to a cache breakpoint is retained and reused. **Prefix-only**: a hit requires a byte-identical prefix.
- Cacheable block types: **tools, system, messages** (text, images, documents, `tool_use`).
- **4 breakpoints** maximum, with a 20-block lookback per breakpoint.
- Cache pre-warming with `max_tokens: 0` removes first-request latency.
- ZDR-eligible: only the KV cache and cryptographic hashes are held, not raw prompts.

### 4.2 Pricing multipliers

| Operation | Multiplier vs base input | Opus 5 illustration ($5/MTok base) |
|---|---|---|
| 5-minute cache **write** | **1.25×** | $6.25/MTok |
| 1-hour cache **write** | **2.0×** | $10.00/MTok |
| Cache **read** | **0.10×** | **$0.50/MTok** |

TTL is 5 minutes by default (free, and **refreshed at no cost on each hit**) or 1 hour at the 2× write premium.

### 4.3 Minimum cacheable prefix — model-dependent, and it fails silently

| Models | Minimum prefix |
|---|---|
| Opus 5 / Fable 5 / Mythos 5 | **512 tokens** |
| Opus 4.8, Sonnet 5, Sonnet 4.6 | **1,024 tokens** |
| Opus 4.7 / 4.6 / 4.5, Haiku 4.5 | **2,048–4,096 tokens** |

> **Below the minimum you get no caching and no error.** Detect it by checking that **both** `cache_creation_input_tokens` and `cache_read_input_tokens` are 0. Log these two fields from day one; otherwise you will believe you are caching when you are not.

### 4.4 Invalidation traps

- **Any change to tool definitions invalidates the entire cache.**
- Toggling web search or citations invalidates system + messages.
- Adding or removing images invalidates messages.
- `tool_choice` changes invalidate messages.
- **A timestamp inside a cached block destroys all cache hits.** This is the classic bug — "current date: 2026-08-17 14:32" in a system prompt gives you a 0% hit rate.

Place the breakpoint **after the last identical block**.

### 4.5 What this changes economically

1. **Break-even is ~1.3 reads.** A 1.25× write plus one 0.1× read costs 1.35× for two calls, vs 2.0× uncached. **Anything reused even twice is cheaper cached.**
2. **Steady-state discount is ~10× on the cached prefix.** In most RAG and agent systems **this is a larger cost lever than changing model.**
3. **It makes static-to-dynamic prompt ordering mandatory:** `[tools] → [system] → [long stable corpus/examples] → BREAKPOINT → [retrieved chunks] → [user query]`. Any dynamic content placed early poisons everything after it. Note this ordering also happens to satisfy §3.4's "question last."
4. **It partially rehabilitates stuffing a large stable corpus into context** — but only where the corpus is stable *and* re-queried within the TTL, and only up to the effective-context limits in §3.4. **Caching does not make long context free or smart**: prefill is skipped, but decode still attends over the full length, and quality degradation is entirely unaffected by caching.
5. **It makes LLM-based preprocessing viable.** Anthropic's Contextual Retrieval costs **$1.02 per million document tokens** to generate per-chunk context — economically possible *only* because the document is cached across the many per-chunk calls ([contextual retrieval](https://www.anthropic.com/engineering/contextual-retrieval)). See [file 05](rag-and-vector-search.md) §6.
6. **Multi-turn agents:** cache the conversation prefix. The 5-minute TTL matches human think-time poorly — the **1-hour TTL is often worth its 2× write premium** for interactive internal tools where a user wanders off to a meeting.
7. `UNVERIFIED` for other vendors: OpenAI and Google both offer prompt/context caching with different mechanics (OpenAI's largely automatic with a smaller discount; Google offers explicit context caching with **storage-time billing**). Since your closed-model key is Google's, verify Google's current multipliers and storage charges directly — these numbers move.

---

## 5. Vocabulary that trips people up

### 5.1 Parameters vs weights — and why bytes is the number to track

Effectively synonymous in casual use. Pedantically: *parameters* = all learned values (weights + biases + embeddings + norm scales); *weights* = the matrix entries specifically.

"7B parameters" is a **count**. Memory is `count × bytes_per_parameter`:

| Precision | Bytes/param | A "7B model" is |
|---|---|---|
| bf16 | 2 | ~14 GB |
| int8 | ~1 | ~7 GB |
| int4 | ~0.5 | ~3.5 GB |

> **Track bytes, not parameters — plus the KV cache (§3.2).** Those two numbers determine whether a model fits your hardware and how much concurrency you get. The parameter count alone determines nothing.

### 5.2 Inference vs training

Training = forward + backward + optimizer update; needs roughly **4–6× the model's bytes** in memory (weights + gradients + Adam moments + activations); runs once. Inference = forward only, runs forever, and is where your bill lives.

**Fine-tuning is training. Prompting and RAG are inference-only.** Corollary worth internalising: a change that helps inference cost (quantization, GQA, prompt caching) is usually worth far more in production than an equivalent training-side win.

### 5.3 Base vs instruct vs reasoning

Per [Raschka](https://sebastianraschka.com/faq/docs/base-vs-instruct-vs-reasoning-model.html):

| Type | Post-training | Behaviour | Use for |
|---|---|---|---|
| **Base** | none (next-token pretraining only) | **continues** your text rather than answering it — feed it a question and get more questions | fine-tuning starting points, raw completion, representation research |
| **Instruct / chat** | SFT on instruction/response pairs + preference optimisation (RLHF/DPO) | follows instructions and formats | the default for applications |
| **Reasoning** | RL on verifiable tasks and/or distillation from another reasoning model | emits extended internal reasoning before answering | multi-step problems |

Reasoning models are better on multi-step problems but carry **higher latency and token cost**, and will happily burn budget on trivial queries. Many 2026 models are hybrid with a controllable thinking budget — Qwen3 exposes `/think` / `/no_think` plus a token budget that **force-terminates** deliberation ([arXiv:2505.09388](https://arxiv.org/html/2505.09388v1)).

> **Reasoning tokens are billed as output tokens.** A reasoning model on a classification task can cost ~20× an instruct model for the same answer. On Italian output, at the 4–5× output rate, with a +36% token premium, this compounds three ways.

### 5.4 Logits

The raw pre-softmax scores over the vocabulary at each position — one real number per vocab entry. **Everything downstream operates on logits**: temperature, top-k, top-p, constrained decoding for structured output, `logprobs`, classifier-free guidance. `logprobs` in APIs = log-softmax of logits; useful for confidence estimation and cheap classification.

### 5.5 Temperature / top-p / top-k, including the interaction order

Mechanics per [Raschka](https://sebastianraschka.com/faq/docs/temperature-topk-topp-sampling.html):

- **Temperature** rescales logits before softmax: `p_i = exp(z_i/T) / Σ exp(z_j/T)`. `T=1` unchanged; `0<T<1` sharpens toward the argmax; `T>1` flattens.
- **Top-k** keeps the k highest-scoring tokens, renormalises, samples. **Fixed** candidate count regardless of distribution shape.
- **Top-p (nucleus)** keeps the smallest prefix of the sorted distribution whose cumulative probability ≥ p. **Adaptive** candidate count — narrow when the model is confident, wide when it is not.

> **Interaction order matters and is rarely taught.** Temperature is applied **first**, so it changes cumulative probabilities and therefore changes the *size* of the top-p nucleus. Temperature does **not** change top-k's rank ordering, so it does not change top-k's candidate set. With both set, the pool is restricted by both.

`T=0` is **undefined in the formula** (division by zero). APIs interpret it as a convention for **greedy decoding** — take the argmax. And note: greedy decoding is still **not bit-reproducible** in practice, because of batching and kernel non-determinism.

Guidance: change one knob, not both. Greedy for debugging and for extraction/classification. Moderate T with top-p for open-ended text.

> **These knobs only affect diversity. They cannot add knowledge or reasoning.** A wrong answer at T=0 will not become right at T=0.9 — it will become *differently wrong*. Tuning temperature to fix a factual error is a category mistake, and a common one.

### 5.6 System prompts

A distinct role whose content is (a) placed in a privileged position by the chat template and (b) trained during post-training to outrank user instructions.

**It is a strong prior, not a security boundary.** Do not treat it as a sandbox. Operationally it is also the **most valuable thing to cache** — it is the stable prefix (§4.5), and per §2.3 it should be in English.

### 5.7 Chat templates and the `add_generation_prompt` bug

A chat template is the Jinja-ish function mapping `[{role, content}, ...]` → the exact token string the model was post-trained on. **This is a correctness issue, not cosmetics.** Mistral-7B-Instruct uses `[INST]...[/INST]`; Zephyr-7B uses `<|user|>`/`<|assistant|>`; ChatML uses `<|im_start|>`. **Using the wrong template degrades quality badly.** Details per [HF chat templating docs](https://huggingface.co/docs/transformers/main/en/chat_templating):

- **`add_generation_prompt=True`** appends the assistant-turn opener (e.g. `<|im_start|>assistant\n`). > **Without it, the model continues the user's message instead of replying to it.** This is the single most common local-inference bug, and it presents as "the model is weirdly bad" rather than as an error.
- Use **`add_generation_prompt=False`** when building **training** data — those tokens are not in the target.
- **`continue_final_message=True`** prefills an assistant turn (e.g. seeding `{"name": "` to force JSON). Mutually exclusive with `add_generation_prompt`.
- Classic second bug: `apply_chat_template(tokenize=False)` and then tokenizing with `add_special_tokens=True` → **duplicated BOS/special tokens**. Prefer `tokenize=True`.

Hosted chat APIs apply the template for you. You only touch this with local/self-hosted models, or when fine-tuning.

### 5.8 Structured output vs JSON mode

Per [OpenAI structured outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs):

| Feature | Guarantees |
|---|---|
| **JSON mode** | *syntactically valid JSON*. Nothing about fields, types, or enums. |
| **Structured Outputs** (`strict: true` + JSON Schema) | valid JSON **and schema adherence** — required fields present, enums respected |

Structured Outputs is implemented by **constrained decoding**: a grammar/FSM masks invalid tokens at the logit level. That implementation is exactly why only a **subset of JSON Schema** is supported (some keywords excluded for performance/technical reasons; `$ref` recursion *is* supported).

> **Neither guarantees semantic correctness.** Schema-valid nonsense is fully possible — the right shape with wrong values. **Validate values, not just shape.**

### 5.9 Function calling vs tool use

Same mechanism, different vocabulary — "function calling" was OpenAI's original name; "tool use" is the broader current term. The flow ([Anthropic tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)):

1. You pass `tools: [{name, description, input_schema}]`.
2. The model returns a `tool_use` block with `stop_reason: "tool_use"` — **it emits a structured request; it does not execute anything.**
3. **Your code** executes it.
4. You send back a `tool_result` keyed by `tool_use_id`.
5. The model answers.

**Client tools** (yours, plus provider-schema'd ones like bash/text_editor/memory) execute in your process. **Server tools** (web_search, web_fetch, code_execution) run on the provider's infrastructure and return results inline — no handler needed.

With `tool_choice: auto`, a tool is called "when the request maps to that tool's described capability and the answer isn't already in context"; the model answers directly for stable knowledge and conversational turns. Force with `tool_choice: {"type":"tool","tool_name":...}`.

Two practical facts that matter more than the flow:

- **Tool definitions cost context tokens on every single call**, and **changing them invalidates the prompt cache entirely** (§4.4).
- **Description quality is the main determinant of correct tool selection.** The description *is* the prompt. Write it in English (§2.3).

Structured Outputs vs function calling: use **function calling to connect to systems and data**; use **`response_format` to shape the reply to the user**.

### 5.10 Multimodality

Two generations:

- **Bolted-on** — a separate vision encoder projects image patches into the LLM's token space (LLaVA-style).
- **Native / omni** — a single model pretrained jointly on interleaved text/image/audio/video tokens. The 2026 default.

Engineer-relevant fact: **non-text inputs are billed as tokens** via modality-specific tokenizers, and the conversion rate matters a lot. As a concrete example of the scope, Gemini Embedding 2 accepts text, up to 6 images, ≤120 s video, ≤180 s audio and ≤6 PDF pages in one call ([Gemini embeddings docs](https://ai.google.dev/gemini-api/docs/embeddings)).

`UNVERIFIED`: exact image→token rates per model in 2026. **Check each provider's vision pricing page** — they differ by an order of magnitude, and by image resolution / detail setting. If you plan to run Gewiss technical drawings or wiring diagrams through a vision model, price a representative sample before committing.

---

## 6. Mixture-of-Experts

### 6.1 What MoE changes about "parameters → capability"

Classical dense scaling ties **capacity** (parameters) to **cost** (FLOPs per token) — you cannot have one without the other. MoE **decouples them**: replace each FFN with N expert FFNs plus a router that activates only k of them per token. Total parameters grow; per-token FLOPs do not.

So "how big is the model?" splits into **three numbers you must quote separately**:

| Number | Determines |
|---|---|
| **Total parameters** | memory/VRAM you must hold — **all experts must be resident (or paged) even though most are idle** |
| **Active parameters** | FLOPs per token → compute cost, and partly latency |
| **Attention parameters** | shared by all tokens; unaffected by expert routing |

Verified examples ([Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)):

| Model | Total | Active | Experts | Activated/token |
|---|---|---|---|---|
| Qwen3-30B-A3B | 30B | **3B** | 128 | 8 |
| Qwen3-235B-A22B | 235B | **22B** | 128 | 8 |
| DeepSeek-V3 ([arXiv:2412.19437](https://arxiv.org/html/2412.19437v1)) | 671B | **37B** | — | — |

Qwen3 uses fine-grained expert segmentation, **no shared experts** (a change from Qwen2.5-MoE), and a global-batch load-balancing loss to force specialisation. Headline efficiency claim: with identical pretraining data, **Qwen3 MoE base models match dense Qwen3 base models using ~1/5 the activated parameters.**

Current architecture comparison table: [Raschka's big LLM architecture comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison).

### 6.2 The √(total × active) folk heuristic and its limits

The widely used rule of thumb: an MoE behaves roughly like a dense model of **√(total × active)** parameters. So 235B/22B ≈ √(235·22) ≈ **72B-dense-equivalent**.

`UNVERIFIED` **as a precise law.** It is a folk approximation with no paper behind it, and the actual scaling-law literature is more careful — and more useful:

- **There is an optimal sparsity, and maximising sparsity is wrong.** *"Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for MoE"* ([arXiv:2501.12370](https://arxiv.org/abs/2501.12370)) finds that under a given parameter-size or total-compute constraint there is an **interior optimum** that improves both training efficiency and quality.
- Joint MoE scaling laws showing MoE can be **memory** efficient too: [arXiv:2502.05172](https://arxiv.org/pdf/2502.05172).
- Efficiency-leverage scaling laws: [arXiv:2507.17702](https://arxiv.org/abs/2507.17702).
- A direct "can MoE beat dense under strictly equal resources?" study: [arXiv:2506.12119](https://arxiv.org/html/2506.12119v1).

Use √(total × active) to sanity-check a vendor claim, never to justify a procurement decision.

### 6.3 Serving consequences

- **Serving is memory-heavy, compute-light.** You buy VRAM for *total* params and get the throughput of *active* params. This flips the usual cost calculus: MoE is excellent for high-batch cloud serving and awkward for single-user local inference — which is the configuration most course exercises and most laptops are in.
- **Latency is not simply proportional to active params.** Expert routing introduces all-to-all communication in distributed serving, and at low batch sizes you are memory-bandwidth-bound reading experts you barely use.
- **Load balancing is a real failure mode.** Without a balancing loss, routers collapse onto a few experts; capacity is wasted and quality drops. Both Qwen3 and DeepSeek use explicit balancing objectives.
- **Fine-tuning MoE is harder than dense.** Routing can shift, experts can be starved by small datasets, and LoRA-on-MoE has its own gotchas. **If a course exercise involves fine-tuning, prefer a dense model.**

> ⚠️ **Reading-comprehension trap: when a vendor says "1T parameters," ask for *active*.** Marketing quotes total; your bill tracks active; your hardware requirement tracks total. Three different numbers, one headline.

### 6.4 The claim that has flipped

⚠️ **"MoE models are lower quality per parameter — a cheap trick"** was defensible in 2022 and is wrong in 2026. Essentially all frontier open-weight models are MoE: DeepSeek-V3, the Qwen3 flagship, Kimi, Llama 4, GLM. And HybridNorm reports gains on **sparse as well as dense** variants ([arXiv:2503.04598](https://arxiv.org/abs/2503.04598)) — MoE is now the mainline architecture that new techniques are validated *against*, not a side branch.

---

## 7. Corrections table — claims that are wrong or outdated

| Claim | Status |
|---|---|
| "Attention needs O(n²) **memory**" | **Wrong since 2022.** [FlashAttention](https://arxiv.org/abs/2205.14135) computes exact attention with linear memory. The surviving quadratic is **prefill compute**. |
| **"Italian costs ~2× English tokens"** | **Outdated.** The 2.01×/2.18× figures are from a 2023 paper ([arXiv:2305.15425](https://arxiv.org/pdf/2305.15425)) on GPT-2 and cl100k. The 2026 parallel-corpus figure is **1.67 vs 1.23 tok/word ≈ +36%** ([arXiv:2605.24718](https://arxiv.org/html/2605.24718)). Budget +30–40%, and measure on your model. |
| "Up to 15× token-cost difference between languages" | **Same stale source.** Worst 2026 European-language case is Greek/Maltese ≈ 2.5× English. |
| "SentencePiece is a tokenization algorithm" | **No** — it is a *library* running BPE or Unigram on raw text with whitespace as a real character. |
| "Token counts are roughly the same across models, so any tokenizer will do for estimates" | **False.** Vocab size, corpus mix, pre-tokenization regex, byte- vs char-level base, and special tokens all differ. Italian fertility spans **1.67 (Llama-3.1) to 2.05 (Mistral-7B) on the same corpus**. |
| "ALiBi extrapolates, so it solves long context" | **Conflates perplexity with retrieval.** RoPE + YaRN won in practice; see NoLiMa for why extrapolated perplexity means little. |
| "1M context means you can stop chunking and paste everything" | **Contradicted by four independent lines**: NoLiMa (effective context ~2K for a 128K Llama 3.1 70B), Context Rot (18 models), EMNLP 2025 whitespace padding (−24% MMLU at 30K *with 97% retrieval success*), and the 2026 position-bias theory. |
| "A green needle-in-a-haystack heatmap proves long-context capability" | **No.** Re-adding literal keyword overlap restores NoLiMa accuracy to >98% at 32K — NIAH measures lexical matching. |
| "Lost in the middle is a training artefact that better models will fix" | **Architectural.** Present at initialization, untrained, **with RoPE removed** ([arXiv:2603.10123](https://arxiv.org/html/2603.10123v1)). Put the question last, permanently. |
| "Prompt caching makes long context free" | **Half true.** ~10× cheaper reads on the cached prefix; **prefill is skipped, decode is not, and quality degradation is entirely unaffected.** |
| "Set temperature lower to make the model more accurate" | **Category error.** Sampling knobs affect only diversity. A wrong answer at T=0 becomes differently wrong at T=0.9. |
| "JSON mode guarantees my schema" | **No.** JSON mode guarantees syntax only. Structured Outputs (`strict: true` + schema) guarantees schema — and **neither guarantees semantics**. |
| "The model executes the tool" | **No.** It emits a `tool_use` request; **your code executes it** and returns a `tool_result`. |
| "T=0 gives reproducible output" | **Not bit-reproducible** — batching and kernel non-determinism. Also, T=0 is undefined in the softmax formula; it is an API convention for greedy decoding. |
| "MoE is a cheap trick with worse quality per parameter" | **Flipped.** Nearly all 2026 frontier open-weight models are MoE, and new techniques are validated on sparse variants. |
| "The model is 1T parameters" | **Ask for *active*.** Total → your VRAM. Active → your bill. Attention params → neither. |
| "MoE ≈ dense model of √(total × active)" | **Folk heuristic, `UNVERIFIED` as a law.** The literature finds an *interior* optimal sparsity ([arXiv:2501.12370](https://arxiv.org/abs/2501.12370)); maximising sparsity is not optimal. |
| "Decoder-only is simply the better architecture" | **Too strong.** After instruction tuning, encoder-decoder matches or beats decoder-only downstream with better inference efficiency ([arXiv:2510.26622](https://arxiv.org/html/2510.26622v1)). Decoder-only wins the *pretraining-scaling + ICL* regime. |

---

## 8. Items I could not verify

- **First-party Italian tokenization measurements.** Tokenizer BPE-file downloads were blocked in the research environment, so every Italian fertility figure here is from the two cited papers, not measured. **Run the §2.5 script on Gewiss documents** — this is the highest-value 10 minutes in these notes.
- **Gemini per-model context and output limits for 2026.** The models doc page listed the lineup but not the token limits. Check [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) — this one matters, since Google is your only closed-model key.
- **OpenAI and Google prompt/context-caching multipliers as of Aug 2026.** Mechanics differ from Anthropic's (Google bills storage time); the Anthropic numbers in §4 are verified, the others are not.
- **Per-model image→token conversion rates in 2026.** Differ by an order of magnitude and by resolution/detail setting.
- **The ~43%-of-context "intelligence cliff"** ([arXiv:2601.15300](https://arxiv.org/html/2601.15300v1)) — single 7B model, single task family, self-defined threshold, no evident peer review. Anecdote.
- **√(total × active) as a quantitative MoE-to-dense conversion.** No primary source; folk heuristic only.
- **Qwen3-235B-A22B attention-parameter count** and per-model attention-vs-expert splits generally — the technical report gives total/active and expert counts, not a clean three-way breakdown per model.

**Where a number drives a decision, re-check it against the provider's own current page.** Pricing multipliers, minimum cacheable prefixes, and context limits all move on a timescale of months.
