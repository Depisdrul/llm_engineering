<!-- Generated copy. Source: .knowledge-extraction/improvements-from-chrome-session/05ragandvectorsearch.md
     Edit the source file; this copy is overwritten on every extraction run. -->

# Week 5 — RAG, embeddings, and vector search

**How this was built:** written from primary sources — arXiv papers, ACL Anthology, official OpenAI/Google/Voyage/Cohere/HuggingFace/FAISS/Chroma/MTEB/RAGAS docs, Anthropic and Microsoft engineering blogs — not from lecture transcripts. Where the source material contradicts what a 2024/2025-era course teaches, that is flagged. Sources are linked inline; unconfirmed claims are marked `UNVERIFIED`.

**Read the corrections table (§10) first.** Four techniques this week is likely to teach as best practice have negative published evaluations against them, and one of them (semantic chunking) has two independent ones.

**Companion file:** [01-llm-foundations.md](llm-foundations.md) — §3.4 (effective context), §4 (prompt caching) and §2.3 (Italian tokenization) are load-bearing for the decisions below and are cross-referenced rather than repeated.

---

## 0. The one-paragraph version

Almost every measured improvement in RAG comes from boring things: parse the documents properly, split them recursively, retrieve wide, **rerank with a cross-encoder**, and pass few chunks. The techniques that sound clever mostly lose: semantic chunking has **two independent negative evaluations**, HyDE came **last of ten methods** on a 23,088-question benchmark and lost to plain BM25, and cluster-based methods underperform in both chunking studies. Meanwhile hybrid BM25+dense with a reranker delivered **0.816 Recall@5 vs 0.587 for dense alone** — a single API call with no reindexing, and the largest measured win available. For you specifically, two Italian facts dominate everything else: **Italian retrieval quality is roughly half of English** (0.282 nDCG@10 vs 0.638 on the same models), and **a 560M open-weight model came within 0.003 nDCG of Google's frontier embedding API at one-seventh the latency** — so the expensive closed model buys you almost nothing on Italian, and you should plan for extra retrieval depth, hybrid search and reranking as *baseline*, not as optimisations. Finally: if your corpus is under ~50K tokens, **do not build RAG at all** — put it in a cached long-context prompt.

---

## 1. Embeddings

### 1.1 What an embedding model does

Maps variable-length text to a fixed-length dense vector such that **semantic similarity ≈ geometric proximity**. Architecturally: usually an encoder-only (bidirectional) transformer plus a pooling step (mean-pool, or the `[CLS]`/`[EOS]` token), then optionally a projection. Decoder-only LLMs are now also used as embedding backbones (Qwen3-Embedding, NV-Embed, Stella) with last-token pooling.

Training is contrastive — InfoNCE with hard negatives, on `(query, positive, negatives)` triples. **That training shape is the reason for the next subsection.**

### 1.2 The query/passage prefix bug — the most common silent RAG failure

Because the model was trained on asymmetric (query, passage) pairs, **the query and the document must be encoded the way the model was trained.** Many models require asymmetric prefixes: `query: ` / `passage: ` for the E5 family, task instructions for Qwen3-Embedding.

> **Forgetting the prefix costs several points of nDCG and produces no error message.** Nothing crashes, nothing logs, retrieval is merely mediocre. It is the single most common silent bug in production RAG, and it is invisible unless you compare against a gold set.

Instruction-aware embedders make this a lever rather than just a trap: Qwen3-Embedding reports **1–5% gains** from a task-appropriate instruction, and the model card explicitly recommends customising it per scenario, task **and language** — [Qwen3-Embedding-8B card](https://huggingface.co/Qwen/Qwen3-Embedding-8B). For an Italian corpus, that means the instruction is a tunable you should actually sweep.

**Checklist for any new embedding model:** read the model card for (a) required prefixes or instruction format, (b) pooling method, (c) whether outputs are L2-normalised, (d) max input tokens. All four are silent failures if you get them wrong.

### 1.3 Dimensionality and Matryoshka truncation

Typical range 384 → 4096. Verified current numbers:

| Model | Default / range | Max input tokens | Notes |
|---|---|---|---|
| `text-embedding-3-small` | 1536 | 8,192 | `dimensions` param truncates |
| `text-embedding-3-large` | 3072 | 8,192 | **truncated to 256 dims still beats full 1536-dim ada-002** ([OpenAI](https://developers.openai.com/api/docs/guides/embeddings)) |
| `gemini-embedding-2` | **128–3072**, recommended 768/1536/3072 | **8,192** | 100+ languages, multimodal (text/image/video/audio/PDF), explicit MRL ([Google](https://ai.google.dev/gemini-api/docs/embeddings)) |
| `gemini-embedding-001` | 3072 / MTEB 68.17 @1536, 67.99 @768, 67.55 @512 | 2,048 | note the small drop from 1536→512 |
| `Qwen3-Embedding-8B` | up to 4096, custom 32–4096 | **32K** | 100+ languages incl. code, Apache-2.0 ([card](https://huggingface.co/Qwen/Qwen3-Embedding-8B)) |
| `voyage-4-large` / `-4` / `-4-lite` / `voyage-code-4` | **256–2048** | **32,000** | `voyage-law-2` 16K/1024; `voyage-finance-2` 1024; `voyage-4-nano` open-weight ([docs](https://docs.voyageai.com/docs/embeddings)) |

**Why truncation works: Matryoshka Representation Learning (MRL)** — the model is trained so that *prefixes* of the vector are themselves valid embeddings at multiple nested dimensions ([arXiv:2205.13147](https://arxiv.org/html/2205.13147)). A 2025 critique/improvement specifically for retrieval compression: SMEC, [arXiv:2510.12474](https://arxiv.org/abs/2510.12474).

**Cost of dimensionality is linear-to-superlinear in the vector store:** memory ≈ `n_vectors × dim × 4 bytes` (fp32) plus index overhead. **1M docs × 3072 dims × 4B ≈ 12.3 GB** before the HNSW graph. FAISS HNSW memory is `(d × 4 + M × 8)` bytes per vector ([FAISS index guidelines](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)).

> **Truncating 3072 → 768 on an MRL model is a 4× memory saving for typically <1 point nDCG. It is usually the correct first optimisation** — before switching stores, before quantization, before tuning `ef`.

Note the tension with long-context findings ([01-llm-foundations.md](llm-foundations.md) §3.4): **higher dimension is not better retrieval beyond a point**, and it directly increases index cost and query latency.

### 1.4 Cosine vs dot product vs Euclidean — the equivalence, and where it bites

- **Dot product** `a·b = Σ aᵢbᵢ` — magnitude-sensitive. Rewards long vectors, which in some models correlates with document length or token frequency (usually an artefact, not signal).
- **Cosine similarity** `a·b / (‖a‖‖b‖)` — dot product with magnitudes divided out; pure angle. Range [−1, 1].
- **Euclidean (L2)** `‖a − b‖` — a *distance*, so smaller is better; invert for similarity.

**The equivalence, stated precisely.** If all vectors are L2-normalised (`‖a‖ = ‖b‖ = 1`), then:

- `cosine(a,b) = a·b` — the denominator is 1, so **cosine and dot product produce identical scores**; and
- `‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b = 2 − 2(a·b)` — a strictly monotone **decreasing** function of the dot product, so **L2 produces the identical *ranking*** (different scores, same order).

> **On normalised embeddings, all three metrics give the same top-k.** Use dot product: it is the cheapest — no square roots, no division — and maps directly to a SIMD/BLAS inner product.

OpenAI states this explicitly for their models: embeddings are normalised to length 1, so cosine can be computed as a plain dot product, and cosine rankings match Euclidean rankings exactly ([docs](https://developers.openai.com/api/docs/guides/embeddings)).

**Where it bites in practice — this is the part worth remembering:**

1. **Not all models return normalised vectors.** Some HF models do not; some do only with `normalize_embeddings=True` in sentence-transformers.
2. **If you truncate an MRL embedding, you must re-normalise.** The prefix of a unit vector is not a unit vector. Truncate-then-forget is a real and quiet bug.
3. **Your vector store's configured metric must match how the model was trained.** A model trained with a cosine objective, queried with unnormalised dot product, silently degrades. Chroma, FAISS, pgvector and Qdrant all happily let you pick the wrong one.
4. In FAISS: `IndexFlatIP` + normalised vectors = cosine. `IndexFlatL2` on normalised vectors = the same ranking. On **un**normalised vectors these diverge, and one of them is wrong for your model.

⚠️ The long blog debates about "cosine vs dot product for RAG" are a **non-issue** for normalised embeddings — which is nearly all of them. **Spend the effort on normalisation hygiene and metric configuration instead.**

---

## 2. MTEB / MMTEB, and how to actually choose a model

**MTEB** is the standard embedding benchmark. **MMTEB** (ICLR 2025, [arXiv:2502.13595](https://arxiv.org/abs/2502.13595)) expanded it enormously: **>500 quality-controlled tasks across 250+ languages**, adding instruction-following, long-document retrieval and code retrieval, plus an inter-task-correlation downsampling method that keeps rankings stable at lower compute.

> **MMTEB's headline finding is the one to internalise: the best publicly available model was `multilingual-e5-large-instruct` at 560M parameters, beating multi-billion-parameter LLM-based embedders overall. Size is not the signal.**

**Current benchmark variants** (official list: [docs.mteb.org](https://docs.mteb.org/overview/available_benchmarks/)): MTEB(eng, v1/v2), MTEB(Multilingual, v1/v2), ~13 language-specific variants including **MTEB(Europe, v1)** — the slice you want for Italian — plus domain variants (Medical, Law, Code, ChemTEB), **BEIR**, **BRIGHT** (reasoning-intensive retrieval), **LongEmbed**, **FollowIR** (instruction following), **ViDoRe/JinaVDR** (visual document retrieval), NanoBEIR.

### 2.1 The honest selection procedure

1. **Filter by hard constraints first.** Licence (Qwen3-Embedding Apache-2.0 vs `llama-embed-nemotron-8b` non-commercial), **max input tokens** (2,048 vs 8,192 vs 32,000 — *this constrains your chunk size*, see §4.3), self-host vs API, data residency, dimension budget.
2. **Look at the retrieval sub-score for your language, not the overall average.** The overall MTEB score averages retrieval, classification, clustering, STS, reranking and summarisation. A model that is excellent at clustering and mediocre at retrieval can outrank a retrieval specialist. Modal's caveat is well put: the overall score "should not be thought of as the whole story," and the best overall model is not always the right choice for your workload ([Modal](https://www.modal.com/blog/mteb-leaderboard-article)).
3. **Assume leaderboard contamination.** MTEB is public and heavily optimised against; several models are trained on task-adjacent data. **Treat rank differences under ~1 point as noise.**
4. **Build a 50–200 query gold set from your own corpus and measure recall@k yourself.** Non-negotiable, and it takes about a day. Every serious comparison found in research concluded that **leaderboard order did not survive contact with a specific domain corpus.**
5. **Consider latency explicitly** — see §3, where a small local model was 7× faster at equal Italian quality.

### 2.2 Current strong embedding models (as of Aug 2026)

| Model | Why it's on the list | Licence/access |
|---|---|---|
| **Gemini Embedding 2** | 100+ languages, 8,192 tokens, 128–3072 MRL dims, multimodal — [docs](https://ai.google.dev/gemini-api/docs/embeddings) | API (you already hold this key) |
| **voyage-4-large / -4 / -4-lite** | 32K context, 256–2048 dims, positioned as best general-purpose multilingual retrieval; **voyage-code-4** for code — [docs](https://docs.voyageai.com/docs/embeddings) | API; `voyage-4-nano` open-weight |
| **Qwen3-Embedding** (0.6B/4B/8B) | 32K context, up to 4096 dims, 100+ languages + code, instruction-aware — [card](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | **Apache-2.0** |
| **OpenAI text-embedding-3-large/small** | 3072/1536 dims, 8,192 tokens; MTEB 64.6 / 62.3 per OpenAI's own docs. Several generations old and mid-pack, but cheap, ubiquitous and normalised | API |
| **BGE-M3** | **dense + sparse + ColBERT-style multi-vector in one model**, 8K context — genuinely useful because it gives you hybrid search from a single model | open |
| **multilingual-e5-large-instruct** | 560M, MMTEB's best public model overall; the Italian result in §3 | open |
| **Cohere embed** (multilingual v3/v4 lineage) | pairs with their reranker | API; `UNVERIFIED` on exact current v4 specs |
| **embeddinggemma-300m**, **stella_en_1.5B_v5**, **llama-embed-nemotron-8b** | small/on-device; strong English; strong but **non-commercial** respectively | mixed |

> ⚠️ **Freshness caveat.** The Qwen3-Embedding-8B card claims **#1 on MTEB multilingual with 70.58 — "as of June 2025,"** which is 14 months stale. `UNVERIFIED` for August 2026 rankings: the live leaderboard could not be scraped (the HF Space renders client-side). **Check [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard) yourself**, filtered to MTEB(Multilingual, v2) or MTEB(Europe, v1), Retrieval task type. Any specific ranking written into course notes is wrong within a quarter. **Learn the procedure, not the snapshot.**

---

## 3. Italian retrieval — the section that should change your architecture

This is the most directly useful result in the whole cluster. **"Benchmarking Google Embeddings 2 against Open-Source Models for Multilingual Dense Retrieval and RAG"** (2026) built **IT-RAG-Bench** — **3,200 Italian passages** (Italian Wikipedia, public-administration FAQs, and the **Italian civil code**) with **640 queries** — and compared six models: [arXiv:2605.23618](https://arxiv.org/html/2605.23618).

| Model | English BEIR avg nDCG@10 | **Italian IT-RAG-Bench nDCG@10** | Latency |
|---|---|---|---|
| Google Embeddings 2 (GE2) | 0.638 | **0.282** | 231.6 ms |
| multilingual-e5-large (mE5-L) | 0.546 | **0.279** | **31 ms** |
| E5-large | 0.538 | 0.262 | — |
| BGE-M3 | 0.437 | — | — |
| LaBSE | 0.188 | — | — |

**Four things to take from this table.**

1. **Absolute Italian retrieval quality (0.28 nDCG@10) is far below English (0.64) — even for the best model.** The multilingual gap is not closed. > **An Italian-corpus RAG system needs more retrieval depth, hybrid search and reranking than an equivalent English system just to reach parity. Plan for it in the architecture rather than discovering it in evaluation.** Concretely: start at `k=50–100` retrieved, always rerank, always add BM25, and do not treat any of those as optional Tier-2 optimisations.

2. **A 560M open-weight model came within 0.003 nDCG of a frontier commercial API on Italian, at 1/7 the latency.** For Italian, the expensive model buys essentially nothing. This is directly relevant to your situation: your only closed-model key is Google's, and **GE2 is the Google model that this benchmark says is not worth its latency on Italian.** Self-hosting `multilingual-e5-large` (or `multilingual-e5-large-instruct`, MMTEB's overall best public model) is both cheaper and faster, keeps embedding traffic inside your boundary, and costs you 0.003 nDCG.

3. **English leaderboard rank does not transfer.** BGE-M3 and LaBSE rank very differently on English vs multilingual. **Rank by the multilingual/Europe leaderboard slice**, per §2.1 step 2.

4. ⚠️ **LaBSE (0.188 BEIR) is a bitext-mining / sentence-alignment model, not a retrieval model.** It appears in "multilingual embeddings" lists constantly and is a **bad default for RAG**. If a tutorial hands you LaBSE for a multilingual RAG demo, that is a red flag for the whole tutorial.

### 3.1 Additional Italian notes

- Italian is a **high-resource language** and is well covered by all major multilingual models — the situation is far better than for, say, Maltese. But "covered" ≠ "English-parity," and the table above quantifies the gap.
- **MTEB has an explicit MTEB(Europe, v1) slice. Use it.**
- **Cross-lingual retrieval (English query → Italian docs, or vice versa) is a separate capability** from multilingual monolingual retrieval. If Gewiss users query in Italian against a mixed IT/EN corpus — English datasheets, Italian manuals, English standards, Italian internal procedures — **test that case specifically.** Many models degrade sharply on it, and the degradation does not show up in a monolingual benchmark.
- **Domain matters more than language.** Legal/administrative Italian (IT-RAG-Bench's civil-code subset) is harder than Italian Wikipedia. Normative and technical Italian — CEI/IEC standard language, installation procedures, certification text — sits at the hard end.
- Compounding factor from [01-llm-foundations.md](llm-foundations.md) §3.4: **Context Rot found that low query–passage semantic similarity degrades sharply with context length.** Colloquial Italian queries against formal normative Italian documents is exactly the low-similarity regime, at exactly the length where it hurts most. This is an argument for **fewer, better chunks** rather than a large `k`.
- Token-side compounding from §2.3 of the foundations file: **Italian corpora hit the context ceiling ~35% sooner** than English, so the same `k` costs you ~36% more prompt tokens.

---

## 4. Chunking

### 4.1 The strategies

1. **Fixed-size** — split every N tokens/characters with O overlap. Dumb, fast, deterministic, and the baseline that keeps winning.
2. **Recursive character splitting** — try separators in descending order of semantic strength, falling back only when a piece is still too big. LangChain's default separator list is `["\n\n", "\n", " ", ""]` — paragraph, then line, then word, then character. Effectively "fixed-size that respects natural boundaries when it can." **This is the correct default.**
3. **Semantic chunking** — embed sentences, split where consecutive-sentence embedding distance exceeds a threshold (breakpoint-based), or cluster semantically similar sentences (clustering-based).
4. **Document-structure-aware** — split on the document's own hierarchy: Markdown headings, HTML tags, PDF section/heading detection, code AST (function/class boundaries), spreadsheet rows, slide boundaries. Usually implemented as structure-first, then recursive-split anything oversized, with the heading path carried into metadata.
5. **Late chunking** ([arXiv:2409.04701](https://arxiv.org/abs/2409.04701)) — invert the order: embed the **whole long document's tokens first**, then pool into chunks *after* the transformer. Each chunk embedding therefore carries whole-document context. Works on most long-context embedding models **without retraining**.
6. **Contextual chunking** — an LLM writes a 50–100-token situating preamble per chunk before embedding. See §6.

### 4.2 What the evidence says about semantic chunking

⚠️ **This is the biggest live myth in RAG tutorials.** Two independent studies:

**"Is Semantic Chunking Worth the Computational Cost?"** (NAACL 2025 Findings, [aclanthology 2025.findings-naacl.114](https://aclanthology.org/2025.findings-naacl.114.pdf)). Compared fixed-size vs breakpoint-semantic vs clustering-semantic across **10 document-retrieval datasets, 5 evidence-retrieval datasets from RAGBench**, plus answer generation, primarily on F1@5.

- Conclusion: the computational costs of semantic chunking **are not justified by consistent performance gains**.
- Fixed-size often **matched or beat** semantic on real-world documents.
- Semantic chunking helped mainly on **synthetically stitched documents with artificially high topic diversity** — i.e. precisely the setup that makes it look good in a demo.
- Crucially: **embedding-model quality mattered more than chunking strategy.**

**"Evaluating Chunking Strategies for RAG on Academic Texts"** (July 2026, [arXiv:2607.01852](https://arxiv.org/html/2607.01852v1)). 13 academic theses, RAGAS evaluation, fixed-size (150 words / 15-word overlap) vs recursive (150 words) vs cluster-based semantic.

- Recursive and fixed-size comparable (median Context F1 ≈ 0.5 and 0.3).
- **Recursive had the tightest interquartile range** — the most *consistent* strategy, which matters more than a median.
- **Cluster-based semantic underperformed: median Answer Quality Score 0.40 vs 0.65.**
- Verdict: semantic chunking yielded no consistent improvement and adds computing complexity.
- Their **biggest actual problem was preliminary pages and formatting artefacts** polluting retrieval — not the chunker.

> **Use recursive splitting. Spend the saved effort on (a) document parsing and cleanup, (b) a better embedding model, (c) reranking.** Semantic chunking is a plausible-sounding idea with two negative evaluations against it.

### 4.3 Chunk size and overlap — the actual evidence

**"Rethinking Chunk Size for Long-Document Retrieval"** ([arXiv:2505.21700](https://arxiv.org/html/2505.21700v2)) tested 64/128/256/512/1024 tokens across NarrativeQA, NQ, NewsQA, COVID-QA, TechQA and SQuAD, with two embedders (Stella, decoder-based; Snowflake-Arctic, encoder-based):

| Dataset shape | Result |
|---|---|
| **SQuAD** (short, factoid, answer-local) | **best at 64 tokens — 64.1% Recall@1.** Small chunks win. |
| **NarrativeQA** (long, dispersed answers) | 64→1024 tokens improved Recall@1 from **4.2% → 10.7%** — a 2.5× relative gain from *larger* chunks. |
| Embedder interaction | Stella gained **5–8% recall** over Snowflake with 512–1024 chunks on long documents; Snowflake was better with small chunks. |

> **There is no universal optimal chunk size.** The determinants are (i) **answer locality** — is the answer in one sentence or spread across pages, (ii) document structure, (iii) embedding-model architecture.

**Sane starting point: 512 tokens with ~10–15% overlap, recursive splitter** — then sweep 256/512/1024 against your own gold set. For reference points: Anthropic used **800-token chunks** for Contextual Retrieval; the academic-texts paper used **150 words**.

**The tradeoff axes:**

| | Small chunks | Large chunks |
|---|---|---|
| Embedding quality | precise — one vector, one idea → higher retrieval precision | **topic dilution** — the vector becomes a blurry average of several topics and is close to nothing |
| Context | fragment reasoning; lose antecedents ("*l'azienda*" — which one?) | self-contained, preserve context |
| Retrieval | need to retrieve more of them | fewer needed |
| Index cost | larger index | cheaper index |
| Generation | — | burn context on irrelevant surrounding text, which **measurably degrades generation** (foundations §3.4) |

**Overlap** cheaply insures against splitting an answer across a boundary. Its cost is index size plus duplicate hits in top-k (**dedupe before assembling context**). **10–20% is the practical range; >25% is usually waste. Zero overlap is a real risk with fixed-size splitting.**

> **A hard constraint people forget: your chunk must fit the *embedding model's* input limit, not the LLM's.** 2,048 tokens (`gemini-embedding-001`) vs 8,192 (OpenAI 3-large, Gemini Embedding 2) vs 32,000 (Voyage 4, Qwen3-Embedding). **Silent truncation past the limit is a classic bug** — the tail of the chunk is simply not embedded, so it is unretrievable while still sitting in your database looking fine.

### 4.4 What actually goes wrong with bad chunking

- **Split answers** — the answer spans a boundary; neither half scores well enough to retrieve, and if one does, the generator sees half an answer and confabulates the rest.
- **Orphaned references** — pronouns, "as described above," "the aforementioned clause," "see Table 3" with no Table 3 in the chunk. This is the specific problem that contextual retrieval and late chunking exist to solve.
- **Topic dilution in oversized chunks** — the vector sits between several topics and is close to nothing. Manifests as *"retrieval returns vaguely on-topic documents but never the right one."*
- **Table and code destruction** — a table split mid-row, or a header separated from its rows, is worse than useless: it is **confidently wrong data**. Code split mid-function loses the signature. This is the #1 argument for structure-aware splitting on those content types specifically. For Gewiss technical documentation — product tables, wiring specifications, parameter tables — this is likely your dominant failure mode.
- **Boilerplate domination** — headers/footers/nav/copyright/legal preamble repeated in every chunk make all chunks mutually similar and crowd out signal. The academic-texts paper found exactly this. **Clean before chunking.**
- **Duplicate near-identical chunks** eating your top-k with the same information — very common with overlap plus versioned documents — starving the generator of diversity.

**Fast diagnostic:** if **recall@20 is good but recall@3 is bad** → chunk sizing/precision problem, add reranking. If **recall@50 is also bad** → the information is not retrievable at all: parsing, chunking, or embedding-model problem.

### 4.5 Metadata attachment — underrated, high ROI

Attach at ingest, because **you cannot recover it later.**

- **Provenance**: source URI, document title, version, page/slide number, section-heading path (`Manuale > Sicurezza > 4.2 Procedure`), author, timestamps.
- **Filter keys**: document type, department, language, product/version, **access-control labels**, validity dates.
- **Structural**: `chunk_index`, `prev_chunk_id`/`next_chunk_id`, `parent_doc_id` (enables parent-document / small-to-big retrieval, §6), and the raw untruncated text.

> **Tenant and ACL labels must be a *filter*, never a post-hoc trim. An unfiltered retrieval that you filter afterwards is a data leak** — the wrong tenant's content has already been embedded into a prompt, logged, and possibly cached. Filter in the query.

Why metadata pays:

1. **Pre-filtering collapses the search space** — a metadata filter is usually a bigger recall win than any embedding upgrade. (Caveat: filtered ANN search on HNSW can degrade recall badly if the filter is very selective — §5.5.)
2. **Citations require it.** Without provenance you cannot show sources, and the system is unverifiable and therefore untrustworthy.
3. **Recency and version disambiguation** — **the single most common production RAG failure is confidently answering from a superseded document.** For a manufacturer with revised standards, product generations and updated installation procedures, this is the failure that will actually hurt.
4. **Prepending a compact metadata header into the embedded text** (title + section path) is a cheap approximation of contextual retrieval, and often recovers most of the orphaned-reference loss for free.

---

## 5. Vector stores and indexing

### 5.1 Exact vs approximate nearest neighbour

- **Exact (brute force / flat)**: compute similarity against every vector. **Recall = 100% by definition.** O(n·d) per query. On modern SIMD this is genuinely fast to surprisingly large n — tens of thousands to low hundreds of thousands of vectors is comfortable.
- **ANN**: trade a small recall loss for orders-of-magnitude speedup. **Every ANN index has a knob that trades recall for latency.**
- **FAISS's own guidance is to use exact search more than people do**: use `Flat` "for exact results or few searches (<1000–10000)" ([FAISS index guidelines](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)).

> ⚠️ **"You need a vector database" is over-applied.** For a corpus of a few thousand documents — which describes most course projects and a great many production systems — a flat index (or `pgvector` without an index, or a numpy matmul) is **exact, simpler, has no recall risk, and supports arbitrary filtering for free**. Introduce ANN when you have *measured* a latency problem.

### 5.2 HNSW

**Hierarchical Navigable Small World**: a multi-layer proximity graph. Upper layers are sparse long-range "express lanes"; you greedily descend to progressively denser layers, ending with a fine-grained search at layer 0.

| Parameter | What it does | When it costs you |
|---|---|---|
| **M** | links per node (FAISS range 4–64) | higher M = better recall, **more memory**, slower build |
| **efConstruction** | candidate-list size at build time | higher = better graph quality, **much slower build**, no query-time cost |
| **ef / efSearch** | candidate-list size at query time | **this is the recall/latency dial you actually tune in production.** Raise for recall, lower for latency. Must be ≥ k |

Memory: `(d × 4 + M × 8)` bytes per vector — **the graph itself is a real cost on top of the vectors.**

Properties: best recall/latency curve of the mainstream options; supports **incremental insertion** (crucial for live corpora); **deletion is awkward** (usually tombstoning plus periodic rebuild); **no GPU support in FAISS**; RAM-hungry.

> **HNSW is the default index in Chroma, Qdrant, Weaviate, Milvus, Elasticsearch and pgvector (`hnsw`).** If you are not sure what index you are using, it is probably HNSW — which means `ef` is probably your untouched dial.

Still an active research area: adaptive/distribution-aware exploration to set `ef` per query rather than globally — [arXiv:2512.06636](https://arxiv.org/html/2512.06636v1).

### 5.3 IVF

**Inverted File**: k-means the vectors into `nlist` cells; at query time search only the `nprobe` nearest cells. Recall/latency dial = **`nprobe`**.

Requires a **training step** on a representative sample (unlike HNSW). **Adding vectors far from the trained distribution degrades it** — a real issue if your corpus grows into new domains.

FAISS's dataset-size recipes (same source):

| Dataset size | Index |
|---|---|
| < 1M | `IVF(4√N … 16√N)` |
| 1M–10M | `IVF65536_HNSW32` |
| 10M–100M | `IVF262144_HNSW32` |
| 100M–1B | `IVF1048576_HNSW32` |

Note that at scale **the coarse quantizer is itself HNSW.**

**Compression** for memory-bound cases: moderate → `OPQ_D,...,PQx4fsr` (~M/2 bytes/vector); severe → `OPQx,...,PQ` (M bytes/vector); maximum → **`RaBitQ`** (`d/8 + 8` bytes/vector — a 1-bit-per-dimension binary quantization scheme, notably newer than most tutorials cover).

**IVF and Flat support GPU in FAISS; HNSW and RaBitQ do not.**

**Practical split: HNSW when RAM is plentiful and you need the best recall/latency plus live inserts. IVF+PQ when the dataset is huge and memory-constrained, or when you want GPU.**

### 5.4 The recall/latency tradeoff

**Every ANN system is a curve, not a point. Benchmark it as one:** sweep the knob (`ef` / `nprobe`), plot **recall@k vs QPS / p99 latency**, and pick your operating point deliberately.

> **Measure recall against your own exact/flat search on a sample — never against a vendor's number.** Recall depends on your data's intrinsic dimensionality and clustering, which is not the benchmark dataset's.

Two traps:

1. **Index recall@k ≠ end-to-end retrieval quality.** 95% index recall is usually invisible in answer quality, because a reranker recovers the ordering and the generator is robust to one missing near-duplicate. **Do not over-optimise index recall; do measure the end-to-end metric.**
2. **Filtered search is where ANN quietly breaks.** A highly selective metadata filter combined with graph traversal can return **far fewer than k results, or drastically reduced recall**, because the graph's navigable structure does not respect your filter. Systems handle this very differently (pre-filter, post-filter, filtered-HNSW variants). > **Test filtered recall explicitly.** This is a common production surprise, especially for multi-tenant systems — and it is precisely the case where §4.5's ACL-as-filter advice collides with §5.2's index behaviour.

### 5.5 Chroma and FAISS, positioned honestly

**FAISS** ([github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)) is an **index library, not a database.** No persistence semantics beyond `write_index`, no metadata filtering, no concurrency model, no updates-in-place story, no server. It is the right tool for: research, batch/offline retrieval, embedded use where you control everything, and GPU-accelerated billion-scale ANN. It is the reference implementation everything else is measured against, and **its index-selection wiki is the single best free document on ANN engineering.**

**Chroma** ([docs.trychroma.com](https://docs.trychroma.com)) — "open-source data infrastructure for AI," Apache-2.0. Deployment: in-process/local via SDK, client-server, and Chroma Cloud (managed serverless). Stores embeddings with metadata, supports **dense, sparse and hybrid** search and multimodal content. It is the correct choice for **learning and prototyping**: one `pip install`, no server, metadata filtering included, and an embedding-function abstraction so you do not hand-roll batching.

- `UNVERIFIED`: Chroma's introduction page does **not** document its index algorithm, distance functions, or scale limits. (Historically: HNSW via hnswlib, with `l2`/`ip`/`cosine` selectable via collection metadata.) **Verify against current docs before quoting** — and note this matters for §1.4, since you must know which metric you configured.
- Worth knowing for this course: **Chroma the company publishes the Context Rot research** cited in the foundations file. Good source, but they have a commercial interest in "long context is not enough."

### 5.6 Production options (Aug 2026) and the decision heuristic

`UNVERIFIED` on specific benchmark numbers — the entire comparison space is dominated by SEO content farms and vendor-run benchmarks, and **no credible neutral 2026 benchmark was found.** Positioning only:

| Store | Positioning |
|---|---|
| **pgvector / pgvectorscale** | Postgres extension. HNSW + IVFFlat, real transactions, real SQL joins, real backups, one fewer system to operate, and metadata filtering is just a `WHERE` clause. Ceiling is real (tens of millions of vectors gets painful) but most projects never reach it. **Strongly recommended default for a working engineer.** |
| **Qdrant** (Rust) | purpose-built; **excellent filtered-search support**, quantization, good ergonomics. The usual "I outgrew pgvector" answer. |
| **Milvus / Zilliz** | the scale-out option; most index types, most knobs, most operational complexity |
| **Weaviate** | built-in hybrid (BM25 + dense) and modules; good developer story |
| **Pinecone** | fully managed, serverless; you pay to not think about it |
| **LanceDB** | embedded/on-disk columnar (Lance format); strong for multimodal and for "FAISS but with persistence and filtering" |
| **Turbopuffer** | object-storage-backed, very cheap at rest, latency tradeoff; notable for large low-QPS corpora |
| **Elasticsearch / OpenSearch** | if you already run it, you get **BM25 + dense + RRF in one place** — a real architectural advantage for hybrid |

> **Decision heuristic:** already on Postgres and <10M vectors → **pgvector**. Need heavy filtered search or >10M → **Qdrant**. Already have Elastic → **use it, hybrid comes free**. Prototyping → **Chroma**. Research/offline/GPU → **FAISS**.

For a manufacturer with an existing Postgres estate and a document corpus in the millions-of-chunks range at most, pgvector is almost certainly the right answer, and the Elastic option is the right answer if Gewiss already runs Elastic for anything.

---

## 6. The pipeline end to end, and how to diagnose it

### 6.1 The stages

```
[1 ingest/parse] → [2 clean] → [3 chunk] → [4 embed] → [5 index]
                                                          ↓
[6 query transform] → [7 retrieve] → [8 rerank] → [9 assemble context] → [10 generate] → [11 cite/verify]
```

### 6.2 Where each stage fails

| Stage | Failure modes | Signature |
|---|---|---|
| **1 Ingest/parse** | PDF text extraction garbles multi-column layout; tables become word salad; scanned pages need OCR; headers/footers interleave into body text; DOCX/PPTX structure lost; images/charts silently dropped | Information is *provably in the source* but nothing retrieves it, at any k. **Most underestimated stage — grep your extracted text for a known answer string before blaming anything else.** |
| **2 Clean** | boilerplate, nav, legal preambles, TOC/preliminary pages not stripped; duplicates across document versions | All chunks mutually similar; retrieval returns front-matter for content questions (documented in the academic-texts paper) |
| **3 Chunk** | split answers, orphaned references, topic dilution, tables/code destroyed, **silent truncation past the embedder's input limit** | recall@50 is bad, or retrieved chunks are on-topic but incomplete |
| **4 Embed** | wrong model for the language/domain; **missing `query:`/`passage:` prefix or instruction**; unnormalised vectors + wrong metric; forgot to re-normalise after MRL truncation; **query and corpus embedded with different model versions** | Retrieval is mediocre-but-not-random across the board. **The version-mismatch case is catastrophic and produces near-random results.** |
| **5 Index** | ANN recall too low (`ef`/`nprobe` too small); stale index after document updates; **filtered search silently under-returning**; deleted docs still present | Flat search finds it, ANN does not → pure index problem. Easy to isolate, so **always test this way** |
| **6 Query transform** | user query is a pronoun-laden follow-up ("*e per il 2024?*") with no standalone meaning; vocabulary mismatch between user language and document language; acronyms | Multi-turn RAG works turn 1, fails turn 3 |
| **7 Retrieve** | k too small; pure-dense missing exact identifiers/codes/names; no metadata filter so wrong tenant/version/date wins; near-duplicates consuming all k | recall@k low but recall@50 fine → increase k + rerank |
| **8 Rerank** | absent; reranker truncating long chunks past **its own** token limit; reranker's language coverage worse than the embedder's | correct chunk is at rank 15 and gets dropped |
| **9 Assemble** | too many chunks → measurable degradation (foundations §3.4); best chunk buried in the middle → position bias; no dedup; no source labels so the model cannot cite; prompt-cache-hostile ordering | **Answer quality *drops* when you increase k** — the OP-RAG inverted U |
| **10 Generate** | model ignores context and uses parametric knowledge; hedges instead of answering; hallucinates when context is genuinely insufficient rather than saying so; contradictory chunks resolved arbitrarily | see §6.3 |
| **11 Cite/verify** | citations fabricated or not checkable; no groundedness gate | Users lose trust; the whole system is unfalsifiable |

### 6.3 The diagnostic decision tree — the most operationally valuable thing in this file

> **The core discipline: never debug RAG end-to-end. Log the retrieved chunks for every query, then run this tree.**

**1. Read the retrieved chunks. Is the answer in them?**

- **YES, and the answer is still wrong / hedged / hallucinated → generation problem.** Fix, in order: the prompt (explicitly instruct the model to answer only from the provided context and to say "not in the provided documents" otherwise); chunk ordering (**best chunk last, question last** — foundations §3.4); **reduce k** (length alone degrades, even with perfect retrieval); or a stronger model. If contradictory chunks are present, add recency/version metadata into the prompt so the model can prefer the current one.
- **NO → retrieval problem. Continue.**

**2. Raise k to 50–100. Is it there now?**

- **YES → ranking problem, not a recall problem.** Fix: add a **cross-encoder reranker** (largest single win available — §7), add hybrid search, or improve the embedding model. **This is the most common real diagnosis.**
- **NO → continue.**

**3. Search the raw extracted text (grep/BM25 over the pre-chunk corpus) for the answer string. Is it there?**

- **NO → parsing/ingest problem.** The information never entered the system. Fix the extractor/OCR. **No amount of retrieval tuning helps.**
- **YES → continue.**

**4. Is the chunk containing it well-formed and self-contained when you actually look at it?**

- **NO** (answer split, orphaned reference, table destroyed) **→ chunking problem.** Fix: chunk size/overlap, structure-aware splitting, contextual retrieval, or parent-document retrieval.
- **YES → continue.**

**5. Does exact/flat search retrieve it while ANN does not?**

- **YES → index problem.** Raise `ef`/`nprobe`, rebuild, check for filter-induced recall collapse, check for a stale index.
- **NO → embedding problem.** Check, in this order: prefix/instruction convention; normalisation and configured metric; **query-vs-corpus model version parity**; and whether the model handles your language and domain at all. **Sanity check by embedding the gold chunk's own text as the query — if that does not retrieve itself, the pipeline is broken, not merely weak.**

**6. If the query itself is the problem** (pronouns, ellipsis, acronyms, cross-lingual) **→ query rewriting / conversational query reformulation.**

> **Instrument these three numbers per query and you can answer "which stage" in seconds:** `recall@k`, `recall@50`, and a boolean **`answer_present_in_assembled_context`**. The third one alone cleanly partitions retrieval failures from generation failures, and **most teams do not log it.**

---

## 7. Beyond naive RAG — and which techniques earn their complexity

### 7.1 Hybrid search (BM25 + dense), fused with RRF

BM25 is lexical: exact terms, rare tokens, product codes, error strings, names, acronyms, numbers. Dense is semantic: paraphrase, synonymy, intent. **They fail on disjoint query sets**, which is exactly why fusing helps.

**Reciprocal Rank Fusion** combines ranked lists by `Σ 1/(k + rank_i)` with k≈60 conventionally. It uses **only ranks**, so there is no score normalisation across incomparable scales — that robustness is why RRF beat score-weighted fusion in practice and became the default.

**Verified data** — T2-RAGBench: 23,088 questions over 7,318 financial documents from FinQA/ConvFinQA/TAT-DQA, average 920 tokens, mixed text + tables ([arXiv:2604.01733](https://arxiv.org/html/2604.01733v1)):

| Method | Recall@5 | MRR@3 |
|---|---|---|
| **Hybrid + Cohere Rerank** | **0.816** | **0.605** |
| Contextual Hybrid | 0.717 | 0.454 |
| Hybrid RRF | 0.695 | 0.433 |
| **BM25 alone** | **0.644** | 0.411 |
| **Dense alone** | **0.587** | 0.351 |
| HyDE | 0.544 | 0.318 |

Two claims worth flagging:

- ⚠️ **BM25 alone beat dense alone** (0.644 vs 0.587) on this financial/tabular corpus. **"Dense retrieval supersedes keyword search" is wrong for entity- and number-heavy domains.** Gewiss product codes, article numbers, CEI/IEC standard references and measurement values are exactly that kind of corpus — **BM25 is not a legacy fallback for you, it is a first-class retriever.**
- ⚠️ **HyDE was the *worst* method tested** — worse than plain BM25.

**Two-stage (hybrid → rerank) beat every single-stage method "by a large margin": +39.7% relative MRR@3 over unreranked hybrid.**

### 7.2 Reranking with cross-encoders

A **bi-encoder** (your embedding model) encodes query and document **independently** — which is what makes precomputation and ANN possible, and also what caps its accuracy. A **cross-encoder** feeds `[query, document]` jointly through a transformer and outputs one relevance score, with full cross-attention between query and document terms. Much more accurate, and **O(candidates) forward passes per query**, so it cannot be precomputed — hence its use strictly as a second stage over ~top-50–100.

Current options ([Cohere Rerank docs](https://docs.cohere.com/docs/rerank)): **`rerank-v4.0-pro`** (multilingual, SOTA-positioned) and **`rerank-v4.0-fast`** (low-latency/high-throughput); `rerank-v3.5` and the v3.0 English/multilingual pair at **4,096-token** context.

> **Documents exceeding the reranker's limit are auto-chunked and processed in multiple inferences** — worth knowing, because it changes latency and cost *unpredictably* for long chunks. If you moved to 1024-token chunks, check what your reranker is actually doing to them.

Open-weight: **bge-reranker-v2-m3**, the **Qwen3-Reranker** family, **jina-reranker-v2**. Middle ground: **ColBERT / late-interaction** models — token-level multi-vector matching, cheaper than a cross-encoder, more accurate than a bi-encoder, but **~10–100× the index size**.

`UNVERIFIED`: current head-to-head nDCG/latency numbers across Cohere v4 / Voyage / Jina / BGE. Every source found was content-marketing. **Benchmark on your own gold set — and check the reranker's Italian coverage specifically**, since a reranker with worse language coverage than your embedder will actively undo the embedder's work.

### 7.3 Query rewriting and expansion

- **Conversational reformulation** — resolve pronouns and ellipsis into a standalone query. **Near-mandatory for any multi-turn RAG.** Cheap, high value.
- **Multi-query expansion** — generate k paraphrases, retrieve for each, fuse with RRF. Reliable recall gain, k× retrieval cost, no reindex.
- **Query decomposition** — split a multi-hop question into sub-questions. Necessary for genuinely compositional queries; overkill otherwise.

### 7.4 HyDE — and why it is over-recommended

Have an LLM write a *fake answer* to the query, embed that, and retrieve against it. The insight: a hypothetical answer lives in the same region of embedding space as real answers, closing the query–document asymmetry gap. The paper's own framing is that the hypothetical document "captures relevance patterns but is unreal and may contain false details," with the encoder acting as a lossy filter — [arXiv:2212.10496](https://arxiv.org/abs/2212.10496).

Reported strong gains **over an unsupervised dense retriever (Contriever), zero-shot**, comparable to fine-tuned retrievers, across web search / QA / fact verification, multilingually.

> ⚠️ **The caveat is decisive. HyDE's win was measured against a *weak unsupervised* baseline in 2022.** Against 2026 instruction-tuned embedders on a specialised corpus it **lost to plain BM25** (0.544 vs 0.644 Recall@5, §7.1) and came **last of ten methods**. It also adds an LLM call to every query's critical path, and it can hallucinate the query *away* from the answer in domains the model does not know — **which is precisely the domain where you needed RAG in the first place.** For Italian technical/normative content the model has never seen, this is the worst case for HyDE.

### 7.5 Parent-document / small-to-big retrieval

Index **small** chunks (precise embeddings, good retrieval) but **return the larger parent** (full section or document) to the generator. This resolves the small-vs-large chunk dilemma directly rather than compromising between them. Requires `parent_doc_id` in metadata (§4.5).

Variant: index a summary or generated questions per chunk, return the chunk. Same principle — **decouple what you match on from what you feed the model.**

**Low complexity, high value, essentially no query-time cost. One of the best complexity-to-payoff ratios available.**

### 7.6 Contextual retrieval, and the cheaper alternative

Before embedding, prepend an LLM-generated 50–100-token situating preamble to each chunk ("This chunk is from ACME's Q2 2023 10-Q, in the section on revenue recognition; it follows the discussion of..."). Index the contextualised chunk with **both** dense embeddings and BM25.

Verified numbers — top-20-chunk **retrieval failure rate**, baseline 5.7%, Gemini text-004 embeddings, 800-token chunks over 8K-token documents, across codebases, fiction, arXiv and scientific papers ([Anthropic](https://www.anthropic.com/engineering/contextual-retrieval)):

| Configuration | Failure rate | Relative reduction |
|---|---|---|
| Baseline | 5.7% | — |
| Contextual embeddings | **3.7%** | **−35%** |
| \+ contextual BM25 | **2.9%** | **−49%** |
| \+ reranking | **1.9%** | **−67%** |

Cost: **$1.02 per million document tokens**, one-time — **affordable only because of prompt caching** the document across the many per-chunk calls (foundations §4.5).

Independently corroborated: "Contextual Hybrid" ranked **2nd of 10** methods (0.717 Recall@5) in the T2-RAGBench study.

> **Late chunking** ([arXiv:2409.04701](https://arxiv.org/abs/2409.04701)) **achieves a related goal — chunk embeddings that carry document context — with no LLM calls at all**, by embedding the whole document's tokens first and pooling afterward. Strictly cheaper. **If you are on a long-context embedding model (Voyage 4 at 32K, Qwen3-Embedding at 32K), try late chunking before paying for contextual retrieval.**

### 7.7 GraphRAG — and the correction that changes the decision

Extract entities and relations with an LLM into a knowledge graph, build hierarchical community summaries, and answer via graph traversal plus summaries. Purpose-built for **global/aggregative questions** ("what are the main themes across this corpus?") and **multi-hop entity chains** — which vector RAG genuinely cannot do, because no single chunk contains the answer.

> ⚠️ **"GraphRAG is prohibitively expensive to index" was true of full GraphRAG in 2024 and is no longer the only option.** Microsoft's **LazyGraphRAG**: indexing cost **identical to vector RAG and 0.1% of full GraphRAG**; matches GraphRAG Global Search answer quality on global queries at **>700× lower query cost**; and at 4% of GraphRAG global-search query cost it "significantly outperforms all competing methods on both local and global query types" — [Microsoft Research](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/). **If GraphRAG is being considered, evaluate LazyGraphRAG, not the 2024 version.**

Still: it is a whole additional system — extraction quality, entity resolution, graph maintenance, reindexing on update. Justified when your query distribution is genuinely aggregative or multi-hop-relational. **Not justified for "find the passage that answers this."**

### 7.8 Agentic RAG

The LLM orchestrates retrieval: decides whether to retrieve, formulates and reformulates queries, judges relevance, retrieves again, and stops when satisfied. Named variants: **Self-RAG** (self-reflection tokens on retrieval necessity and support), **CRAG / Corrective RAG** (a lightweight evaluator grades retrieved docs and triggers web search or decomposition on failure), ReAct-style retrieval loops. Surveys: [arXiv:2501.09136](https://arxiv.org/abs/2501.09136), [arXiv:2507.09477](https://arxiv.org/abs/2507.09477), [arXiv:2506.10408](https://arxiv.org/html/2506.10408v1).

The relevant 2026 evidence is *"Is Agentic RAG worth it? An experimental comparison of RAG approaches"* (ACL 2026, [arXiv:2601.07711](https://arxiv.org/abs/2601.07711)), comparing **Enhanced RAG** (targeted modules fixing specific weaknesses) vs **Agentic RAG** (LLM orchestrates everything).

- `UNVERIFIED`: **the abstract does not state the quantitative verdict** — it promises "practical insights into the trade-offs" and guidance on selecting a RAG design. **Pull the full PDF.** The framing (modular fixes vs LLM-orchestrated) is exactly the right question, and this is the current best reference for it.
- Indirect signal: **CRAG appeared in the T2-RAGBench comparison and did not top the table.** Hybrid+Rerank did, at a fraction of the latency.

### 7.9 Which of these are worth the complexity — ranked

Ordered by payoff per unit of complexity, based on the evidence above.

**Tier 1 — do these essentially always.** Cheap, robust, evidence-backed.

1. **Reranking with a cross-encoder over top-50.** +39.7% relative MRR@3 in one shot; the biggest single measured win in every study here. One API call, **no reindexing**.
2. **Hybrid BM25 + dense with RRF.** Fixes an entire class of failures (identifiers, codes, names, numbers) that dense retrieval structurally cannot. **BM25 alone beat dense alone** on financial documents.
3. **Metadata filtering + good metadata.** Correctness and security, not just quality.
4. **Conversational query rewriting**, if you have multi-turn.
5. **Parent-document / small-to-big.** Almost free; resolves the chunk-size dilemma.
6. **Recursive chunking + serious document parsing.** Unglamorous; **highest leverage of anything on this list.**

**Tier 2 — worth it once Tier 1 is exhausted *and measured*.**

7. **Late chunking** (free if your embedder is long-context) **or contextual retrieval** ($1.02/M doc tokens; −35% failure alone, −49% with contextual BM25). Corroborated twice. **Choose late chunking first on cost grounds.**
8. **Multi-query expansion.** Reliable recall gain, k× retrieval cost, no reindex.
9. **Domain-adapted or better embedding model.** The NAACL chunking paper found embedder quality mattered more than chunking strategy.

**Tier 3 — only with a specific, identified trigger.**

10. **LazyGraphRAG** — trigger: your evaluation shows failures on *global/aggregative* or *multi-hop relational* queries that no amount of chunk retrieval fixes. Do not adopt for lookup workloads.
11. **Agentic / corrective RAG** — trigger: queries requiring a genuinely variable number of retrieval steps, or a need to fall back to web search. Costs multiple LLM round-trips of latency per query and makes behaviour **non-deterministic and hard to evaluate**.
12. **HyDE** — trigger: honestly, hard to justify in 2026. Try it, measure it, expect it to lose.

> **The discipline that makes this list actionable: add one technique at a time and measure. Every study cited here found that intuitively appealing techniques — semantic chunking, HyDE, cluster-based methods — lost to boring baselines on real corpora.**

**For an Italian corpus specifically, Tier 1 items 1, 2 and 6 are not optimisations — they are the baseline**, because §3's 0.28 nDCG@10 starting point does not leave room to skip them.

---

## 8. RAG evaluation

### 8.1 Why retrieval and generation must be measured separately

A RAG system has two independent failure modes **with opposite fixes**. If you only measure end-to-end answer quality, you get a single scalar that tells you something is wrong and nothing about what. **This is the most common evaluation mistake.**

The two failures are also not independent in a convenient direction: **perfect retrieval + bad generation looks identical, from the outside, to bad retrieval + faithful generation.** You cannot disentangle them post hoc.

Retrieval metrics have an enormous practical advantage: **they are cheap, deterministic and reproducible.** You need a gold set of (query → relevant chunk ids) once; after that, evaluating a retrieval change takes seconds and costs nothing. Generation metrics need LLM judges — slow, expensive, noisy, and **version-dependent** (upgrade the judge and your historical numbers move).

> **Therefore: iterate on retrieval using retrieval metrics, and gate releases on generation metrics.** Do not iterate on retrieval using LLM-judged answer quality — you will spend $100 and a day to learn what recall@10 would have told you in three seconds.

### 8.2 Retrieval metrics

| Metric | Definition | When it's the right one |
|---|---|---|
| **Recall@k** | fraction of relevant documents appearing in the top k | **The most important RAG retrieval metric** — the generator sees all k, so if it's in the window at all it can be used. **Nothing recovers information that was not retrieved.** |
| **Precision@k** | fraction of top-k that is relevant | Matters more than people assume in 2026: irrelevant chunks measurably degrade generation, and **even a single distractor hurts** (Context Rot). **Low precision is no longer free.** |
| **MRR** | mean of `1/rank_of_first_relevant` | Single correct passage per query; blind to whether you found the other four |
| **nDCG@k** | DCG normalised by the ideal ordering, logarithmic position discount | The only one here that handles **graded relevance** and rewards good *ordering* throughout. The IR-community standard; what BEIR and MTEB report (nDCG@10) |
| **Hit rate / success@k** | did *any* relevant doc appear | Crude, but the easiest gold set to build and a fine starting point |
| **MAP** | precision averaged at each relevant document's rank | Order-sensitive across all relevant items |

Metric definitions with worked examples: [Weaviate](https://weaviate.io/blog/retrieval-evaluation-metrics).

**Report `recall@k` for your actual k *and* at a large k (50).** The gap between them tells you whether you have a **ranking** problem or a **recall** problem — this is step 2 of §6.3, expressed as a metric.

**Which to use:** `Recall@k` for "can the generator possibly succeed"; `nDCG@10` for comparing retrievers (comparable to published numbers, handles graded relevance, position-aware); `MRR` for single-answer factoid workloads; `Precision@k` to justify a smaller k. **Report recall@k and nDCG@k together** — that pair covers both questions.

**Choosing k** is driven by (a) how many chunks the generator can use before degradation sets in, and (b) reranker cost. **Standard shape: retrieve 50–100 → rerank → pass 3–8.** The OP-RAG inverted-U (§9) says more is not better.

### 8.3 Generation metrics

- **Faithfulness / groundedness** — is every claim in the answer supported by the retrieved context? **This is *the* RAG-specific metric.** Measured by decomposing the answer into atomic claims and checking each against the context (LLM judge or NLI model). **Independent of whether the answer is correct** — a faithfully-reported wrong retrieval scores high faithfulness. That is a feature: it isolates the generator.
- **Answer relevance** — does the answer address the question asked? Catches hedging, over-generalising, and answering an adjacent question.
- **Factual correctness / answer accuracy** — vs a ground-truth answer. Requires labelled data; closest to what users care about.
- **Context relevance / noise sensitivity** — how much of the retrieved context was actually needed, and does the system get worse when irrelevant context is added. Straddles the two halves and is a **direct probe of the distractor problem** (foundations §3.4).
- **Answer completeness** — did it capture all the required information, not just some.

⚠️ **ROUGE/BLEU/METEOR for RAG answers is close to meaningless.** N-gram overlap against a reference answer punishes a perfect answer phrased differently. They persist in papers for comparability with older work. **Do not build a pipeline on them.**

### 8.4 Frameworks

**RAGAS** — the most-used RAG-specific framework. Current metric families ([docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)):

- *Retrieval-side*: Context Precision, Context Recall, Context Entities Recall, Noise Sensitivity, Multimodal Faithfulness/Relevance
- *Generation-side*: Faithfulness, Response Relevancy, Factual Correctness, Semantic Similarity
- *NVIDIA set*: Answer Accuracy, Context Relevance, Response Groundedness
- *Agentic*: Topic Adherence, Tool Call Accuracy, Tool Call F1, Agent Goal Accuracy
- *SQL*: Execution-based Datacompy Score, SQL Query Equivalence
- *General-purpose judges*: Aspect Critic, Simple Criteria Scoring, Rubrics-based Scoring
- *Traditional NLP*: BLEU, ROUGE, CHRF, Exact Match

RAGAS's original selling point was **reference-free** evaluation — faithfulness and answer relevance need no ground truth; context recall and factual correctness do. `UNVERIFIED`: the current docs page does **not** tabulate reference-required vs reference-free per metric. **Verify per metric before designing a gold set around it.**

**Others worth knowing:** **TruLens** (the "RAG triad" — context relevance / groundedness / answer relevance — a good minimal mental model), **DeepEval** (pytest-style, CI-friendly), **ARES** (trains lightweight judges to cut LLM-judge cost), **Phoenix/Arize** and **LangSmith/LangFuse** (tracing + eval, which matters more than the metrics for debugging), **Braintrust**, **promptfoo**.

**Benchmarks worth knowing:** **BEIR** (zero-shot retrieval, the standard), **BRIGHT** (reasoning-intensive retrieval — where 2026 embedders still do badly), **RAGBench**, **T2-RAGBench** (financial text + tables), **LongEmbed**, **NoLiMa** (long-context retrieval beyond literal matching), and **MTEB(Europe)** plus **IT-RAG-Bench** for Italian.

### 8.5 Practical gold-set advice

- **Build the gold set first, before any optimisation.** 50–200 real queries from real users, with the correct chunk ids labelled. **This single artefact is worth more than any framework.** Without it, every "improvement" is vibes.
- **Bootstrap with synthetic queries** — ask an LLM to write questions answerable only by chunk X, and you get query→chunk pairs for free — **then have a human review a sample.** > **Synthetic-only gold sets are systematically biased toward literal lexical overlap** — precisely the flaw NoLiMa exposed in needle-in-a-haystack testing. For Italian, also make sure some queries are in colloquial Italian against formal document language, or you will never see the low-similarity degradation §3.1 warns about.
- **LLM-as-judge hygiene**: pin the judge model and version; prefer pairwise comparison to absolute scoring; measure judge agreement with human labels on a sample before trusting it; control for position bias in pairwise setups.
- **Always report the naive baseline** — recursive chunking + a good embedder + top-5, no tricks. Every study cited in §7 found that fancy techniques frequently lose to it.
- **Log the retrieved chunks in production, permanently.** Offline evaluation cannot generate the query distribution your users will. **Production logs are your real gold set.**

---

## 9. RAG vs long context vs fine-tuning

### 9.1 What each actually solves

| | **RAG** | **Long context** | **Fine-tuning** |
|---|---|---|---|
| **Solves** | *which* knowledge to bring | *processing* a specific large input | *how* the model behaves — format, style, task, tone, domain vocabulary |
| **Knowledge freshness** | update the index; instant | per-request; instant | requires retraining |
| **Corpus scale** | unbounded (millions of docs) | bounded by the window (~1M tokens) | bounded by training-data curation |
| **Citations/provenance** | native | possible but weaker | none |
| **Access control** | native (metadata filters) | all-or-nothing per request | **impossible** — knowledge is baked in for everyone |
| **Per-query cost** | low | high (linear in context) | low at inference; high one-off |
| **Latency** | retrieval + short prefill | long prefill | same as base |
| **Complexity** | a whole pipeline to build and maintain | trivial to build | MLOps, data curation, eval, versioning |

> **These are not substitutes, and the "vs" framing is misleading.** RAG and long context are both *inference-time knowledge injection* and they **compose** (retrieve, then be generous with what you pass). Fine-tuning is *behaviour shaping* and composes with both. The genuinely common production shape is **fine-tuned-for-format + RAG-for-knowledge.**

### 9.2 The evidence, chronologically — the tradeoff has genuinely moved

1. **ChatQA-2 (2024)** — complementary, not competing. On ultra-long (>100K) benchmarks, **RAG with 30 chunks of 1,200 tokens scored 64.55 vs 64.29 for full long-context** — a tie, at a fraction of the tokens. The paper's own framing: RAG can still outperform GPT-4-Turbo-level long-context models given a sufficient number of top-k chunks. [arXiv:2407.14482](https://www.arxiv.org/pdf/2407.14482v3)
2. **OP-RAG (2024)** — the finding to remember: **as retrieved chunks increase, answer quality rises then falls — an inverted U.** There are "sweet points" achieving higher quality **with far fewer tokens** than feeding the whole context. Also: **order-preserving chunk arrangement** (keep chunks in original document order rather than relevance order) beats relevance-ordered. [arXiv:2409.01666](https://arxiv.org/abs/2409.01666) — `UNVERIFIED` on the exact F1/token figures; the abstract page does not carry them, fetch the PDF.
3. **Context Rot (2025) + EMNLP 2025 Findings + NoLiMa (2025)** — long context degrades measurably, even with perfect retrieval and even with **whitespace-only** padding (foundations §3.4). This is the empirical case *against* "just paste it all."
4. **"The Token Tax of Epistemic Accuracy" (June 2026)** — the most current head-to-head, and **it cuts *against* RAG on accuracy.** Manufacturing-safety domain (Bridgeport mill, Haas CNC lathe, UR5e robot), expert-validated benchmark, **972 answers over 162 questions**, GPT-5.4-mini and GPT-5.4-nano, comparing semantic retrieval / keyword retrieval / long-context prompting ([arXiv:2606.20898](https://arxiv.org/html/2606.20898v1)):

| Measure | Long context | Semantic RAG |
|---|---|---|
| Correct | **73.1%** | 65.4% |
| Significance | **+7.7pp, McNemar's p = 0.0049** | |
| Input tokens/query | ~247,754 | **~8,399** |
| Cost/query | $0.1181 | **$0.0045** (**26× cheaper**) |
| Latency | no significant difference (4.69–5.40 s means) | |

   Read it carefully: this is a **small, bounded, homogeneous corpus that fits in a window**. It says long context wins *when the whole corpus fits and you will pay 26×*. It says nothing about a 10M-token corpus. Note also the domain — industrial machine safety documentation — is unusually close to Gewiss's.

5. **Fine-tuning vs RAG for novel knowledge (Jan 2026)** — 7B models, QASC plus a purpose-built 10,000+ question set from 2024 Wikipedia events (post-cutoff). Results: **continual pretraining alone ≈ no help. RAG gave substantial, consistent gains, especially on temporally novel information. Supervised fine-tuning scored highest overall** across models and datasets. [arXiv:2601.07054](https://arxiv.org/abs/2601.07054)
   - **The nuance people miss:** SFT winning on a *benchmark* is not SFT winning on *knowledge injection*. SFT is teaching the model the task format, and on a fixed benchmark that is worth a lot. For information the model has never seen **and that changes**, retrieval is the mechanism.
6. Earlier and still directionally right: fine-tuning is poor at injecting *facts*, especially long-tail ones, and RAG beats it there — [arXiv:2403.01432](https://arxiv.org/html/2403.01432v3), [arXiv:2312.05934](https://ar5iv.labs.arxiv.org/html/2312.05934).

### 9.3 Decision criteria

**Choose RAG when:** the corpus exceeds the context window, or changes often, or needs per-user/per-tenant access control, or answers must cite sources, or per-query cost matters at volume, or you need auditability. **This is most enterprise document work.**

**Choose long context when:** the relevant material genuinely fits *and* the whole of it is plausibly relevant (one contract, one codebase, one meeting transcript, one PDF); the corpus is stable enough to prompt-cache; you are prototyping and want an answer today; **or retrieval is structurally the wrong tool because the query is holistic** ("summarise this whole document," "find inconsistencies across these 40 pages") — retrieval cannot answer questions about the *whole*.

**Choose fine-tuning when:** you need a specific output format, style or tone reliably; you need to teach a *task* rather than facts; you want to distill a large model's behaviour into a small cheap one (usually the strongest business case); you need to shrink prompts by baking in instructions; or you need domain vocabulary/jargon fluency. **Not for facts that change.**

**Combine — the realistic answer:** fine-tune small for format and task + RAG for knowledge; or **RAG to select the right ~50K tokens and then let long context reason over them generously** (the "generous retrieval" middle ground, which is effectively what ChatQA-2's 30×1200 configuration is).

### 9.4 Rules of thumb (2026)

| Corpus size | Stable? | Recommendation |
|---|---|---|
| **< ~50K tokens** | either | **Long context, cached. Do not build RAG.** This is the biggest change from 2023 and **the most common over-engineering mistake in courses and startups alike.** |
| ~50K–500K tokens | stable, re-queried | **Long context + prompt caching**, or RAG with generous k. **Measure both** — a genuine coin flip that depends on your accuracy/cost weights |
| ~50K–500K tokens | changing per query | **RAG.** Cache misses destroy long context's economics |
| > 1M tokens | any | **RAG.** Not a choice |
| Any size, needs ACLs / citations / audit | any | **RAG.** Architecturally required |
| Behaviour/format/style is wrong | — | **Fine-tune.** RAG cannot fix this, and more context will not either |
| Need cheap + fast at high volume | — | **Fine-tune a small model** (distillation) **+ RAG** |

Adjust the token thresholds down by ~35% for Italian corpora — the Italian token premium (foundations §2.3) means the same document set hits the ceiling sooner.

### 9.5 How prompt caching and 1M windows shifted this

**The 2023 answer was "RAG, always" and it is no longer right.** Three things changed: (1) windows went 4K → 128K → 1M; (2) **prompt caching made a stable long prefix ~10× cheaper on reads** (foundations §4), which specifically attacks RAG's cost advantage in the stable-corpus case; (3) frontier models genuinely got better at long context, even if not as good as advertised.

**What caching changes concretely:** if your corpus is stable and re-queried within the TTL, long context's marginal cost drops toward 0.1× the base input rate. **The Token Tax paper's 26× becomes ~2.6× in the fully-cached steady state** — which, against a +7.7pp accuracy gain, can be a rational trade. `UNVERIFIED` **as an arithmetic extrapolation**: that paper measured *uncached* costs. But it is the right calculation to run, and you should run it with your own numbers.

**What caching does *not* change:** quality degradation with length (foundations §3.4), corpus-size limits, access control, freshness, and citations. **Caching makes long context cheaper, not smarter.**

**And the tradeoff shifted *toward* RAG in one way:** 2025–26 long-context research is far more pessimistic about *effective* context than 2024 marketing was. NoLiMa-style findings ("effective context ~2K for a 128K model") mean the window number was never the operative constraint in the first place.

---

## 10. Corrections table — claims that are wrong or outdated

| Claim | Status |
|---|---|
| "Semantic chunking beats fixed-size/recursive splitting" | **Two independent negative evaluations.** NAACL 2025: costs "not justified by consistent performance gains," and gains appeared mainly on *synthetically stitched* documents. July 2026 academic-texts study: cluster-based semantic **median Answer Quality 0.40 vs 0.65**. Recursive had the tightest IQR. |
| "Chunking strategy is where the wins are" | **Embedding-model quality mattered more than chunking strategy** (NAACL 2025). Parsing and cleanup mattered more than either (July 2026). |
| "There is an optimal chunk size (512 tokens)" | **No universal optimum.** SQuAD best at **64 tokens (64.1% R@1)**; NarrativeQA improved **4.2% → 10.7%** going 64→1024. Optimum depends on answer locality, structure, *and* embedder. |
| "HyDE is a standard RAG improvement" | **Lost to plain BM25** (0.544 vs 0.644 Recall@5) and came **last of 10 methods** on T2-RAGBench. Its 2022 gains were against a weak *unsupervised* baseline. |
| "Dense retrieval supersedes keyword search" | **BM25 alone beat dense alone** (0.644 vs 0.587) on 23,088 financial/tabular questions. For codes, part numbers, standards references and names, BM25 is a first-class retriever. |
| "GraphRAG is prohibitively expensive to index" | **True of full GraphRAG in 2024 only.** LazyGraphRAG: indexing = vector-RAG cost (**0.1% of full GraphRAG**), **>700× lower query cost** at global-query parity. Evaluate LazyGraphRAG, not the 2024 version. |
| "Cosine vs dot product is an important RAG choice" | **Non-issue on normalised embeddings** — cosine and dot give identical scores, L2 gives identical *ranking*. Spend the effort on normalisation hygiene and store metric configuration. |
| "LaBSE is a good multilingual embedding model for RAG" | **Misapplied.** It is a bitext-mining/sentence-alignment model; **0.188 BEIR**. Bad default. |
| "Pick the top model on the MTEB overall leaderboard" | **Wrong axis.** Overall averages six task types; look at **retrieval for your language** (MTEB(Europe, v1) for Italian). Rank differences under ~1 point are noise, and MTEB is heavily optimised against. |
| "Bigger embedding model = better retrieval" | **MMTEB's best public model was 560M** (`multilingual-e5-large-instruct`), beating multi-billion-parameter LLM embedders. On Italian, a 560M model came within **0.003 nDCG** of a frontier API at **1/7 the latency**. |
| "You need a vector database" | **Over-applied.** FAISS's own guidance is `Flat` for exact results or <1,000–10,000 searches. A few thousand documents → flat index or pgvector without an index: exact, simpler, arbitrary filtering for free. |
| "95% ANN recall means 5% worse answers" | **No.** Index recall ≠ end-to-end quality; a reranker recovers ordering. But **filtered ANN search can collapse recall silently** — test filtered recall explicitly. |
| "More retrieved chunks is better" | **OP-RAG inverted U**, plus the single-distractor evidence from Context Rot. **Retrieve wide (50–100), rerank hard, pass few (3–8).** |
| "ROUGE/BLEU are fine for RAG answer quality" | **Close to meaningless** for free-form generation. Use faithfulness/groundedness + answer relevance + factual correctness. |
| "Measure end-to-end answer quality and iterate on that" | **The most common evaluation mistake.** Perfect retrieval + bad generation is externally indistinguishable from bad retrieval + faithful generation. Measure the two halves separately; iterate on retrieval metrics, gate on generation metrics. |
| "1M context killed RAG" | **No.** Cost (26× uncached), corpus scale, ACLs, freshness, citations, and measured quality degradation all still favour retrieval for most enterprise work. |
| "RAG is always cheaper *and* better than long context" | **Also no.** June 2026: long context **+7.7pp more accurate** (73.1% vs 65.4%, p=0.0049) on a bounded corpus. Say **"cheaper," not "better."** |
| "Fine-tune to teach the model your documents" | **The most expensive way to get the worst version of RAG.** Continual pretraining alone gave near-zero gains on novel knowledge (Jan 2026). **Fine-tune for behaviour; retrieve for facts.** |
| "Build RAG for your document set" *(when the set is small)* | **Under ~50K tokens: don't.** Cached long context is simpler, cheaper to build, and likely more accurate. |
| "Filter tenant/ACL after retrieval" | **That is a data leak**, not an optimisation. Filter in the query. |

---

## 11. Items I could not verify

- **Live MTEB / MMTEB rankings for Aug 2026.** The HF leaderboard Space renders client-side and returned no data to a fetcher. The Qwen3-Embedding-8B card's "#1 multilingual, 70.58" claim is dated **June 2025** and is 14 months stale. **Check the leaderboard in a browser**, filtered to MTEB(Multilingual, v2) / MTEB(Europe, v1), Retrieval.
- **The quantitative verdict of "Is Agentic RAG worth it?" (ACL 2026, [arXiv:2601.07711](https://arxiv.org/abs/2601.07711)).** The abstract withholds it. **Pull the PDF** — it is the best current reference on modular-vs-agentic.
- **Current reranker head-to-head nDCG and latency** across Cohere v4 / Voyage / Jina / BGE. Only content-marketing sources exist. Benchmark on your own gold set, and check Italian coverage specifically.
- **Chroma's index algorithm, distance functions and scale limits.** Absent from Chroma's own introduction page. Historically HNSW via hnswlib with selectable `l2`/`ip`/`cosine`.
- **Any credible neutral 2026 vector-database benchmark.** None found; §5.6 is positioning only, not measurement.
- **Cohere embed v4 exact specs.**
- **RAGAS's per-metric reference-required vs reference-free split.** Not tabulated on the current docs page — verify before designing a gold set around a specific metric.
- **OP-RAG's exact F1/token figures.** Not on the abstract page; fetch the PDF.
- **The 26× → ~2.6× cached-cost extrapolation in §9.5.** That is arithmetic on top of a paper that measured uncached costs, not a published result.
- **Italian-specific reranker and cross-lingual (EN query → IT docs) retrieval numbers.** IT-RAG-Bench covers monolingual Italian dense retrieval only. **If your users query across a mixed IT/EN corpus, that case is unmeasured in the literature reviewed here — you have to measure it.**

**Where a number drives a decision, re-check it.** Several of the above are client-side-rendered leaderboards that a browser can read and a fetch tool cannot.
