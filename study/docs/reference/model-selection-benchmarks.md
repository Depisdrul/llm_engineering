<!-- Generated copy. Source: study/notes/04-model-selection-benchmarks-leaderboards.md
     Edit the source file; this copy is overwritten on every extraction run. -->

# Week 4 — Model selection, scaling laws, benchmarks, leaderboards

**Covers lectures 85–105** (Week 4: "LLM Showdown: Evaluating Models for Code Gen & Business Tasks").

**How this was built:** written from primary sources — arXiv papers, official benchmark repos, first-party leaderboard methodology pages — not from lecture transcripts. Where the source material contradicts what a 2024/2025-era course teaches, that's flagged explicitly. Sources are linked inline; claims I could not verify are marked `UNVERIFIED`.

**Read the corrections table (§7) first.** Several things this week teaches are outdated, and one of them will actively mislead you.

---

## 0. The one-paragraph version

Almost every number you will see this week is a property of the *measurement harness*, not of the model. GPT-4o's SWE-bench score doubled from 16% to 33% with no model change, purely from fixing the benchmark. AIME has 30 questions and a 5–15 point standard deviation across random seeds. GPQA-Diamond has 198 four-option questions and frontier models now exceed the human expert baseline on it. The correct posture for the rest of your career is: **a public benchmark number shortlists candidates and predicts nothing about your task.** Build a golden set of 100–300 items from your own domain and measure that. Everything below is in service of that conclusion.

---

## 1. Scaling laws (lecture 86 — "The Chinchilla Scaling Law")

### 1.1 What Chinchilla actually claims

Hoffmann et al. 2022, *Training Compute-Optimal Large Language Models* — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

- **The claim:** to minimise training loss for a *fixed training compute budget*, scale parameters `N` and training tokens `D` **equally**. Formally `N_opt ∝ C^a`, `D_opt ∝ C^b` with `a ≈ b ≈ 0.5`.
- **Evidence base:** >400 models, 70M–16B params, 5B–500B tokens, three independent estimation methods that roughly agree (exponents 0.50/0.50, 0.49/0.51, 0.46/0.54).
- **The "20 tokens per parameter" figure is derived, not the headline.** It falls out of the paper's Table 3 projections and stays roughly flat across scale — ~20 at 400M params, ~21 at 175B, ~21.6 at 10T. *That flatness is the actual content of "scale equally."*
- **The demonstration:** Gopher was 280B params on 300B tokens (~1.07 tokens/param). Chinchilla used **the same compute budget** — 5.76e23 FLOPs — but reallocated it to **70B params on 1.4T tokens** (exactly 20:1). Chinchilla beat Gopher by >7 points on MMLU (67.5% 5-shot) and also beat GPT-3 (175B), Jurassic-1 (178B) and MT-NLG (530B). Conclusion: the big models of the day were *significantly undertrained*.
- **Parametric loss form:** `L(N,D) = E + A/N^α + B/D^β`. `E` is irreducible entropy, `A/N^α` the finite-capacity penalty, `B/D^β` the finite-optimisation penalty.

### 1.2 Kaplan 2020 and why it disagreed

Kaplan et al., *Scaling Laws for Neural Language Models* — [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

Recommended `a = 0.73, b = 0.27` — grow the model ~2.7× faster than the data, and *stop training well before convergence*. This is why GPT-3 (175B/300B tokens) and Gopher look the way they do.

**Why it differed:** Kaplan used a **fixed learning-rate schedule and fixed token count across all runs**, so the LR schedule never matched the actual token budget. A cosine schedule truncated early leaves the model at a worse loss than it could reach, which systematically penalises long-token runs and biases the apparent optimum toward larger models. Hoffmann's own diagnosis. A secondary cause (Pearce & Song 2024) is **embedding-parameter accounting** — Kaplan counted non-embedding params, and embeddings are a disproportionate share of small models.

> **Don't say "Kaplan was wrong."** The power-law *form* survives intact. Only the compute-allocation exponents were an artefact of methodology.

### 1.3 The replication problem — flag this if the lecture quotes the coefficients

Epoch AI, *Chinchilla Scaling: A replication attempt* ([arXiv:2404.10102](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt)) found that Hoffmann's **Approach 3 parametric fit fits their reconstructed data poorly**, that the reported confidence intervals are **implausibly tight** (consistent with hundreds of thousands of observations, not ~400 models), and that Approach 3's implied policy **contradicts the paper's own other two approaches and the 20:1 ratio actually used to train Chinchilla**.

Corrected fit: `L = 1.8172 + 482.01/N^0.3478 + 2085.43/D^0.3658` — which *is* consistent with 20 tokens/param.

**Practical upshot:** the widely-quoted `E=1.69, A=406.4, B=410.7, α=0.34, β=0.28` are the *published but probably mis-fit* values. Quote them as "as published," not as true. Approaches 1 and 2 — and therefore the 20:1 rule of thumb — hold up fine.

### 1.4 The important part: nobody optimises the Chinchilla objective anymore

Chinchilla minimises **training** compute. It says nothing about **inference** cost. If you serve millions of requests, lifetime FLOPs are dominated by inference, so you want the *smallest model that clears your quality bar* — which means training far past 20 tokens/param.

Meta said this out loud in the Llama-1 paper (Feb 2023): the preferred model is not the fastest to train but the fastest at inference, and a 7B model kept improving past 1T tokens even though Chinchilla would have stopped at 140B.

| Model | Params | Tokens | tokens/param |
|---|---|---|---|
| GPT-3 | 175B | 300B | ~1.7 |
| Gopher | 280B | 300B | ~1.1 |
| **Chinchilla** | **70B** | **1.4T** | **20** |
| Llama-1 7B | 7B | 1.0T | ~143 |
| Llama-2 7B | 7B | 2.0T | ~286 |
| Llama-3 8B | 8B | ~15T | ~1875 |
| Llama-3.1 405B | 405B | 15.6T | ~38.5 |
| Llama-4 Scout | 17B active / 109B total | ~40T | ~2353 active / ~367 total |
| Qwen3 0.6B | 0.6B | 36T | ~60,000 |

Sources: [Llama-1](https://arxiv.org/pdf/2302.13971), [Llama-3](https://ar5iv.labs.arxiv.org/html/2407.21783), [Llama-4 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md), [Qwen3](https://arxiv.org/pdf/2505.09388).

Note that **both patterns coexist inside one family**: the Llama-3 paper states 405B is approximately compute-optimal for its budget, while the 8B and 70B are deliberately trained "much longer than is compute-optimal" because they perform better *at the same inference budget*.

**Quantified trend** ([Epoch AI](https://epoch.ai/data-insights/training-tokens-per-parameter), Aug 2025 / rev. Nov 2025): ~10 tokens/param in 2022 → **~300 in 2025**, growing **3.1× per year** (90% CI 2.1–4.9×). Stated driver: cheaper inference serving.

> **MoE denominator trap.** Epoch lists DeepSeek-V3 at ~22 tokens/param using *total* params (671B/14.8T); on *active* params (37B) it's ~400. Always state which denominator you're using, or your comparison is meaningless.

### 1.5 Work that supersedes it

- **Inference-aware scaling** — Sardana & Frankle et al., [arXiv:2401.00448](https://arxiv.org/abs/2401.00448). Minimises training *plus* inference compute. Teams expecting ~1B requests should train smaller and longer than Chinchilla-optimal. Quality kept improving out to **10,000 tokens/param**. Also warns that fitting Chinchilla-form laws on data collected at normal ratios **overestimates** the value of extra tokens in the over-trained regime.
- **Over-training + downstream prediction** — Gadre et al., [arXiv:2403.08540](https://arxiv.org/abs/2403.08540). 104 models, multipliers to 32× over-optimal. Predicted a 1.4B/900B run's loss from experiments using **300× less compute**, and downstream top-1 error from **20× less**. Proposes a power law linking perplexity to average downstream error — the missing bridge from loss-scaling to benchmark scores.
- **Data-constrained scaling** — Muennighoff et al., [JMLR v26](https://jmlr.org/papers/volume26/24-1000/24-1000.pdf). ~4 epochs of repeated data is roughly as good as fresh data; then returns collapse.
- **Test-time compute** — Snell et al., [arXiv:2408.03314](https://arxiv.org/abs/2408.03314). Difficulty-adaptive per-prompt allocation is **>4× more compute-efficient** than best-of-N. In FLOPs-matched comparison test-time compute can beat a **14× larger model** — but only where the smaller model already has non-trivial success. On the hardest problems, pretraining scale still wins. This is the formal basis for the o1/R1-era shift.
- **Cheap test-time scaling** — s1, [arXiv:2501.19393](https://arxiv.org/abs/2501.19393). **1,000 curated examples**; "budget forcing" (append "Wait" to extend thinking) pushed AIME24 from 50% → 57%. Relevant to Week 7: tiny, well-chosen datasets transfer a lot of reasoning.
- **2026 refinement** — Bryant & Liu, [arXiv:2605.09189](https://arxiv.org/html/2605.09189), argues the Chinchilla form structurally cannot represent overfitting (it decreases monotonically in N) or distinguish total from unique examples. Proposes a saturating replacement bounded by `[E, L₀]`. Recent, single group — `UNVERIFIED` replication status.

**Framing to carry forward:** there are now **four** scaling axes — pretraining params, pretraining tokens (including repetition), post-training RL compute, and test-time compute. Chinchilla describes one 2D slice under one objective.

---

## 2. Benchmarks (lecture 87 — "GPQA, MMLU-Pro, and HLE")

### 2.1 MMLU

[arXiv:2009.03300](https://arxiv.org/abs/2009.03300) · 57 subjects · 4-option MC · ~14,000 test items · macro-averaged accuracy over subjects (matters — a micro-average gives a different number) · 5-shot standard · **random baseline 25%**.

Frontier models now ~90–93%. **Effectively saturated** — Artificial Analysis has removed it from its index entirely, which is the strongest available evidence.

**The critical weakness:** [Gema et al., NAACL 2025](https://arxiv.org/abs/2406.04127) found **~6.49% of MMLU questions have ground-truth errors** — and the **Virology subset is 57% erroneous**. They released MMLU-Redux (5,700 re-annotated items) and found that **model rankings change after correction**. So **~93.5% is roughly the label ceiling**, and scores above ~90% are within noise of it.

Also heavily contaminated: ChatGPT and GPT-4 could recover *masked wrong answer options* at 52% and 57% exact match ([arXiv:2311.09783](https://arxiv.org/abs/2311.09783)) — far above chance, which is only possible if they'd seen the items.

### 2.2 MMLU-Pro

[arXiv:2406.01574](https://arxiv.org/abs/2406.01574) · **12,032 questions** · **14 disciplines** · **10 options** (random baseline drops to 10%) · CoT is the intended setting.

Built from **6,810 cleaned MMLU items + 5,222 new** items from STEM sites, TheoremQA, SciBench. Effects vs MMLU: accuracy drops 16–33 points; prompt-sensitivity falls from 4–5% to ~2%; **CoT now helps, where it didn't on MMLU** — the paper's own evidence it captures more reasoning.

Inherits MMLU's label errors and contamination for the 6,810 retained items. Published June 2024, so now well inside training windows. **Dropped from the Artificial Analysis index.** No published audit of its own label-error rate — `UNVERIFIED`.

### 2.3 GPQA and GPQA-Diamond

[arXiv:2311.12022](https://arxiv.org/abs/2311.12022) · biology, physics, chemistry · written by domain PhDs · designed to be **Google-proof** · 4-option MC (**so a 25% floor**).

| Subset | Size | Criterion |
|---|---|---|
| Extended | 546 | all validated |
| Main | 448 | ≥1/2 experts agree, ≤2/3 non-experts correct |
| **Diamond** | **198** | **2/2 experts agree, ≤1/3 non-experts correct** |

**Human baselines on Diamond: experts 81.3%, non-experts with 30+ min of web access 21.9%.** GPT-4 at release: 39%.

Frontier now **~94–95%** ([Artificial Analysis](https://artificialanalysis.ai/evaluations/gpqa-diamond): Grok 4.6 high 94.9%, Gemini 3.7 Flash high 94.5%, GPT-5.6 Sol max 94.1%).

> **198 items is far too small for the differences people quote.** Binomial standard error at p=0.94, n=198 is ≈1.7pp; one question is 0.51pp. **Gaps under ~4pp between top models are statistically meaningless.** (This SE is my arithmetic, not a published figure.)

Models now **exceed the expert baseline**, so the benchmark can no longer discriminate at the top. And "GPQA = PhD-level intelligence" is a bad shorthand: it's closed-book multiple-choice in three sciences, on 198 items.

The repo ships a **canary string** to help detect leakage — which only works if labs filter on it, and only proves presence, never absence.

### 2.4 Humanity's Last Exam

[arXiv:2501.14249](https://arxiv.org/abs/2501.14249) · [lastexam.ai](https://lastexam.ai/) · **2,500 public questions**, 100+ subjects, plus a **private held-out set** · ~1,000 expert contributors from 500+ institutions · items filtered by requiring then-frontier models to fail them.

Reports **accuracy plus calibration error** — unusual and genuinely useful.

**Two incompatible score regimes. Never quote "the HLE score" without saying which.**

- **Official site:** Gemini 3 Pro 38.3% (calibration error 57.2%), GPT-5 25.3%, Grok 4 24.5%. *(The site footer reads "Latest Update: April 3rd, 2025" while listing much later models — the label is stale.)*
- **Artificial Analysis**, on **2,158 text-only** items, pass@1 with an LLM equality checker: **Claude Fable 5 (max) 55.5%, Claude Opus 5 (max) 54.9%** — [link](https://artificialanalysis.ai/evaluations/humanitys-last-exam)
- A **with-tools** variant scores higher again.

**Known problems:** FutureHouse (July 2025) estimated **~30% of HLE chemistry and biology text-only answers may be wrong**, and the HLE team partially confirmed it; the response was **HLE-Rolling**, a continuously revised version. *(Attributed via Wikipedia; I did not read the FutureHouse primary report — `UNVERIFIED`.)* Calibration error is 50–57% at the top, i.e. **models confidently assert wrong answers** — arguably more decision-relevant than the accuracy figure. And adversarial construction means HLE measures a frontier-of-failure, not a representative distribution of expert work.

### 2.5 SWE-bench — the best teaching example in the whole field

[arXiv:2310.06770](https://arxiv.org/abs/2310.06770) · [swebench.com](https://www.swebench.com/) · input is a repo snapshot + a real GitHub issue, output is a patch, **no test files given** · execution-scored: **FAIL_TO_PASS** tests must now pass (the fix works), **PASS_TO_PASS** must still pass (no regression).

| Variant | Instances |
|---|---|
| Full | 2,294 |
| Lite | 300 |
| **Verified** | **500** ← the de facto standard |
| Multimodal | 517 |
| Multilingual | 300 |

**SWE-bench Verified** ([OpenAI, Aug 2024](https://openai.com/index/introducing-swe-bench-verified/)): 93 experienced Python developers annotated 1,699 instances, 3 annotators each. **68.3% of original samples were filtered out** — 38.3% for underspecified problem statements, 61.1% for unit tests that unfairly reject valid solutions.

> **GPT-4o went from 16% to 33.2% with no model change.** Memorise this example. It is the cleanest proof that a benchmark number describes the harness, not the model.

**Contamination is documented.** *The SWE-Bench Illusion* ([arXiv:2506.12286](https://arxiv.org/abs/2506.12286)) gives models *only the issue text* and asks which file holds the bug: **up to 76% file-path accuracy on SWE-bench Verified vs 53% on repos outside it** (23pp gap); 35% vs 18% on consecutive-5-gram function reproduction. Both the repos and their fix commits are in every pretraining corpus.

**SWE-Bench Pro** ([arXiv:2509.16941](https://arxiv.org/pdf/2509.16941)) is the contamination-resistant successor, and its central result is the most important number in this section:

| Set | Repos | Best score |
|---|---|---|
| Public (731) | 11 GPL repos | **Claude Sonnet 4.5 — 43.6%** |
| Commercial (276) | 18 proprietary startup repos | **Claude Opus 4.1 — 17.8%** |
| Held-out (858) | 12 GPL repos | reserved for overfitting checks |

Same models, same harness, same task type, **unfamiliar code → ~60% relative drop.** The contamination-resistance trick is deliberately choosing **strong-copyleft GPL repos**, on the theory that labs avoid GPL in training corpora. Tasks are long-horizon: reference solutions average **107.4 LOC across 4.1 files**.

### 2.6 AIME — the worst-variance benchmark in common use

Real human competition: **15 problems per exam, 2 exams/year → 30 problems**, integer answers 000–999 (exact-match gradable, ~0.1% guess rate).

**Saturated** — [Artificial Analysis](https://artificialanalysis.ai/evaluations/aime-2025) shows multiple models at 100%, and AIME is absent from its current index.

The variance is the lesson. [Hochlehnert et al., COLM 2025](https://arxiv.org/pdf/2504.07086):

- **Standard deviation across 20 random seeds: 5–15 percentage points.**
- Temperature variation → up to **15%** swing; top_p → up to **8%**.
- Same model, same container, same GPU type, **different physical cluster → up to 8% difference**, with a 4pp gap persisting.
- Variance "extreme" at K=1, still large at K=5, stabilising only at **K≥30 seeds**.

> **30 problems means one problem = 3.33pp.** Almost every single-run AIME number in a model launch post is inside the noise band. Demand seeds, temperature, and error bars.

### 2.7 The current-generation benchmarks worth knowing

**Terminal-Bench 2.1** — [arXiv:2601.11868](https://arxiv.org/abs/2601.11868) · **89 containerised tasks** (SWE, ML training, security, data processing, sysadmin) · run repeatedly with published CIs because the same agent+model is non-deterministic. Version 2.1 fixed **28 of 89 tasks** (drifted Docker images, insufficient resource budgets, instructions conflicting with tests).

Two harnesses, same benchmark version: [Snorkel](https://snorkel.ai/leaderboard/terminal-bench-2-1/) has Claude Code + Claude 5 Fable at 83.8% ±1.2; [Artificial Analysis](https://artificialanalysis.ai/evaluations/terminalbench-v2-1) (Terminus 2 harness, 3 repeats, e2b sandbox) has GPT-5.6 Sol xhigh at 89.5%. **~6pp difference at the top from the harness alone.** A score here is an *agent+model system* score, not a model score.

**GDPval** — [arXiv:2510.04374](https://arxiv.org/abs/2510.04374) · **44 occupations** across the top 9 GDP-contributing US sectors, **220 open-sourced gold tasks**, built from real work products of professionals averaging ~14 years' experience · graded by **human expert blind pairwise comparison against a human deliverable**. The most business-relevant benchmark in wide use, and the highest-weighted component of the AA index.

**AA-Omniscience** — [arXiv:2511.13029](https://arxiv.org/html/2511.13029v1) · 6,000 questions, 42 topics · **Omniscience Index = 100·(c−i)/(c+p+i+a)**, range −100 to +100, where **abstaining is neutral and hallucinating is penalised**. Headline: **only three models scored above zero; best was Claude 4.1 Opus at 4.8.** Key finding: **general intelligence does not predict factual reliability** — some small models beat larger ones. *This is the most useful single benchmark for a RAG engineer,* because calibration and abstention are what break production systems.

**BrowseComp** — [OpenAI](https://openai.com/index/browsecomp/) · 1,266 problems built by *inversion* (find an obscure fact, write a question hard to locate but trivial to verify). Human trainers solved 29.2%. Model results: **GPT-4o 0.6% → GPT-4o with browsing 1.9% → o1 9.9% → Deep Research 51.5%.** The 0.6→1.9 jump is the teaching point: **a tool is nearly worthless without the reasoning to use it strategically.**

**ARC-AGI** — [arcprize.org](https://arcprize.org/arc-agi) · measures *skill-acquisition efficiency on novel tasks*, scored **pass@2**, with **cost as a scored axis** (leaderboard capped at $10k/run). ARC-AGI-2 human baseline: 100% of tasks solved by ≥2 humans in ≤2 attempts. Current: Opus 4.5 Thinking 37.6% at $2.20/task; Gemini 3 Pro + Poetiq refinement 54% at **$30/task** — 16pp for 14× the money. A 7M-param specialised model gets 45% on ARC-AGI-1, showing task-specific inductive bias beating scale. **ARC-AGI-3** (launched 25 Mar 2026) moved to interactive game environments with no stated rules: **humans 100%, frontier AI ~0.51%.**

**CritPt** — [arXiv:2509.26574](https://arxiv.org/abs/2509.26574) · 71 research challenges / 190 checkpoints from 50+ physicists' unpublished work · **best base model 5.7%, ~10% with code execution.** Nowhere near saturated.

**AA-LCR** — [link](https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning) · 100 questions over 10k–100k-token documents. Explicit finding: **a large context window does not guarantee a model can reason over long documents.** (100 questions → ±~4pp SE; noisy.)

**OSWorld** — 369 tasks in real desktop VMs, execution-scored. Human baseline 72.36%; best agent at release 12.24%. **OSWorld-Verified** (July 2025) changed the scoring — do not compare pre- and post-Verified numbers.

**FrontierMath** — Epoch AI, held private, tiers 1–4. **Governance incident worth knowing:** per [Epoch's own disclosure](https://epoch.ai/latest/openai-and-frontiermath), OpenAI commissioned and owns the 300 core problems and has access to problems *and solutions* except for a 50-problem holdout. The funding relationship was only disclosed publicly around the o3 announcement. [TechCrunch coverage](https://techcrunch.com/2025/01/19/ai-benchmarking-organization-criticized-for-waiting-to-disclose-funding-from-openai/).

**LiveCodeBench** — [arXiv:2403.07974](https://arxiv.org/abs/2403.07974) · four scenarios (generation, self-repair, execution, test-output prediction) from LeetCode/AtCoder/Codeforces, **problems date-tagged so you can evaluate only post-cutoff windows**. That temporal design is the canonical contamination defence. Current version, problem count, and top scores `UNVERIFIED` (client-side rendering).

**IFEval / IFBench** — IFEval ([arXiv:2311.07911](https://arxiv.org/abs/2311.07911)) checks *code-verifiable* constraints ("more than 400 words", "all lowercase"), ~500 prompts, 25 instruction types. Now **heavily overfit** — RLVR trains directly on these. **IFBench** ([arXiv:2507.02833](https://arxiv.org/abs/2507.02833)) exists because "most models strongly overfit on a small set of verifiable constraints," and adds **58 out-of-domain constraints**.

**BBH** — [arXiv:2210.09261](https://arxiv.org/abs/2210.09261) · 23 tasks chosen because no LM had beaten the average human rater. Historically important for showing that **few-shot without CoT substantially underestimates capability** (with CoT, Codex beat the human rater on 17/23). **Saturated and retired.**

---

## 3. Benchmark limitations (lecture 88 — "Data Contamination and Overfitting")

### 3.1 Contamination taxonomy

Survey: [arXiv:2502.17521](https://arxiv.org/html/2502.17521v2)

- **Exact** — verbatim overlap between train and test.
- **Syntactic** — matches after transformations (punctuation, whitespace, synonyms, prefixes) that preserve meaning. **This is what most detection misses.**
- **Input-only** (saw the question) vs **input+label** (saw question and answer) — the latter far more damaging.
- **Indirect** — the model saw a *derivative*: a blog post analysing the benchmark, an editorial solution, a leaderboard-tuning set.

### 3.2 Detection methods

| Method | How | Limitation |
|---|---|---|
| n-gram / retrieval over corpus | search training data for benchmark strings | needs corpus access → impossible for closed models |
| Canary strings | benchmark embeds a GUID; reproduction proves leakage | only if labs don't filter on it; proves presence, never absence |
| Membership inference / perplexity | anomalously low loss vs paraphrases | noisy, format-sensitive |
| **TS-Guessing** | mask the *wrong* MCQ options, ask the model to fill them | **works on closed models** — ChatGPT 52% / GPT-4 57% on MMLU |
| Permutation tests | shuffle options; contaminated models are order-sensitive | |
| **Held-out twin** | build a distribution-matched fresh benchmark | **strongest evidence** — GSM1K, HLE private, SWE-Bench Pro held-out |
| Temporal cutoff | evaluate only post-cutoff items | LiveBench, LiveCodeBench; shrinks sample → raises variance |
| Diagnostic probes | ask for info only memorisation could supply (file paths, function bodies) | SWE-Bench Illusion |

### 3.3 Documented cases

1. **MMLU wrong-option recovery: ChatGPT 52%, GPT-4 57%** — [arXiv:2311.09783](https://arxiv.org/abs/2311.09783)
2. **GSM8K → GSM1K** ([arXiv:2405.00332](https://arxiv.org/abs/2405.00332)): distribution-matched fresh set, **accuracy drops up to 8%**, systematic overfitting in several families, Spearman r²=0.36 between a model's probability of *generating* GSM8K examples and its performance gap. **Correct a common misquote: the max drop is ~8pp, not 13pp** — and the paper's own conclusion is moderate: frontier models mostly generalised.
3. **SWE-bench Verified memorisation**: 76% vs 53% file-path, 35% vs 18% 5-gram — [arXiv:2506.12286](https://arxiv.org/abs/2506.12286)
4. **SWE-Bench Pro public vs commercial: 43.6% → 17.8%** — [arXiv:2509.16941](https://arxiv.org/pdf/2509.16941)
5. **Llama 4 / LMArena, April 2025.** Meta submitted `Llama-4-Maverick-03-26-Experimental`, an unreleased variant "optimized for conversationality," scored highly, and shipped a different model. The **released** vanilla Maverick ranked **32nd** — below then-months-old GPT-4o, Claude 3.5 Sonnet and Gemini 1.5 Pro. [TechCrunch](https://techcrunch.com/2025/04/11/metas-vanilla-maverick-ai-model-ranks-below-rivals-on-a-popular-chat-benchmark/). *The Leaderboard Illusion* identified **27 private Llama variants** tested in the run-up.
6. **FrontierMath / OpenAI ownership** — a *governance* contamination case (§2.7).

### 3.4 Benchmark label errors — a separate failure mode

- **MMLU: 6.49% overall, Virology 57%** — [arXiv:2406.04127](https://arxiv.org/abs/2406.04127)
- **HLE: ~30% of chem/bio text-only answers possibly wrong** (partially confirmed by the HLE team)
- **GSM8K: 88 invalid of 997 (≈8.8%)** — [Stanford, NeurIPS 2025](https://ai.stanford.edu/blog/fantastic-bugs); 93.3% of detected problems were *grading* issues rather than wrong keys
- **SWE-bench original: 68.3% of samples problematic**

### 3.5 Goodhart channels

Once a benchmark becomes the optimisation target it stops measuring the construct. The documented channels:

1. **Training on the benchmark's format** — IFEval's 25 constraint types became RLVR targets.
2. **Best-of-N private variant selection** — test 27 variants, publish the max. With LMArena-scale noise this is free points.
3. **Harness engineering instead of model improvement** — mini-SWE-agent hit 65% on Verified in ~100 lines of Python.
4. **Reasoning-effort dialling** — publish the max-effort score next to the base price. Standard soft deception; cost differs 10×+.
5. **Subset shopping** — HLE full vs text-only vs with-tools; SWE-bench Full vs Verified vs Lite; ARC public vs semi-private eval.
6. **Single-run reporting inside the noise band** — see AIME.

Systematic critique: [*Measuring what Matters: Construct Validity in LLM Benchmarks*](https://arxiv.org/abs/2511.04703) — **445 benchmarks reviewed by 29 expert reviewers**, finding pervasive problems in how benchmarks operationalise the phenomena they claim to measure.

### 3.6 Benchmark score vs business performance — the evidence

**METR randomised controlled trial, July 2025** — [link](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/). 16 experienced open-source developers, 246 real issues (~2h each), in repos averaging 22,000+ stars and 1M+ LOC.

> **With AI tools, developers took 19% LONGER. They predicted a 24% speedup beforehand. Afterwards — having been slowed down — they still believed they'd been sped up by 20%. The perception gap is ~39 percentage points.**

Cite this whenever someone justifies a rollout with developer satisfaction surveys. METR's own caveats: doesn't generalise to less experienced developers, non-software work, or future models; possible sampling bias; only ~50h of tool exposure so limited learning effects.

Everything else points the same way: SWE-Bench Pro's 43.6%→17.8% on unfamiliar code; AA-LCR showing context length ≠ context capability; AA-Omniscience showing only 3 models net-positive on knowledge-minus-hallucination; BrowseComp's 0.6%→1.9% from tool access alone.

**The structural reason benchmarks don't transfer:** benchmarks have a single verifiable answer, no ambiguity, no stakeholders, no tacit context, no proprietary data, no need to say "I don't know", no cost or latency constraint, no multi-day horizon, and no consequence for confident error. Your tasks have all of these.

### 3.7 The saturation ladder

| Gen | Benchmark | Released | Now |
|---|---|---|---|
| 1 | GLUE / SuperGLUE | 2018/19 | saturated ~2019/20 |
| 2 | MMLU | 2020 | ~90–93%, at the ~93.5% label ceiling |
| 2 | GSM8K, HumanEval | 2021 | saturated |
| 3 | BBH | 2022 | retired |
| 3 | GPQA-Diamond | Nov 2023 | ~94–95%, above the 81.3% expert baseline |
| 3 | SWE-bench | Oct 2023 | → Verified → Pro |
| 4 | MMLU-Pro | Jun 2024 | dropped from AA index |
| 4 | AIME 2024/25 | 2024/25 | **100% for several models** |
| 5 | HLE | Jan 2025 | 38% (site) to ~55% (AA text-only), rising |
| 5 | ARC-AGI-2 | Mar 2025 | 24%–54% depending on budget |
| 5 | Terminal-Bench 2.x | 2025/26 | ~84–90%, saturating |
| 6 | **ARC-AGI-3** | Mar 2026 | **~0.51% vs 100% human** |
| 6 | **CritPt** | Sep 2025 | **5.7%** |
| 6 | **AA-Omniscience** | Nov 2025 | **best 4.8/100** |
| 6 | **SWE-Bench Pro (commercial)** | Sep 2025 | **17.8%** |

The churn is structural, not fashion: a benchmark only carries information in its unsaturated band; publication turns the answers into training data; optimisation pressure destroys the construct; label errors cap the achievable score; and the capability target itself keeps moving (knowledge → reasoning → tool use → long-horizon agentic → interactive → real economic deliverables).

---

## 4. Leaderboards (lectures 90–93)

### 4.1 Artificial Analysis

[Methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking). Answers: *"across providers, what is the current capability × price × speed frontier, measured consistently by one third party?"* Maintains **internal copies of all eval datasets and runs everything itself** — not vendor self-reports.

**Intelligence Index v4.1.1 composition — this is very different from what most write-ups still say:**

| Category | Weight | Benchmarks |
|---|---|---|
| **Agents** | **34%** | GDPval-AA v2, τ³-Banking |
| Scientific Reasoning | 24% | HLE, GPQA Diamond, CritPt |
| Coding | 24% | Terminal-Bench v2.1, SciCode |
| General | 18% | AA-LCR, AA-Omniscience |

MMLU-Pro, AIME, LiveCodeBench, IFBench, MATH-500 are **all out**. Agentic and economic tasks now dominate. **Any course material describing the index as "MMLU-Pro + GPQA + AIME + LiveCodeBench" is outdated.**

**Performance metric definitions** (worth internalising):

- **Output speed** = tokens/sec *after the first token*, normalised to OpenAI tokens.
- **TTFT** = request to first token — for a reasoning model this is the first *reasoning* token, so it's nearly meaningless.
- **End-to-end response time** = input processing + reasoning + generation. **This is what users actually feel.**
- **Blended price** assumes a **7:2:1 ratio of cache-hit : input : output tokens.** That is almost certainly not your workload — RAG is input-heavy, agents are output-heavy, uncached chat gets nothing from the 70% cache assumption. **Recompute with your own ratio.**

**When it misleads:** reasoning-effort variants are listed as separate entries, so comparing one model's max-effort score to another's default is apples-to-oranges at 10×+ the cost; aggregating nine benchmarks into one number destroys the profile you need; speed is measured on some route at some time, not yours; open-weight speed depends entirely on which host.

### 4.2 LM Arena / Chatbot Arena

[arXiv:2403.04132](https://arxiv.org/abs/2403.04132) · now at [arena.ai](https://arena.ai/leaderboard) (lmarena.ai redirects).

Answers: *"which model do anonymous users prefer, on prompts they chose themselves, in a blind A/B?"* That is a **preference** question, not a correctness question.

> **It is not Elo, despite everyone calling it that.** Elo is a sequential, order-dependent online update. The Arena fits a static **Bradley-Terry maximum-likelihood** model over the full battle history, with active sampling toward informative pairs and bootstrapped CIs. The scale is Elo-*like* (~1000-anchored), which is why the name stuck.

**Read the confidence intervals.** Top-10 text (Aug 2026) spans 1506±5 down to 1489±7 — **17 points across ranks 1–10, with CIs of ±3 to ±10.** Most of the top 10 is a statistical tie. "Rank #1" is not a defensible claim from this table.

**Style bias is measured and large.** LMArena's own [style-control analysis](https://blog.lmarena.ai/blog/2024/style-control/) regresses out four confounders. Fitted coefficients: **response length 0.249 (dominant)**, markdown lists 0.031, headers 0.024, bold 0.019. Applying style control moved GPT-4o-mini from rank 6 → 11, Grok-2-mini 6 → 18, Claude 3.5 Sonnet 6 → 4. **Up to 12 rank positions were pure formatting.**

***The Leaderboard Illusion*** ([arXiv:2504.20879](https://arxiv.org/abs/2504.20879)):

- **27 private Llama variants** tested before Llama-4's release → best-of-N score selection.
- **Data-access asymmetry: Google ~19.2% and OpenAI ~20.4% of all Arena data each; 83 open-weight models combined got ~29.7%.** Arena battles are a training signal for the arena distribution — the paper measures **relative gains up to 112%** on that distribution from modest additional arena data.
- Proprietary models get more battles and are deprecated less often, distorting the historical rating field.

Other biases: self-selected prompt distribution (heavy on creative writing and simple coding, not your enterprise mix); non-expert voters cannot verify correctness on hard content, so the vote measures plausibility; sycophancy is rewarded.

**Good tie-breaker on user-facing feel. Bad primary selection criterion.**

### 4.3 Vellum

[vellum.ai/llm-leaderboard](https://www.vellum.ai/llm-leaderboard) — the best one-page **spec sheet**: context windows, input/output prices, generation speed, TTFT, knowledge cutoffs, max output length, alongside HLE / GPQA-D / SWE-Bench Verified / AutoBench / OSWorld / BrowseComp / Terminal-Bench columns. Only covers models released after April 2024 and excludes saturated benchmarks like MMLU.

**The caveat that matters:** data is explicitly *a mix* — "from model providers as well as independently run evaluations." **Partly vendor self-reported**, no error bars, no stated reasoning-effort or agent harness. A vendor-published SWE-bench Verified number and an independently-run one are not the same measurement, and here they sit in the same column.

### 4.4 Scale AI SEAL

[labs.scale.com/leaderboard](https://labs.scale.com/leaderboard). Design intent: **private held-out prompt sets** judged by Scale's own domain experts — the strongest contamination story of any public leaderboard, because private prompts can't be trained on. Also hosts the SWE-Bench Pro boards.

**Current maintenance status `UNVERIFIED`** — the pages render client-side. Given Meta's June 2025 investment in Scale and Alexandr Wang's departure, **check freshness before citing.** The SWE-Bench Pro boards are demonstrably active.

**Two structural problems:** private prompts mean **you cannot inspect the items or audit the grading** — you're trusting Scale. And Scale sells data-labelling and evaluation services to the labs it ranks.

### 4.5 LiveBench

[arXiv:2406.19314](https://arxiv.org/abs/2406.19314) · [livebench.ai](https://livebench.ai/) · six categories (math, coding, reasoning, language, instruction following, data analysis).

Three contamination defences: frequently-updated questions from recent sources; **automatic scoring against objective ground truth — no LLM judge, no human preference**, which removes judge bias entirely; and a wide task variety. Monthly refresh cadence; top models were below 70% at release.

The cleanest contamination-resistant *general* design. **But current status `UNVERIFIED`** — the site renders client-side, and rolling benchmarks are expensive and several have quietly stalled. Verify the refresh is still happening before relying on it.

Excludes open-ended and judgement-heavy work by construction (everything must be auto-gradable), so it misses most business tasks. Monthly question sets are small → month-to-month movement includes sampling noise.

### 4.6 HuggingFace Open LLM Leaderboard — RETIRED

> **Archived 13 March 2025**, after evaluating **13,000+ models** over ~2 years. [Archive notice](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) · [announcement](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard/discussions/1135)

Its v2 set was IFEval, BBH, MATH Level 5, GPQA, MuSR, MMLU-Pro. Stated reason for closure: as capabilities changed, the leaderboard was "slowly becoming obsolete" and risked pushing researchers toward irrelevant directions.

**No single successor.** HF points to a [leaderboard finder](https://huggingface.co/spaces/OpenEvals/find-a-leaderboard) and 200+ community boards.

**A great deal of tutorial material still tells engineers to "check the HuggingFace Open LLM Leaderboard." That advice has been dead since March 2025.** For open weights in 2026: Artificial Analysis (covers open weights and per-provider hosting), LiveBench if it's still refreshing, and task-specific boards (Snorkel for Terminal-Bench, swebench.com, ARC Prize).

### 4.7 Selection summary

| Leaderboard | Actually answers | Trust for | Do NOT use for |
|---|---|---|---|
| Artificial Analysis | independent capability × price × speed | shortlisting, cost/latency budgeting | your specific task; literal blended price |
| LM Arena | anonymous user preference | user-facing feel, tie-breaking | correctness, procurement |
| Vellum | spec sheet + headline scores | prices, context windows, cutoffs | rigorous comparison (mixed data) |
| Scale SEAL | expert judgement on private prompts | contamination-resistant signal *if fresh* | anything auditable |
| LiveBench | objectively-graded post-cutoff tasks | contamination-resistant general capability | open-ended or agentic work |
| HF Open LLM | *(retired)* | history ≤ Mar 2025 | anything current |
| Epoch AI | independently-run, with error bars | rigour | task breadth |
| ARC Prize | novel-task skill acquisition, cost-capped | genuine generalisation | predicting business performance |
| Snorkel / tbench.ai | agentic terminal work, CIs, repeats | agent selection | non-coding tasks |

---

## 5. Model selection (lectures 85, 95–96, 99, 101–105)

### 5.1 The dimensions and what each costs

**Capability** — decompose it, don't score it. Useful axes: agentic/tool-use, reasoning, coding, knowledge+calibration, long-context reasoning, instruction adherence, multilingual. **Italian-language quality is not implied by English benchmark scores** — check multilingual evals (MMMLU, Global-MMLU) separately. Current Italian-specific numbers `UNVERIFIED`.

**Cost** — three traps:

- Blended-price assumptions aren't your workload (see §4.1).
- **Reasoning tokens are billed output tokens.** A max-effort reasoning model can emit 10–100× the visible answer in hidden thinking. **Cost per *task* is the only meaningful unit.** ARC Prize's `$/task` axis is the right model to copy: 37.6% at $2.20/task vs 54% at $30/task.
- **Prompt caching changes RAG economics more than model choice does.**

**Latency** — TTFT matters for streaming chat and is meaningless for reasoning models; output tok/sec matters for long generations; **end-to-end is what a user or downstream service experiences.** Reasoning effort is a latency dial as much as a quality dial — batch workloads can afford max effort, interactive ones usually can't.

**Context window** — an advertised length is a *capacity* claim, not a *capability* claim. Llama 4 Scout advertises 10M tokens; that tells you nothing about retrieval accuracy at 2M. AA-LCR's finding is explicit on this. Long contexts also cost input tokens linearly and degrade latency.

**Licence / open weights** — what you actually buy: deployment control (data never leaves your boundary — often the *only* clean answer to strict residency); **version pinning** (no silent model updates breaking your evals — a real operational risk with hosted APIs); fine-tuning and distillation rights. **Licence gotchas:** Llama licences are *not* OSI-open — acceptable-use policies, attribution requirements, and a monthly-active-user threshold above which you need a separate licence. Apache-2.0 (Qwen, some Mistral) and MIT (some DeepSeek) are cleaner. **Read the actual licence each time; "open source" is used loosely.** The cost of open weights: you own inference ops, capacity planning, serving optimisation, security patching, and eval regression testing. **At low volume this is more expensive than an API, not less.**

**Rate limits** — the most common cause of a working prototype dying in production. Check TPM/RPM per tier, how tiers are earned, burst behaviour, whether provisioned throughput exists, 429 backoff behaviour, and whether you can multi-home the same model across providers.

**Provider concentration** — one vendor, one region, one model version is three single points of failure. Build the abstraction layer before you need it.

### 5.2 When a small model is the right answer

- **Classification, extraction, routing, tagging, reformatting, short summarisation with a fixed schema.** Largely solved; frontier capability is wasted.
- **Grounded in retrieved context**, where the model only reads and cites. Here **calibration matters far more than intelligence** — and AA-Omniscience shows the two are decoupled, with some small models beating larger peers.
- **High volume, low per-item value.** This is exactly the inference-cost regime that motivated over-training small models past Chinchilla (§1.4) — modern small models are trained at hundreds-to-thousands of tokens/param *specifically to be cheap to serve*.
- **Tight user-visible latency.**
- **Distillation** — use a frontier model to generate traces, fine-tune a small one. s1 shows 1,000 well-chosen examples transfer a lot of reasoning.
- **Routing/cascade** — cheap model first, escalate on low confidence. Snell et al.'s >4× efficiency gain from difficulty-adaptive allocation is the theoretical justification.

**When you need the frontier:** long-horizon agentic work with many tool calls (compounding error punishes weak models brutally); unfamiliar large codebases (43.6% → 17.8%); genuinely novel reasoning (CritPt 5.7%, ARC-AGI-3 0.51%); high-stakes ambiguity resolution.

### 5.3 EU / GDPR / residency — relevant to Gewiss specifically

**EU AI Act timeline** ([Commission](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)):

| Date | What applies |
|---|---|
| 1 Aug 2024 | entered into force |
| 2 Feb 2025 | prohibited practices + **Art. 4 AI literacy obligations** (already live, frequently missed) |
| 2 Aug 2025 | GPAI provider obligations (technical docs, training-data summary, copyright policy) |
| 2 Aug 2026 | general applicability + **Art. 50 transparency** (disclose AI interaction, mark synthetic content) |
| 2 Dec 2026 | CSAM/NCII prohibition; watermarking |
| **2 Dec 2027** | **stand-alone high-risk (Annex III)** — ⚠️ **deferred from Aug 2026** |
| 2 Aug 2028 | high-risk embedded in regulated products (Annex I) |

> **"High-risk obligations start August 2026" is now WRONG.** The **Digital Omnibus** (adopted 19 Nov 2025, political agreement 7 May 2026, **in force 27 July 2026**) pushed Annex III to Dec 2027 and Annex I to Aug 2028. [White & Case analysis](https://www.whitecase.com/insight-alert/eu-agrees-digital-omnibus-deal-simplify-ai-rules) · [Council release](https://www.consilium.europa.eu/en/press/press-releases/2026/06/29/artificial-intelligence-council-gives-final-green-light-to-simplify-and-streamline-rules/). It also clarified that systems merely assisting users don't automatically become high-risk absent health/safety risk. The Omnibus reportedly touches GDPR on legitimate-interest grounds for AI training and pseudonymised data — **`UNVERIFIED`, get counsel on that specifically.**

**Seven questions to ask every provider:**

1. Is a **DPA** available, and does it name sub-processors?
2. **Where is data processed**, and separately, **where is it stored at rest?** Providers often answer only the second.
3. Retention period — **is zero-retention available?**
4. Is our data used for training? *(Must be "no by default" on API/enterprise tiers.)*
5. **Abuse-monitoring / human review**: done at all, retained how long, by staff in which jurisdiction, and can we opt out?
6. Transfer mechanism for any US touchpoint — SCCs and/or EU-US DPF certification.
7. **PII in prompts** — lawful basis, DPIA, and Art. 22 consideration if outputs drive decisions about individuals.

**What providers offer:**

- **OpenAI API** ([enterprise privacy](https://openai.com/enterprise-privacy/)): API inputs/outputs removed after **30 days** unless legally required; **zero data retention available on request** for eligible endpoints; API data **not used for training by default** since 1 Mar 2023; DPA executable. ⚠️ The page fetched did **not** document EU residency options — OpenAI does offer regional processing for eligible customers, but **confirm current terms directly.**
- **Azure OpenAI / AI Foundry** — the most granular residency controls of the big three ([data privacy doc](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/data-privacy)): **Regional** deployment keeps all input/output/logs and data at rest in your chosen region; **EU Data Zone** processes in-EU with **abuse-monitoring human review performed by EU-based personnel**; **Global Standard** may process anywhere — **avoid for regulated data.** Microsoft does not train on customer data, and you can apply for **modified abuse monitoring** to disable logging entirely.
- **Anthropic** — DPA available, commercial data not used for training by default, custom retention for Enterprise. ⚠️ **EU-region processing is the weak spot** — [reporting from July 2026](https://www.infoq.com/news/2026/07/claude-foundry-ga-europe/) indicates Claude reached GA on Microsoft Foundry *without* EU-region availability. **Verify before designing around it.**
- **AWS Bedrock / Google Vertex** — both have EU regions (AWS includes **Milan**) and let you pin inference, but **cross-region inference profiles can move data** — check which mode you're in. `UNVERIFIED`.
- **Self-hosted open weights in an EU VPC** — the only option where residency is a property of your architecture rather than a contractual promise.

**Italy-specific:** the **Garante** has been the most aggressive EU DPA on LLMs — it temporarily blocked ChatGPT in Italy in March 2023 and later fined OpenAI. Italy also passed a national AI law in 2025 layering on the AI Act. Details `UNVERIFIED` — but the practical implication is real: **assume above-average regulatory scrutiny and document your DPIA, lawful basis, and residency choices.**

### 5.4 Technical vs business metrics (lecture 102)

| Layer | Metrics | Who cares |
|---|---|---|
| **Model** | MMLU-Pro, GPQA-D, HLE, Terminal-Bench, tok/sec, $/1M tokens | shortlisting only |
| **System** | accuracy on *your* golden set; groundedness/citation rate; **abstention rate on unanswerable questions**; schema-validity rate; p50/p95 end-to-end latency; **cost per task**; regression vs previous model version | engineering |
| **Product** | task completion rate, human escalation rate, **edit distance between draft and shipped output**, retry rate, containment rate | product |
| **Business** | cost per resolved case, hours returned per FTE/week, cycle-time reduction, defect rate vs baseline, **cost of a bad output** (rework + liability + trust) | sponsor |

**Six non-negotiables:**

1. **Build a golden set of 100–300 real items from your own domain before choosing a model.** Every source in §3.6 says public benchmarks won't predict your task.
2. **Always measure a no-AI baseline.** METR's result was only discoverable because there was a control arm.
3. **Report cost per *successfully completed* task**, not per token.
4. **Track abstention and calibration explicitly.** AA-Omniscience's formula (correct − incorrect, abstention neutral) is a good template to reuse.
5. **Run ≥5 seeds and publish internal CIs.** +3pp on 100 items is noise.
6. **Version-pin and re-run the golden set on every provider model update.**

---

## 6. Commercial framing (lecture 94 — "Automation, Augmentation & Agentic AI")

| | **Automation** | **Augmentation** | **Agentic** |
|---|---|---|---|
| Human role | specifies once, receives output | in the loop, iterating | sets goal, approves, monitors |
| Task shape | bounded, repetitive, verifiable | open-ended, judgement-heavy | multi-step, tool-using, long-horizon |
| Failure mode | silent wrong output at scale | wasted human time | compounding error, unintended side effects |
| Metric | throughput, cost/item, error rate | time-to-draft, quality lift | completion rate, steps-to-completion, intervention rate, $/task |
| Guardrail | validation schema + sampling QA | review is inherent | permissions, sandboxing, approval gates, rollback |

**Examples.** *Automation:* route 50k support emails/day; extract invoice line items into ERP; translate a product catalogue into 12 languages; redact PII from call transcripts. *Augmentation:* IDE completion; a lawyer marking up a contract draft; an analyst querying a 200-page annual report; a bid team drafting a tender from a document library. *Agentic:* a coding agent that reads an issue, explores the repo, edits files, runs tests until green, opens a PR (this *is* SWE-bench); a research agent doing multi-hop web investigation (BrowseComp); a service agent that queries account systems and executes a refund (τ²-bench).

**The benchmark→category mapping is a clean structure:** automation quality ≈ IFBench / AA-Omniscience; augmentation ≈ LM Arena / GDPval / AA-LCR; agentic ≈ SWE-bench & Pro / Terminal-Bench / OSWorld / BrowseComp / τ-bench. And note **the AA index now weights Agents at 34%** — the field's evaluation centre of gravity has moved to the third column.

### 6.1 Measured reality of the split

**Anthropic Economic Index** — the only large-scale primary measurement I'm aware of.

[Feb 2025 report](https://www.anthropic.com/news/the-anthropic-economic-index): **57% augmentation / 43% automation**; 37.2% of conversations in computer & mathematical occupations. **~36% of occupations showed AI use across ≥25% of their tasks; only ~4% across ≥75%.** *That last statistic is the single best counter to "AI replaces jobs" — adoption is task-level, not occupation-level.*

[March 2026 report](https://www.anthropic.com/research/economic-index-march-2026-report): augmentation up slightly on Claude.ai, **automation down sharply in API data**; personal use rose to 42%; API is more task-complexity-sensitive in model selection (2.8pp per $10 of task wage vs 1.5pp).

[June 2026 report](https://www.anthropic.com/research/economic-index-june-2026-report): **Claude Code sessions show +0.37 higher AI autonomy (1–5 scale) than chat**, and **~2/3 of that gap is users delegating more completely**, not a different task mix — the agentic surface changes behaviour, not just capability. 68% report learning more with AI, with no reduction among heavy delegators (self-assessed, so weak evidence).

> ⚠️ **The automation/augmentation boundary moved between reports.** Feb 2025 counted directive + feedback loop as automation (43%). March 2026 counted only directive. June 2026 again counted directive + feedback loop. **Do not compare automation-share numbers across reports without checking each definition.** Exactly the kind of silent definitional drift that makes cross-report trend claims wrong.

### 6.2 Why deployments fail

**MIT NANDA/Media Lab, "The GenAI Divide" (Aug 2025)** — the widely-cited **~95% of enterprise GenAI pilots deliver no measurable P&L return**. ⚠️ **Treat this with caution.** The primary report is hard to obtain; sample size, sampling frame, and the definition of "ROI" are unverified, and the number is repeated everywhere with almost no methodological scrutiny. **Cite as a widely-reported industry finding, not established fact.** The reported *diagnosis* — failures come from workflow integration, absent feedback loops, and lack of process redesign rather than model capability — is consistent with METR and SWE-Bench Pro and is the more useful takeaway anyway.

**The pattern across all of it: the model is rarely the bottleneck.** The bottlenecks are no golden eval set, no baseline, no feedback loop, no workflow redesign, unmeasured cost of wrong outputs, and self-reported rather than measured benefit.

### 6.3 Framing for a sponsor

1. **Pick the category deliberately.** Automation for bounded/verifiable/high-volume; augmentation for judgement-heavy; agentic only where you can verify the outcome cheaply and bound the side effects. **Agentic without cheap verification is the highest-risk configuration** — which is precisely why every good agentic benchmark is execution-verified rather than judged.
2. **Verification asymmetry is the selection criterion.** BrowseComp's design principle — answers hard to find but easy to verify — is also the right filter for which business tasks to automate first. **If you cannot cheaply check the output, don't automate it; augment instead.**
3. **Baseline first, then pilot.** Without a control arm you will get METR'd.
4. **Budget in cost-per-completed-task**, including retries, escalations and rework.
5. **Compliance is a design input, not a review gate.** Residency, retention, DPA, abuse-monitoring opt-out, AI Act role (provider vs deployer), Art. 4 literacy and Art. 50 transparency all *eliminate options*. Decide them before picking a model.
6. **Expect the model to change under you.** Version-pin, keep the golden set in CI, design the provider abstraction on day one.

---

## 7. Corrections table — claims that are wrong or outdated

| Claim | Status |
|---|---|
| "Chinchilla says 20 tokens/param, so more is wasteful" | **Misleading.** Minimises *training* FLOPs only. Inference-aware optima are 10–1000× higher; 2025 median is ~300; Llama-3 8B is ~1875. |
| "Chinchilla's fitted E=1.69, A=406.4, B=410.7, α=0.34, β=0.28" | **Published but likely mis-fit.** Epoch's replication found poor fit, implausible CIs, and contradiction with the paper's own other approaches. Corrected: 1.8172 / 482.01 / 2085.43 / 0.3478 / 0.3658. |
| "Kaplan et al. was just wrong" | **Too strong.** Power-law form survives; exponents were a fixed-LR-schedule artefact plus embedding-param accounting. |
| "MMLU 90%+ = expert level" | **Wrong ceiling.** ~6.49% label errors (Virology 57%) put the achievable max near ~93.5%. |
| "GPQA-Diamond 94% = PhD-level intelligence" | **Overclaim.** 198 four-option MCQs in 3 sciences; expert baseline 81.3%; SE ~1.7pp so top gaps are noise. |
| "GSM1K showed 13% drops" | **Wrong number.** Up to **8%**, and the paper concludes frontier models mostly generalise. |
| "SWE-bench has 2,294 tasks" *(when quoting a score)* | **Wrong set.** Reported scores are nearly always **Verified (500)**. |
| "AIME scores show clear reasoning gains" | **Usually noise.** 30 problems, 3.33pp each, seed SD 5–15pp, 8% swing across compute clusters. |
| "LM Arena uses Elo" | **No** — Bradley-Terry MLE on an Elo-like scale. |
| "Check the HuggingFace Open LLM Leaderboard" | **Retired 13 March 2025** after 13,000+ models. |
| "The AA index = MMLU-Pro + GPQA + AIME + LiveCodeBench" | **Outdated.** v4.1.1 is Agents 34%, Sci-Reasoning 24%, Coding 24%, General 18% — and MMLU-Pro, AIME and LiveCodeBench are all out. |
| "EU AI Act high-risk obligations apply from 2 Aug 2026" | **Wrong since the Digital Omnibus** (in force 27 Jul 2026): Annex III → **2 Dec 2027**, Annex I → **2 Aug 2028**. |
| "HLE top score is ~38%" | **Harness-dependent.** 38.3% on lastexam.ai; ~55% on AA's 2,158 text-only items; higher again with tools. |
| "Big context window = can use big context" | **False.** AA-LCR disproves it directly. |
| "Developers report AI makes them faster, so it does" | **METR: −19% actual, +20% perceived, +24% predicted.** |

---

## 8. Items I could not verify

MMLU test-split size (14,079 paper vs 14,042 HF); MMLU-Pro discipline count (14 vs 15) and its own label-error rate; current MMLU/MMLU-Pro top scores from a first-party source; ARC-AGI-1 split sizes; LiveCodeBench current version/count/scores; LiveBench and Scale SEAL current maintenance status; IFEval's four metric definitions; IFBench prompt counts; HLE's MC/short-answer split and multimodal share (Wikipedia only); the FutureHouse HLE error-rate report (primary not read); GDPval grading mechanics and win rates; τ³-Banking spec; SciCode; current SWE-bench Verified top resolve rate; Epoch AI hub current scores; AA measurement frequency and mean-vs-median; LMArena vote count and default style-control setting; AI Act Art. 51 10²⁵ FLOPs threshold; Digital Omnibus GDPR provisions; AWS Bedrock / Vertex EU residency specifics; Anthropic EU-region availability as of Aug 2026; Garante current position and Italy's national AI law; the MIT 95% figure's methodology; current Italian-language eval numbers.

**Where a number matters to a decision, re-check it.** Half of the above are client-side-rendered leaderboards that a browser can read and a fetch tool cannot.
