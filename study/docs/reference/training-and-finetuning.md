<!-- Generated copy. Source: study/notes/06-training-and-finetuning.md
     Edit the source file; this copy is overwritten on every extraction run. -->

# Weeks 6–7 — From traditional ML to fine-tuning (frontier + QLoRA)

**Covers Week 6 ("frontier fine-tuning") and Week 7 ("open-source fine-tuning with QLoRA"), plus the price-prediction capstone that runs through both.** *(Exact lecture numbers are not recorded in the research file — don't rely on the numbering in these notes.)*

**How this was built:** written from primary sources — arXiv papers, official OpenAI/HuggingFace/PEFT/TRL/bitsandbytes docs, the llama.cpp repo — not from lecture transcripts. Course-repo facts (`pyproject.toml`, `uv.lock`) come from direct inspection of the repo. Where the source material contradicts what a 2023–2025-era course teaches, that is flagged explicitly. Claims that could not be verified are marked `UNVERIFIED`.

> **Read §1 and §2 before you start Week 6 or Week 7.** §1 may make the Week 6 exercise literally impossible to run on your account. §2 explains why Week 7 will not run on your laptop no matter how the environment is set up. Both are structural, not bugs you can debug.

**Then read the corrections table (§12).** A significant fraction of what this part of the course teaches is two generations stale.

---

## 0. The one-paragraph version

Two hard facts frame these two weeks. First: **OpenAI is winding down self-serve fine-tuning.** Since **2026-05-07**, an organisation that has never previously run a fine-tuning job [cannot create one at all](https://developers.openai.com/api/docs/deprecations) — so if your Gewiss/personal OpenAI org is fine-tuning-virgin, the Week 6 exercise is not executable as written, and no amount of debugging will fix it. Second: **Week 7 was never provisioned to run locally** — the course repo's `pyproject.toml` and `uv.lock` contain no `peft`, `trl`, `bitsandbytes`, or `accelerate`, which means QLoRA is a Colab-with-CUDA exercise by design. Beyond the logistics, the single most important conceptual correction is this: **fine-tuning changes the conditional distribution over outputs given inputs; it does not reliably install new facts, and attempting to do so measurably increases hallucination rate** ([Gekhman et al., arXiv:2405.05904](https://arxiv.org/abs/2405.05904)). Fine-tuning buys format conformance, style, prompt compression, and model downsizing. It does not buy knowledge — that is retrieval's job. And for the capstone specifically: before you fine-tune anything on a text→price task, build the frozen-embedding + gradient-boosting baseline. If a fine-tuned 7B LoRA does not beat it by more than your bootstrap confidence interval, the LLM is not earning its inference cost, and that is the actual result you should report.

---

## 1. Lead item: the 2026 OpenAI fine-tuning wind-down

**This invalidates a large fraction of LLM-course material written 2023–2025, including, most likely, the Week 6 exercise.**

Source: [OpenAI deprecations page](https://developers.openai.com/api/docs/deprecations), announced **2026-05-07**.

| Date | What happens |
|---|---|
| **2026-05-07** | Organisations that had **never previously run a fine-tuning job** can no longer create fine-tuning jobs **at all** |
| **2026-07-02** | Orgs that have **not run inference on a fine-tuned model in the past 60 days** lose the ability to create new jobs |
| **2027-01-06** | "Active existing customers will no longer be able to create new fine-tuning jobs on this date" |
| ongoing | Existing fine-tuned models keep serving inference "until the base models are deprecated"; disabled "only when the underlying base model is deprecated" |

**Corroborating evidence from the [pricing page](https://developers.openai.com/api/docs/pricing):** fine-tuning pricing is now listed **only for `o4-mini-2025-04-16`** (reinforcement fine-tuning), not for the gpt-4.1 family, and the page states the platform is being wound down and is not accessible to new users.

> **Do this check first, before you spend an evening on Week 6.** Open the deprecations page, then check whether your org has ever created a fine-tuning job. If it hasn't, the Week 6 notebook will fail at job creation, and the failure will look like an auth or quota problem. It isn't. It's policy.

### What to do with Week 6 instead

Treat frontier fine-tuning as **conceptual and historical** — the concepts in §4 are still exactly right and still transfer to every other provider — and spend the hands-on hours on:

1. The **baseline ladder** (§11.4) against a prompted frontier model. This is the part of Week 6 that still runs and is the part that matters commercially.
2. **Open-weights LoRA/QLoRA** (§5–§6), which is now the default hands-on fine-tuning route for everyone, not just people avoiding API costs.
3. Reading the OpenAI [model-optimization guide](https://developers.openai.com/api/docs/guides/model-optimization) and [SFT guide](https://developers.openai.com/api/docs/guides/supervised-fine-tuning) for the dataset-format and dataset-size discipline (§4.2) — that discipline is provider-independent and you will reuse it on any managed fine-tuning service.

**Practical consequence for your day job:** frontier-model fine-tuning at OpenAI is now primarily an **enterprise / custom-models** conversation with a sales motion attached, not a self-serve API call. If Gewiss needs a specialised model, the realistic 2026 options are (a) prompt + retrieval on a frontier model, (b) LoRA/QLoRA on open weights you host, or (c) a managed fine-tuning service from another provider. Option (a) is right far more often than course material implies.

### Also on that page: the 2026 model shutdown dates

| Date | What is deprecated |
|---|---|
| 2026-09-24 | Sora 2 / Videos API |
| 2026-10-23 | Legacy gpt-3.5-turbo / gpt-4 / o1 / o3 snapshots |
| 2026-12-11 | GPT-5 and o3 snapshots |

This matters for §4.3: **a fine-tuned model dies with its base model.** Every base-model deprecation is a forced re-training and re-evaluation project. That is the cost centre that kills fine-tuning projects, and the 2026 schedule is the proof.

---

## 2. Week 7 practicalities: it is deliberately not provisioned to run locally

**Course-repo fact:** the repo's `pyproject.toml` and `uv.lock` contain **no `peft`, no `trl`, no `bitsandbytes`, and no `accelerate`.** Those four packages are the entire QLoRA fine-tuning stack. Their absence is not an oversight to work around — it is the signal that Week 7 is designed to run in **Google Colab on a CUDA GPU**, not in your local `uv` environment.

Why that is the right design rather than a defect:

- **`bitsandbytes` is CUDA-first.** The NF4/4-bit primitives QLoRA depends on ([bitsandbytes docs](https://huggingface.co/docs/bitsandbytes/main/en/index)) target NVIDIA GPUs. On an Apple-Silicon or CPU-only machine there is no path to a 4-bit backward pass.
- **Backpropagation needs a differentiable dequantise-to-compute path**, which is a GPU kernel concern (§7.5). Inference-only quantisation stacks (GGUF/llama.cpp, MLX) will happily run a model on your laptop and cannot train one.
- Installing `peft`/`trl`/`bitsandbytes`/`accelerate` into the local lock file would give you an environment that resolves, imports, and then fails at the first `load_in_4bit=True`. Leaving them out is honest.

**What this means for how you plan your study block:**

| Week | Local GPU needed? | Where the compute lives | Practical setup |
|---|---|---|---|
| 6 | no | OpenAI's servers | API key — **but check §1 first** |
| **7** | **no, but a CUDA GPU is required** | **Colab (or any rented CUDA box)** | `pip install peft trl bitsandbytes accelerate` **in the notebook**, not in the repo's `uv` env |
| **8** | **no** | **Modal, rented per-second** | `GPU = "T4"` in `pricer_service2.py`, `pricer_ephemeral.py`, `llama.py` |

> **Don't fight your local environment on Week 7.** If you spend the first hour of a monthly study block trying to make `bitsandbytes` import locally, you have lost the hour to a problem with no solution. Open Colab, select a GPU runtime, `pip install` the four packages in cell 1, and move on.

**Colab-specific gotchas worth knowing in advance:**

- Confirm you actually got a GPU (`nvidia-smi`) and which one — a T4 is 16 GB and bf16-incapable in the way an Ampere+ card is (§8.7 on precision). Free-tier allocation is not guaranteed.
- Sessions are killed on idle. Checkpoint adapters to Drive or the Hub, not to local disk.
- The adapter is small — the LoRA paper reports per-task checkpoints going from **350 GB → 35 MB** for GPT-3 175B ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) — so pushing adapters to the Hub after every run is cheap and is the correct habit.

**Week 8 needs no local GPU at all**, because the GPU is rented from Modal per-second and specified in code (`GPU = "T4"`). See the companion note `07-agents-and-deployment.md` §1.

---

## 3. When classical ML still beats an LLM

### 3.1 The primary result

Grinsztajn, Oyallon, Varoquaux, *"Why do tree-based models still outperform deep learning on typical tabular data"* — NeurIPS 2022 Datasets & Benchmarks, [arXiv:2207.08815](https://arxiv.org/abs/2207.08815)

- Benchmark: **45 datasets** across domains, with **~20,000 compute-hours of hyperparameter search per learner** — so "the neural net was under-tuned" is not an available objection.
- Conclusion: tree ensembles remain state of the art on **medium-sized tabular data (~10K samples)**, including when compute cost is accounted for.
- Three named inductive-bias gaps neural nets must close: (1) robustness to **uninformative features**; (2) preserving **data orientation** — NNs are rotationally invariant, trees are not, and tabular features are not exchangeable axes; (3) ability to learn **irregular / non-smooth target functions**.

### 3.2 The 2025–2026 nuance that must be taught alongside it

Hollmann et al., *"Accurate predictions on small data with a tabular foundation model"* (TabPFN v2) — [Nature, 2025](https://www.nature.com/articles/s41586-024-08328-6)

- A **prior-data fitted network**: a transformer pretrained on **millions of synthetic datasets** that does in-context learning — no per-dataset training at all.
- Operating envelope: **up to ~10,000 samples and ~500 features.**
- Headline: beats an ensemble of the strongest baselines tuned for **4 hours** in classification, in **2.8 seconds** → reported **5,140× speedup** (classification) and **3,000× speedup** (regression); normalised score **0.952 vs CatBoost 0.822** in the tuned classification setting.
- The authors' own stated limits: inference is **slower than optimised CatBoost at serve time**; **memory scales linearly with dataset size**; the evaluation did not cover >10K samples or >500 features.
- Follow-ups: [*A Closer Look at TabPFN v2*](https://arxiv.org/html/2502.17361v1), [TabPFN-2.5 report](https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report), [TabICLv2](https://arxiv.org/html/2602.11139).

**Flag as outdated:** "deep learning never wins on tabular data." The accurate 2026 statement is: *gradient boosting still wins on medium-to-large tabular data with heterogeneous, noisy, non-smooth features; tabular foundation models (TabPFN-class) now win on small data (<10K rows) at dramatically lower tuning cost; a plain MLP or from-scratch transformer on tabular data is still usually a mistake.*

### 3.3 Decision rule: does an LLM belong here at all

| Use classical ML when | Use an LLM (or LLM-embedding + classical head) when |
|---|---|
| Features are already **structured/numeric** and the signal is in the columns, not in language | The predictive signal lives in **unstructured text/images** — free-text product descriptions, tickets, contracts |
| n is small-to-medium and you need calibrated numeric output | Schema is unstable / features are open-vocabulary |
| You need cheap, deterministic, auditable inference and feature importances | You need generation or reasoning, not just a number |
| Target is a scalar and the loss is a proper regression loss | |

### 3.4 The hybrid baseline that is under-taught and usually wins

**Frozen text-embedding model → GBDT or ridge regression on the embedding vector, plus any structured columns.**

This is the correct baseline to beat before fine-tuning an LLM on a text→number task. It costs one embedding pass and minutes of CPU. It is also the baseline that most course write-ups and blog posts skip, which is why fine-tuned-LLM results look more impressive than they are.

**How to know, concretely:**

1. Build the trivial baselines first: global mean/median; group median by an obvious categorical; linear regression on a handful of hand-features.
2. Build the embedding + GBDT baseline.
3. **Only then** fine-tune.
4. Report all four on the *same* held-out split with the *same* metric.
5. **If a fine-tuned 7B LoRA does not beat embedding+GBDT by a margin larger than your bootstrap CI, the LLM is not earning its inference cost.** Write that up as the finding. It is a real, publishable, career-useful result.

---

## 4. Frontier ("OpenAI-style") fine-tuning: what it does and does not change

### 4.1 The methods on offer

Source: [OpenAI model-optimization guide](https://developers.openai.com/api/docs/guides/model-optimization). All subject to the wind-down in §1.

| Method | What it is for | Model family listed |
|---|---|---|
| **SFT** | Classification, nuanced translation, specific formatting, fixing instruction-following failures | `gpt-4.1-2025-04-14`, `gpt-4.1-mini`, `gpt-4.1-nano` |
| **Vision fine-tuning** | SFT with image inputs | `gpt-4o-2024-08-06` |
| **DPO** | Contrast correct vs incorrect responses; tone/style/summarisation | gpt-4.1 family |
| **RFT** (reinforcement fine-tuning) | Grader-scored reasoning chains for complex domain reasoning | requires `o4-mini-2025-04-16` |

The docs' own framing of *why* fine-tune: handle more input variety than fits in a context window; **reduce token cost via shorter prompts**; train on sensitive data without sending it per-request; make a **smaller/cheaper model** good enough for a narrow task.

**Note what the docs do not claim: that fine-tuning teaches new knowledge.** They frame it consistently as specialising existing capability. That framing is correct and §4.4 is the evidence.

### 4.2 Dataset format and size

Source: [OpenAI SFT guide](https://developers.openai.com/api/docs/guides/supervised-fine-tuning)

- **JSONL**, one JSON object per line, each with a `messages` array in chat format; optional `tools` and `parallel_tool_calls` fields.

```
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

- **Minimum: 10 examples** (the file must have ≥10 lines).
- Improvements observable from **50–100 examples**; the docs recommend **starting with 50 well-crafted demonstrations and evaluating.**
- **The iteration rule is the valuable part:** if 50 examples produce no improvement, **rethink the task or the prompt before adding data.** If it improves, add data incrementally. This is the opposite of the "collect 10,000 examples then train" instinct.
- Hyperparameters surfaced on the docs page: `n_epochs`, `batch_size`, `learning_rate_multiplier`. The rendered defaults retrieved were `n_epochs: 10`, `batch_size: 1`, `learning_rate_multiplier: 1` — **`UNVERIFIED`.** These read like JSON-schema defaults rather than the historical `"auto"` behaviour. Do not teach or rely on them without re-checking the live page.
- Per-example token limits were **not** stated on the page fetched — **`UNVERIFIED`**; check the per-model context limit.

### 4.3 The cost model — and the cost centre nobody budgets for

- The only fine-tuning price currently on the [pricing page](https://developers.openai.com/api/docs/pricing): **`o4-mini` RFT — training $100.00/hour**; fine-tuned inference **$4.00 / $1.00 cached / $16.00 per 1M tokens** (standard), halved under Batch. **50% inference discount if you enable data sharing** when creating the job.
- Historical SFT pricing (token-based training price plus a premium on fine-tuned inference vs base) is no longer listed for gpt-4.1 — consistent with the wind-down. **Treat any "$X per 1M training tokens for gpt-4.1" figure in course material as stale.**

**The real cost model has three centres:**

1. **One-off training cost.** The one everyone quotes. Usually the smallest.
2. **Persistently higher per-token inference price** on the fine-tuned endpoint, forever.
3. **Engineering + eval + re-training on every base-model deprecation.**

> **(3) is the one that kills projects**, and the 2026 deprecation schedule in §1 is the proof: three shutdown waves in a single quarter. Every one is a forced migration of every fine-tuned model built on those snapshots. Budget fine-tuning as an ongoing maintenance commitment, not a project.

### 4.4 When it helps, and when it demonstrably does not

**Helps:**

- **Format/schema conformance** — always emit this JSON shape, this taxonomy, this label set.
- **Style/tone/persona consistency.**
- **Prompt compression** → latency and cost reduction: you delete a 2,000-token system prompt plus a few-shot block from every single request.
- **Model downsizing** — a fine-tuned nano/mini matching a much larger base model on a narrow task.
- **Fixing systematic instruction-following failures** that prompting cannot stabilise.

**Does not help — adding facts.** Primary evidence: Gekhman et al., *"Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?"* — [arXiv:2405.05904](https://arxiv.org/abs/2405.05904)

- Examples that introduce **new** knowledge are learned **significantly more slowly** than examples consistent with what the model already knows.
- As the model does eventually fit the new-knowledge examples, they **linearly increase the model's tendency to hallucinate.**
- Authors' conclusion: LLMs mostly acquire factual knowledge in **pretraining**; fine-tuning teaches them to **use it more efficiently.**

> **The corollary matters at Gewiss specifically.** For "the model must know our 2026 product catalogue," the answer is **retrieval or tool access, not SFT.** Fine-tuning can teach the *shape* of the answer; RAG supplies the *content*. Fine-tuning to install catalogue facts will make the model more fluent and more wrong at the same time, which is the worst possible combination for a technical-sales or specification assistant.

Also does not help: a fundamentally underpowered base model on reasoning-heavy tasks. That is what RFT/RLVR is for, and it needs a grader (§10.4).

**The sentence to memorise:** *fine-tuning changes the conditional distribution over outputs given inputs; it does not reliably install new facts, and attempting to do so measurably increases hallucination rate.*

---

## 5. LoRA mechanics

Paper: Hu et al., 2021 — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) ([full-text mirror](https://ar5iv.labs.arxiv.org/html/2106.09685))

### 5.1 Mechanism

- Freeze the pretrained `W0 ∈ R^{d×k}`. Learn `ΔW = BA` with `B ∈ R^{d×r}`, `A ∈ R^{r×k}`, `r ≪ min(d,k)`.
- Forward pass: **`h = W0 x + BA x`**, with the update scaled by **`α/r`**.
- Initialisation: **A ~ Gaussian random, B = 0** ⇒ `ΔW = 0` at step 0, so the adapted model starts *exactly* at the base model. (PEFT's default is Kaiming-uniform A / zeros B; `init_lora_weights="gaussian"` reproduces the diffusers convention.)
- **No inference latency:** merge `W = W0 + BA` before deployment. This is the structural advantage over bottleneck adapters, which add sequential layers you cannot merge away.

### 5.2 `r` and `α`, and how they interact

- Effective update magnitude is `(α/r)·BA`. Because A is randomly initialised at a scale tied to fan-in, **increasing α at fixed r is approximately equivalent to increasing the LoRA learning rate.**
- The authors' own stance: *"we simply set alpha to the first r we try and do not tune it"* — because with Adam, tuning α is roughly equivalent to tuning the LR when init is scaled appropriately.
- ⇒ **Teach α as a fixed convention, not a tuned knob.** Conventions: `α = r` (scaling 1.0) or `α = 2r` (scaling 2.0).
- **rsLoRA** (`use_rslora=True` in PEFT) replaces `α/r` with **`α/√r`**, which keeps the update magnitude better behaved as r grows. Use it if you intend to use high ranks. [PEFT LoRA conceptual guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora), citing arXiv 2312.03732.

### 5.3 Which modules to adapt — and the correction to the 2021 default

The paper's experiments adapt **self-attention projections** (`Wq, Wk, Wv, Wo`) with MLPs frozen; headline results use **`Wq, Wv`**.

- **Budget ablation (§7.1, Tables 5–6):** under a *fixed* parameter budget, spreading across **more matrices at lower rank beats one matrix at high rank** — `{Wq,Wv}` at r=4 outperforms `Wq` alone at r=8 on WikiSQL/MNLI.
- **Rank ablation (§7.2):** r ∈ {1,2,4,8,64} — *"a very low rank … can be one or two … suffices even when the full rank … is as high as 12,288."* Gains saturate fast; very large r does not help on these tasks.

**Flag as outdated:** "apply LoRA to `q_proj` and `v_proj`." That was the 2021 paper's compute-constrained choice, and modern evidence contradicts it as a default.

- **QLoRA** found adapting **all linear layers of the transformer block** is *critical* to matching 16-bit full fine-tuning (§6.1).
- **Thinking Machines, ["LoRA Without Regret" (2025)](https://thinkingmachines.ai/blog/lora/):**
  - LoRA matches full fine-tuning when (a) applied to **all layers, especially MLP/MoE**, and (b) **not capacity-constrained** — trainable params exceed the information content of the dataset.
  - *"attention-only LoRA significantly underperforms MLP-only LoRA, and does not further improve performance on top of LoRA-on-MLP"* — attention-only at **r=256** underperformed MLP-only at **r=128** despite similar parameter counts.
  - **Optimal LoRA LR is ~10× the optimal full-FT LR**, and this held across supervised and RL settings. This is the single most useful practical number in the post.
  - LoRA is **more sensitive to large batch sizes** than full FT; attributed to the product-of-matrices parametrisation, not to rank.
- **Biderman et al., ["LoRA Learns Less and Forgets Less"](https://arxiv.org/abs/2405.09673) (arXiv:2405.09673):**
  - Regimes tested: instruction tuning (~100K prompt–response pairs) and **continued pretraining (20B tokens)**, in code and math.
  - At typical low ranks, LoRA **underperforms full FT on the target domain**, but **better preserves** base-model performance outside it.
  - LoRA mitigates forgetting **more than weight decay or dropout**, and maintains more diverse generations ⇒ **LoRA functions as a regulariser.**
  - Full FT learns perturbations with rank **10–100× higher** than typical LoRA configs — the mechanistic explanation for the gap.
  - **Practical reading:** LoRA ≈ full FT for **style/format/task-shape adaptation on modest datasets**; LoRA < full FT for **large-scale domain absorption**. Do not overclaim "LoRA always recovers full FT quality."
- *"LoRA vs Full Fine-tuning: An Illusion of Equivalence"* — argues LoRA solutions differ structurally ("intruder dimensions") even at equal task score. **`UNVERIFIED`** — not fetched; cite only after checking.

### 5.4 Why it saves memory — the arithmetic, explicitly

The key fact: **optimizer and gradient state is proportional to *trainable* parameters, not total parameters.**

Full fine-tune, mixed precision + AdamW, bytes per parameter:

```
bf16 weights                2
bf16 gradients              2
fp32 master weights         4
Adam m (fp32)               4
Adam v (fp32)               4
------------------------------
                           16 bytes/param   (some accountings reach 18–20 with an fp32 grad copy)
```

- **7B full FT ⇒ 7e9 × 16 = 112 GB** of static state, before activations. Does not fit one 80 GB GPU.
- QLoRA's paper uses a **12 bytes/param** accounting (bf16 W + bf16 grad + 8 B Adam): 65e9 × 12 = **780 GB** — which is exactly the ">780GB" figure in that abstract.

LoRA on a 7B, bf16 base, r=16 on all linear layers (~40M trainable, ~0.6% of params):

```
frozen bf16 base          7e9 × 2  = 14.0 GB
LoRA params (16 B/param)  4e7 × 16 =  0.64 GB
--------------------------------------------
static state                       ≈ 14.6 GB   + activations
```

⇒ **~7.7× less static state** than full FT for the same model.

The paper's headline "**GPU memory requirement reduced by 3×**" (GPT-3 175B: **1.2 TB → 350 GB**) is a *smaller* ratio than this, because at 175B the **frozen weights themselves dominate** the remaining footprint. That is precisely the gap QLoRA closes by quantising them (§6).

Other paper figures: **trainable parameters reduced up to 10,000×**; per-task **checkpoint 350 GB → 35 MB** (~10,000×); **25% training throughput gain** (43.1 vs 32.5 tokens/s/V100 for GPT-3 175B).

> ⚠️ **Arithmetic discrepancy to flag before you put a number on a slide.** The GPT-3 table lists 4.7M and 37.7M trainable params. For 96 layers, d=12288, adapting `{Wq,Wv}`, trainable = `96 × 2 × 2 × 12288 × r = 4.72M × r` — which gives **4.7M at r=1** and **37.7M at r=8**. A retrieval of the table labelled 4.7M as "r=4"; that is inconsistent with the arithmetic. **`UNVERIFIED` which label is correct — verify against the PDF.** Note also that 175B/17.5M = 10,000×, so the headline 10,000× corresponds to ~17.5M trainable params (≈ r=4 on `{Wq,Wv}`), not to the 4.7M row.

### 5.5 Practical LoRA defaults (2026)

From [Unsloth's LoRA hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) — practitioner-authoritative, not peer-reviewed.

| Knob | Default | Notes |
|---|---|---|
| `r` | **16 or 32** | range {8,16,32,64,128} |
| `lora_alpha` | `r` or `2r` | keep `α/r ≥ 1`; `α/√r` if `use_rslora=True`; **do not tune it** |
| `target_modules` | **`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`** | attention *and* MLP; PEFT also accepts `target_modules="all-linear"` |
| learning rate | **2e-4** for LoRA/QLoRA SFT; **5e-6** for DPO/GRPO | see §8.1 for the 10× transfer rule |
| epochs | **1–3** | |
| warmup | **5–10% of steps** | |
| weight decay | 0.01–0.1 | |
| schedule | linear or cosine | |
| `lora_dropout` | 0.0 | → 0.1 if overfitting |

---

## 6. QLoRA

Paper: Dettmers et al., 2023 — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) ([full text](https://ar5iv.labs.arxiv.org/html/2305.14314))

Headline claim as stated: backpropagate through a **frozen 4-bit quantised** model into LoRA adapters, reducing memory enough to finetune a **65B model on a single 48 GB GPU while preserving full 16-bit finetuning task performance.**

### 6.1 The contributions, and what each actually buys

**1. NF4 — 4-bit NormalFloat.** Described as *"information theoretically optimal for normally distributed weights."*

- Construction: estimate `2^k + 1` **quantiles of a theoretical N(0,1)**, normalise into `[-1,1]`, quantise weights **blockwise with block size 64** — each block gets its own absmax scale.
- Asymmetric: `2^(k-1)` bins for negatives, `2^(k-1)+1` for positives, so **exact zero is representable** (matters for padding, masks, sparsity).
- What it buys: better fidelity per bit than FP4 or int4 for weights that are approximately Gaussian post-LayerNorm.
- **It is a *storage* dtype.** Compute happens after dequantisation into `bnb_4bit_compute_dtype`. This is the single most misunderstood point about QLoRA.

**2. Double quantization — quantise the quantisation constants.**

- Layer 1: NF4, block size **64**. Layer 2: the fp32 absmax scales are quantised to **8-bit floats, block size 256**.
- Explicit arithmetic:

```
Without DQ: one fp32 scale per 64 weights  = 32 / 64                = 0.500 bits/param
With DQ:                                     8/64 + 32/(64 × 256)
                                           = 0.125 + 0.00195        = 0.127 bits/param
Saving                                                              = 0.373 bits/param
For 65B: 65e9 × 0.373 / 8 / 1e9                                     ≈ 3.03 GB
```

That matches the paper's "roughly 3 GB for a 65B model."

- **Effective footprint: 4.127 bits/param ≈ 0.516 bytes/param.**

**3. Paged optimizers.** Uses **NVIDIA unified memory** to page optimizer states between GPU and CPU RAM, absorbing the **memory spikes** that otherwise OOM long-sequence gradient-checkpointed runs.

> **Paged optimizers buy *robustness*, not average-case footprint.** They convert intermittent OOM crashes into slowdowns. If you are OOMing at step 400 of 1,000 on a long-sequence run, this is your fix — not a smaller batch.

**4. (Underrated) Target all linear layers.** QLoRA applies adapters to **all linear layers in every transformer block**, and the paper identifies this as *critical* to matching 16-bit full finetuning. This is the finding later reinforced by "LoRA Without Regret" (§5.3). If a tutorial gives you QLoRA with `q_proj, v_proj` only, it has removed the load-bearing part.

### 6.2 Memory arithmetic, worked

Weights only, at 0.516 bytes/param (NF4 + double quantisation):

| Model | Weights (NF4+DQ) |
|---|---|
| 7B | 3.6 GB |
| 13B | 6.7 GB |
| 33B | 17.0 GB |
| 65B | 33.5 GB |
| 70B | 36.1 GB |
| 405B | 209 GB |

65B end-to-end on 48 GB: 33.5 GB weights + ~0.5–1 GB LoRA state (r=64, all-linear) + paged Adam + gradient-checkpointed activations at their sequence length ⇒ **under 48 GB**, versus **over 780 GB** for 16-bit full FT (`65e9 × 12`).

That is the paper's central comparison: **~16× reduction**, driven mostly by (a) eliminating optimizer/gradient state on 65B params and (b) 4-bit weight storage.

**Sanity-check for your Colab run:** a 7B/8B model at 3.6 GB of NF4 weights fits comfortably on a 16 GB T4 with room for activations. That is why the course can use a free-tier-class GPU at all.

### 6.3 Quality-vs-memory claim, stated precisely

- "Preserving full 16-bit finetuning task performance" — NF4 with DQ **matches BF16 full-finetuning performance** on their benchmark suite.
- **Guanaco** (their best model): **99.3% of ChatGPT's performance level on the Vicuna benchmark**, after **24 hours of finetuning on a single GPU**.
- Reported LR range 1e-5 to 5e-4, batch sizes 8–128, and **low sensitivity to r** (Appendix A).

### 6.4 Caveats to teach alongside it

> **The 99.3%-of-ChatGPT number is one of the most over-cited figures in the field.** It is a **GPT-4-judged Vicuna-benchmark** result from **2023** on a small prompt set. It is not a general capability claim and carries no information about 2026 models. Do not repeat it as evidence about anything current.

- **QLoRA trades speed for memory.** The dequantise-on-the-fly path makes steps slower than bf16 LoRA. **If the model fits in bf16, prefer bf16 LoRA.** QLoRA is a memory-constraint solution, not a free upgrade.
- **Merging a LoRA into a 4-bit base is lossy.** The clean path is: train on 4-bit → merge into the **bf16** base → re-quantise for serving. **LoftQ** (`init_lora_weights="loftq"`) initialises adapters to compensate for quantisation error and reduces this mismatch ([PEFT docs](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)).

### 6.5 Config in practice

[PEFT quantization guide](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# then: model = prepare_model_for_kbit_training(model)
```

PEFT lists training-capable quantisation backends: **bitsandbytes (4/8-bit), GPTQ (2/3/4/8), AQLM (down to 2-bit), EETQ (8-bit), HQQ, torchao (int8 weight-only), Intel INC (FP8/HPU), Transformer Engine (FP8).**

[bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index) itself provides three things: **8-bit optimizers** (blockwise quantisation of Adam states, claimed to hold 32-bit performance), **LLM.int8()** (vector-wise int8 with a separate fp16 path for outlier features; ~50% memory cut, no degradation claimed), and the **4-bit NF4/FP4 primitives** QLoRA is built on.

---

## 7. Quantization more broadly

### 7.1 Numeric formats

| Format | Bits | Exp/mantissa | Where used | Note |
|---|---|---|---|---|
| fp32 | 32 | 8/23 | master weights, optimizer states, loss accumulation | the reference |
| fp16 | 16 | 5/10 | legacy mixed precision | narrow dynamic range → needs **loss scaling**; overflow/NaN risk |
| **bf16** | 16 | 8/7 | **default for training in 2026** | same exponent range as fp32 ⇒ no loss scaling; less mantissa precision |
| fp8 (E4M3/E5M2) | 8 | — | H100+ training and inference | mainstream for serving; W8A8-FP |
| int8 | 8 | integer + scale | inference (LLM.int8, W8A8-INT) | needs outlier handling |
| int4 / NF4 | 4 | integer / quantile | QLoRA training storage; W4A16 serving | group/blockwise scales mandatory |

HuggingFace's own guidance: **fp16 is not actually memory-optimal for training**, because gradients are converted back to fp32 for the optimizer step, so you effectively hold two copies; **bf16 is the preferred alternative.** [Docs](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one)

### 7.2 The methods

**GPTQ** — Frantar et al., [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

One-shot **post-training** weight quantisation using approximate **second-order (Hessian) information**, layer by layer. Quantises a **175B model in ~4 GPU hours** to **3–4 bits/weight** with "negligible accuracy degradation"; more than doubles the compression of prior one-shot methods; "reasonable accuracy" even at 2-bit/ternary. End-to-end inference speedups over FP16: **~3.25× on A100, ~4.5× on A6000.** First method to run a 175B model for generative inference **inside a single GPU**.

**AWQ** — Lin et al., [arXiv:2306.00978](https://arxiv.org/abs/2306.00978), MLSys 2024 Best Paper

Insight: *"not all weights in an LLM are equally important"*, and importance is revealed by the **activation distribution**, not by weight magnitude. Protecting only **~1% of salient weight channels** greatly reduces quantisation error. Implemented as an **equivalent per-channel scaling transform** with scales from **offline activation statistics** — deliberately avoiding hardware-inefficient mixed precision. Weight-only, INT3/INT4 group quantisation. TinyChat: **>3× speedup over HF FP16** on desktop and mobile GPUs; enables 70B Llama-2 on mobile GPUs.

**Practical difference vs GPTQ:** AWQ needs no Hessian/backprop-like solve, is often faster to produce and more robust on instruction-tuned models; GPTQ has broader bit-width options (2/3/4/8).

**GGUF / llama.cpp** — the CPU / Apple-Silicon / consumer-GPU ecosystem. A single-file container with metadata plus quantised tensors, plus **k-quants** (`Q4_K_M` etc.) and **i-quants** (`IQ2_XXS` etc., importance-matrix based).

Real measured bits-per-weight and sizes for **Llama-3.1-8B**, from the [llama.cpp quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md):

| Type | bits/weight | size (GiB) | vs F16 |
|---|---|---|---|
| F16 | 16.00 | 14.96 | — |
| Q8_0 | 8.50 | 7.95 | −47% |
| Q6_K | 6.56 | 6.14 | −59% |
| Q5_K_M | 5.70 | 5.33 | −64% |
| **Q4_K_M** | **4.89** | **4.58** | **−69%** |
| Q4_K_S | 4.67 | 4.36 | −71% |
| Q3_K_M | 4.00 | 3.74 | −75% |
| Q3_K_S | 3.64 | 3.41 | −77% |
| Q2_K_S | 2.97 | 2.78 | −81% |
| IQ3_XXS | 3.25 | 3.04 | −80% |
| IQ2_XXS | 2.38 | 2.23 | −85% |
| IQ1_S | 2.00 | 1.87 | −87% |

> **The bits/weight exceed the nominal name.** `Q4_K_M` is **4.89 bpw, not 4.0**, because of per-block scales/mins and unquantised or higher-precision tensors (embeddings, output head). **That gap is why naive "N-bit ⇒ N/8 bytes × params" estimates undershoot GGUF file sizes by 15–25%.** Size your disk and VRAM from the table, not from arithmetic.

The README publishes **speed, not perplexity deltas** — do not cite perplexity numbers to it.

### 7.3 What degrades, at what bit width

Kurtic et al., *"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization* — [arXiv:2411.02355v4](https://arxiv.org/html/2411.02355v4). Llama 3.1 8B / 70B / 405B; W8A8-INT, W8A8-FP, W4A16-INT.

| Evaluation | Recovery vs BF16 |
|---|---|
| Open LLM Leaderboard V1 | **~99.75% at 8-bit**, **~99.36% for W4A16-INT**; all formats ≈99% |
| Harder V2 tasks | W8A8-FP holds **>99%** across 8B/70B/405B; **W8A8-INT lowest at 97.8%** (405B on MMLU-Pro); W4A16-INT beat W8A8-INT on several tasks |
| HumanEval | **99.9% at 8-bit, 98.9% at 4-bit** |
| Long context (RULER) | **≥98%** across formats |
| Arena-Hard | CIs overlap the BF16 baseline |

**The genuinely useful part — deployment mapping:**

- **Latency-bound / low concurrency** (memory-bandwidth-bound decode) → **W4A16-INT** wins; 2–3× cost reduction for 8B/70B, **5–7× for 405B**.
- **Throughput-bound / high concurrency** (compute-bound) → **W8A8** formats maximise throughput.

The paper explicitly refutes earlier claims of 10%+ degradation from W8A8-INT: with proper tuning, losses are minor.

### 7.4 The 2026 counterweight — "4-bit is basically free" is no longer true

Ouyang et al., *"Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens"* — [arXiv:2411.17691v2](https://arxiv.org/html/2411.17691v2)

- **Quantization-induced degradation (QiD) increases with the number of training tokens.** Undertrained models quantise nearly for free; **fully-trained models degrade meaningfully.**
- Pythia-12B: 3-bit QiD is *negligible* up to ~**10^11 tokens**, then becomes pronounced.
- Projection: models trained on **~100T tokens** face "undesirable" low-bit performance, especially at smaller parameter counts.

> **Flag as outdated: "4-bit quantization is essentially free."** It was near-free for 2023-era models at their token budgets. As token budgets grow, the same bit width costs more accuracy — **and it costs *more* on small, heavily-trained models**, which is exactly the class of model you would want to quantise. Always re-measure on your own model and task rather than importing a 2023 number.

Constructive reading: QiD is a **proxy for how undertrained a checkpoint is.**

### 7.5 Training vs inference quantization — the distinction to hammer

[HF Transformers quantization overview](https://huggingface.co/docs/transformers/main/en/quantization/overview)

- **For training (PEFT-compatible):** bitsandbytes, AQLM, AWQ, AutoRound, compressed-tensors, GPT-QModel, EETQ, HQQ, SINQ. These keep a **differentiable path** (dequantise-to-compute) so LoRA gradients can flow. bitsandbytes is the default because it quantises **on the fly at load time** — no calibration step.
- **Inference-only (PTQ):** GGUF, Metal, optimum-quanto, VPTQ, Quark. **Not for fine-tuning.**
- **On-the-fly** (bitsandbytes, torchao, HQQ, GGUF-load, fp8 variants) vs **pre-quantised / calibration-required** (GPTQ, AWQ, AQLM, compressed-tensors, VPTQ, SpQR). The calibration requirement is the real ergonomic difference: AWQ/GPTQ need a representative calibration set and a one-off quantisation job.
- Selection heuristic from the docs: **bitsandbytes** for simplicity and hardware breadth (and it is what you use during QLoRA training); **GPTQ/AWQ** for maximum serving compression; **compressed-tensors** for extreme bit reduction; **torchao** for PyTorch-native + `torch.compile`; **Metal/GGUF/HQQ/quanto** on Apple Silicon.

**The key asymmetry:** **training quantization targets frozen weights only** — activations, gradients and adapters stay in bf16. **Serving quantization can additionally quantise activations (W8A8) and the KV cache**, which is where the throughput wins actually come from.

---

## 8. Hyperparameters that actually matter — with the failure signature of each

Ordered by how often each is the *actual* cause of a bad run.

| Rank | Knob | Sane value | Signature when wrong |
|---|---|---|---|
| 1 | **Learning rate** | LoRA/QLoRA SFT **2e-4** | too high: loss spikes then plateaus/NaN, repetition collapse. too low: adapter behaves like a no-op |
| 2 | **Warmup + schedule** | warmup **5–10% of steps**, cosine/linear decay | no warmup: unrecoverable spike in first ~20 steps. no decay: eval oscillates, best checkpoint is mid-run |
| 3 | **Effective batch size** | 16–128 sequences | too small: noisy loss, unstable eval. too large: beautifully smooth loss and still underfits |
| 4 | **Epochs** | **1–3** | train loss ↓ while val loss ↑; verbatim regurgitation; diversity collapse |
| 5 | **Gradient checkpointing** | on when OOM-bound | off: OOM scaling with seq×batch. on unnecessarily: ~20–30% throughput loss |
| 6 | **Sequence length / packing** | p95 of your data | truncated targets → model learns to truncate. wrong completion mask → model parrots prompts |
| 7 | **Precision / optimizer** | bf16 on Ampere+ | NaN at step 1–50 in fp16 |

### 8.1 Learning rate (dominant)

- LoRA/QLoRA SFT: **1e-4 to 3e-4**, default **2e-4**. Full FT of a 7B+: **1e-5 to 2e-5**. DPO full: **1e-6** (TRL default); DPO with adapters: **1e-5**. GRPO: **~5e-6**.
- **Key transfer rule (Thinking Machines): optimal LoRA LR ≈ 10× optimal full-FT LR.** If you ever port a full-FT recipe to LoRA, this is the single adjustment that matters.
- **Too high:** loss spikes then plateaus high or goes NaN; gradient-norm spikes; sudden degeneration into repetition; the model loses base-model competence on unrelated prompts (catastrophic forgetting).
- **Too low:** loss decreases smoothly but barely; eval metric ≈ base model; the adapter behaves like a no-op after merge.

### 8.2 Schedule and warmup

- **Warmup = 5–10% of total steps** (or a flat 5–20 steps for tiny runs). Cosine or linear decay to ~0, or to 10% of peak.
- **No warmup:** a loss spike in the first ~20 steps that the run never fully recovers from. Worst with high LR, large batch, or bf16.
- **No decay:** eval metric oscillates and never settles; the best checkpoint is mid-run rather than final. (If you are not checkpointing on the eval metric, you will ship the worse final model.)

### 8.3 Effective batch size and gradient accumulation

- `effective_batch = per_device_batch × grad_accum_steps × num_devices`. Gradient accumulation buys effective batch size **without extra activation memory**, at the cost of proportionally more forward/backward passes per optimizer step. [HF example](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one): `per_device_train_batch_size=4, gradient_accumulation_steps=16` ⇒ effective 64.
- Typical SFT target: effective batch **16–128 sequences**, or better, think in **tokens per step** — more portable across sequence lengths.
- **Too small:** noisy loss curve, unstable eval, sensitivity to data order.
- **Too large:** loss curve is beautifully smooth and the model still underfits, because too few optimizer steps were taken. **LoRA specifically degrades more than full FT at large batch** (Thinking Machines). If you scale batch up, scale LR up too — cautiously, with LoRA.
- **Gotcha:** with gradient accumulation, per-token loss normalisation must be done across the whole accumulation window, not per micro-batch, or short sequences get overweighted. This is a known class of bug in older trainer code — verify your library version.

### 8.4 Epochs and overfitting

- **1–3 epochs** for instruction/format SFT. Datasets under ~1k examples often want 2–3; large ones 1.
- **Overfitting signature:** **training loss keeps falling while validation loss turns up** — the classic hockey stick; plus verbatim regurgitation of training examples; plus collapse of output diversity. [Unsloth's rule of thumb](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide): **training loss below ~0.2 indicates memorisation.** Treat 0.2 as a heuristic tripwire, not a law.
- **Fixes in priority order:** fewer epochs → lower LR → more/better data → `weight_decay` 0.01–0.1 → `lora_dropout` 0.1 → lower r → early stopping on validation.
- **Underfitting signature:** both losses flat and high; eval ≈ base model. Fixes: higher LR, more epochs, higher r/α, better-targeted modules (add the MLPs), better data.

### 8.5 Gradient checkpointing

- Origin: Chen et al., *Training Deep Nets with Sublinear Memory Cost* — [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
  - **O(√n) memory for an n-layer network, at the cost of one extra forward pass per mini-batch.**
  - Reported real result: a **1,000-layer ResNet from 48 GB → 7 GB (~6.9×) with 30% additional runtime** on ImageNet.
  - Extreme variant: O(log n) memory for O(n log n) extra forward compute.
- HF states the practical cost as **~20% slower training**.
- **Off when you need it:** OOM that scales with sequence length × batch, *not* with model size. **On when you don't:** throughput mysteriously ~20–30% below expectation.
- **Interaction:** gradient checkpointing creates the memory *spikes* that QLoRA's **paged optimizers** exist to absorb (§6.1).

> **The silent-failure mode to memorise.** With PEFT you often need `model.enable_input_require_grads()` (or `use_reentrant=False`) or the checkpointed graph has **no grad path to the adapter** — and the symptom is that the loss does not move at all while everything appears to run correctly. If your grad norm is exactly zero, this is why.

### 8.6 Sequence length and packing

- Memory and time scale with sequence length; activation memory is roughly linear in `batch × seq_len` (with FlashAttention), and the KV/attention term is quadratic without it.
- **Set `max_seq_length` to the 95th percentile of your actual examples, not to the model's max.** Doubling seq_len to accommodate 2% of your data is a pure tax.
- **Packing** (concatenating short examples into full-length sequences) massively improves token throughput when examples are short, but requires correct attention masking and position resetting, or examples bleed into each other.
- **Too short:** silent truncation of the assistant target → the model learns to produce truncated answers. **Always log the truncation rate.**
- **Completion mask wrong:** you are training on the prompt tokens as well as the completion. Shows up as a suspiciously low loss and a model that parrots prompts. Use the trainer's completion-only / assistant-only loss masking, and verify by decoding one batch's labels.

### 8.7 Precision and optimizer

- **bf16 on Ampere+**; fp16 only on older hardware, and then with loss scaling. *(Relevant to Colab: a T4 is Turing, pre-Ampere — check what your runtime actually supports before assuming bf16.)*
- `adamw_torch` → `adamw_bnb_8bit` (`optim="adamw_bnb_8bit"`) cuts optimizer state from 8 to ~2 bytes/param. Marginal for LoRA (few trainable params), large for full FT.
- **NaN loss at step ~1–50 in fp16 ⇒ switch to bf16 or fix loss scaling.**

---

## 9. Training loss vs validation loss vs downstream metric

Three different things. Conflating them is the most common evaluation error in fine-tuning.

| Quantity | What it is | What it tells you |
|---|---|---|
| **Training loss** | token-level cross-entropy on the training distribution | measures fit; monotone decrease is expected and proves almost nothing |
| **Validation loss** | same objective on held-out data | detects overfitting — **but it is not the objective you care about** |
| **Downstream task metric** | the business objective (MAE, hit-rate, JSON validity, pass rate) | **this is the model-selection criterion. Checkpoint on it.** |

> **Validation loss can rise while your task metric improves.** A model that becomes terser and more decisive raises per-token cross-entropy while improving exact-match. Conversely, a lower val loss can come with worse task behaviour. If you select checkpoints on val loss, you will sometimes ship the worse model and have no idea why.

### 9.1 Early-warning checklist — run this in the first 10% of every run

1. **Loss at step 0** should ≈ the base model's loss on your data. If it is wildly off, your chat template, tokenisation, or label masking is wrong. *Check this before letting a 4-hour run proceed.*
2. **Grad norm** should be stable and O(1)-ish after warmup. Spiking → LR too high or bad examples. **Exactly zero → your adapter is not in the graph** (the checkpointing / `requires_grad` bug from §8.5).
3. **Truncation rate** and **mean target-token count**, logged from the actual collated batches — not from your assumptions about the data.
4. **Decode 3 generations at step 0, again at 5%, again at 20%.** Loss curves hide degeneration, repetition loops, template leakage, and EOS problems. **This catches more real bugs than any metric.**
5. **Val loss at ~10%** should be below step-0 val loss. If not, stop; something is structurally wrong.
6. **Diversity check** — distinct-n or output-length distribution on a fixed prompt set. Collapse is an early overfit signal that loss curves miss.
7. **Regression check on off-task prompts** — a small set of unrelated prompts to detect catastrophic forgetting (the thing LoRA is comparatively good at avoiding, per [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)).
8. **Cheap task metric every N steps**, on a fixed 100–300 example subset. **If it is not moving by 20–30% of the run, kill the run** rather than waiting for it to finish.

### 9.2 DPO/GRPO-specific early signals (TRL logs these)

[TRL DPO trainer docs](https://huggingface.co/docs/trl/main/en/dpo_trainer)

- `rewards/accuracies` — fraction of pairs where chosen > rejected. Should climb above 0.5 quickly. Stuck at ~0.5 means no learning; jumping to ~1.0 within a few hundred steps means memorising, or β too low.
- `rewards/margins` — should grow steadily. Explosive growth ⇒ reward hacking / drift.
- `rewards/chosen` and `rewards/rejected` — **if both go strongly negative, the policy is degrading overall** while merely widening the gap between chosen and rejected. This is the classic DPO failure, and **it is invisible in the loss alone.**

---

## 10. Preference tuning: RLHF → DPO → what is actually standard in 2026

### 10.1 RLHF/PPO (the 2022–2023 pipeline)

SFT → train a **reward model** on human preference pairs → optimise the policy with **PPO** against the RM plus a KL penalty to the reference policy. Requires up to **four models resident**: policy, reference, critic/value network, reward model. Expensive, memory-heavy, notoriously hyperparameter-sensitive.

### 10.2 DPO (Rafailov et al., 2023)

[arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

- Introduces *"a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form,"* letting you *"solve the standard RLHF problem with only a simple classification loss."*
- **Removes three things:** the explicit reward-model fitting step, sampling from the LM during fine-tuning, and RL optimisation itself.
- Claims: *"stable, performant, and computationally lightweight"*; *"substantially simpler to implement and train."*
- Results: **exceeds** PPO-RLHF at sentiment control; **matches or improves** response quality on summarisation and single-turn dialogue.

### 10.3 DPO as actually implemented (TRL, 2026)

[TRL DPO trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)

- Data: `{prompt, chosen, rejected}`, standard or conversational form; the chat template is applied automatically.
- **β (default 0.1)** scales the log-ratio term: higher β = stronger tether to the reference policy; lower β = more aggressive preference optimisation and more drift.
- `ref_model=None` ⇒ the initial policy is used as reference. **With PEFT you get the reference for free by disabling the adapter** — no second copy of the model in memory. This is a genuinely useful trick.
- Loss variants shipped: `sigmoid` (default, Bradley–Terry), `ipo` (less overfitting-prone), `hinge` (RSO), `robust` (noisy labels via `label_smoothing`), `bco_pair`, `nca_pair`, `exo_pair`, **`sigmoid_norm`** (length-normalised, SimPO-style — addresses DPO's well-documented length/verbosity bias), and `sft`.
- **MPO-style multi-loss** is now first-class: `loss_type=["sigmoid","bco_pair","sft"]` with `loss_weights=[0.8,0.2,1.0]`. **Mixing an SFT term into DPO counteracts the "both rewards go negative" degradation** from §9.2.
- LR: **1e-6 full FT, 1e-5 for adapters.**

### 10.4 What is actually standard by 2026

The centre of gravity moved from *preference tuning* to **RL with verifiable rewards (RLVR)** and **GRPO-family** algorithms, driven by reasoning models.

**GRPO** as implemented in [TRL](https://huggingface.co/docs/trl/main/en/grpo_trainer), originally from DeepSeekMath (arXiv 2402.03300):

- **No value/critic model.** For each prompt, sample a **group of G completions** (`num_generations`, default 4, typical 2–8), score them, and use the **group-relative normalised advantage** `Â = (r_i − mean(r)) / std(r)`.
- **Reward *functions*, not reward models, are first-class:** any Python callable returning a float per completion — exact-match graders, unit tests, regex/schema validators, math checkers. `reward_funcs` accepts a list, can be `async`, and can return `None` to skip non-applicable tasks in multi-task training.
- **`beta` (KL penalty) defaults to 0.0 — KL regularisation is OFF by default**, with the docs noting recent papers show it is not essential. **This is a real break from the PPO-RLHF mental model**, where the KL term was load-bearing.
- Loss variants for known pathologies: `dapo` (better for long chain-of-thought, reduces length bias), `dr_grpo` (removes response-length bias by dividing by a constant L rather than per-response length; pair with `scale_rewards=False`), `sapo` (soft gating / smoother trust region).
- Example config from the docs: `learning_rate=5e-6`, `num_generations=4`, `loss_type="dapo"`, `num_iterations=1`.

**Practical 2026 selection rule:**

| You have | Use |
|---|---|
| Demonstrations of the right output | **SFT (+ LoRA)** |
| Pairwise human/model preferences, offline, compute-constrained | **DPO** (consider `sigmoid_norm`, or the `+sft` mixin) |
| A **programmatic verifier** — tests, exact answer, schema, tool success | **GRPO / RLVR** — this is where the field is |
| Genuinely need a learned scalar RM + on-policy RL at scale | PPO (rare outside frontier labs) |

> **For the price-prediction capstone, the third row is worth noticing.** "Is the predicted price within X% of truth" *is* a programmatic verifier. A GRPO-style setup with a numeric-closeness reward function is a legitimate, modern alternative to SFT on this task — and it is not what the course teaches. Whether it beats SFT here is an empirical question, not a settled one.

**Flag as outdated:** "RLHF means PPO + reward model, and DPO is the simpler new alternative." By 2026 that framing is two generations old. Also note that OpenAI's hosted **DPO** is subject to the same fine-tuning wind-down as SFT (§1). And **`UNVERIFIED`**: claims that DPO is "strictly worse than online methods" — the on-policy-vs-off-policy comparison is still contested and depends heavily on data freshness.

Supporting overviews (secondary, for framing only, not for figures): [HF post-training algorithms guide](https://huggingface.co/blog/karina-zadorozhny/guide-to-llm-post-training-algorithms), [llm-stats 2026](https://llm-stats.com/blog/research/post-training-techniques-2026), [Turingpost](https://www.turingpost.com/p/reasoning-rl-in-2026).

---

## 11. Evaluating a regression-style business task (the price-prediction capstone)

*Canonical shape: predict a numeric price from a free-text product description. This is also, not coincidentally, exactly the shape of a real Gewiss problem — predicting a list price or a bid price from a catalogue description.*

### 11.1 Why this task is a trap for LLM fine-tuning

The model emits **text**, not a float. So you have **two coupled failure modes**:

1. **Parsing failure** — non-numeric output, wrong units, currency symbols, ranges like "€40–€50", refusals.
2. **Accuracy failure** — the number is parseable and wrong.

> **Report parse-failure rate as a first-class metric, and define the policy for unparseable outputs *before* computing error.** Silently excluding them inflates every score, and **this is the single most common way these evaluations get quietly cheated** — including in published tutorials. Constrain the output shape with Structured Outputs or grammar-constrained decoding so parse failure ≈ 0.

### 11.2 Metrics, and what each is for

Let `y` = true price, `ŷ` = predicted.

| Metric | Formula | Optimal predictor | What it is for / against |
|---|---|---|---|
| **MAE** | mean\|ŷ − y\| | conditional **median** | In currency units, directly interpretable ("off by €14 on average"). Robust to outliers. **Usually the headline metric.** |
| **RMSE** | √mean(ŷ − y)² | conditional **mean** | Penalises large errors quadratically. On a price distribution spanning 3 orders of magnitude, RMSE is essentially "how well do you do on the expensive tail." |
| **MAPE / sMAPE** | mean\|ŷ−y\|/\|y\| | — | Scale-free, but **explodes for small y** and is **asymmetric** (bounded above by 100% for under-prediction, unbounded for over-prediction) → it **structurally rewards under-prediction.** Use with care, or not at all, when prices approach zero. |
| **RMSLE** | √mean(log(1+ŷ) − log(1+y))² | — | **The right default for prices.** See §11.3. |
| **hit@X%** | fraction within ±X% | — | **The metric non-technical stakeholders actually understand**, and the one that maps to a product decision ("is this listing mispriced?"). Report a ladder: hit@5% / 10% / 20%. |

Supporting metrics worth reporting: **R²**, or better, **skill score vs baseline** = `1 − MAE_model/MAE_baseline`; **median absolute error**; **calibration/bias** = `mean(ŷ − y)` to detect systematic under/over-prediction; and **error by price decile**, to expose where the model is actually bad.

### 11.3 Why log space matters for prices

1. **Prices are right-skewed and approximately log-normal, spanning orders of magnitude.** In raw space, **one €5,000 item contributes as much squared error as 10,000 €50 items.** Your metric — *and your training loss* — becomes a report on the tail.
2. **RMSLE measures *relative* error.** Since `log(1+ŷ) − log(1+y) ≈ log(ŷ/y)` for non-small values, RMSLE ≈ RMS of the log-ratio. Being off by €10 on a €20 item and by €10 on a €2,000 item are treated very differently — correctly, because the business consequence differs.
3. **RMSLE is asymmetric: it penalises under-prediction more than over-prediction.** `log(1+ŷ)−log(1+y)` for ŷ = y/2 is `−0.69`; for ŷ = 2y it is `+0.69` — symmetric in *ratio*, but in absolute currency terms halving is a much smaller absolute miss than doubling, so **relative to RMSE, RMSLE shifts penalty onto underestimates.** Know which direction your business prefers and pick accordingly.
4. The `1+` guards `log(0)` and keeps the metric defined for free items.
5. **The consequence for training, not just reporting:** if RMSLE or relative error is your objective, **train in log space** — have the model predict `log(price)`, or use a log-transformed target for your GBDT baseline. Minimising squared error on raw price when you will be scored on RMSLE is a straightforward objective mismatch.

> **The bias trap in step 5.** `exp(mean of log predictions)` is the **geometric mean**, which is a *biased* estimator of the arithmetic mean. If you need unbiased currency predictions — e.g. for summing a basket, or for revenue forecasting — apply a smearing/Duan correction, or predict in raw space with an appropriate loss. Reporting a log-space model's back-transformed predictions as if they were unbiased euro amounts is a real and common error.

Background reading, all secondary: [Kaggle discussion](https://www.kaggle.com/discussions/questions-and-answers/466599), [RMSE vs RMSLE](https://medium.com/analytics-vidhya/root-mean-square-log-error-rmse-vs-rmlse-935c6cc1802a).

### 11.4 Building a fair baseline — the ladder

This is where most write-ups cheat. All rungs on the **identical split** with the **identical metric**:

| # | Baseline | Cost | Note |
|---|---|---|---|
| 1 | **Constant predictor** | seconds | training-set **median** for MAE, **mean** for RMSE, **geometric mean** for RMSLE. Anything that doesn't beat this is broken. |
| 2 | **Simple group statistic** | seconds | median price per category/brand token. **Astonishingly strong on real catalogues.** |
| 3 | **Classical text baseline** | minutes, CPU | TF-IDF (word + char n-grams) → Ridge or LightGBM, on `log(price)` |
| 4 | **Frozen-embedding baseline** | one embedding pass + minutes | sentence-embedding model → GBDT/Ridge on `log(price)`. **This is the baseline to beat.** |
| 5 | **Prompted frontier model** | API cost | zero-shot and few-shot, same output constraint. Establishes "do I need to train anything at all." |
| 6 | **Fine-tuned open-weights model** | Colab GPU hours | LoRA/QLoRA |

### 11.5 Fairness rules

- **Split before doing anything.** For catalogues, split by **product/seller group** *and* by **time**. Random row splits leak near-duplicate listings and inflate scores enormously. **If prices drift, a temporal split is the only honest one** — and in an electrical-components catalogue with annual price-list revisions, prices absolutely drift.
- **Fit every transform (log, scaler, vectoriser, target encoding) on train only.** Target encoding of categorical columns is the classic leak.
- **Never touch the test set until the end.** Tune on validation.
- **Give every baseline a comparable tuning budget**, or explicitly disclose the asymmetry. The [arXiv:2207.08815](https://arxiv.org/abs/2207.08815) authors spent 20,000 compute-hours per learner precisely to remove this objection.
- **Report uncertainty.** Bootstrap the test set (e.g. 1,000 resamples) and give a CI on MAE and hit-rate. **A 2% MAE improvement with overlapping CIs is not a result.**
- **Report cost and latency alongside accuracy** — €/1k predictions and p50/p95 latency. **A fine-tuned 7B that beats GBDT by 3% MAE at 500× the inference cost is a negative result for most businesses**, and saying so out loud is the mark of someone who has actually shipped.
- **Add slice metrics** — error by price decile, by category, by description length. Aggregate MAE hides that the model is useless on the cheap tail.

---

## 12. Corrections table — claims that are wrong or outdated

| Claim | Status |
|---|---|
| "Let's fine-tune a GPT model via the OpenAI API" | **Not executable for new orgs since 2026-05-07.** Existing customers cut off 2027-01-06. Teach conceptually; check the [deprecations page](https://developers.openai.com/api/docs/deprecations) live. |
| "Fine-tuning teaches the model your data/facts" | **Wrong and actively harmful.** New-knowledge examples are learned slowly and **linearly increase hallucination rate** ([arXiv:2405.05904](https://arxiv.org/abs/2405.05904)). Use RAG for content, fine-tuning for shape. |
| "Fine-tuning costs = the training bill" | **Misses two of three cost centres:** persistently higher inference price, and forced re-training on every base-model deprecation. |
| "$X per 1M training tokens for gpt-4.1" | **Stale.** No longer listed on the pricing page; only `o4-mini` RFT at **$100/hour** remains. |
| "Apply LoRA to `q_proj` and `v_proj`" | **Superseded.** Target **all linear layers**; MLP matters more than attention (QLoRA; "LoRA Without Regret" — attention-only at r=256 lost to MLP-only at r=128). |
| "Tune `lora_alpha`" | **Don't.** The authors set it once and don't tune it; it is approximately equivalent to an LR change. Use `α=r` or `2r`, or rsLoRA's `α/√r` at high rank. |
| "LoRA always matches full fine-tuning" | **Domain-dependent.** True for style/format/task-shape on modest data with all-layer targeting; **false for large-scale domain absorption** ([arXiv:2405.09673](https://arxiv.org/abs/2405.09673)). |
| "Use the same LR you'd use for full fine-tuning" | **~10× too low for LoRA** (Thinking Machines). |
| "QLoRA is free — just always use 4-bit" | **It trades speed for memory.** If the model fits in bf16, prefer bf16 LoRA. |
| "Guanaco reached 99.3% of ChatGPT, so QLoRA loses nothing" | **Over-cited.** GPT-4-judged Vicuna benchmark, small prompt set, **2023**. Carries no information about 2026 models. |
| "Merge the LoRA into the 4-bit base" | **Lossy.** Merge into the **bf16** base, then re-quantise. Consider LoftQ init. |
| "4-bit quantization is essentially free" | **Model- and token-budget-dependent.** QiD grows with training tokens ([arXiv:2411.17691](https://arxiv.org/html/2411.17691v2)) and is worse on small heavily-trained models. |
| "Q4_K_M is 4 bits per weight" | **It is 4.89 bpw.** Nominal names understate real size by 15–25% ([llama.cpp README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)). |
| "W8A8-INT loses 10%+ accuracy" | **Refuted** by [arXiv:2411.02355](https://arxiv.org/html/2411.02355v4) — ~99.75% recovery at 8-bit with proper tuning. |
| "fp16 is the memory-efficient training dtype" | **No** — gradients go back to fp32 for the optimizer step, so you hold two copies. **bf16** is preferred on Ampere+. |
| "Watch validation loss to pick the best checkpoint" | **Insufficient.** Val loss can rise while the task metric improves. **Checkpoint on the downstream metric.** |
| "Tree models always beat deep learning on tabular data" | **Still true on medium/large tabular data; false on small data (<10K rows)**, where TabPFN-class models now win ([Nature 2025](https://www.nature.com/articles/s41586-024-08328-6)). |
| "RLHF = PPO + reward model; DPO is the new simple alternative" | **Two generations stale.** GRPO/RLVR with programmatic reward functions is 2026 practice, and TRL's GRPO **defaults KL off** (`beta=0.0`). |
| "The KL penalty is essential in RL fine-tuning" | **Not per TRL's GRPO defaults** (`beta=0.0`), which cite recent work showing it is not essential. |
| "Week 7 will run locally once I set up the environment" | **No.** The repo has no `peft`/`trl`/`bitsandbytes`/`accelerate`; `bitsandbytes` is CUDA-first. **Colab with a CUDA GPU is the intended path.** |
| "MAPE is a good scale-free price metric" | **Structurally rewards under-prediction** and explodes for small y. Prefer RMSLE + a hit-rate ladder. |
| "Random train/test split on a product catalogue" | **Leaks.** Near-duplicate listings inflate scores enormously. Split by product/seller group and by time. |

---

## 13. Items I could not verify

- **OpenAI SFT effective hyperparameter defaults** — the retrieved `n_epochs: 10, batch_size: 1, learning_rate_multiplier: 1` look like JSON-schema defaults rather than the historical `"auto"` behaviour. Also: per-example token limits were not stated on the page fetched.
- **The LoRA paper's Table 4 label for the 4.7M-parameter GPT-3 row** — the arithmetic says r=1 on `{Wq,Wv}`; a retrieval said r=4. Verify against the PDF before using either number.
- **"LoRA vs Full Fine-tuning: An Illusion of Equivalence"** (intruder dimensions) — not fetched; do not cite yet.
- **CrewAI / framework details, MCP specifics, deployment figures** — see the companion note `07-agents-and-deployment.md` §11 for that file's unverified list.
- **Exact lecture numbers** for Weeks 6–7 in Ed Donner's course — the research file does not record them, so the section-to-lecture mapping in these notes is not authoritative.
- **Whether a T4 Colab runtime supports the bf16 path** these recipes assume — Turing is pre-Ampere. Check at runtime; this determines whether `bf16=True` or `fp16=True` (with loss scaling) is correct for your session.
- **Any figure not present in the research file has been left absent rather than estimated.** Notably: there are no published perplexity deltas in the llama.cpp quantize README (speed only), and no measured Colab wall-clock times for the Week 7 recipe.

**Where a number matters to a decision, re-check it against the primary source.** The OpenAI deprecations and pricing pages in particular are live documents that changed materially in 2026 and will change again.
