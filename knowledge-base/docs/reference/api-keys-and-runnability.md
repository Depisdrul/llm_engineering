<!-- Generated copy. Source: notes/08-api-keys-and-runnability.md
     Edit the source file; this copy is overwritten on every extraction run. -->

# Operational — which notebooks you can actually run with a Google key

**Why this file exists:** you hold a Google API key for closed models and nothing else. The course is written OpenAI-first. This maps what runs, what doesn't, and the one change that fixes most of it.

**How this was built:** by grepping every core notebook in a fresh clone of `ed-donner/llm_engineering` for provider references. Counts are *mentions*, so treat the classification as a strong signal, not a contract — verify per notebook before a study block.

---

## 0. The one-paragraph version

Roughly **half the core notebooks name OpenAI and nothing else**, so with a Google key alone they fail at the first cell. The fix is not editing 30 notebooks: it's Google's **OpenAI-compatibility endpoint**, which lets the unmodified `OpenAI()` client talk to Gemini by changing two arguments — `base_url` and `api_key`. Set those once in `.env`, patch the client construction in each notebook (usually one line), and most OpenAI-only notebooks run against Gemini. Add **Ollama** for the open-model work and you need no paid key at all for a large fraction of the course. Do this setup *before* your next block, not during it.

---

## 1. The runnability map

**Legend:** ✅ runs as written · 🔁 needs a provider swap · 🔑 needs a HuggingFace token · ⛔ Colab stub, nothing local · 🖥️ needs CUDA GPU

| Notebook | Providers named in-notebook | Status with a Google key only |
|---|---|---|
| `week1/day1` | OpenAI, Anthropic, Google, Ollama | ✅ multi-provider already |
| `week1/day2` | OpenAI, Google, Ollama (heavy) | ✅ this is the Ollama/alternatives day |
| `week1/day4` | OpenAI only | 🔁 swap |
| `week1/day5` | OpenAI only | 🔁 swap |
| `week2/day1` | OpenAI, Anthropic, Google, Ollama | ✅ multi-provider — the 3-model comparison day |
| `week2/day2` | OpenAI, Anthropic, Google | ✅ |
| `week2/day3` | OpenAI only | 🔁 swap |
| `week2/day4` | OpenAI, Ollama | 🔁 swap (tool use / function calling) |
| `week2/day5` | OpenAI only | 🔁 swap — multimodal; check Gemini image/audio parity |
| `week3/day1–day5` | — | ⛔ Colab stubs. day4/day5 also 🖥️ |
| `week4/day3` | OpenAI, Anthropic, Google | ✅ |
| `week4/day4` | OpenAI, Anthropic, Google, Ollama | ✅ |
| `week4/day5` | OpenAI, Anthropic, Google, Ollama | ✅ |
| `week5/day1` | OpenAI only | 🔁 swap |
| `week5/day2` | OpenAI only | 🔁 swap + 🔑 |
| `week5/day3` | OpenAI only | 🔁 swap |
| `week5/day4` | — *(ships with outputs — readable without running)* | ✅ |
| `week5/day5` | OpenAI only | 🔁 swap |
| `week6/day1` | — | 🔑 only |
| `week6/day2` | OpenAI, Ollama | 🔁 light |
| `week6/day3` | — | ✅ |
| `week6/day4` | OpenAI, Anthropic, Google, HF | ✅ + 🔑 |
| `week6/day5` | OpenAI (heavy) | 🔁 **this is the frontier fine-tuning day — see §4** |
| `week6/redemption_*` | — | 🔑, and `redemption_train` wants a GPU (use `redemption_run` + pretrained weights) |
| `week7/day2` | HF | 🔑 — the one real week-7 notebook, CPU-fine |
| `week7/day1, day3&4, day5` | — | ⛔ + 🖥️ Colab only |
| `week8/day1–day5` | OpenAI | 🔁 swap; day1/day2 also 🔑. **No local GPU needed** — Modal rents a T4 |
| `guides/09_ai_apis_and_ollama` | OpenAI, Google, Ollama, Anthropic | ✅ **read this first — it is the alternatives guide** |

**Counts:** ~13 notebooks need a swap, ~9 run multi-provider as written, ~9 are Colab stubs or GPU-bound, and week 5 is the worst-affected local week (4 of 5 notebooks are OpenAI-only).

---

## 2. The fix: Google's OpenAI-compatibility endpoint

Google publishes an OpenAI-compatible surface — [official docs](https://ai.google.dev/gemini-api/docs/openai). You keep the `openai` SDK and change two arguments:

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",          # substitute a current model id
    messages=[{"role": "user", "content": "..."}],
)
```

Every notebook that does `openai = OpenAI()` then `openai.chat.completions.create(model="gpt-4o-mini", ...)` becomes runnable by changing the constructor and the model string. That is **two edits per notebook**, not a rewrite.

**Set it once in `.env`** so you are not editing constructors by hand:

```
GOOGLE_API_KEY=...
OPENAI_API_KEY=...                                              # same Google key
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

The `openai` SDK reads `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the environment automatically, so a bare `OpenAI()` picks both up and **many notebooks then need only the model-name change.** This is the highest-leverage 15 minutes you can spend on this course.

> **Caveats, and they are real.** [Likely] Coverage of the compatibility layer is not complete — streaming, tool/function calling, structured outputs, embeddings, and multimodal inputs each have their own support status, and there are [reported mismatches around Chat Completions vs the Responses API](https://github.com/twentyhq/twenty/issues/16213). Expect `week2/day4` (tool use), `week2/day5` (multimodal) and `week5` (embeddings) to be where it breaks. Check the compatibility doc's support table before assuming a notebook will work, and keep the native `google-genai` SDK as the fallback for those.

**A cleaner alternative already in the repo:** `pyproject.toml` includes **`litellm`**, which normalises provider APIs behind one call signature. If the compatibility endpoint fights you on tool use or embeddings, routing through litellm is the more robust swap — one import change, provider-prefixed model strings (`gemini/gemini-2.5-flash`).

---

## 3. Ollama covers the rest, for free

`week1/day2` and `guides/09_ai_apis_and_ollama.ipynb` are built around Ollama specifically as the no-paid-key path — the README calls guide 09 out for exactly this. Ollama runs open-weight models locally through an **OpenAI-compatible endpoint of its own** (`http://localhost:11434/v1`), so the same two-argument swap works:

```python
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

This matters beyond cost: for anything involving Gewiss documents, a local model means **the data never leaves your machine**, which sidesteps the DPA/residency questions in `04-model-selection-benchmarks-leaderboards.md` §5.3 entirely. For prototyping RAG over Italian internal documents, that is the right default regardless of budget.

**Embeddings too.** Week 5's OpenAI-only notebooks need an embedding model, and per `05-rag-and-vector-search.md` the finding that matters for you is that **`multilingual-e5-large` (560M, open-weight, runs locally) came within 0.003 nDCG of Google Embeddings 2 on Italian retrieval at 1/7 the latency.** So for week 5 the local path isn't a compromise — on Italian it is the better engineering choice. Swap `OpenAIEmbeddings` for a `sentence-transformers` model and you lose nothing measurable.

---

## 4. Week 6 day 5 — plan for this to be blocked

`week6/day5.ipynb` is the frontier fine-tuning notebook and the most OpenAI-dependent file in the repo. Two independent reasons it may not run for you:

1. **The compatibility endpoint does not cover fine-tuning.** It's a chat-completions shim; there is no Gemini fine-tuning behind it.
2. **[Likely] OpenAI has been winding down self-serve fine-tuning.** Per its published deprecations, organisations that never previously ran a fine-tuning job can no longer create one. You've said Gewiss uses Anthropic internally and AI use is still exploratory — which makes it very likely nobody there has ever run an OpenAI fine-tuning job, so the account would fall on the blocked side of that line.

**Do this rather than discovering it mid-block:** check the Udemy Q&A for other students reporting it (cheapest signal), and separately try to create a trivial fine-tuning job on your own key before the block. If it's blocked, treat week 6 as conceptual and spend the hands-on time on the baseline ladder in `06-training-and-finetuning.md` — the classical-ML and frozen-embedding baselines are the part of week 6 with lasting value anyway, they need no paid key, and they're what week 7's fine-tuned model has to beat for the capstone to mean anything.

---

## 5. Suggested order for your next block

1. **Before the block** (10 min, no study time): set the three `.env` lines, `uv sync`, install Ollama and pull one model, test one cell of `week1/day4`.
2. **`guides/09_ai_apis_and_ollama.ipynb`** — 87 OpenAI references and 37 Ollama ones; it is the Rosetta stone for every swap in the table above. Reading this first makes the other 13 swaps mechanical.
3. **Weeks 4 and 6 (partial)** — the multi-provider notebooks already accept a Google key. Lowest friction, so put them where your energy is lowest.
4. **Week 5 with local embeddings** — reframe it as "build Italian RAG with an open-weight embedder," which is both the correct engineering choice and removes the OpenAI dependency.
5. **Week 3 and 7 Colab downloads** — see `nb2md-README.md`. Do the downloads at the *start* of a block while you have GPU quota and patience.
6. **Week 8 last** — needs Modal (which rents its own T4) plus the swap; most moving parts.

---

## 6. Unverified / check before relying on it

- Exact coverage of Google's OpenAI-compatibility layer for tool calling, structured outputs, embeddings, streaming, and multimodal input, as of your study date. The support matrix moves; read [the doc](https://ai.google.dev/gemini-api/docs/openai).
- Current Gemini model IDs and whether the free tier's rate limits are sufficient for the notebooks that loop over many items (week 6 data curation and week 8 embedding builds are the volume-heavy ones).
- Whether OpenAI fine-tuning is creatable on the Gewiss account — only an actual attempt answers this.
- The provider counts in §1 are grep-based mention counts on the `main` branch as cloned; a notebook could import a provider indirectly through `pricer/` or `agents/` modules that the grep didn't cover.

Sources: [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai) · [reported Chat Completions mismatch](https://github.com/twentyhq/twenty/issues/16213) · [course resources page](https://edwarddonner.com/2024/11/13/llm-engineering-resources/)
