# Verified facts — `ed-donner/llm_engineering`

Verified by cloning and inspecting the repo directly, then re-verified against the fork on 17 Aug 2026 after merging `upstream/main` at `109b683e`. Re-verify before relying on any of it; the repo is actively maintained.

Re-run the audit at the start of a study block:

```bash
python study/nb2md.py . --recursive \
  --exclude community-contributions --exclude community_contributions --audit
```

Not the same course: `ed-donner/agents` (Agentic Track), `ed-donner/agentic` (an O'Reilly live event — a near-miss name trap), `ed-donner/production` (Production Track), `ed-donner/tech2ai`.

## Structure

Eight `week*` folders, plus `guides/` (14 notebooks), `setup/`, `extras/`, `assets/`, `community-contributions/`.

**65 core notebooks. ~3,044 community-contribution notebooks across ~273 folders.** Always exclude both `community-contributions` and `community_contributions` from repo-wide operations — the spelling is inconsistent between weeks, and week 7 uses the underscore form.

Notebooks are `day1.ipynb` … `day5.ipynb`. Week 4 has no `day1`/`day2` notebook. Week 7 has a file literally named `day3 and 4.ipynb` (spaces included).

## Setup

**uv only.** `uv sync`, per `setup/SETUP-new.md`. There are **no** per-OS setup files (`SETUP-mac.md` etc. all 404) and **no `requirements.txt` anywhere**. `environment.yml` is a dead legacy file — zero references to conda in the README, setup docs, or any guide.

`.python-version` pins **3.12.12**. `pyproject.toml` only floors at `>=3.11`. Cursor is the assumed IDE, not JupyterLab.

Also in `setup/`: `troubleshooting.ipynb`, `diagnostics.ipynb`, `diagnostics.py`.

## Colab stubs — 8 notebooks contain only a Drive link

`week3/day1`–`day5`, `week7/day1`, `week7/day3 and 4`, `week7/day5`.

**`week7/day2` is the exception** — real local code, ~13 cells, CPU-fine (loads a HF dataset, tokenizes, plots token histograms).

Weeks 1, 2, 4, 5, 6, 8 are entirely local. There are **no Colab badges and no filename convention** — a stub is just a near-empty notebook holding one markdown cell with a `colab.research.google.com/drive/...` link.

## GPU reality

| Needs CUDA | Doesn't |
|---|---|
| `week3/day4`, `week3/day5` (4-bit quantized inference, Whisper) | `week3/day1` (Colab tour), `week3/day3` (tokenizers — Colab by convention only) |
| All of week 7 | **All of week 8** — GPU is rented from Modal (`GPU = "T4"` in `pricer_service2.py`, `pricer_ephemeral.py`, `llama.py`) |
| `week6/redemption_train.ipynb` (~hours, marked very optional) | `week6/redemption_run.ipynb` — loads pretrained weights instead |

**Week 7 cannot run locally, notebooks or not:** `pyproject.toml` and `uv.lock` contain **no `peft`, `trl`, `bitsandbytes`, or `accelerate`**. Deliberately not provisioned for QLoRA.

## Notebook outputs are never stripped

No root `.gitattributes`, no `.pre-commit-config.yaml`, no `nbstripout`, no `.github/` directory, no filter in a fresh clone's `.git/config`. `.gitignore` contains only `.ipynb_checkpoints` for notebooks.

**Executed outputs commit verbatim.** The clear-outputs rule is a human PR convention documented in `guides/03_git_and_github.ipynb` and `guides/05_notebooks.ipynb`, enforced by nothing.

Current state in this fork: **12 executed, 45 stripped, 8 Colab stubs** of 65. Upstream ships 4 executed; the other 8 are notebooks Bea ran locally and committed with their outputs, which is deliberate — a saved output is the evidence the extraction pipeline reads.

## litellm is the repo's own idiom

`litellm>=1.77.5` in `pyproject.toml` (1.79.1 locked). `from litellm import completion` appears in **11 files**: `week2/day1`, `week5/day5`, `week5/evaluation/eval.py`, `week5/pro_implementation/answer.py`, `week5/pro_implementation/ingest.py`, `week6/day2`, `week6/day4`, `week6/pricer/preprocessor.py`, `week8/day2`, `week8/agents/messaging_agent.py`, `week8/agents/preprocessor.py`.

`week6/pricer/preprocessor.py` defaults to `"groq/openai/gpt-oss-20b"`. **The OpenAI-hardcoded notebooks are the inconsistent ones.** Converting one is a two-line diff:

```python
# from
from openai import OpenAI
r = OpenAI().chat.completions.create(model="gpt-4o-mini", messages=msgs)
# to
from litellm import completion
r = completion(model="gemini/gemini-2.5-flash", messages=msgs)
```

Provider prefixes route the call: `gemini/`, `ollama/`, `groq/`, `anthropic/`. Response is OpenAI-shaped, so downstream `.choices[0].message.content` is untouched.

Also in `pyproject.toml`: `torch>=2.8.0`, `transformers>=4.56.2`, `sentence-transformers`, `chromadb`, `gradio<6.0`, `modal`, `xgboost`; pins `datasets==3.6.0`, `protobuf==3.20.2`.

## Provider dependencies per notebook

Grep-based mention counts, so a strong signal rather than a contract — a notebook may import a provider indirectly via `pricer/` or `agents/` modules.

**OpenAI-only (~13):** `week1/day4`, `week1/day5`, `week2/day3`, `week2/day4`, `week2/day5`, `week5/day1`, `week5/day2`, `week5/day3`, `week5/day5`, `week6/day5`, `week8/day1`–`day5`.

**Already multi-provider:** `week1/day1`, `week1/day2`, `week2/day1`, `week2/day2`, `week4/day3`, `week4/day4`, `week4/day5`, `week6/day4`.

**Week 5 is worst hit** — 4 of 5 local notebooks are OpenAI-only.

**HF token needed:** `week5/day2`, `week6/day1`, `week6/day4`, `week6/day5`, `week6/redemption_*`, `week7/day2`, `week8/day1`, `week8/day2`.

**`guides/09_ai_apis_and_ollama.ipynb`** is the alternatives guide — 87 OpenAI references, 37 Ollama, 14 Google. The README calls it out as the free path. Read it before doing any provider swap.

## Slides

Google Drive folder linked from `edwarddonner.com/2024/11/13/llm-engineering-resources/` — **not on the course platform**.

```text
AI Engineer Core Track/          <- drive.google.com/drive/folders/1GMXbdgkqnZfCRcIdoUVBBB-hxeN4Lo06
  Week 1/ … Week 8/
    Copy of LLM - Week N Day 1     ← Google Slides, 12–25 MB
    … Day 2, Day 3, Day 4, Day 5
```

**Per-day decks, five per week.** All shared by "ed", modified 20 Dec 2025. Week 5 Day 1's deck is 11 slides and covers lectures 106–111.

## Guides worth knowing

`guides/` holds 14 notebooks, an often-overlooked part of the repo: command line, git/GitHub (incl. the PR checklist), technical foundations, notebooks in Cursor, Python foundations, vibe coding, debugging survival guide, **LLM APIs and Ollama beyond OpenAI**, intermediate Python, async Python, starting your project, frontend crash course, Docker/Terraform.

External: `edwarddonner.com/faq` (Q7 `.env`, Q11 uv problems, Q15 corporate SSL), `edwarddonner.com/pr`, and the resources page with the slide links.

## Course shape

210 lectures, 33.5 hours, last updated June 2026, instructor Ed Donner under the Ligency brand. Eight weeks: foundations → frontier APIs/multimodal → open source with HuggingFace → LLM selection and code gen → RAG → fine-tuning a frontier model → QLoRA on open source → agentic finale.
