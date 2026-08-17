# Handoff — the single re-entry point

**Last updated: 2026-08-17.** Read this first. It is the only handoff document;
the cloud/browser session that started this work is wound down, and its notes are
merged in here. It records *why* things are shaped the way they are, which the
code cannot tell you.

**Reader:** Bea (Italian, Gewiss). Studies this course in 6–8 hour blocks roughly
once a month. Wants challenge and correction over agreement, confidence tags on
claims, and the uncomfortable point first.

**Division of labour going forward:** everything happens in a local session in
this repo, with one exception — **one-shot lecture distillation stays in a
browser session**, because it needs to read an open Udemy transcript panel and an
open Google Slides tab. That session's only output is a
`study/notes/lectures/weekN/NNN-slug.md` file, filed here. See
[LECTURE-DISTILL.md](LECTURE-DISTILL.md).

---

## 1. The layering decision — the one that matters

The knowledge base has **three layers, and they are not equals.**

| Layer | Lives in | Built from | Authority |
| --- | --- | --- | --- |
| **Durable** | `study/notes/01-…08-*.md` | primary sources: arXiv, ACL Anthology, provider docs | highest — wins every disagreement |
| **Provenance (lectures)** | `study/notes/lectures/weekN/*.md` | Udemy transcript + slide deck | what the course claims, corrections flagged |
| **Provenance (notebooks)** | `study/docs/topics/*` | generated from the `.ipynb` files | which notebook demonstrates what, and what it output |

The generated topic pages do **not** explain concepts. Each opens with a callout
naming its reference and stating that the reference supersedes it; topics with no
reference say so rather than implying authority they don't have.

**Do not let the generator compete with the references.** A heading scraper
cannot produce a retrieval diagnostic tree. If a generated page starts looking
like it explains something, that content belongs in a reference instead.

Lecture notes should get **thinner** over time: on each integration pass, general
facts migrate up into the topic references and leave a link behind.

### Reference coverage, and the one real gap

| Reference | Covers | Topics mapped |
| --- | --- | --- |
| `01-llm-foundations.md` | transformers, tokenization, context, KV cache, prompt caching, MoE | prompt-engineering, tokenization |
| `04-model-selection-benchmarks-leaderboards.md` | scaling laws, benchmarks, contamination, leaderboards, EU/GDPR | model-selection |
| `05-rag-and-vector-search.md` | embeddings, chunking, vector stores, reranking, RAG eval | rag-systems, embeddings, vector-databases |
| `06-training-and-finetuning.md` | LoRA, QLoRA, quantization, hyperparameters, DPO | fine-tuning, dataset-prep, evaluation, gpu-computing |
| `07-agents-and-deployment.md` | agent framing, multi-agent, MCP, deployment, Gradio | agent-architectures, tool-use, multi-agent, deployment, ui-interaction |
| `08-api-keys-and-runnability.md` | what runs with a Google-only key, litellm swaps | llm-apis, multi-model |
| `LECTURE-DISTILL.md` | the per-lecture distillation workflow | — |

**`dev-tools` is the only topic with no durable layer**, and none is planned —
Jupyter and git workflow are covered well enough by `guides/`.

`04-model-selection-benchmarks-leaderboards.md` opens with "read the corrections
table (§7) first — several things this week teaches are outdated, and one of them
will actively mislead you." Take that literally before doing week 4.

**Numbering gap `02-`/`03-` is deliberate** — weeks 1–3 were merged into `01-`.
Leave the gap or fill it if week 2/3 material grows.

Each topic file follows one shape: a `## 0. The one-paragraph version` opener,
inline source links on every figure, `UNVERIFIED` tags where a claim could not be
confirmed, a **corrections table** near the end, and a closing list of unverified
items. Match that structure if you add to them.

---

## 2. Where everything lives

Everything added to this fork lives under `study/`. Nothing else in the repo is
touched, so `git log -p study/` is the whole history of this work and the diff
against upstream stays trivially separable.

```text
study/
  README.md            ← points here
  notes/               ← HAND-WRITTEN. The only files you edit by hand.
    HANDOFF.md         ← this file
    01-…08-*.md        ← topic references
    LECTURE-DISTILL.md ← distillation spec
    OPEN-QUESTIONS.md  ← merged `Open` sections, grouped by theme
    lectures/
      INDEX.md         ← generated
      weekN/NNN-slug.md
      _slides/         ← gitignored; decks are read from a Drive tab
  pipeline/            ← the extractor: code, config, templates, tests
  docs/                ← MkDocs content. ALL BUILD OUTPUT except index.md
  mkdocs.yml
  nb2md.py, nb2mdREADME.md   ← the repo's single notebook parser
  _site/               ← rendered HTML, gitignored
```

**`study/knowledge-base` does not exist and must not** — `week5/knowledge-base/`
is the course's RAG corpus, and the name was ambiguous while both existed.

`study/docs/reference/*` and `study/docs/lectures/*` are copies of
`study/notes/`, overwritten on every run. Edit `study/notes/`.

Publishing is opt-in: a reference goes live only once listed in
`REFERENCE_NOTES` in `study/pipeline/src/generators/reference_sync.py`, so
drafts can sit in `study/notes/` without appearing on the site. Lecture notes
publish automatically, but individual pages are deliberately left out of the
MkDocs nav — there are 210 lectures; `lectures/index.md` is the entry point.

---

## 3. Facts verified against this clone — trust these

`ed-donner/agentic`, `ed-donner/agents`, `ed-donner/production` are *different
courses*, not a rebrand of this one.

**Structure:** 8 `week*` folders, `guides/` (14 notebooks), `setup/`, `extras/`,
`community-contributions/`. Always exclude `community-contributions` **and**
`community_contributions` (the spelling is inconsistent) from any repo-wide
operation.

**Notebook audit, run 2026-08-17 after the upstream merge:**

```bash
python study/nb2md.py . --recursive \
  --exclude community-contributions --exclude community_contributions --audit
```

> 66 scanned — **11 executed, 47 stripped, 8 Colab stubs.**

The 8 stubs are exactly `week3/day1`–`day5`, `week7/day1`, `week7/day3 and 4`,
`week7/day5`. `week7/day2` is the exception — real local code, CPU-fine. Weeks 1,
2, 4, 5, 6, 8 are entirely local. Re-run the audit at the start of a block; if
the stub list changes, this section needs re-verifying.

**Setup is uv-only.** `uv sync`, driven by `setup/SETUP-new.md`. No per-OS setup
files, no `requirements.txt` (upstream removed it). `environment.yml` is dead
legacy — zero conda references in the README, setup docs or guides.
`.python-version` pins **3.12.12**; `pyproject.toml` only floors at `>=3.11`.

**GPU reality:** `week3/day4`, `day5` and all of week 7 need CUDA. `week3/day1`
and `day3` need no GPU and are Colab-only by convention. **Week 8 needs no local
GPU** — it rents a T4 from Modal (`pricer_service2.py`, `pricer_ephemeral.py`,
`llama.py`). `week6/redemption_train.ipynb` is marked very optional and takes
hours; `redemption_run.ipynb` loads pretrained weights instead.

**Week 7 cannot run locally, notebooks or not:** no `peft`, `trl`,
`bitsandbytes` or `accelerate` in `pyproject.toml` or `uv.lock`. Deliberately
not provisioned for QLoRA.

**Nothing strips notebook outputs.** No root `.gitattributes`, no
`.pre-commit-config.yaml`, no `nbstripout`, no `.github/`. The clear-outputs rule
is a human PR convention (`guides/03_git_and_github.ipynb`). **Executed outputs
you save will commit verbatim** — which is the whole basis of the extraction
pipeline, since a saved output is evidence of what the code actually did.

**litellm is already the repo's own idiom** — `litellm>=1.77.5` in
`pyproject.toml`, and `from litellm import completion` in 11 files across weeks
2, 5, 6 and 8. `week8/agents/preprocessor.py` defaults to
`"groq/openai/gpt-oss-20b"`. The OpenAI-hardcoded notebooks are the inconsistent
ones, not the litellm ones.

**Slides** live in a Google Drive folder linked from
`edwarddonner.com/2024/11/13/llm-engineering-resources/` — **not on Udemy**.
Eight week folders, **five per-day decks each** (`Copy of LLM - Week N Day D`,
Google Slides, 12–25 MB). One deck covers all of that day's lectures; Week 5
Day 1 is 11 slides across lectures 106–111.

---

## 4. Two live constraints that change what to do

**(a) Only a Google API key.** No OpenAI, no Anthropic key. Gewiss uses Anthropic
internally but there is no key for this course. Roughly **13 core notebooks name
OpenAI and nothing else**; week 5 is worst hit (4 of 5 local notebooks).
Per-notebook table in `08-api-keys-and-runnability.md`.

The fix is the **litellm** swap, not the `OPENAI_BASE_URL` redirect — it keeps
the OpenAI-shaped response object, reads the right env var per provider, and
doesn't silently redirect every OpenAI call in the process:

```python
# from
from openai import OpenAI
r = OpenAI().chat.completions.create(model="gpt-4o-mini", messages=msgs)
# to
from litellm import completion
r = completion(model="gemini/gemini-2.5-flash", messages=msgs)
```

For week 5 specifically, **don't swap to Gemini — go local.**
`multilingual-e5-large` (560M, open-weight) came within **0.003 nDCG** of Google
Embeddings 2 on Italian retrieval at **1/7 the latency**. On Italian corpora the
paid API buys nothing measurable, and local keeps Gewiss data in-boundary.

**(b) Week 6's frontier fine-tuning exercise may be inexecutable.** [Likely]
OpenAI has been winding down self-serve fine-tuning for orgs that never ran a
job, which is probably Gewiss. Unresolved — see `OPEN-QUESTIONS.md`. If blocked,
redirect week 6's hands-on time to the classical-ML and frozen-embedding baseline
ladder in `06-training-and-finetuning.md`; it needs no paid key and it is what
week 7's fine-tuned model must beat for the capstone to mean anything.

---

## 5. Pipeline decisions

**nb2md is the single notebook parser.** `study/nb2md.py` is
stdlib-only and standalone; `src/extractors/nb2md_loader.py` loads it by path and
caches it. The `nbformat` dependency is gone. This is what gives the extractor
stub detection — without it, week 3 reported "1 unique concept across 5 days"
when in truth all five notebooks are Colab links with no local content.

**A code cell with saved output is always kept**, even when the code looks
trivial. The output is the evidence; discarding it defeats the point.

**Heading scrapes are filtered for lecture narration.** The notebooks are
conversational, so raw headings yield "Donezo! On to Step 2" as a learning
objective. Filter is `is_narration()` — a heuristic. Add new patterns to
`NARRATION_PREFIXES` / `NARRATION_SUBSTRINGS` **with a test case**.

**Generated docs are committed.** Rendered Markdown diffs readably in git, so
`git log -p` shows how understanding changed between study blocks. That is the
same reason `notes/` is committed and `_slides/` is not.

**Paths in generated docs are repo-relative POSIX**, and **console output is
ASCII-only** — a cp1252 Windows terminal crashes on `✓`.

---

## 6. Deliberately not done

- **`docs/projects/*` and `docs/quick-ref/*` are still placeholders.** Nothing
  generates them. Decide whether they deserve a generator or should be
  hand-written like the references.
- **`--use-llm` has never been run against the topic-page path.** Wired, one call
  per topic, 18 calls — unproven. Try Ollama first; it's free. If it doesn't earn
  its place, delete the code path rather than leaving an untested option.
- **Website scraping is untested.** `extract_website_phase` warns and continues
  by default; `--strict-website` makes it fail loudly. Nobody knows if it ever
  worked.
- **README MD059 warnings are left unfixed.** Rewriting Ed's "[here]" link text
  would conflict on every future upstream merge.

---

## 7. Upstream (surveyed 2026-08-17)

Merged `upstream/main` at `109b683e`; the fork is level with upstream. 1,169
commits, small core impact: 9 notebooks, `requirements.txt` removed in favour of
pyproject/uv, README refreshed. Only `README.md` conflicted.

**`week3/QUESTHER_V3_WEEK3.md` and `week4/QUESTHER_V3_WEEK4.md` were removed
upstream.** They were stray root copies of a contributor's submission, never
yours; the originals remain under `community-contributions/`. Do not build a
portfolio narrative on them and do not imitate their style — unsubstantiated
claims ("Success Rate: > 90%") with no benchmark behind them.

**Community contributions — low yield.** 409 new directories: week1 302, week2
110, week5 65, week4 36, week3 28, week6 8, weeks 7–8 zero. Three-quarters is
week 1–2, i.e. hundreds of variants of "summarize a website". The weeks where an
alternative approach would teach something have 8 directories between them.
Scope any attempt to weeks 5–6, or skip it.

**nb2md is novel upstream** — no notebook-to-Markdown renderer exists there. The
two near-misses do different jobs (`week4_auto_markdown_comments.ipynb` annotates
code; `files_to_markdown_converter.py` prepares RAG inputs). A PR is worth
making — branch off a clean `upstream/main`, not `linting-setup-improvements`,
which carries unrelated linting and study-tree changes.

---

## 8. Next steps, in priority order

1. **Run the distillation loop.** Everything built so far is scaffolding for it.
   The concept-only lectures are the ones worth doing — they are the ones with no
   other source. 208 lectures remain; you do not need them all. Lectures 109 and
   110 were watched but skipped; see `OPEN-QUESTIONS.md`.
2. **Answer the two cheap items in `OPEN-QUESTIONS.md`** — the token size of
   `week5/knowledge-base/`, and whether the OpenAI fine-tuning account is
   blocked. Both change what you do next, and both take minutes.
3. **Fact-check pass on the topic references.** Never run. Highest-risk items are
   listed in `OPEN-QUESTIONS.md`; the `UNVERIFIED` tags mark the ones that need a
   browser rather than a fetch tool, and they cluster in `04-…md`.
4. **PR nb2md upstream** (§7).
5. **Download the 8 Colab stubs**, if week 3 / week 7 content matters to you.

---

## 9. Standing constraints — don't relitigate

- **No bulk transcript extraction.** No crawling the course, no scripted
  pagination, no pacing to evade bot detection. One lecture at a time, on a page
  Bea has open, on explicit request.
- **Distillations are condensed notes, not copies.** Substantially shorter,
  restructured, merged with slide content and primary sources. A note that reads
  like a transcript has failed.
- **Don't reconstruct slide decks from video frames.** The decks are in Drive.
- **Don't commit slide decks or downloaded course materials.**
- **Never invent a figure or a source URL.** If a number isn't available, write
  that it isn't. Every topic file carries an explicit unverified list; keep that
  habit.

---

## 10. Running it

```bash
# regenerate everything from the notebooks (free, no LLM)
python study/pipeline/extract_all.py --skip-website

# regenerate pages only, from the cached extraction - this is the usual one
python study/pipeline/extract_all.py --generate-only

# tests
python -m unittest discover -s study/pipeline -t study/pipeline

# serve the site
cd study && python -m mkdocs serve
```

`mkdocs build --strict` must stay clean — it is what catches broken cross-links
between the references, the lecture notes and the generated pages. The extraction
cache is `study/pipeline/.cache/extracted.json`, gitignored.
