# Decisions — why this is shaped the way it is

**Last updated: 2026-08-17.** The standing record of choices that the code
cannot explain. Not a facts sheet, not a command reference, not a task list —
those live elsewhere and are linked below.

**Reader:** Bea (Italian, Gewiss). Studies this course in 6–8 hour blocks roughly
once a month. Wants challenge and correction over agreement, confidence tags on
claims, and the uncomfortable point first.

| Looking for | Read |
| --- | --- |
| Layout, commands, entry point | [`study/README.md`](../README.md) |
| What the pipeline does, and what is unproven | [`study/pipeline/README.md`](../pipeline/README.md) |
| Verified facts about the course repo | `.claude/skills/course-distill/references/repo-facts.md` |
| The distillation workflow | `.claude/skills/course-distill/SKILL.md` |
| Where lecture notes go and how they publish | [`LECTURE-DISTILL.md`](LECTURE-DISTILL.md) |
| Which days are worth distilling first | [`SLIDE-COVERAGE.md`](SLIDE-COVERAGE.md) |
| What is still unresolved | [`OPEN-QUESTIONS.md`](OPEN-QUESTIONS.md) |

---

## 1. The layering decision — the one that matters

Three layers, and they are **not equals**.

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

### Reference coverage

| Reference | Covers | Topics mapped |
| --- | --- | --- |
| `01-llm-foundations.md` | transformers, tokenization, context, KV cache, prompt caching, MoE | prompt-engineering, tokenization |
| `04-model-selection-benchmarks-leaderboards.md` | scaling laws, benchmarks, contamination, leaderboards, EU/GDPR | model-selection |
| `05-rag-and-vector-search.md` | embeddings, chunking, vector stores, reranking, RAG eval | rag-systems, embeddings, vector-databases |
| `06-training-and-finetuning.md` | LoRA, QLoRA, quantization, hyperparameters, DPO | fine-tuning, dataset-prep, evaluation, gpu-computing |
| `07-agents-and-deployment.md` | agent framing, multi-agent, MCP, deployment, Gradio | agent-architectures, tool-use, multi-agent, deployment, ui-interaction |
| `08-api-keys-and-runnability.md` | what runs with a Google-only key, litellm swaps | llm-apis, multi-model |

**`dev-tools` is the only topic with no durable layer**, and none is planned —
Jupyter and git workflow are covered well enough by `guides/`.

**Numbering gap `02-`/`03-` is deliberate** — weeks 1–3 were merged into `01-`.
Leave the gap or fill it if week 2/3 material grows.

Each reference follows one shape: a `## 0. The one-paragraph version` opener,
inline source links on every figure, `UNVERIFIED` tags where a claim could not be
confirmed, a **corrections table** near the end, and a closing list of unverified
items. Match it if you add to them.

`04-…md` opens by telling you to read its corrections table first, because "one
of them will actively mislead you." Take that literally before doing week 4.

---

## 2. Division of labour

Everything happens in a local session in this repo, with one exception —
**one-shot lecture distillation stays in a browser session**, because it needs to
read an open Udemy transcript panel and an open Google Slides tab. That session's
only output is a `study/notes/lectures/weekN/NNN-slug.md` file, filed here.

The workflow is the `course-distill` skill. The browser holds its own copy at
`claude.ai/customize/skills`, and the two can drift with nothing to flag it;
mirror any edit into the repo copy in the same sitting.

---

## 3. Pipeline decisions

**nb2md is the single notebook parser.** `study/nb2md.py` is stdlib-only and
standalone; `nb2md_loader.py` loads it by path and caches it. No `nbformat`
dependency. This is what gives the extractor stub detection — without it, week 3
reported "1 unique concept across 5 days" when in truth all five notebooks are
Colab links with no local content.

**A code cell with saved output is always kept**, even when the code looks
trivial. The output is the evidence; discarding it defeats the point. It is also
why executed notebooks are committed with their outputs, against the course's own
PR convention.

**Heading scrapes are filtered for lecture narration.** The notebooks are
conversational, so raw headings yield "Donezo! On to Step 2" as a learning
objective. `is_narration()` is a heuristic — add patterns **with a test case**.

**Generated docs are committed.** Rendered Markdown diffs readably in git, so
`git log -p` shows how understanding changed between study blocks. Same reason
`notes/` is committed and `_slides/` is not.

**Everything added to this fork lives under `study/`** — one collision surface
with upstream, one folder to separate for a PR, and `git log -p study/` is the
whole history. `study/knowledge-base` must never exist: `week5/knowledge-base/`
is the course's RAG corpus, and that ambiguity is what the restructure removed.

**Rendered notebooks are read locally, not published.** `study/nb2md.py`
renders every course notebook into the gitignored `study/notebooks/`. The
render is a pure function of the committed `.ipynb` files and nb2md is
stdlib-only, so rebuilding costs seconds and no API key — which is why the
tree is disposable and the executed notebooks behind it are not. Nothing under
`study/docs/` may link into it, the same constraint that applies to
`_slides/`.

This is the exception to committing generated output, above. The argument for
committing renders is that their diffs show understanding changing; these
diffs only show which notebook was last executed, which `git log` already
says.

**No notebook in the course carries an image output** — verified across all
65. So nb2md's `_assets/` path has never fired, and `.gitignore`'s blanket
`*.png` rule has never silently eaten anything. It would eat one the first time a
saved plot lands in a tracked directory, which is a second reason the render
tree is ignored wholesale rather than by file type.

**Paths in generated docs are repo-relative POSIX**, and **console output is
ASCII-only** — a cp1252 Windows terminal crashes on `✓`.

---

## 4. Constraints that change what to do

**Only a Google API key.** ~13 core notebooks name OpenAI and nothing else; week
5 is worst hit at 4 of 5 local notebooks. The fix is the litellm swap, not an
`OPENAI_BASE_URL` redirect. For week 5 specifically, go local rather than swap to
Gemini — `multilingual-e5-large` came within 0.003 nDCG of Google Embeddings 2 on
Italian retrieval at 1/7 the latency, and local keeps Gewiss data in-boundary.
Details and the per-notebook table: `08-api-keys-and-runnability.md`.

**Week 6's frontier fine-tuning may be inexecutable** — see `OPEN-QUESTIONS.md`.
If blocked, redirect that time to the baseline ladder in
`06-training-and-finetuning.md`; it needs no paid key and it is what week 7's
fine-tuned model must beat for the capstone to mean anything.

---

## 5. Upstream (surveyed 2026-08-17)

Merged `upstream/main` at `109b683e`; the fork is level with it. 1,169 commits,
small core impact: 9 notebooks, `requirements.txt` removed in favour of
pyproject/uv, README refreshed. Only `README.md` conflicted.

**`week3/QUESTHER_V3_WEEK3.md` and `week4/QUESTHER_V3_WEEK4.md` were removed
upstream.** They were stray root copies of a contributor's submission, never
yours; the originals remain under `community-contributions/`. Do not build a
portfolio narrative on them and do not imitate their style — unsubstantiated
claims ("Success Rate: > 90%") with no benchmark behind them.

**Community contributions — low yield.** 409 new directories: week1 302, week2
110, week5 65, week4 36, week3 28, week6 8, weeks 7–8 zero. Three-quarters is
week 1–2, i.e. hundreds of variants of "summarize a website". The weeks where an
alternative approach would teach something have 8 directories between them. Scope
any attempt to weeks 5–6, or skip it.

**nb2md is novel upstream** — no notebook-to-Markdown renderer exists there. The
two near-misses do different jobs (`week4_auto_markdown_comments.ipynb` annotates
code; `files_to_markdown_converter.py` prepares RAG inputs). A PR is worth
making — branch off a clean `upstream/main`, not `linting-setup-improvements`,
which carries unrelated linting and study-tree changes.

**README MD059 warnings are left unfixed.** Rewriting Ed's "[here]" link text
would conflict on every future upstream merge.

---

## 6. Next steps, in priority order

1. **Run the distillation loop.** Everything built so far is scaffolding for it.
   The concept-only lectures are the ones worth doing — they are the ones with no
   other source. 208 lectures remain; you do not need them all. Lectures 109 and
   110 were watched but skipped. [`SLIDE-COVERAGE.md`](SLIDE-COVERAGE.md) ranks the
   days by how much they teach that no other source records.
2. **Answer the two cheap items in `OPEN-QUESTIONS.md`** — the token size of
   `week5/knowledge-base/`, and whether the OpenAI fine-tuning account is
   blocked. Both change what you do next, and both take minutes.
3. **Fact-check pass on the topic references.** Never run. The `UNVERIFIED` tags
   mark the claims needing a browser rather than a fetch tool; they cluster in
   `04-…md`.
4. **Decide `--use-llm`'s fate.** Run it against Ollama once and either keep it
   or delete the code path.
5. **PR nb2md upstream** (§5).
6. **Download the 8 Colab stubs**, if week 3 / week 7 content matters to you.
