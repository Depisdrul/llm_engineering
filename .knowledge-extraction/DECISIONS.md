# Knowledge base — decisions and next steps

**Last session: 2026-08-17.** Read this first in a new session; it is the re-entry
point for the knowledge-base work. It records *why* things are shaped the way
they are, which the code cannot tell you.

---

## 1. The layering decision — the one that matters

The knowledge base has **two layers, and they are not equals**.

**Authoritative layer — `improvements-from-chrome-session/*.md`.** Hand-written
references built from primary sources (arXiv, ACL Anthology, provider docs),
with corrections tables flagging where the 2024/2025-era course is now wrong.
These are the reason the knowledge base is worth having.

**Provenance layer — `knowledge-base/docs/topics/*`.** Generated from the course
notebooks. It answers "which notebook demonstrates this, what was the code, what
did it output" — nothing more. It does not explain concepts.

Every generated topic page opens with a callout naming its reference and stating
that the reference supersedes it. Topics with no reference yet say so instead of
implying authority they don't have.

**Do not let the generator compete with the references.** A heading scraper
cannot produce a retrieval diagnostic tree. If a generated page starts looking
like it explains something, that content belongs in a reference instead.

### Reference coverage, and the gaps

| Reference | Covers | Topics mapped |
| --- | --- | --- |
| `01llmfoundations.md` | transformers, tokenization, context, KV cache, prompt caching, MoE | prompt-engineering, tokenization |
| `05ragandvectorsearch.md` | embeddings, chunking, vector stores, reranking, RAG eval | rag-systems, embeddings, vector-databases |
| `06trainingandfinetuning.md` | LoRA, QLoRA, quantization, hyperparameters, DPO | fine-tuning, dataset-prep, evaluation, gpu-computing |
| `07agentsanddeployment.md` | agent framing, multi-agent, MCP, deployment, Gradio | agent-architectures, tool-use, multi-agent, deployment, ui-interaction |
| `08apikeysandrunnability.md` | what runs with a Google-only key, Ollama fallback | llm-apis, multi-model |
| `LECTUREDISTILL.md` | the per-lecture distillation workflow | — |

**Two topics have no reference:** `dev-tools` and `model-selection`.
`LECTUREDISTILL.md` §1 names a planned `04-model-selection-benchmarks-leaderboards.md`
that was never written. That is the largest content gap.

Mapping lives in `config/taxonomy.yaml` as a `reference:` key per topic.
Publishing is driven by `REFERENCE_NOTES` in `src/generators/reference_sync.py` —
only files listed there go live, so drafts can sit in the source folder.

---

## 2. Architecture decisions

**nb2md is the single notebook parser.** `knowledge-base/nb2md.py` is stdlib-only
and standalone; `src/extractors/nb2md_loader.py` loads it by path and caches it.
`notebook_extractor.py` uses its `load_notebook`, `classify`, `source_text` and
`cell_language`. The `nbformat` dependency is gone.

Why it matters: the extractor previously had no notion of Colab stubs, so week 3
reported "1 unique concept across 5 days" when in truth all five notebooks are
Colab links with no local content. It also ignored saved outputs entirely.

**A code cell with saved output is always kept**, even when the code looks
trivial. The output is the evidence of what happened; discarding it defeats the
point of merging with nb2md.

**Heading scrapes are filtered for lecture narration.** The course notebooks are
conversational, so raw headings yield "Donezo! On to Step 2" as a learning
objective. Filter lives in `is_narration()`. It is a heuristic and will need new
patterns as new weeks are read — add them to `NARRATION_PREFIXES` /
`NARRATION_SUBSTRINGS` with a test case.

**Generated docs are committed.** Consistent with how the repo already treats
`docs/`, and per `nb2mdREADME.md`: rendered Markdown diffs readably in git, so
`git log -p` shows how understanding changed between study blocks.

**`docs/reference/*.md` are build output.** Edit the source in
`improvements-from-chrome-session/`; the copies are overwritten every run.
Cross-links are rewritten on sync because the notes link each other by a numbered
filename convention the files on disk don't use.

**Paths in generated docs are repo-relative POSIX.** Absolute Windows paths were
leaking into published pages.

**Console output is ASCII-only.** A cp1252 Windows terminal crashes on `✓`/`⚠`.

---

## 3. What is deliberately not done

- **`docs/projects/*` and `docs/quick-ref/*` are still placeholders.** Nothing
  generates them. Decide whether they are worth a generator or should be
  hand-written like the references.
- **`--use-llm` has never been run against the new topic-page path.** It is wired
  (one `generate_topic_summary` call per topic, 18 calls) but unproven. Try
  Ollama first — it is free.
- **Website scraping is untested.** `extract_website_phase` warns and continues
  on failure by default; `--strict-website` makes it fail loudly. Nobody knows if
  it ever worked.
- **README MD059 warnings are left unfixed.** Rewriting Ed's "[here]" link text
  would conflict on every future upstream merge.

---

## 4. Upstream survey (as of 2026-08-17)

Merged `upstream/main` at commit `109b683e`; the fork is now level with upstream.

- 1,169 commits merged. Core impact was small: 9 notebooks, `requirements.txt`
  removed in favour of pyproject/uv, README refreshed, `assets/core.jpg` added.
- Only `README.md` conflicted. Resolved by taking upstream's content and
  re-running markdownlint.
- **`week3/QUESTHER_V3_WEEK3.md` and `week4/QUESTHER_V3_WEEK4.md` were removed
  upstream.** They were stray root copies of a contributor's (ijosh) submission,
  never yours. The originals remain under `community-contributions/`. Do not
  build a portfolio narrative on them, and do not imitate their style — they are
  all unsubstantiated claims ("Success Rate: > 90%") with no benchmark.

### Community contributions — assessment revised downward

409 new contribution directories, distributed:

| week | new dirs |
| --- | --- |
| week1 | 302 |
| week2 | 110 |
| week5 | 65 |
| week4 | 36 |
| week3 | 28 |
| week6 | 8 |
| week7 / week8 | 0 |

Three-quarters is week 1–2, i.e. hundreds of variants of "summarize a website".
The weeks where alternative approaches would teach something — 6, 7, 8 — have 8
new directories between them. **Mining this corpus is low-yield.** Scope any
attempt to weeks 5–6, or skip it.

### nb2md is novel upstream

No notebook-to-Markdown renderer exists in `upstream/main`. The two near-misses
do different jobs: `week4/community-contributions/week4_auto_markdown_comments.ipynb`
annotates code, `week5/community-contributions/vic_knowledge_worker_with_RAG/files_to_markdown_converter.py`
prepares RAG inputs. A PR is still worth making.

---

## 5. Next steps, in priority order

1. **Run the LECTUREDISTILL loop.** This is the highest-value work; everything
   built so far is scaffolding for it. See `improvements-from-chrome-session/LECTUREDISTILL.md`.

   Blocker to clear first: the slides in `notes/lectures/_slides/` are eight
   Google-Drive zips of per-day `.pptx` files, not the `weekN.pdf` the workflow
   expects. Either export to PDF, or extract text with `python-pptx` and adapt
   the workflow to per-day decks. `notes/` is untracked — decide whether it
   should be committed or gitignored before it fills up.

2. **Write the missing `model-selection` reference.** Named in LECTUREDISTILL §1,
   never written, and it is one of the two topics with no authoritative layer.

3. **PR nb2md upstream.** `knowledge-base/nb2md.py` and `nb2mdREADME.md` are now
   committed to the fork, so there is something to PR. Branch off a clean
   `upstream/main` — do not PR from `linting-setup-improvements`, which carries
   unrelated linting and knowledge-base changes.

4. **Try `--use-llm --provider ollama`** and judge whether LLM synthesis earns
   its place on topic pages. If it does not, delete the code path rather than
   leaving it as an untested option.

5. **Community contributions, weeks 5–6 only** — or drop it.

---

## 6. Running it

```bash
# regenerate everything from the notebooks (free, no LLM)
python .knowledge-extraction/extract_all.py --skip-website

# regenerate pages only, from the cached extraction
python .knowledge-extraction/extract_all.py --generate-only

# tests
python -m unittest discover -s .knowledge-extraction -t .knowledge-extraction

# serve the site
cd knowledge-base && python -m mkdocs serve
```

`mkdocs build --strict` must stay clean — it is what catches broken cross-links
between the references and the generated pages.

The extraction cache lives at `.knowledge-extraction/.cache/extracted.json` and
is gitignored.
