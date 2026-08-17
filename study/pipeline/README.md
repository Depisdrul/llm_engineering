# study/pipeline — the extraction pipeline

> **New session? Read [`study/README.md`](../README.md) to orient, then
> [`notes/DECISIONS.md`](../notes/DECISIONS.md) for why things are shaped the way
> they are.** This README covers only how to run the pipeline and what it
> actually does.

Reads the course notebooks and the hand-written notes in [`../notes/`](../notes/),
and writes the MkDocs content in `../docs/`. Everything in `../docs/` is build
output except `index.md`.

## What it does, in five phases

| Phase | Does | Notes |
| --- | --- | --- |
| 1. Extract | Parses every `weekN/*.ipynb` via `../nb2md.py` | Cached to `.cache/extracted.json` |
| 2. Website | Scrapes edwarddonner.com | **Untested — see below** |
| 3. Summarize | One LLM call per notebook | Only with `--use-llm` |
| 4. Publish notes | Copies `../notes/` into `../docs/reference/` and `../docs/lectures/`, regenerates `../notes/lectures/INDEX.md` | |
| 5. Generate | Week summaries, topic pages, site index | One LLM call per topic with `--use-llm` |

## Running it

```bash
# regenerate everything from the notebooks (free, no LLM) - the usual command
python study/pipeline/extract_all.py --skip-website

# pages only, from the cached extraction - fast, and what you want after
# editing anything in study/notes/
python study/pipeline/extract_all.py --generate-only

# one week at a time
python study/pipeline/extract_all.py --weeks 5,6 --skip-website

# tests
python -m unittest discover -s study/pipeline -t study/pipeline

# read the site
cd study && python -m mkdocs serve
```

`mkdocs build --strict` must stay clean. It is the only thing that catches a
cross-link broken while moving content between layers, so run it before
committing.

The pipeline finds the repo root by walking up for a directory holding both
`.git` and `study/`, so it runs from any working directory.

## What is unproven — read before trusting output

**`--use-llm` has never been run against the current topic-page path.** It is
wired: one `generate_topic_summary` call per topic, 18 calls. Nothing has
validated the output or the cost. Try Ollama first — it is free.

```bash
ollama serve                 # in another terminal
ollama pull llama3.2
python study/pipeline/extract_all.py --use-llm --provider ollama --skip-website
```

Defaults per provider: `ollama` → `llama3.2`, `openai` → `gpt-4.1-mini`,
`anthropic` → `claude-3-5-sonnet-20241022`. Override with `--model`.

The cost figure the run prints for paid providers is a rough
`tokens × rate` estimate with rates hardcoded in `summarize_phase` — treat it as
an order of magnitude, not a bill.

**Website scraping is untested.** `extract_website_phase` warns and continues on
failure by default; `--strict-website` makes it fail loudly. Nobody knows whether
it has ever worked. `--skip-website` is the honest default.

**`docs/quick-ref/*` and `docs/projects/*` are placeholders.** No generator
writes them. They are hand-editable files that nothing overwrites, which also
means nothing keeps them true.

**Three `LLMSummarizer` methods are dead code** — `generate_quickref_entry`,
`extract_troubleshooting_info` and `classify_topic` have no caller. Delete them
or wire them up; leaving them reads as capability the pipeline does not have.

## Layout

```text
study/
├── notes/                   # Hand-written. The pipeline reads these.
├── pipeline/
│   ├── src/
│   │   ├── extractors/      # nb2md_loader, notebook_extractor, web_scraper
│   │   ├── processors/      # llm_summarizer
│   │   └── generators/      # content_generator, reference_sync, lecture_sync
│   ├── templates/           # Jinja2: topic_page.md.j2, quickref.md.j2
│   ├── config/taxonomy.yaml # Topics, and which reference is authoritative for each
│   ├── test_extraction.py
│   └── extract_all.py
├── docs/                    # Generated. reference/ and lectures/ mirror notes/.
├── nb2md.py                 # The repo's single notebook parser
├── mkdocs.yml
└── _site/                   # Rendered HTML, gitignored
```

## Changing things

**Add a topic:** append to `config/taxonomy.yaml`, then `--generate-only`. Add a
`reference:` key naming a published slug if a hand-written reference covers it;
without one the page says so rather than implying authority it lacks.

```yaml
  - id: new-topic-id
    name: New Topic Name
    category: core-concepts
    folder: 02-core-concepts
    description: What it covers
    weeks: [1, 2]
    reference: llm-foundations
```

A topic also needs a nav entry in `../mkdocs.yml`, or the strict build reports
the page as unlisted.

**Publish a new reference:** add it to `REFERENCE_NOTES` in
`src/generators/reference_sync.py` and to the nav. Files in `../notes/` that are
not listed stay unpublished, so drafts can sit there safely.

**Change page shape:** edit the Jinja2 templates in `templates/`.

**Narration filtering:** the course notebooks are conversational, so raw headings
yield things like "Donezo! On to Step 2" as learning objectives. `is_narration()`
in `src/extractors/notebook_extractor.py` filters them. It is a heuristic — add
new patterns to `NARRATION_PREFIXES` / `NARRATION_SUBSTRINGS` **with a test
case**.

## Troubleshooting

**No notebooks found.** The root walk needs a directory containing both `.git`
and `study/`. Check you are inside the repo.

**A lecture note is missing from `INDEX.md`.** Its filename does not match
`NNN-slug.md`. Off-template names are skipped without an error — see
[`../notes/LECTURE-DISTILL.md`](../notes/LECTURE-DISTILL.md) §2.

**`mkdocs build --strict` fails on a link.** Most often a reference cross-link
pointing at a filename that is not a published slug. `rewrite_cross_links()` maps
source stems and `aliases` onto slugs; add the missing alias.

**Unicode errors on Windows.** Console output is deliberately ASCII-only; a
cp1252 terminal crashes on `✓`. Keep it that way.
