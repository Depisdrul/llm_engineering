# nb2md — make the course repo reviewable in monthly study blocks

Renders every notebook in the course repo to Markdown with the **actual saved
outputs** inline, so a study block starts with reading rather than re-running.
Standard library only — no install, no dependencies, works under `uv` or plain
`python3`.

---

## The problem this solves

Two separate things were blocking review, and they need different fixes.

**1. Colab stubs.** In the repo, `week3/day1–day5.ipynb` and
`week7/day1, day2, "day3 and 4", day5.ipynb` are *not the lectures*. Each is a
single markdown cell holding a `colab.research.google.com/drive/...` link. Eight
notebooks are pure stubs; `week7/day2.ipynb` is the one exception in week 7 that
holds real local code. No amount of tooling recovers content that isn't in the
clone — this part is manual, once, and described below.

**2. Stripped outputs.** 53 of the 65 core notebooks ship with outputs cleared,
so reading them tells you what the code *is* but not what it *did*. That's the
part `nb2md` fixes, permanently.

Verified on a fresh clone: nothing in the repo strips your outputs. There is no
root `.gitattributes`, no `.pre-commit-config.yaml`, no `nbstripout`, no
`.github/` directory. The clear-outputs rule is a human convention for upstream
PRs only. **Outputs you save are yours to keep and will commit verbatim.**

---

## Quick start

```bash
# 1. See what you're dealing with — classifies, writes nothing
python3 nb2md.py . --recursive --exclude community-contributions --audit

# 2. Render everything into a notes tree, with an index
python3 nb2md.py . -o notes/ --recursive \
    --exclude community-contributions \
    --exclude community_contributions \
    --index
```

The excludes matter: the repo carries ~3,000 community-contribution notebooks
across ~273 folders. Without excluding them you render those too, and it buries
the 65 that are actually the course.

Expected audit output on a clean clone:

```
     4  executed (has saved outputs) - fully reviewable
    53  stripped (code + prose, no outputs)
     8  COLAB STUB - link only, no local content
```

Output layout mirrors the repo, with images extracted next to each file:

```
notes/
  INDEX.md                    every notebook, flagged reviewable / no outputs / STUB
  week1/day1.md
  week5/day4.md
  week5/day4_assets/          day4_out001.png, ...
  guides/09_ai_apis_and_ollama.md
```

---

## Recovering the eight Colab stubs

Once each, per notebook. Ed's Colab links are shared read-only, so you must copy
before you can run.

1. Open the link (it's printed in the stub's rendered `.md`, and in the audit).
2. `File > Save a copy in Drive` — you now own a runnable copy.
3. Run it. Week 3 day 4 and day 5, and all of week 7, need a CUDA GPU; a free or
   low-cost T4 runtime is enough for week 3.
4. `File > Download > Download .ipynb`. Colab serializes executed outputs into
   the file — there is no Colab-side stripping.
5. Save it **next to the stub, not over it**:
   `week3/day4_executed.ipynb`. Overwriting the stub loses the canonical link and
   creates a diff against upstream you'll have to resolve on every `git pull`.
6. Re-run `nb2md`. The `_executed` notebook renders as fully reviewable.

Worth knowing before you start:

- **Week 3 day 1 and day 3 don't need a GPU at all.** Day 1 is a Colab tour; day
  3 is tokenizers — pure CPU work, Colab-only by convention. If you'd rather have
  them local, you can rebuild them locally instead of downloading. Day 3 needs a
  HuggingFace token for the gated Llama tokenizers.
- **Week 7 will not run locally, notebooks or not.** `pyproject.toml` and
  `uv.lock` contain no `peft`, `trl`, `bitsandbytes`, or `accelerate` — the local
  environment is deliberately not provisioned for QLoRA. Downloading the
  notebooks makes week 7 *readable*, which is the goal here; it does not make it
  runnable. Don't burn a study block trying.
- **Week 8 needs no local GPU** despite serving a fine-tuned Llama — the GPU is
  rented from Modal (`GPU = "T4"` in `pricer_service2.py`, `pricer_ephemeral.py`,
  `llama.py`).
- **`week6/redemption_train.ipynb`** is marked very optional and takes hours;
  `redemption_run.ipynb` exists so you can load pre-trained weights instead.

---

## Building the habit that actually saves your time

The renderer only pays off if executed notebooks accumulate. One line, after any
session where you ran something:

```bash
python3 nb2md.py week5/ -o notes/week5 --index
```

Optional git alias so you never think about it:

```bash
git config alias.notes '!python3 nb2md.py . -o notes/ --recursive \
  --exclude community-contributions --exclude community_contributions --index'
# then: git notes
```

Add `notes/` to `.gitignore` if you'd rather not commit the rendered copies —
but committing them is the better call. Rendered Markdown diffs readably in git,
so `git log -p notes/week5/day4.md` shows you how your own understanding of a
notebook changed between study blocks. Raw `.ipynb` diffs are unreadable JSON.

---

## Options

| Flag | Effect |
|---|---|
| `-o, --out DIR` | output directory, repo structure mirrored (default: `.md` beside each `.ipynb`) |
| `-r, --recursive` | descend into subdirectories |
| `--exclude PAT` | skip any path containing `PAT`; repeatable |
| `--audit` | classify and report only, render nothing |
| `--max-output-lines N` | truncate text outputs to N lines, default 40. `0` = unlimited |
| `--no-images` | skip image extraction, leave a placeholder |
| `--index` | write `INDEX.md` linking every rendered file with its status |

`--max-output-lines 0` is the one you'll want for week 6 and week 7 training
logs, where the loss curve printout *is* the content. The default 40 keeps
progress-bar spam from drowning a file.

---

## What gets rendered

- Markdown cells verbatim, so the instructor's prose and the HTML callout boxes
  survive.
- Code cells fenced with the notebook's kernel language.
- Outputs under a `> **Output**` marker: stdout and stderr labelled separately,
  `execute_result` and `display_data` text, images written to
  `<name>_assets/` and linked, and errors with ANSI-stripped tracebacks — worth
  keeping, since a captured traceback from three months ago is often the fastest
  way back into a problem.
- A provenance header on every file: source path, cell counts, how many cells
  carried output. This is what tells you at a glance whether a file is worth
  reading or needs a re-run first.
- Rich HTML output (pandas tables) falls back to the `text/plain` form; the
  notebook itself is still the place to look at a real DataFrame.

`.ipynb_checkpoints` is always skipped. Unreadable or non-notebook `.ipynb`
files are reported to stderr and skipped rather than aborting the run.
