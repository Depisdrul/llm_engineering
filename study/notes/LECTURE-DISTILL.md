# Lecture distillation — how it works in this repo

**The workflow itself lives in the `course-distill` skill**, at
`.claude/skills/course-distill/` — the template, the section discipline, the
correction-hunting patterns, the length targets, and the rules for the session
doing the distilling. Read `SKILL.md` and `references/template.md` for any of
that.

This file covers only what the skill cannot know: where the output goes in this
repo, how it gets published, and what the pipeline parses. Where the two
overlap, **the skill wins** — it is the copy that gets loaded and executed.

**What a lecture note is not:** a transcript archive. Every file is a condensed
study note, substantially shorter than the lecture and merged with slide content
and primary-source facts. You cannot revise 33 hours of speech in a monthly
block, which is the whole reason this exists.

---

## 1. Folder layout

```text
study/notes/
  DECISIONS.md                           ← why this is shaped this way
  01-llm-foundations.md                  ← topic references, the durable layer
  04-model-selection-benchmarks-leaderboards.md
  05-rag-and-vector-search.md
  06-training-and-finetuning.md
  07-agents-and-deployment.md
  08-api-keys-and-runnability.md
  LECTURE-DISTILL.md                     ← this file
  OPEN-QUESTIONS.md                      ← merged from the notes' `Open` sections
  lectures/
    week4/086-chinchilla-scaling-law.md  ← the provenance layer
    week5/108-simple-rag-dictionary-lookup.md
    INDEX.md                             ← generated
    _slides/                             ← gitignored
```

**The two layers.** Topic references (`01-`…`08-`) are organised by concept and
built from primary sources — the durable layer. Lecture notes are organised by
where you actually were — the provenance layer, which gets thinner over time as
content is promoted upward. When they disagree, the reference wins.

`_slides/` is gitignored: decks are the instructor's materials, they're read
live from an open Drive tab, and you may open an upstream PR from this repo
someday.

---

## 2. Publishing

```bash
python study/pipeline/extract_all.py --generate-only
```

Topic references become **Topic References** in the site nav; lecture notes and
`INDEX.md` become **Lecture Notes**. Copies under `study/docs/` are build output
— edit the files above.

A topic reference publishes only once it is listed in `REFERENCE_NOTES` in
`study/pipeline/src/generators/reference_sync.py`, plus a nav line in
`study/mkdocs.yml`. Drafts can sit in `study/notes/` without going live.

Lecture notes publish automatically. Individual pages are deliberately absent
from the nav — 210 lectures would make it unreadable — so `INDEX.md` is the only
way in.

**The filename rule is parsed, not decorative.** The generator reads the H1 for
the title, the `**Week N, Day D**` line for the week, the first `.ipynb` in
backticks for the notebook, and counts bullets under `## Open`. A file whose name
doesn't match `NNN-slug.md` is **skipped silently**. If a note is missing from
the index, check its filename first — this has already happened once.

**The domain is Gewiss.** The skill's template calls the optional section
`## {Domain} angle`; here it is `## Gewiss angle`, and it earns a place only for
a real consequence — EU/GDPR, Italian-language handling, a Google-key-only
constraint, or industrial/manufacturing applicability. Delete it rather than
forcing one.

---

## 3. Your side of the loop

1. Open the lecture on Udemy and open its transcript panel.
2. Open the Drive deck for that **day** — decks are per-day, five per week, and
   one covers all of that day's lectures. Leave it open in its own tab.
3. Say `distill 87`. The skill takes it from there.
4. Glance at the result, click to the next lecture.

Batch 5–10 lectures in one sitting, then run the integration pass. Don't
interleave the two — they need different attention.

---

## 4. Integration pass — the repo side

The skill's `## Integration pass` section covers the editorial work: promoting
durable content upward, merging corrections, collecting `Open` items into
`OPEN-QUESTIONS.md`, reconciling against executed notebooks. Afterwards:

```bash
python study/pipeline/extract_all.py --generate-only
cd study && python -m mkdocs build --strict
```

The strict build is what catches a cross-link broken while moving content
between layers. Then commit — `git log -p study/notes/` becomes the record of
how your understanding changed between blocks, which raw `.ipynb` diffs never
give you.
