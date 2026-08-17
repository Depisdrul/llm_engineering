# Lecture distillation — workflow, conventions, and template

**What this is.** A spec for turning individual lectures into study notes, one at a time, driven by you. It is written so that *any* session — this one, or a local Claude Code session in your repo — can pick it up and produce consistent output. Read it as instructions, not prose.

**What this is not.** A transcript archive. Every output file is a **condensed study note**: substantially shorter than the lecture, restructured, and merged with the slide content and with primary-source facts. If a note ever reads like a transcript, it has failed its purpose — you cannot revise 33 hours of speech in a monthly block, which is the whole reason this exists.

---

## 1. Folder layout

```text
notes/
  HANDOFF.md                             ← re-entry point; read it first
  01-llm-foundations.md                  ← topic references (already written)
  04-model-selection-benchmarks-leaderboards.md
  05-rag-and-vector-search.md
  06-training-and-finetuning.md
  07-agents-and-deployment.md
  08-api-keys-and-runnability.md
  LECTURE-DISTILL.md                     ← this file
  OPEN-QUESTIONS.md                      ← merged from the notes' `Open` sections
  lectures/
    week1/
      001-introduction.md
    week4/
      086-chinchilla-scaling-law.md
      087-ai-model-benchmarks.md
    week5/
      108-simple-rag-dictionary-lookup.md
    INDEX.md                             ← regenerated, links every lecture note
```

Everything here is published into the MkDocs site by
`python .knowledge-extraction/extract_all.py --generate-only`: the topic
references become **Topic References**, the lecture notes and `INDEX.md` become
**Lecture Notes**. The copies under `knowledge-base/docs/` are build output —
edit the files above. A reference is only published once it is listed in
`REFERENCE_NOTES` in `.knowledge-extraction/src/generators/reference_sync.py`,
so a draft can sit in this folder without going live.

Decks are read live from an open Drive tab (§2), so no deck belongs in git.
`notes/lectures/_slides/` is `.gitignore`d for the case where one gets
downloaded anyway — they're the instructor's materials and you may open an
upstream PR from this repo someday.

**Filename rule:** `NNN-kebab-slug.md`, where `NNN` is the **Udemy lecture number, zero-padded to 3**, and the slug is a short form of the lecture title. Zero-padding makes `ls` and git sort in course order, which matters once you have 200 files. Never renumber — if Udemy reorders lectures, keep the original number and note the drift in the file.

**Why numbered lectures and not one file per topic:** the topic references (`01-`…`08-`) are the *durable* layer, organised by concept. The lecture notes are the *provenance* layer, organised by where you actually were. Keep them separate; cross-link. When a lecture note and a topic reference disagree, the topic reference wins — it was built from primary sources.

---

## 2. The slides — just leave them open in a tab

Don't download them. A browser session reads a Google Slides deck straight from an open tab, so exporting to PDF buys nothing.

Where they live: **not on Udemy** — a Google Drive folder linked from the [course resources page](https://edwarddonner.com/2024/11/13/llm-engineering-resources/). Structure, verified:

```text
AI Engineer Core Track/          ← drive.google.com/drive/folders/1GMXbdgkqnZfCRcIdoUVBBB-hxeN4Lo06
  Week 1/ … Week 8/
    Copy of LLM - Week 5 Day 1   ← Google Slides, ~12–25 MB each
    Copy of LLM - Week 5 Day 2
    …Day 3, Day 4, Day 5
```

**Decks are per-day, not per-week.** Five per week, one per "Day N". So before distilling, open the deck matching the *day* of the lecture — the Day number is in the Udemy lecture title (`107. Day 1 - …`).

Working procedure:

1. Open the Drive folder for the week, double-click the deck for the day → it opens in its own tab.
2. Leave it open. Tell the session the lecture number; it reads both the transcript panel and the deck tab.
3. One deck covers all lectures for that day (Day 1 of week 5 is 11 slides across lectures 106–111), so you open it once per five-or-so lectures, not once per lecture.

**Slides beat narration where they disagree.** They carry the figures, the diagrams, and — most usefully — the instructor's own framing and section labels, which the spoken version only gestures at. They also *pre-announce* the next lectures, so a deck read at lecture 108 tells you what 109–111 will claim, which is when it is cheapest to flag a claim for verification.

If a deck ever won't load in a tab, fall back to `File > Download > PDF`, save it anywhere, and attach it to the conversation.

---

## 3. The per-lecture loop

Per lecture, the cycle is about two minutes of your time:

1. **You:** open the lecture on Udemy and open its transcript panel.
2. **You:** say `distill 87` (or `distill 87 slides 12-18` if you know which slides apply).
3. **Session:** reads the open transcript panel, reads the matching slides from the open *Week N Day D* deck tab, and cross-checks any factual claim against the topic references in `notes/`.
4. **Session:** writes `notes/lectures/weekN/NNN-slug.md` using the template in §4.
5. **You:** glance at it, click to the next lecture.

**Batching:** do a run of 5–10 lectures in one sitting, then let a local session in the repo do the integration pass (§6). Don't interleave distillation and integration — they need different attention.

**The browser session runs this as a skill.** That skill is not stored in this repo, so it can drift from this file without anything flagging it. **This file is the contract** — where the two disagree, this file wins, and the skill is what should change. If you edit the skill, mirror the change here; §4's template and §5's filename rule in particular are what the local index generator parses, and a note that doesn't match them is skipped silently.

**Rules for the session doing the distilling:**

- **Condense hard.** A 9-minute lecture becomes 15–40 lines. If it's longer than that, you're transcribing, not distilling.
- **Slides over speech.** Where a slide states a figure and the narration approximates it, take the slide.
- **Primary sources over both.** If the lecture states something the topic references contradict (see the corrections tables in `04-`, `06-`, `07-`), record what the lecture said *and* the correction, flagged. This is the single most valuable thing these notes can do — a course updated in June 2026 still carries claims from its 2024 recording.
- **Never invent.** If the transcript is unclear on a number, write `[unclear in lecture]`. Do not smooth over a gap.
- **Note what has no notebook.** Concept-only lectures are the ones where the lecture *is* the only source; mark them, because they're the ones worth re-reading.
- **Do not automate the sweep.** One lecture per explicit request, on pages Bea has open. No crawling the course, no scripted pagination.

---

## 4. The template

Copy this structure exactly. Fixed sections make the files diffable, greppable, and mergeable.

```markdown
# NNN — <Lecture title as shown on Udemy>

**Week N, Day D** · <MM> min · `<repo path or "no notebook">` · deck *LLM - Week N Day D* slides <a–b>

## Claim
<One or two sentences: what this lecture is actually asserting. Not "this lecture covers X"
— what does it want you to believe or be able to do.>

## Substance
<3–8 bullets. The facts, figures, definitions, and mechanisms. Slide figures preferred.
Include the numbers; a note without numbers is a note you won't trust in three months.>

## Code / demo
<What was run, which file, what the observable result was. Omit the section if none.
If it maps to a notebook, name the notebook and the cells.>

## Corrections
<Anything the lecture states that the topic references contradict, with a pointer:
"Lecture says the AA index is MMLU-Pro + GPQA + AIME → see 04-…md §4.1, v4.1.1 dropped all three."
Omit if nothing. This section is why the notes are worth more than the lectures.>

## Gewiss angle
<Optional. Only when there's a real one: EU/GDPR, Italian-language handling, Anthropic-internal,
Google-key-only constraints, industrial/manufacturing applicability. Delete if forced.>

## Open
<What you still don't know, or want to test. Carry these into the next block —
this is the re-entry hook that makes a monthly cadence survivable.>

## Links
<Repo path · relevant topic-reference section · primary source URLs.>
```

**Section discipline:** `Claim`, `Substance`, `Open` and `Links` are mandatory. `Code / demo`, `Corrections` and `Gewiss angle` are omitted rather than left empty. An empty heading is noise; a missing heading is information.

---

## 5. Index regeneration

`notes/lectures/INDEX.md` is one line per note, in numeric order, with week, title, whether it has a notebook, and the count of open questions. That file is what you open at the *start* of a block to remember where you were — which is the actual problem being solved here, not note completeness.

It is generated, not written. After each batch:

```bash
python .knowledge-extraction/extract_all.py --generate-only
```

The generator reads the note's H1 for the title, the `**Week N, Day D**` line for the week, the first `.ipynb` in backticks for the notebook, and counts the bullets under `## Open`. A note whose filename is off-template (`NNN-slug.md`) is skipped silently — if a note is missing from the index, check its filename first.

---

## 6. The integration pass — for a local session in the repo

Run this after a batch of distillations, in a local Claude Code session with the repo checked out. Instructions for that session:

1. Read `notes/lectures/weekN/*.md` for the batch just added.
2. **Promote durable content upward.** Anything in a lecture note that is a general fact rather than a course-specific observation belongs in the matching topic reference (`01-` … `08-`). Move it, and leave a link behind in the lecture note. The lecture notes should get *thinner* over time as their content migrates.
3. **Merge the `Corrections` sections** into the topic references' corrections tables. Deduplicate.
4. **Collect all `Open` items** into a single `notes/OPEN-QUESTIONS.md`, grouped by theme rather than by lecture, with the source lecture numbers as references. Themes recur across weeks; lectures don't.
5. **Reconcile against executed notebooks.** For any lecture whose notebook you've run, check that the note's `Code / demo` section matches what actually happened — the rendered Markdown from `nb2md.py` is the ground truth here, not the lecture.
6. Regenerate the site and the index: `python .knowledge-extraction/extract_all.py --generate-only`, then `cd knowledge-base && python -m mkdocs build --strict`. The strict build is what catches a cross-link you broke while promoting content.
7. Commit. `git log -p notes/` then becomes a record of how your understanding changed between blocks, which raw `.ipynb` diffs can never give you.

---

## 7. Scope boundary

Two things this workflow deliberately will not do, so nobody has to relitigate them mid-block:

- **No bulk transcript extraction.** No crawling the course, no scripted pagination, no pacing to evade bot detection. One lecture, on request, on a page you have open. The output is a condensed note, not a copy.
- **No reproduction.** Notes summarise and restructure; they don't reproduce lecture or slide content at length. That's a constraint on the artifact, but it's also what makes the artifact useful — a 200-page verbatim digest would not survive a monthly cadence any better than the videos do.
