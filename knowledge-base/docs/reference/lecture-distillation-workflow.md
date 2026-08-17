<!-- Generated copy. Source: .knowledge-extraction/improvements-from-chrome-session/LECTUREDISTILL.md
     Edit the source file; this copy is overwritten on every extraction run. -->

# Lecture distillation — workflow, conventions, and template

**What this is.** A spec for turning individual lectures into study notes, one at a time, driven by you. It is written so that *any* session — this one, or a local Claude Code session in your repo — can pick it up and produce consistent output. Read it as instructions, not prose.

**What this is not.** A transcript archive. Every output file is a **condensed study note**: substantially shorter than the lecture, restructured, and merged with the slide content and with primary-source facts. If a note ever reads like a transcript, it has failed its purpose — you cannot revise 33 hours of speech in a monthly block, which is the whole reason this exists.

---

## 1. Folder layout

```
notes/
  01-llm-foundations.md                  ← topic references (already written)
  04-model-selection-benchmarks-leaderboards.md
  05-rag-and-vector-search.md
  06-training-and-finetuning.md
  07-agents-and-deployment.md
  08-api-keys-and-runnability.md
  LECTURE-DISTILL.md                     ← this file
  lectures/
    _slides/
      week1.pdf                          ← you download these once per week
      week4.pdf
    week1/
      001-introduction.md
      002-...
    week4/
      086-chinchilla-scaling-law.md
      087-ai-model-benchmarks.md
    INDEX.md                             ← regenerated, links every lecture note
```

**Filename rule:** `NNN-kebab-slug.md`, where `NNN` is the **Udemy lecture number, zero-padded to 3**, and the slug is a short form of the lecture title. Zero-padding makes `ls` and git sort in course order, which matters once you have 200 files. Never renumber — if Udemy reorders lectures, keep the original number and note the drift in the file.

**Why numbered lectures and not one file per topic:** the topic references (`01-`…`08-`) are the *durable* layer, organised by concept. The lecture notes are the *provenance* layer, organised by where you actually were. Keep them separate; cross-link. When a lecture note and a topic reference disagree, the topic reference wins — it was built from primary sources.

---

## 2. Getting the slides — do this once per week

The slides are not on the public site. They live in a Google Drive folder linked from the [course resources page](https://edwarddonner.com/2024/11/13/llm-engineering-resources/), accessible with your Google account.

1. Open the Drive folder, find the deck for the week you're studying.
2. `File > Download > PDF Document`.
3. Save as `notes/lectures/_slides/weekN.pdf`.

That's it. Once the PDF is in the repo, any session with file access can read the relevant pages and fold them into the note. Slides carry the diagrams, the numbers, and the structure that spoken narration only gestures at — they are the higher-value input of the two.

---

## 3. The per-lecture loop

Per lecture, the cycle is about two minutes of your time:

1. **You:** open the lecture on Udemy and open its transcript panel.
2. **You:** say `distill 87` (or `distill 87 slides 12-18` if you know which slides apply).
3. **Session:** reads the open transcript panel, reads the matching pages of `_slides/weekN.pdf` if present, and cross-checks any factual claim against the topic references in `notes/`.
4. **Session:** writes `notes/lectures/weekN/NNN-slug.md` using the template in §4.
5. **You:** glance at it, click to the next lecture.

**Batching:** do a run of 5–10 lectures in one sitting, then let a local session in the repo do the integration pass (§6). Don't interleave distillation and integration — they need different attention.

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

**Week N, Day D** · <MM> min · `<repo path or "no notebook">` · slides `weekN.pdf` pp. <a–b>

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

After each batch, regenerate `notes/lectures/INDEX.md`: one line per note, in numeric order, with week, title, whether it has a notebook, and the count of open questions. That file is what you open at the *start* of a block to remember where you were — which is the actual problem being solved here, not note completeness.

Grouping by week with a per-week open-question count at the top is enough. Don't over-engineer it.

---

## 6. The integration pass — for a local session in the repo

Run this after a batch of distillations, in a local Claude Code session with the repo checked out. Instructions for that session:

1. Read `notes/lectures/weekN/*.md` for the batch just added.
2. **Promote durable content upward.** Anything in a lecture note that is a general fact rather than a course-specific observation belongs in the matching topic reference (`01-` … `08-`). Move it, and leave a link behind in the lecture note. The lecture notes should get *thinner* over time as their content migrates.
3. **Merge the `Corrections` sections** into the topic references' corrections tables. Deduplicate.
4. **Collect all `Open` items** into a single `notes/OPEN-QUESTIONS.md`, grouped by theme rather than by lecture, with the source lecture numbers as references. Themes recur across weeks; lectures don't.
5. **Reconcile against executed notebooks.** For any lecture whose notebook you've run, check that the note's `Code / demo` section matches what actually happened — the rendered Markdown from `nb2md.py` is the ground truth here, not the lecture.
6. Regenerate `INDEX.md`.
7. Commit. `git log -p notes/` then becomes a record of how your understanding changed between blocks, which raw `.ipynb` diffs can never give you.

---

## 7. Scope boundary

Two things this workflow deliberately will not do, so nobody has to relitigate them mid-block:

- **No bulk transcript extraction.** No crawling the course, no scripted pagination, no pacing to evade bot detection. One lecture, on request, on a page you have open. The output is a condensed note, not a copy.
- **No reproduction.** Notes summarise and restructure; they don't reproduce lecture or slide content at length. That's a constraint on the artifact, but it's also what makes the artifact useful — a 200-page verbatim digest would not survive a monthly cadence any better than the videos do.
