---
name: course-distill
description: Distill a single video lecture into a condensed study note that corrects the lecture against primary sources. Use when the user asks to distill a lecture by number, asks to turn a lecture, talk, webinar or course video into study notes, or asks for an integration pass over accumulated lecture notes. Written for the ed-donner/llm_engineering course repo, but the template and discipline generalise to any video course.
---

# Course distillation

Turn one lecture into one condensed study note that is **worth more than the lecture**, by correcting it against primary sources and folding in the slide deck.

The reader studies in long blocks weeks apart. The note has to work cold, months later, with no memory of the session that produced it.

## Non-negotiables

1. **Condense hard.** A 9-minute lecture becomes 30–60 lines. Longer than that and you are transcribing, not distilling. Never reproduce a transcript or a deck at length — summarise and restructure.
2. **The `Corrections` section is the point.** A note that faithfully records what the lecture said is worth less than one that records the *delta* between the lecture and current primary sources. Courses lag their own recordings by 18+ months. Hunting the delta is the job.
3. **Never invent.** No figure, no source URL, no spelling of a proper noun that you did not verify. Write `[unclear in lecture]` rather than smoothing over a gap.
4. **Mark concept-only lectures.** Where a lecture maps to no notebook, the lecture is the *only* source for its content — say so in the header line. Those are the notes worth re-reading, and the ones whose loss is unrecoverable.
5. **Verify the lecture number before writing.** See "Identify the lecture" — users misreport it constantly, and a misfiled note stays misfiled.
6. **No bulk extraction.** One lecture per explicit request, on a page the user already has open. No crawling a course, no scripted pagination, no pacing to evade bot detection. Decline and say why if asked.

## Identify the lecture — do this first, every time

The user's stated number is a hypothesis, not a fact. In practice they name the lecture they *think* they're on while the player has advanced past it.

Check three signals and require at least two to agree:

- **Runtime.** The player shows `mm:ss / mm:ss`. Match the total against the duration in the curriculum sidebar. This is the strongest single signal — durations differ by minutes between adjacent lectures.
- **Content.** Does the transcript match the candidate lecture's *title*? Titles are descriptive enough to disambiguate.
- **Completion state.** The lecture in progress is usually the first one still marked incomplete.
If the signals contradict the user, **distill what is actually on screen and say so plainly in your reply.** Do not silently pick one. Add a `## Note on numbering` section to the file if lectures were skipped.

## Gather sources

**Transcript** — the user opens the lecture's transcript panel; read the page text. Expect ASR garbling of invented product and company names; never trust transcript spelling of a proper noun. Recover correct spellings from the repo's own files.

**Slides** — check whether a deck tab is open before asking. For this course they live in a Google Drive folder linked from the instructor's resources page, **not on the course platform**, organised as `Week N/` → five **per-day** decks. One deck covers all lectures for that day, so it is opened once per five-or-six lectures.

Read decks live from an open tab; do not ask for a download. If a deck will not load in a tab, the fallback is `File > Download > PDF` and attaching it to the conversation — not reconstructing it from video frames. Two things make slides the higher-value source:

- **They carry the instructor's own framing and section labels**, which narration only gestures at.
- **They pre-announce the next lectures.** A deck read at lecture 108 tells you what 109–111 will claim — the cheapest possible moment to flag a claim for verification. Put those in a `## Coming next in this deck` section.
**Topic references** — read `study/notes/0*.md` in the repo before writing. That is where the primary-source material lives and where corrections come from. `references/repo-facts.md` in this skill has the verified repo facts.

## Write the note

Path: `study/notes/lectures/weekN/NNN-kebab-slug.md`, where `NNN` is the platform lecture number **zero-padded to 3**. Padding keeps 200 files sorting in course order. Never renumber — if the platform reorders, keep the original number and note the drift.

Template and section discipline: `references/template.md`.

**The filename rule is load-bearing.** `study/pipeline/` parses `NNN-slug.md` to build the index, and a note named anything else is skipped silently — no error, it simply never appears. The generator also reads the H1 for the title, the `**Week N, Day D**` line for the week, the first `.ipynb` in backticks for the notebook, and counts bullets under `## Open`.

`study/notes/LECTURE-DISTILL.md` documents the repo side — where output goes, how it publishes, what the pipeline parses. **This file owns the workflow**; where the two overlap, this file wins.

Where to look for corrections — the highest-yield patterns, all of which have paid off on this course:

- **A named leaderboard, benchmark suite, or index composition.** These churn every few months; course material describing one is almost always stale.
- **A retired or archived resource** still recommended as current.
- **A famous result stated without its caveat** — the caveat is usually load-bearing.
- **A technique presented as an obvious improvement.** Check whether it has been evaluated. Several field favourites lose to boring baselines.
- **A strawman being replaced.** Ask whether the thing being discarded actually had a real strength, and whether the honest answer is "use both."
- **A best-case example.** Ask what the symmetric case looks like and whether the lecture mentions it.
- **A capability claim tied to a headline number** (context window, parameter count) where the effective figure is much smaller.
- **Provider/API assumptions** the user cannot satisfy with the keys they hold.

## Integration pass

Run after a batch of distillations, not after each one — they need different attention.

1. **Promote durable content upward.** General facts belong in the topic references, not in lecture notes. Move them, leave a link. **Lecture notes should get thinner over time.**
2. **Merge `Corrections` into the topic files' corrections tables.** Deduplicate.
3. **Collect all `Open` items** into `study/notes/OPEN-QUESTIONS.md`, grouped by theme rather than by lecture. Themes recur across weeks; lectures don't.
4. **Reconcile against executed notebooks.** Where a notebook has been run, its rendered output is ground truth — not the lecture.
5. **Regenerate the index and the site:** `python study/pipeline/extract_all.py --generate-only`, then `cd study && python -m mkdocs build --strict`. `study/notes/lectures/INDEX.md` is generated from the notes — one line per note in numeric order, with week, title, whether a notebook exists, and the open-question count. That file is what the user opens at the *start* of a block to re-enter context, which is the actual problem being solved. The strict build catches cross-links broken while promoting content.
6. Commit. `git log -p study/notes/` then records how understanding changed between blocks, which raw notebook diffs never can.

## Hygiene

- Don't commit slide decks or downloaded course materials into the repo — `.gitignore` them. The user may open an upstream PR someday.
- Don't reconstruct slide decks from video frames. Use the shared decks.
- Confidence tags on claims: `[Certain]` with a link, `[Likely]` for well-founded inference, `[Guessing]` for gap-filling. Tag `UNVERIFIED` inline where a source could not be confirmed, and keep a closing list of unverified items in every file.
