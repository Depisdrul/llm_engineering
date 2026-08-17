# The note template

Copy this structure exactly. Fixed sections make files diffable, greppable and mergeable across a 200-file set.

Placeholders are written as `{ ... }` and should be replaced wholesale.

```markdown
# NNN — {Lecture title as shown in the course}

**Week N, Day D** · {MM} min · `{repo path, or: no notebook (concept-only)}` · slides: *{deck name}*, slide {n}

## Claim
{One or two sentences. Not "this lecture covers X" — what does it want you to
believe, or be able to do? If you cannot state it in two sentences the lecture
had no thesis, and that is itself worth writing down.}

## Substance
{3–8 bullets. Facts, figures, definitions, mechanisms. Slide figures preferred
over spoken approximations. Include the numbers — a note without numbers is a
note you will not trust in three months. Bold the one thing the instructor
themselves flags as most important or most commonly misunderstood.}

## Code / demo
{What was run, which file, what the observable result was. Name the notebook and
the relevant cells. Flag ASR-garbled proper nouns and give the correct spelling
from the repo. Omit the whole section if there was no demo.}

## Corrections
{The reason this file exists. Anything the lecture asserts that primary sources
contradict, qualify, or have overtaken — each with a pointer to the topic
reference and a source link. Omit only if you genuinely found nothing, which
should be rare.}

## Coming next in this deck
{Only when the deck's later slides belong to lectures not yet watched. Flag
claims to verify before watching. Cheapest possible moment to catch them.}

## {Domain} angle
{Optional, named for the reader's actual domain. Only when there is a real
consequence for their work — regulatory, language, infrastructure, data. Delete
the section rather than forcing one.}

## Open
{What you still don't know, or want to test. This is the re-entry hook that makes
a monthly study cadence survivable — the first thing read at the start of the
next block. Carry unresolved items forward across lectures rather than dropping
them.}

## Links
{Repo paths · topic-reference sections · primary source URLs · slide location.}

## Note on numbering
{Only when the distilled lecture differs from the one requested, or when
lectures were skipped. State what was skipped and whether it is recoverable.}
```

## Section discipline

**Mandatory:** `Claim`, `Substance`, `Open`, `Links`.

**Omitted rather than left empty:** `Code / demo`, `Corrections`, `Coming next`, `{Domain} angle`, `Note on numbering`.

An empty heading is noise. A missing heading is information.

## Length calibration

| Lecture length | Target note |
|---|---|
| 5–8 min | 25–40 lines |
| 9–12 min | 40–60 lines |
| 13+ min | 60–80 lines |

If a note is running long, the excess is almost always narrative reconstruction — cut it. `Substance` should read as claims, not as a retelling.

## Two layers, kept separate

- **Topic references** (`study/notes/0*.md`) — organised by concept, built from primary sources. The **durable** layer. When a lecture note and a topic reference disagree, the topic reference wins.
- **Lecture notes** (`study/notes/lectures/`) — organised by where the reader actually was. The **provenance** layer. These get thinner over time as content is promoted upward during integration passes.
Cross-link between them constantly. Neither is complete alone.
