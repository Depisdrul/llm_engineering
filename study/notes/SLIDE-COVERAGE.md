# Slide coverage map

**Built 2026-08-17** from all 41 decks (weeks 1-8, one per day plus a promo deck).

It ranks **which days are worth distilling from the Udemy transcript**. The decks
themselves are done — fully extracted and read — so there is no separate pass over
them to schedule. What remains is choosing where transcript time goes.

The ranking is day-level and distillation is lecture-level: a day covers roughly five
lectures, so the Udemy curriculum still translates a chosen day into lecture numbers.
This narrows 41 days to about 8; you pick the lectures inside them.

It paraphrases rather than transcribes, because the decks are the instructor's
material and stay in the gitignored `lectures/_slides/`.

Regenerate the underlying text with:

```bash
python study/pipeline/src/extractors/slides_extractor.py
```

| Looking for | Read |
| --- | --- |
| Why the layers are shaped this way | [`DECISIONS.md`](DECISIONS.md) |
| How to distill a lecture | `.claude/skills/course-distill/SKILL.md` |
| Where lecture notes go | [`LECTURE-DISTILL.md`](LECTURE-DISTILL.md) |

---

## 1. What the corpus actually is

**339 slides across 41 decks — 143 of them (42%) are recap or title slides.** Every
deck opens with a "what you can now do" slide and closes with a "what you'll be able
to do next time" slide, and the 8-week journey map is reprinted in 9 separate decks.
The real content is **196 slides**, roughly 24k tokens of text.

**Decks are per-day, not per-lecture.** 41 decks against 210 lectures means one deck
backs ~5 lectures. A deck gives you a day's skeleton and its numbers; it does not
tell you what was said in any individual lecture. That is still the transcript's job,
and it is why distillation stays a browser task.

**The slides are worth reading before a study block, not during.** They are the
agenda. Ed talks over them, and the talking is where the reasoning lives.

---

## 2. Where the images carry the content

Text extraction returns nothing from a diagram. Four clusters are **diagram-only** —
their text layer is just box labels with no connecting logic:

| Deck | Slides | What the diagram carries |
| --- | --- | --- |
| `week7-day1` | 6-11 | The LoRA build-up: freeze weights → pick target modules → attach low-rank adaptors → apply → the A/B matrix pair. Six progressive slides, text layer is the words "Target Module" and "Low Rank Adaptor" repeated. |
| `week7-day5` | 3-12 | The four steps of training as a loop, drawn twice at two depths. Text layer is `Δ Δ Δ Δ` and `89`. |
| `week5-day1` | 6, 10 | Small idea vs big idea of RAG. The boxes are extracted; the arrows are not, and the arrows are the point. |
| `week8-day1` | 8 | The seven-agent architecture and who calls whom. |

**Everything else with numbers already has them in the text layer.** The parameter-count
charts, the prediction-error bar charts, and the quantization size chart all extracted
cleanly. So the vision pass is worth far less than the 430-image count suggested — the
genuinely image-dependent set is roughly **20 slides**, all box-and-arrow diagrams whose
labels are already in hand. See §6.

---

## 3. Per-week coverage

Priority is for **distillation from the transcript**, not for reading the deck.
"Concept" means the day teaches an idea you cannot recover from the notebook;
"walkthrough" means the day is mostly live coding and the notebook is the record.

### Week 1 — Foundations (60 content slides, the densest week)

| Day | Content | Covers | Priority |
| --- | --- | --- | --- |
| 1 | 11 | Course structure, environment setup, first frontier call | Low |
| 2 | 9 | Model taxonomy: closed vs open frontier families; three ways to use a model (chat UI / cloud API / direct inference) | Medium |
| 3 | 6 | **Base vs Chat/Instruct vs Reasoning vs Hybrid**, budget forcing; frontier strengths and blind spots | **High — concept** |
| 4 | 13 | **Transformer timeline, parameter-count history, why tokens (char → word → subword), token rules of thumb, context window, API cost model** | **High — concept, densest deck in the course** |
| 5 | 2 | Brochure generator build | Low — walkthrough |

The promo deck duplicates day 1 almost slide-for-slide. Ignore it.

### Week 2 — Frontier APIs, UIs, tools (16 content slides)

| Day | Content | Covers | Priority |
| --- | --- | --- | --- |
| 1 | 1 | Multi-provider API setup | Skip |
| 2 | 3 | Gradio intro | Low — walkthrough |
| 3 | 3 | System prompt vs context vs multi-shot, as three distinct levers | Medium |
| 4 | 6 | **Tool calling in theory vs in practice** — the correction that the model only emits tokens and your code executes; tool use cases including the two that become agents | **High — concept** |
| 5 | 3 | First definition of agents; multi-modal assistant | Medium |

### Week 3 — Open source and Hugging Face (13 content slides — the thinnest week)

| Day | Content | Covers | Priority |
| --- | --- | --- | --- |
| 1 | 4 | HF platform surfaces (models/datasets/spaces), library stack, Colab runtime tiers and their GPU RAM | Medium |
| 2 | 3 | The two API levels: pipelines vs tokenizers-and-models | Medium |
| 3 | 2 | Tokenizer concepts: encode/decode, vocab, special tokens, chat templates | Medium |
| 4 | 1 | Names quantization, model internals and streaming — and stops | **High — the deck is empty, so the transcript is the only source** |
| 5 | 3 | Meeting-minutes build, synthetic-data challenge | Low — walkthrough |

**Week 3 is thin in the decks and thin in the repo** — all five notebooks are Colab
stubs. Both sources are missing, which makes the transcript the only record. If you
distill one week for coverage rather than interest, distill this one.

### Week 4 — Model selection and code generation (21 content slides)

| Day | Content | Covers | Priority |
| --- | --- | --- | --- |
| 1 | 6 | **Chinchilla scaling law; the six hard benchmarks with their sizes and human baselines; benchmark limitations including evaluation-awareness** | **High — concept, feeds `04-*.md`** |
| 2 | 4 | The five leaderboards and what each is for; LM Arena and ELO-style rating | **High — concept** |
| 3 | 7 | AI-engineer framing, the 5-step strategy preview, Python→C++ challenge with measured speedups | Medium |
| 4 | 2 | Frontier speedup ranking, then the open-source model's failure on the hard test | Medium |
| 5 | 2 | **Business-centric vs model-centric metrics** — the framing the whole capstone rests on | **High — concept** |

### Week 5 — RAG (21 content slides)

| Day | Content | Covers | Priority |
| --- | --- | --- | --- |
| 1 | 7 | **Auto-encoding vs auto-regressive; what a vector embedding is; small idea vs big idea of RAG** | **High — concept** |
| 2 | 3 | **LangChain's honest pros and cons, including that maturing APIs have reduced the need for it**; encoder and vectorstore options; that vector-DB choice is independent of the embedding | **High — concept** |
| 3 | 4 | RAG pipeline build; the `ingest.py` / `answer.py` / `app.py` split | Low — walkthrough |
| 4 | 2 | **The retrieval metrics — MRR, nDCG, Recall@K, Precision@K — each with a one-line definition, plus LLM-as-judge for answers** | **Highest in the course** |
| 5 | 5 | **The ten advanced RAG techniques**, then dropping LangChain | **High — densest single slide in the course** |

Ed calls evals "perhaps the most important topic of the entire course" on day 4. The
deck agrees with him: it is the only slide that defines its metrics rather than naming
them.

### Week 6 — Data curation and frontier fine-tuning (30 content slides)

| Day | Content | Covers | Priority |
| --- | --- | --- | --- |
| 1 | 11 | Capstone framing; the three participation tiers with cost estimates; dataset sourcing; train/validation/test split | Medium |
| 2 | 9 | **The 5-step strategy in full — Understand, Prepare, Select, Customize, Productionize** — the course's spine | **High — concept** |
| 3 | 5 | Why a baseline matters; generalization vs overfitting | Medium |
| 4 | 2 | The four steps of training; hyper-parameters as trial and error | Medium |
| 5 | 3 | **The full results table, and the finding that fine-tuning the frontier model made it worse**, with the list of what frontier fine-tuning is actually for | **High — concept, and the most useful negative result in the course** |

### Week 7 — QLoRA (28 content slides)

| Day | Content | Covers | Priority |
| --- | --- | --- | --- |
| 1 | 11 | **LoRA from first principles in six diagrams; r / alpha / target-modules with rules of thumb; quantization intuition and the memory ladder** | **High — concept, diagram-dependent** |
| 2 | 2 | Base vs instruct selection; base-model error numbers | Medium |
| 3 | 4 | The five QLoRA and five training hyper-parameters, named as two groups | Medium |
| 4 | 0 | **Nothing.** Three slides, all recap | Transcript-only |
| 5 | 11 | **Forward pass → loss → backprop → optimization, then cross-entropy loss derived properly** (why it is the negative log of the probability assigned to the true token) | **High — concept, diagram-dependent** |

### Week 8 — Agents and the capstone (15 content slides)

| Day | Content | Covers | Priority |
| --- | --- | --- | --- |
| 1 | 6 | **Three successive definitions of an agent, dated**; the seven-agent architecture; Modal | **High — concept** |
| 2 | 2 | The agent workflow with all seven roles named | Medium |
| 3 | 1 | **Structured outputs: Pydantic → JSON schema → system prompt, plus constrained decoding** | Medium |
| 4 | 2 | Hallmarks of an agentic solution | Medium |
| 5 | 4 | Recap and close | Low |

---

## 4. Distillation priority, condensed

If the goal is maximum understanding per hour, take these days first:

1. **w5d4** — retrieval metrics. Nothing else in the course defines them.
2. **w1d4** — tokens, context, cost. Everything downstream assumes it.
3. **w7d1 + w7d5** — LoRA and the training loop. Both diagram-dependent.
4. **w6d5** — the negative fine-tuning result and what it implies.
5. **w4d1 + w4d5** — benchmarks and the metric taxonomy.
6. **w5d5** — the ten advanced techniques.
7. **w2d4** — tool calling in practice.
8. **w3d4** — because the deck and the notebooks are both empty.

Days to skip unless something specific is wanted: w2d1, w1d5, w1d1, w8d5, and the
promo deck.

---

## 5. Discrepancies the decks revealed

These belong in the corrections tables of the topic references once confirmed against
the transcripts.

- **Two different speedup scales in week 4.** Day 3 reports one order of magnitude for
  the simple and hard tests; day 4 reports a different, much larger frontier ranking.
  They are different tests and the numbers are easy to conflate. [certain — both are
  in the decks]
- **Frontier fine-tuning made the model worse** (week 6 day 5), and the fine-tuned
  variant scores below the plain one. The lecture treats this as a deliberate teaching
  moment, not an error. Do not carry the fine-tuned number forward as an improvement.
- **The base open-source model scores worse than the constant baseline** (week 7 day 2).
  That is the point of the week, but out of context it reads as a broken experiment.
- **Model names drift between decks** — week 8 day 1 and day 2 name different GPT
  versions for the same scanner agent. The decks were updated at different times.
  Trust the notebook, not the slide. [likely]
- **A stale figure survives in the week 3 day 1 text layer**: the HF model and dataset
  counts appear twice, old value and new. Confirms the decks are edited in place, so a
  number on a slide has no reliable date.
- Two decks carry the wrong day in a heading (`week3-day4` and `week8-day4` both say
  the previous day). Cosmetic, but it means the heading is not a reliable key.

---

## 6. Verdict on the vision pass

**Not worth running over 430 images.** The measurement changed the picture:

- The charts that looked like the prize — parameter counts, prediction errors,
  quantization sizes — **already have their numbers in the extracted text.** A VLM
  would be re-reading what we have.
- The genuinely image-dependent content is the four diagram clusters in §2, about
  **20 slides**, and what's missing from them is the arrows, not the labels.
- For those 20, the arrows are also described aloud in the lecture, so the transcript
  supplies them during distillation at no extra cost.

**What to do instead:** reconstruct the four diagrams as mermaid, from the extracted
labels plus the transcript, at the point where the matching lecture gets distilled.
That yields a publishable diagram that diffs in git and follows the site theme, and it
skips the hallucination risk entirely.

Keep the local VLM idea in reserve for one case: if a diagram's structure turns out to
be genuinely ambiguous from labels plus transcript, describe that single image rather
than batch-processing the set.
