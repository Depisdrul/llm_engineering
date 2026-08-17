# Open questions

Everything unresolved, grouped by theme rather than by lecture — themes recur
across weeks, lectures don't. Fed by the `## Open` sections of the lecture notes
and by whatever a study block leaves hanging. Answer an item, then delete it and
fold the answer into the matching topic reference.

Source lectures are in brackets.

---

## Retrieval architecture

- **What is the token size of `week5/knowledge-base/`?** [108] Under ~50K tokens
  and the corpus does not need RAG at all — long context with prompt caching
  wins on cost *and* accuracy (`05-rag-and-vector-search.md` §B7). Worth
  answering early, so week 5's architecture is not carried into a Gewiss project
  where the same reasoning applies. Measurable in one command.
- **Does week 5 ever reach hybrid retrieval, or stay dense-only?** [108, 111]
  Lecture 131 mentions re-ranking and GraphRAG, 130 mentions query rewriting —
  but no lecture title mentions lexical or hybrid search, which would be a real
  gap. Dense-alone scored *worse than BM25 alone* on entity- and number-heavy
  corpora (`05-…md` §B5), which is exactly the Gewiss case — part numbers,
  CEI/IEC references, measurements.
- **Which similarity metric does week 5's Chroma setup use, and are the
  embeddings normalised?** [111] Chroma lets you pick `l2`/`ip`/`cosine`. On
  L2-normalised embeddings all three give the identical top-k ranking, so the
  choice is a performance decision; on un-normalised vectors they diverge and one
  is wrong for the model. Degrades silently either way. `05-…md` §B1.
- **Does `week5/day1.ipynb` use `gr.Interface` or `gr.ChatInterface`?** [108]
  The transcript is loose; the notebook settles it.
- **Where does the course "take a peek at LangChain"?** [111] Ed refers back to
  it; week 5 builds on whatever that was.

## Runnability with a Google-only key

- **Is frontier fine-tuning actually blocked for the Gewiss OpenAI account?**
  [likely blocked] OpenAI has been winding down self-serve fine-tuning for orgs
  that never ran a job. Unresolved — check the Udemy Q&A, or attempt a trivial
  job. If blocked, week 6's hands-on time redirects to the classical-ML and
  frozen-embedding baseline ladder in `06-training-and-finetuning.md`, which
  needs no paid key and is what week 7's fine-tuned model must beat for the
  capstone to mean anything.
- **Which of the 13 OpenAI-only core notebooks actually need the litellm swap?**
  Per-notebook table is in `08-api-keys-and-runnability.md`. Week 5 is worst hit
  (4 of 5 local notebooks). For week 5 specifically the recommendation is to go
  local rather than swap to Gemini — `multilingual-e5-large` came within 0.003
  nDCG of Google Embeddings 2 on Italian retrieval at 1/7 the latency.

## Lectures watched but not distilled

- **109 (Vector Embeddings and Encoder LLMs) and 110 (How Vector Embeddings
  Represent Meaning: From word2vec to Encoders).** [111] Both concept-only, no
  notebook, so the lecture is the only source. Slides 8–9 cover them, and two
  flags raised in the 108 note belong to exactly these two: encoder-only vs
  decoder-based embedders, and the `King − Man + Woman = Queen` claim. Reopen and
  request each if you want them.

## Verification debt in the topic references

- **The fact-check pass was never run.** Figures came from research subagents:
  source-linked, but not independently re-verified. Highest risk: scaling-law
  coefficients, benchmark item counts, model version claims, and anything on a
  client-side-rendered leaderboard — several are tagged `UNVERIFIED` precisely
  because a fetch tool cannot read those pages but a browser can.
- **The `UNVERIFIED` tags in `04-…md` are concentrated on leaderboards.**
  Client-side-rendered pages a fetch tool cannot read but a browser can — so
  these are cheap to clear from a browser session and impossible from here.

## Course material and repo

- **The 8 Colab stubs have not been downloaded.** `week3/day1`–`day5`,
  `week7/day1`, `week7/day3 and 4`, `week7/day5`. Procedure in
  `study/nb2mdREADME.md`: *Save a copy in Drive* → run → *Download
  .ipynb* → save alongside the stub as `dayN_executed.ipynb`, never over it.
  Until then, week 3 and most of week 7 have no local content to extract from.
- **Week 7 cannot run locally regardless.** `pyproject.toml` and `uv.lock`
  contain no `peft`, `trl`, `bitsandbytes` or `accelerate` — deliberately not
  provisioned for QLoRA. Decide whether to provision them or accept Colab.
