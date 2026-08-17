<!-- Generated copy. Source: notes/lectures/week5/108-simple-rag-dictionary-lookup.md
     Edit the source file; this copy is overwritten on every extraction run. -->

# 108 — Day 1 - Building a Simple RAG System: Dictionary Lookup and Context Retrieval

**Week 5, Day 1** · 9 min · `week5/day1.ipynb` · deck: *LLM - Week 5 Day 1* (not yet read)

## Claim

A deliberately naive "fake RAG" — split the question on whitespace, look each word up in a Python dict keyed on surnames and product names, prepend whatever matches to the system prompt — is enough to prove that context injection works. Its two failure modes are the actual lesson, and they're what real retrieval exists to fix.

## Substance

- **Mechanism:** lowercase the question, split into words, look each word up in a dict built from the knowledge-base files. Concatenate any hits into a block introduced with "the following additional context might be relevant…", and prepend that to the system message. No embeddings, no scoring, no ranking.
- **Failure mode 1 — false negative.** `who is Lancaster` retrieves; `who is Avery` retrieves nothing, because the dict is keyed on surname only. Matching is case-insensitive, so lowercase `lancaster` still works. Ed notes the fix (also key on first names) is trivial and deliberately *not* applied, so the brittleness stays visible.
- **Failure mode 2 — false positive, and the more dangerous one.** `who is Alex Lancaster` retrieves Avery Lancaster's context and the model answers from it. Word-level lookup has no notion of whether the retrieved context is *about the entity asked for*. So the system is wrong in both directions: it misses things it holds, and it confidently supplies context about a different entity.
- **The conversation-history trap — the most valuable thing in this lecture.** After asking `who is Avery Lancaster`, asking `who is Avery` in the same session **answers correctly even though retrieval returned nothing** — because the full chat history is also in the prompt and already contains the answer. The test appeared to pass while the retrieval component was broken. Only a fresh session exposes it (`I don't have information about Avery`).
- **Assembly:** `system = system_prefix + additional_context`, then `[system, *history, user]` → `chat.completions.create` on a cheap nano model → `choices[0].message.content`. Surfaced through Gradio with `launch(inbrowser=True)`.
- **Pythonic aside:** `results += …` rather than `results = results + …`.
- Deeper questions work identically — "how much does Healthllm cost" is the same surname-lookup mechanism against the product docs, not a different code path.

## Code / demo

`week5/day1.ipynb`, against `week5/knowledge-base/` (`company/`, `contracts/`, `employees/`, `products/` markdown files). Entities exercised: Avery Lancaster (co-founder/CEO), Maxine Thompson (senior data engineer), and the Carllm / Homellm / Healthllm products.

> **Do not trust the transcript's spelling of any proper noun here.** Udemy's ASR garbles the invented company and product names badly — "Insurellm" comes through as "In Serum" and "insurer Elm", "Carllm" as "car LMDh", "Rellm" as "realm". Take spellings from the `knowledge-base/` files in the repo, which are authoritative.

## Corrections

- **The history-contamination point generalises into a real evaluation discipline.** See `05-rag-and-vector-search.md` §B6: retrieval and generation must be measured separately, and you should log `answer_present_in_assembled_context` per query. This lecture is an accidental demonstration of exactly the failure that discipline prevents — and it's worth more than the RAG mechanics it's illustrating.
- **The lecture frames the upgrade path as "replace string matching with fuzzy/embedding matching." That's only half right.** Per `05-…md` §B5, dense retrieval *alone* scored **worse than BM25 alone** on entity- and number-heavy corpora (0.587 vs 0.644 Recall@5, [arXiv:2604.01733](https://arxiv.org/html/2604.01733v1)). The dictionary lookup being discarded here is a degenerate form of lexical retrieval, and lexical retrieval is precisely what handles surnames, part numbers and product codes. **The right upgrade is hybrid — BM25 + dense, fused with RRF — not swapping lexical out for dense.** Watch whether the week gets to this; lecture 131 mentions re-ranking and 130 mentions query rewriting, so it may.
- **Provider:** uses OpenAI. With a Google key only, this is one of the four OpenAI-only notebooks in week 5 — swap via litellm, `08-api-keys-and-runnability.md` §2b.

## Gewiss angle

Failure mode 2 is the one that should worry you for an industrial corpus. A query about part `4.2.1` retrieving `4.2.11`'s data — and the model answering fluently from it — is the same bug with a much worse consequence than a mistaken CEO. Practical implication for anything you build over Gewiss part numbers, CEI/IEC references or measurement values: **identifiers need exact-match filtering, not similarity.** Put them in metadata and filter on them; don't rely on the embedding to distinguish `4.2.1` from `4.2.11`, because it won't.

## Open

- Does `week5/day1.ipynb` use `gr.Interface` or `gr.ChatInterface`? The transcript is loose on this; check the notebook.
- **What is the actual token size of `week5/knowledge-base/`?** This matters more than it sounds: per `05-…md` §B7, a corpus under ~50K tokens should skip RAG entirely and use long context with prompt caching. If the course's knowledge base is that small, week 5 is teaching RAG on a corpus that doesn't need it — fine pedagogically, but worth knowing so you don't generalise the architecture to a case where it's over-engineering.
- Try the fresh-session discipline on your own eval harness before building anything real.
- Whether the week's later lectures add BM25/hybrid or go dense-only.

## Links

- Repo: `week5/day1.ipynb`, `week5/knowledge-base/`
- Topic references: `05-rag-and-vector-search.md` §B5 (hybrid vs dense), §B6 (eval separation), §B7 (RAG vs long context) · `08-api-keys-and-runnability.md` §2b (litellm swap)
- [T2-RAGBench — BM25 vs dense vs hybrid+rerank](https://arxiv.org/html/2604.01733v1)
