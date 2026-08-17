<!-- Generated copy. Source: notes/lectures/week5/111-big-idea-vector-data-stores.md
     Edit the source file; this copy is overwritten on every extraction run. -->

# 111 — Day 1 - Understanding the Big Idea Behind RAG and Vector Data Stores

**Week 5, Day 1** · 7 min · no notebook (concept-only) · slides: *Week 5 Day 1*, slide 10 (the diagram he refers to) + slide 11 (progress report)

## Claim

Replace string lookup with vector proximity. Encode the question with an encoder model, encode every knowledge-base entry with the *same* encoder at ingest time, store text alongside its vector, then retrieve by nearness in that space — which finds matching *meaning* rather than matching words. The retrieved **text**, never the vectors, goes into the prompt.

## Substance

- **The motivating example.** User asks about ticket prices to *Heathrow*; the knowledge base holds ticket prices to *London*. The word "Heathrow" appears nowhere in the corpus, so any string match returns nothing. Vector proximity finds it, because the two phrases sit near each other in embedding space. This is the concrete answer to the brittleness demonstrated in lecture 108.
- **What a "vector data store" is:** the store holds, for each entry, both the original text *and* the vector representing it. Query-time operation is "what do you hold that is closest to this point?" Nearness is the whole retrieval mechanism.
- **The point Ed flags as confusing even professionals, and he's right to.** The **encoder LLM has nothing to do with the generating LLM.** They are two unrelated models doing two unrelated jobs. The encoder exists only to power the fuzzy lookup — vectorise the question, vectorise the corpus. The generator (auto-regressive) receives **natural language only**: the user's question plus the retrieved text. It never sees a vector and would not know what to do with one.
- **Ed's own closing assessment, which is unusually honest for a course and worth keeping:** RAG "is really one big hack," it comes with "a whole zoo of hacks," "hack upon hack upon hack" to get better at picking relevant context. He calls it "very empirical," which he glosses as very experimental — a lot of trial and error.
- Day 1 ends here. Next: LangChain properly, a knowledge base turned into vectors, stored in Chroma.

## Corrections

- **Ed's "it's all a hack, it's empirical" is correct, and the evidence is stronger than he lets on.** Every technique-comparison study in `05-rag-and-vector-search.md` §B5 found that intuitively-appealing methods lose to boring baselines: semantic chunking has **two independent negative evaluations** (NAACL 2025; a July 2026 academic-texts study), and HyDE was the **worst of ten methods** tested on a 23,088-question benchmark — worse than plain BM25. So "empirical" isn't a caveat, it's an instruction. **The actionable form of his point: build a 50–200 query gold set from your own corpus before adopting any technique.** Without it every "improvement" is vibes.
- **The Heathrow/London example is a best case for dense retrieval, not a representative one.** Vocabulary mismatch between query and document is precisely where embeddings win and lexical matching cannot. But the symmetric case is just as real and the lecture never mentions it: on entity- and number-heavy corpora, **BM25 alone beat dense alone** — 0.644 vs 0.587 Recall@5 ([arXiv:2604.01733](https://arxiv.org/html/2604.01733v1)). Hybrid BM25 + dense with RRF reached 0.816. The honest conclusion from Day 1 is not "vectors replace string matching" but "you need both."
- **The example also happens to be the configuration that degrades worst at scale.** Chroma's Context Rot work found that **query–document semantic similarity is a first-order variable**: high-similarity pairs hold up as context grows, low-similarity pairs (cosine ~0.4–0.5) degrade sharply. "Heathrow" → "London" is exactly a low-similarity pair. It works beautifully in a toy corpus and is the first thing to break in a real one. See `05-…md` and `01-llm-foundations.md` §A3.
- **"Closest" needs a metric, and neither the lecture nor the slide says which.** Worth knowing now to avoid a silent bug later: on **L2-normalised** embeddings, cosine, dot product and Euclidean all produce the **identical top-k ranking**, so the choice is a performance decision, not a quality one. On un-normalised vectors they diverge and one of them is wrong for your model. Full derivation in `05-…md` §B1. The trap: if you truncate a Matryoshka embedding you must **re-normalise** — a prefix of a unit vector is not a unit vector.
- **The encoder ≠ generator point has a practical consequence the lecture doesn't draw.** Because they're independent, you choose them independently. For Italian that matters concretely: run `multilingual-e5-large` locally as the encoder (within 0.003 nDCG of Google Embeddings 2 on Italian retrieval, at 1/7 the latency) while using Gemini as the generator. Mixing providers across the two roles is normal, not a compromise.

## Gewiss angle

Two things follow for a Gewiss corpus. First, the Heathrow/London win is real for natural-language queries against normative or descriptive text — an operator asking in colloquial Italian against formal document language is exactly the vocabulary-mismatch case. Second, the *inverse* is true for identifiers: part numbers, CEI/IEC references and measurement values need lexical or exact-match retrieval, because embeddings will happily place `4.2.1` next to `4.2.11`. A Gewiss knowledge worker needs hybrid retrieval plus metadata filtering on identifiers from day one — not as an optimisation, as the baseline architecture.

## Open

- Which similarity metric does week 5's Chroma setup actually use, and are the embeddings normalised? (Chroma lets you pick `l2`/`ip`/`cosine`; getting it wrong for your model degrades silently.)
- **Still unanswered from 108, and it gates everything:** what is `week5/knowledge-base/` in tokens? Under ~50K and the corpus doesn't need RAG at all.
- Does the week ever introduce BM25 or hybrid retrieval, or is it dense-only throughout? Lecture 131 mentions re-ranking and GraphRAG, 130 mentions query rewriting — but no lecture title mentions lexical or hybrid search, which would be a real gap.
- Ed says "we took a peek at LangChain before" — where? Worth finding, since week 5 builds on it.

## Links

- Topic references: `05-rag-and-vector-search.md` §B1 (metrics, normalisation), §B5 (hybrid vs dense, HyDE), §B6 (gold sets) · `01-llm-foundations.md` §A3 (Context Rot, similarity as a length variable)
- Slides: Drive → `Week 5` → `Copy of LLM - Week 5 Day 1`, slides 10–11
- [T2-RAGBench — BM25 vs dense vs hybrid+rerank](https://arxiv.org/html/2604.01733v1)

---

## Note on numbering

Lectures **109** ("Vector Embeddings and Encoder LLMs") and **110** ("How Vector Embeddings Represent Meaning: From word2vec to Encoders") were watched but not distilled — the transcript panel had already advanced to 111 by the time distillation ran. Both are concept-only with no notebook, so if you want them, reopen each and request it. Slides 8–9 cover them, and the two advance flags raised in `108-…md` (encoder-only vs decoder-based embedders; the `King − Man + Woman = Queen` claim) belong to exactly those two lectures.
