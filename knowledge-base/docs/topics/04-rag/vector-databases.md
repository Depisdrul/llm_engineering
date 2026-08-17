---
title: Vector Databases
topics: ['vector-databases']
difficulty: intermediate
weeks: [5]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# Vector Databases

## Overview

!!! abstract "Primary reference: [RAG & Vector Search (Week 5)](../../reference/rag-and-vector-search.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

Chroma, FAISS, persistence, retrieval strategies

Covered in week(s) 5: 5 notebook(s) with content in the repo.

## Core Concepts

### Week 5 Day 2 — day2

- Expert Knowledge Worker
- A question answering agent that is an expert knowledge worker
- To be used by employees of Insurellm, an Insurance Tech company
- The agent needs to be accurate and the solution should be low cost
- PART A: Divide our documents into chunks
- PART B: Make vectors and store in Chroma
- Part C: Visualize!

### Week 5 Day 3 — day3

- RAG Day 3
- Expert Question Answerer for InsureLLM
- Connect to Chroma; use Hugging Face all-MiniLM-L6-v2
- Set up the 2 key LangChain objects: retriever and llm
- These LangChain objects implement the method invoke()

### Week 5 Day 4 — RAG Day 4

- Evaluation!

## Implementation Patterns

**code_example** — Week 5 Day 2 — day2

```python
# We humans find it easier to visalize things in 2D!
# Reduce the dimensionality of the vectors to 2D using t-SNE
# (t-distributed stochastic neighbor embedding)

tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker={"size": 5, "color": colors, "opacity": 0.8},
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(title='2D Chroma Vector Store Visualization',
    scene={"xaxis_title": 'x',"yaxis_title": 'y'},
    width=800,
    height=600,
    margin={"r": 20, "b": 10, "l": 10, "t": 40}
)

fig.show()
```

**code_example** — Week 5 Day 2 — day2

```python
# Let's try 3D!

tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker={"size": 5, "color": colors, "opacity": 0.8},
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='3D Chroma Vector Store Visualization',
    scene={"xaxis_title": 'x', "yaxis_title": 'y', "zaxis_title": 'z'},
    width=900,
    height=700,
    margin={"r": 10, "b": 10, "l": 10, "t": 40}
)

fig.show()
```

**api_call** — Week 5 Day 5 — Let's go PRO!

```python
def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    emb = openai.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection(collection_name)

    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")
```

**code_example** — Week 5 Day 5 — Let's go PRO!

```python
tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker={"size": 5, "color": colors, "opacity": 0.8},
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='3D Chroma Vector Store Visualization',
    scene={"xaxis_title": 'x', "yaxis_title": 'y', "zaxis_title": 'z'},
    width=900,
    height=700,
    margin={"r": 10, "b": 10, "l": 10, "t": 40}
)

fig.show()
```




## Related Topics

- [Prompt Engineering](../02-core-concepts/prompt-engineering.md)
- [RAG Systems](rag-systems.md)
- [Embeddings](embeddings.md)
- [UI & Interaction](../08-production/ui-interaction.md)

## Week References

- [Week 5](../../week-summaries/week5.md)

