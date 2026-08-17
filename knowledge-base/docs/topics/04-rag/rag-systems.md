---
title: RAG Systems
topics: ['rag-systems']
difficulty: intermediate
weeks: [5]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# RAG Systems

## Overview

!!! abstract "Primary reference: [RAG & Vector Search (Week 5)](../../reference/rag-and-vector-search.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

Vector databases, embeddings, retrievers, semantic search, chunking

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

### Week 5 Day 1 — Welcome to RAG week!!

- Expert Knowledge Worker
- A question answering Assistant that is an expert knowledge worker
- To be used by employees of Insurellm, an Insurance Tech company
- The AI assistant needs to be accurate and the solution should be low cost

### Week 5 Day 4 — RAG Day 4

- Evaluation!

## Implementation Patterns

**api_call** — Week 5 Day 2 — day2

```python
# Pick an embedding model

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")
```

**code_example** — Week 5 Day 2 — day2

```python
# Let's investigate the vectors

collection = vectorstore._collection
count = collection.count()

sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
```

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


## Business Applications

- **Week 5 Day 1 — Welcome to RAG week!!**: Business applications of this week's projects RAG is perhaps the most immediately applicable technique of anything that we cover in the course! In fact, there are commercial products that do precisely what we build this week: nuanced querying across large databases of information, such as company contracts or product specs. RAG gives you a quick-to-market, low cost mechanism for adapting an LLM to


## Related Topics

- [Prompt Engineering](../02-core-concepts/prompt-engineering.md)
- [Embeddings](embeddings.md)
- [Vector Databases](vector-databases.md)
- [UI & Interaction](../08-production/ui-interaction.md)

## Week References

- [Week 5](../../week-summaries/week5.md)

