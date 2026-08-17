---
title: Embeddings
topics: ['embeddings']
difficulty: intermediate
weeks: [2, 5, 8]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# Embeddings

## Overview

!!! abstract "Primary reference: [RAG & Vector Search (Week 5)](../../reference/rag-and-vector-search.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

HuggingFace embeddings, vector representations, similarity search

Covered in week(s) 2, 5, 8: 18 notebook(s) with content in the repo.

## Core Concepts

### Week 5 Day 2 — day2

- Expert Knowledge Worker
- A question answering agent that is an expert knowledge worker
- To be used by employees of Insurellm, an Insurance Tech company
- The agent needs to be accurate and the solution should be low cost
- PART A: Divide our documents into chunks
- PART B: Make vectors and store in Chroma
- Part C: Visualize!

### Week 8 Day 2 — The Price is Right

- Week 8 Order of Play
- RAG (Retrieval Augmented Generation) based on a dataset of 800,000 scraped Amazon products
- By calculating vectors for 800,000 scraped products
- Download the Neural Network weights from Week 6 into this directory

### Week 5 Day 3 — day3

- RAG Day 3
- Expert Question Answerer for InsureLLM
- Connect to Chroma; use Hugging Face all-MiniLM-L6-v2
- Set up the 2 key LangChain objects: retriever and llm
- These LangChain objects implement the method invoke()

### Week 2 Day 2 — Gradio Day!

- User Interface time!
- Adding authentication
- Forcing dark mode

### Week 8 Day 1 — Welcome to a very busy Week 8 folder

- We have lots to do this week!
- Week 8 Order of Play
- IMPORTANT - please do read and follow these instructions!
- Troubleshooting
- Added thanks to student Tue H
- We need to set your HuggingFace Token as a secret in Modal
- Super important - please read - this confuses a lot of people!
- Add this to your .env if you want the Preprocessor to use a different model by default

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

**code_example** — Week 5 Day 2 — day2

```python
# Let's investigate the vectors

collection = vectorstore._collection
count = collection.count()

sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
```

**code_example** — Week 5 Day 3 — day3

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
```




## Related Topics

- [LLM APIs & Integration](../01-foundations/llm-apis.md)
- [Prompt Engineering](../02-core-concepts/prompt-engineering.md)
- [Model Selection & Comparison](../02-core-concepts/model-selection.md)
- [Tokenization & Context](../02-core-concepts/tokenization.md)
- [Multi-Model Orchestration](../03-integration/multi-model.md)
- [RAG Systems](rag-systems.md)
- [Vector Databases](vector-databases.md)
- [Agent Architectures](../05-agents/agent-architectures.md)

## Week References

- [Week 2](../../week-summaries/week2.md)
- [Week 5](../../week-summaries/week5.md)
- [Week 8](../../week-summaries/week8.md)

