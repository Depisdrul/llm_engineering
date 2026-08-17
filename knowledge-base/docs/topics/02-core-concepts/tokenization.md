---
title: Tokenization & Context
topics: ['tokenization']
difficulty: intermediate
weeks: [1, 2]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# Tokenization & Context

## Overview

!!! abstract "Primary reference: [LLM Foundations (Weeks 1-3)](../../reference/llm-foundations.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

Token counting, context windows, tiktoken, encoding formats

Covered in week(s) 1, 2: 12 notebook(s) with content in the repo.

## Core Concepts

### Week 2 Day 1 — Welcome to Week 2!

- Frontier Model APIs
- Setting up your keys - OPTIONAL!
- Adding API keys to your .env file
- Training vs Inference time scaling
- Testing out the best models on the planet
- A spicy challenge to test the competitive spirit
- Going local
- Gemini and Anthropic Client Library

### Week 1 Day 4 — Day 4

- Tokenizing with code
- The Illusion of "memory"
- You should be very comfortable with what the next cell is doing!
- A message to OpenAI is a list of dicts
- Wait, wha??
- To recap

## Implementation Patterns

**code_example** — Week 1 Day 4 — Day 4

```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4.1-mini")

tokens = encoding.encode("Hi my name is Ed and I like banoffee pie")
```

**code_example** — Week 2 Day 1 — Welcome to Week 2!

```python
print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")
print(f"Total cost: {response._hidden_params['response_cost']*100:.4f} cents")
```

**code_example** — Week 2 Day 1 — Welcome to Week 2!

```python
print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")
print(f"Total cost: {response._hidden_params['response_cost']*100:.4f} cents")
```

**code_example** — Week 1 Day 4 — Day 4

```python
for token_id in tokens:
    token_text = encoding.decode([token_id])
    print(f"{token_id} = {token_text}")
```


## Business Applications

- **Week 2 Day 1 — Welcome to Week 2!**: Business relevance This structure of a conversation, as a list of messages, is fundamental to the way we build conversational AI assistants and how they are able to keep the context during a conversation. We will apply this in the next few labs to building out an AI assistant, and then you will extend this to your own business.


## Related Topics

- [LLM APIs & Integration](../01-foundations/llm-apis.md)
- [Prompt Engineering](prompt-engineering.md)
- [Development Tools](../01-foundations/dev-tools.md)
- [Model Selection & Comparison](model-selection.md)
- [Multi-Model Orchestration](../03-integration/multi-model.md)
- [Embeddings](../04-rag/embeddings.md)

## Week References

- [Week 1](../../week-summaries/week1.md)
- [Week 2](../../week-summaries/week2.md)

