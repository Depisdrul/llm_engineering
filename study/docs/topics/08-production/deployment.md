---
title: Production Deployment
topics: ['deployment']
difficulty: intermediate
weeks: [8]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# Production Deployment

## Overview

!!! abstract "Primary reference: [Agents & Deployment (Week 8)](../../reference/agents-and-deployment.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

Modal.com, containerization, API endpoints, scaling, cost optimization

Covered in week(s) 8: 6 notebook(s) with content in the repo.

## Core Concepts

### Week 8 Day 2 — The Price is Right

- Week 8 Order of Play
- RAG (Retrieval Augmented Generation) based on a dataset of 800,000 scraped Amazon products
- By calculating vectors for 800,000 scraped products
- Download the Neural Network weights from Week 6 into this directory

### Week 8 Day 4 — The Price is Right

- Week 8 Order of Play
- Start with some test data

## Implementation Patterns

**code_example** — Week 8 Day 2 — The Price is Right

```python
# How much does our favorite distortion pedal cost?

test[0].price
```

**function** — Week 8 Day 4 — The Price is Right

```python
def notify_user_of_deal(description: str, deal_price: float, estimated_true_value: float, url: str) -> str:
    """
    This tool notifies the user of a great deal, given a description of it, the price of the deal, and the estimated true value
    """
    print(f"Fake function to notify user of {description} which costs {deal_price} and estimate is {estimated_true_value}")
    return "notification sent ok"
```




## Related Topics

- [Embeddings](../04-rag/embeddings.md)
- [Agent Architectures](../05-agents/agent-architectures.md)
- [Tool Use & Function Calling](../05-agents/tool-use.md)
- [Multi-Agent Systems](../05-agents/multi-agent.md)
- [UI & Interaction](ui-interaction.md)

## Week References

- [Week 8](../../week-summaries/week8.md)

