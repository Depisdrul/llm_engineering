---
title: UI & Interaction
topics: ['ui-interaction']
difficulty: intermediate
weeks: [5, 8]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# UI & Interaction

## Overview

!!! abstract "Primary reference: [Agents & Deployment (Week 8)](../../reference/agents-and-deployment.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

Gradio interfaces, chatbots, streaming responses, conversational AI

Covered in week(s) 5, 8: 11 notebook(s) with content in the repo.

## Core Concepts

### Week 5 Day 1 — Welcome to RAG week!!

- Expert Knowledge Worker
- A question answering Assistant that is an expert knowledge worker
- To be used by employees of Insurellm, an Insurance Tech company
- The AI assistant needs to be accurate and the solution should be low cost

### Week 5 Day 3 — day3

- RAG Day 3
- Expert Question Answerer for InsureLLM
- Connect to Chroma; use Hugging Face all-MiniLM-L6-v2
- Set up the 2 key LangChain objects: retriever and llm
- These LangChain objects implement the method invoke()

### Week 8 Day 5 — The Price is Right

- Just hit shift + enter in the next cell, and let the deals flow in!!

## Implementation Patterns

**imports** — Week 5 Day 1 — Welcome to RAG week!!

```python
import glob
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
```

**imports** — Week 5 Day 3 — day3

```python
import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
```

**function** — Week 8 Day 5 — The Price is Right

```python
# Updated to change from height to max_height due to change in Gradio v5
# With much thanks to student Ed B. for raising this

with gr.Blocks(title="The Price is Right", fill_width=True) as ui:

    initial_deal = Deal(product_description="Example description", price=100.0, url="https://cnn.com")
    initial_opportunity = Opportunity(deal=initial_deal, estimate=200.0, discount=100.0)
    opportunities = gr.State([initial_opportunity])

    def get_table(opps):
        return [[opp.deal.product_description, opp.deal.price, opp.estimate, opp.discount, opp.deal.url] for opp in opps]

    with gr.Row():
        gr.Markdown('<div style="text-align: center;font-size:24px">"The Price is Right" - Deal Hunting Agentic AI</div>')
    with gr.Row():
        gr.Markdown('<div style="text-align: center;font-size:14px">Deals surfaced so far:</div>')
    with gr.Row():
        opportunities_dataframe = gr.Dataframe(
            headers=["Description", "Price", "Estimate", "Discount", "URL"],
            wrap=True,
            column_widths=[4, 1, 1, 1, 2],
            row_count=10,
            col_count=5,
            max_height=400,
        )

    ui.load(get_table, inputs=[opportunities], outputs=[opportunities_dataframe])

ui.launch(inbrowser=True)
```


## Business Applications

- **Week 5 Day 1 — Welcome to RAG week!!**: Business applications of this week's projects RAG is perhaps the most immediately applicable technique of anything that we cover in the course! In fact, there are commercial products that do precisely what we build this week: nuanced querying across large databases of information, such as company contracts or product specs. RAG gives you a quick-to-market, low cost mechanism for adapting an LLM to


## Related Topics

- [Prompt Engineering](../02-core-concepts/prompt-engineering.md)
- [RAG Systems](../04-rag/rag-systems.md)
- [Embeddings](../04-rag/embeddings.md)
- [Vector Databases](../04-rag/vector-databases.md)
- [Agent Architectures](../05-agents/agent-architectures.md)
- [Tool Use & Function Calling](../05-agents/tool-use.md)
- [Multi-Agent Systems](../05-agents/multi-agent.md)
- [Production Deployment](deployment.md)

## Week References

- [Week 5](../../week-summaries/week5.md)
- [Week 8](../../week-summaries/week8.md)

