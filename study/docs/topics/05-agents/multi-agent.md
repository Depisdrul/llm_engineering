---
title: Multi-Agent Systems
topics: ['multi-agent']
difficulty: intermediate
weeks: [8]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# Multi-Agent Systems

## Overview

!!! abstract "Primary reference: [Agents & Deployment (Week 8)](../../reference/agents-and-deployment.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

Agent collaboration, coordination, frameworks (CrewAI, AutoGen, LangGraph)

Covered in week(s) 8: 6 notebook(s) with content in the repo.

## Core Concepts

### Week 8 Day 2 — The Price is Right

- Week 8 Order of Play
- RAG (Retrieval Augmented Generation) based on a dataset of 800,000 scraped Amazon products
- By calculating vectors for 800,000 scraped products
- Download the Neural Network weights from Week 6 into this directory

### Week 8 Day 5 — The Price is Right

- Just hit shift + enter in the next cell, and let the deals flow in!!

### Week 8 Day 3 — The Price is Right

- Week 8 Order of Play
- We are going to ask GPT-5-mini to summarize deals and identify their price
- Introducing Pushover

### Week 8 Day 4 — The Price is Right

- Week 8 Order of Play
- Start with some test data

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

**code_example** — Week 8 Day 2 — The Price is Right

```python
from agents.frontier_agent import FrontierAgent

agent = FrontierAgent(collection)
agent.price("Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio")
```

**function** — Week 8 Day 5 — The Price is Right

```python
agent_framework = DealAgentFramework()
agent_framework.init_agents_as_needed()

with gr.Blocks(title="The Price is Right", fill_width=True) as ui:

    initial_deal = Deal(product_description="Example description", price=100.0, url="https://cnn.com")
    initial_opportunity = Opportunity(deal=initial_deal, estimate=200.0, discount=100.0)
    opportunities = gr.State([initial_opportunity])

    def get_table(opps):
        return [[opp.deal.product_description, opp.deal.price, opp.estimate, opp.discount, opp.deal.url] for opp in opps]

    def do_select(opportunities, selected_index: gr.SelectData):
        row = selected_index.index[0]
        opportunity = opportunities[row]
        agent_framework.planner.messenger.alert(opportunity)

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
# ... truncated
```

**code_example** — Week 8 Day 3 — The Price is Right

```python
from agents.messaging_agent import MessagingAgent

agent = MessagingAgent()
agent.push("SUCH A MASSIVE DEAL!!")
```

**code_example** — Week 8 Day 2 — The Price is Right

```python
from agents.neural_network_agent import NeuralNetworkAgent

agent = NeuralNetworkAgent()
```




## Related Topics

- [Embeddings](../04-rag/embeddings.md)
- [Agent Architectures](agent-architectures.md)
- [Tool Use & Function Calling](tool-use.md)
- [Production Deployment](../08-production/deployment.md)
- [UI & Interaction](../08-production/ui-interaction.md)

## Week References

- [Week 8](../../week-summaries/week8.md)

