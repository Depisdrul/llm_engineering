---
title: Tool Use & Function Calling
topics: ['tool-use']
difficulty: intermediate
weeks: [8]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# Tool Use & Function Calling

## Overview

!!! abstract "Primary reference: [Agents & Deployment (Week 8)](../../reference/agents-and-deployment.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

OpenAI tools, function calling patterns, structured outputs

Covered in week(s) 8: 6 notebook(s) with content in the repo.

## Core Concepts

### Week 8 Day 4 — The Price is Right

- Week 8 Order of Play
- Start with some test data

### Week 8 Day 2 — The Price is Right

- Week 8 Order of Play
- RAG (Retrieval Augmented Generation) based on a dataset of 800,000 scraped Amazon products
- By calculating vectors for 800,000 scraped products
- Download the Neural Network weights from Week 6 into this directory

### Week 8 Day 3 — The Price is Right

- Week 8 Order of Play
- We are going to ask GPT-5-mini to summarize deals and identify their price
- Introducing Pushover

### Week 8 Day 1 — Welcome to a very busy Week 8 folder

- We have lots to do this week!
- Week 8 Order of Play
- IMPORTANT - please do read and follow these instructions!
- Troubleshooting
- Added thanks to student Tue H
- We need to set your HuggingFace Token as a secret in Modal
- Super important - please read - this confuses a lot of people!
- Add this to your .env if you want the Preprocessor to use a different model by default

### Week 8 Day 5 — The Price is Right

- Just hit shift + enter in the next cell, and let the deals flow in!!

## Implementation Patterns

**code_example** — Week 8 Day 4 — The Price is Right

```python
tools = [{"type": "function", "function": scan_function},
 {"type": "function", "function": estimate_function},
 {"type": "function", "function": notify_function}
 ]
```

**code_example** — Week 8 Day 4 — The Price is Right

```python
scan_function = {
        "name": "scan_the_internet_for_bargains",
        "description": "Returns top bargains scraped from the internet along with the price each item is being offered for",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    }

estimate_function = {
    "name": "estimate_true_value",
    "description": "Given the description of an item, estimate how much it is actually worth",
    "parameters": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "The description of the item to be estimated"
            },
        },
        "required": ["description"],
        "additionalProperties": False
    }
}

notify_function = {
    "name": "notify_user_of_deal",
    "description": "Send the user a push notification about the single most compelling deal; only call this one time",
# ... truncated
```

**api_call** — Week 8 Day 4 — The Price is Right

```python
import json
import logging

import chromadb
from agents.scanner_agent import ScannerAgent
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
openai = OpenAI()
MODEL = "gpt-5.1"
```

**function** — Week 8 Day 4 — The Price is Right

```python
def handle_tool_call(message):
    """
    Actually call the tools associated with this message
    """
    results = []
    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
    return results
```




## Related Topics

- [Embeddings](../04-rag/embeddings.md)
- [Agent Architectures](agent-architectures.md)
- [Multi-Agent Systems](multi-agent.md)
- [Production Deployment](../08-production/deployment.md)
- [UI & Interaction](../08-production/ui-interaction.md)

## Week References

- [Week 8](../../week-summaries/week8.md)

