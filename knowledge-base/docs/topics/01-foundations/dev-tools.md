---
title: Development Tools
topics: ['dev-tools']
difficulty: intermediate
weeks: [1]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# Development Tools

## Overview

!!! note "No primary-source reference yet"

    This page is assembled from the course notebooks only. It has not been
    cross-checked against primary sources, so treat dated claims with caution.

Jupyter, debugging, environment setup, git workflows

Covered in week(s) 1: 5 notebook(s) with content in the repo.

## Core Concepts

### Week 1 Day 1 — YOUR FIRST LAB

- Your first Frontier LLM Project
- If you're new to working in "Notebooks" (also known as Labs or Jupyter Lab)
- I am here to help
- More troubleshooting
- If this is old hat!
- If necessary, install Cursor Extensions
- Troubleshooting if you have problems
- Types of prompts

### Week 1 Day 5 — A full business solution

- Use a call to gpt-5-nano to read the links on a webpage, and respond in structured JSON
- Second step: make the brochure!

## Implementation Patterns

**api_call** — Week 1 Day 1 — YOUR FIRST LAB

```python
# Load environment variables in a file called .env

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# Check the key

if not api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif not api_key.startswith("sk-proj-"):
    print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
elif api_key.strip() != api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
else:
    print("API key found and looks good so far!")
```

**imports** — Week 1 Day 5 — A full business solution

```python
# imports
# If these fail, please check you're running from an 'activated' environment with (llms) in the command prompt
import json
import os

from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from scraper import fetch_website_contents, fetch_website_links
```

**code_example** — Week 1 — End of week 1 exercise

```python
# set up environment
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "gemma3:1b"
```


## Business Applications

- **Week 1 Day 5 — A full business solution**: Business applications In this exercise we extended the Day 1 code to make multiple LLM calls, and generate a document. This is perhaps the first example of Agentic AI design patterns, as we combined multiple calls to LLMs. This will feature more in Week 2, and then we will return to Agentic AI in a big way in Week 8 when we build a fully autonomous Agent solution. Generating content in this way is


## Related Topics

- [LLM APIs & Integration](llm-apis.md)
- [Prompt Engineering](../02-core-concepts/prompt-engineering.md)
- [Tokenization & Context](../02-core-concepts/tokenization.md)

## Week References

- [Week 1](../../week-summaries/week1.md)

