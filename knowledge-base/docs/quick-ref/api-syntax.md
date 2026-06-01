---
title: API Syntax Quick Reference
type: cheatsheet
last_updated: 2026-06-01
---

# API Syntax Quick Reference

*Content will be generated during extraction phase*

## OpenAI API

```python
from openai import OpenAI

openai = OpenAI()
response = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User prompt"}
    ]
)
```

## Anthropic Claude API

```python
from anthropic import Anthropic

anthropic = Anthropic()
response = anthropic.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "User prompt"}
    ]
)
```

## Common Patterns

*To be filled during extraction*
