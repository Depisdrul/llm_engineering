---
title: Multi-Model Orchestration
topics: ['multi-model']
difficulty: intermediate
weeks: [2, 3]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# Multi-Model Orchestration

## Overview

!!! abstract "Primary reference: [API Keys & Runnability](../../reference/api-keys-and-runnability.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

LiteLLM, LangChain, OpenRouter, Ollama integration

Covered in week(s) 2, 3: 7 notebook(s) with content in the repo, 5 Colab stub(s).

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

### Week 2 — Special Extra!!

- This uses OpenRouter.ai so that we easily access the latest models
- In Week 4 we will have more scientific ways to compare models

### Week 2 Day 4 — Project - Airline AI Assistant

- Tools
- Getting OpenAI to use our Tool
- Exercise

### Week 2 Day 2 — Gradio Day!

- User Interface time!
- Adding authentication
- Forcing dark mode

### Week 2 Day 5 — Project - Airline AI Assistant

- A bit more about what Gradio actually does
- Price alert: each time I generate an image it costs about 4 cents - don't go crazy with images!
- The 3 types of Gradio UI

## Implementation Patterns

**api_call** — Week 2 Day 1 — Welcome to Week 2!

```python
# Connect to OpenAI client library
# A thin wrapper around calls to HTTP endpoints

openai = OpenAI()

# For Gemini, DeepSeek and Groq, we can use the OpenAI python client
# Because Google and DeepSeek have endpoints compatible with OpenAI
# And OpenAI allows you to change the base_url

anthropic_url = "https://api.anthropic.com/v1/"
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
deepseek_url = "https://api.deepseek.com"
groq_url = "https://api.groq.com/openai/v1"
grok_url = "https://api.x.ai/v1"
openrouter_url = "https://openrouter.ai/api/v1"
ollama_url = "http://localhost:11434/v1"

anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_url)
gemini = OpenAI(api_key=google_api_key, base_url=gemini_url)
deepseek = OpenAI(api_key=deepseek_api_key, base_url=deepseek_url)
groq = OpenAI(api_key=groq_api_key, base_url=groq_url)
grok = OpenAI(api_key=grok_api_key, base_url=grok_url)
openrouter = OpenAI(base_url=openrouter_url, api_key=openrouter_api_key)
ollama = OpenAI(api_key="ollama", base_url=ollama_url)
```

**setup** — Week 2 — Special Extra!!

```python
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if OPENROUTER_API_KEY and OPENROUTER_API_KEY.startswith("sk-or-"):
    print("OPENROUTER_API_KEY looks good so far")
else:
    print("OPENROUTER_API_KEY doesn't seem right")
```

**api_call** — Week 2 Day 1 — Welcome to Week 2!

```python
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
grok_api_key = os.getenv('GROK_API_KEY')
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set (and this is optional)")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:2]}")
else:
    print("Google API Key not set (and this is optional)")

if deepseek_api_key:
    print(f"DeepSeek API Key exists and begins {deepseek_api_key[:3]}")
else:
    print("DeepSeek API Key not set (and this is optional)")

if groq_api_key:
# ... truncated
```

**api_call** — Week 2 — Special Extra!!

```python
openrouter = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
```


## Business Applications

- **Week 2 Day 1 — Welcome to Week 2!**: Business relevance This structure of a conversation, as a list of messages, is fundamental to the way we build conversational AI assistants and how they are able to keep the context during a conversation. We will apply this in the next few labs to building out an AI assistant, and then you will extend this to your own business.
- **Week 2 Day 4 — Project - Airline AI Assistant**: Business Applications Hopefully this hardly needs to be stated! You now have the ability to give actions to your LLMs. This Airline Assistant can now do more than answer questions - it could interact with booking APIs to make bookings!


## Related Topics

- [LLM APIs & Integration](../01-foundations/llm-apis.md)
- [Prompt Engineering](../02-core-concepts/prompt-engineering.md)
- [Model Selection & Comparison](../02-core-concepts/model-selection.md)
- [Tokenization & Context](../02-core-concepts/tokenization.md)
- [Embeddings](../04-rag/embeddings.md)
- [GPU Computing](../07-gpu/gpu-computing.md)

## Week References

- [Week 2](../../week-summaries/week2.md)
- [Week 3](../../week-summaries/week3.md)

