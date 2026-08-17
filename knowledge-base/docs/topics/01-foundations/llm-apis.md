---
title: LLM APIs & Integration
topics: ['llm-apis']
difficulty: intermediate
weeks: [1, 2]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# LLM APIs & Integration

## Overview

!!! abstract "Primary reference: [API Keys & Runnability](../../reference/api-keys-and-runnability.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

OpenAI, Anthropic, Gemini API patterns, authentication, error handling

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

### Week 2 Day 2 — Gradio Day!

- User Interface time!
- Adding authentication
- Forcing dark mode

### Week 2 Day 4 — Project - Airline AI Assistant

- Tools
- Getting OpenAI to use our Tool
- Exercise

### Week 1 Day 2 — Welcome to the Day 2 Lab!

- We will start by calling OpenAI again - but don't worry non-OpenAI people, your time is coming!
- Do you know what an Endpoint is?
- THIS IS OPTIONAL - but if you wish to try out Google Gemini, please visit
- Download llama3.2 from meta
- Recap on installation of Ollama

### Week 2 Day 5 — Project - Airline AI Assistant

- A bit more about what Gradio actually does
- Price alert: each time I generate an image it costs about 4 cents - don't go crazy with images!
- The 3 types of Gradio UI

### Week 2 Day 3 — Day 3 - Conversational AI - aka Chatbot!

- The job of this function

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

**api_call** — Week 2 Day 2 — Gradio Day!

```python
# Connect to OpenAI, Anthropic and Google; comment out the Claude or Google lines if you're not using them

# openai = OpenAI()

# anthropic_url = "https://api.anthropic.com/v1/"
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

# anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_url)
gemini = OpenAI(api_key=google_api_key, base_url=gemini_url)
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

**api_call** — Week 2 Day 2 — Gradio Day!

```python
# Load environment variables in a file called .env
# Print the key prefixes to help with any debugging
# You can choose whichever providers you like - or all Ollama

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")
```

> Output

```text
OpenAI API Key not set
Anthropic API Key not set
Google API Key exists and begins AIzaSyBH
```


## Business Applications

- **Week 2 Day 1 — Welcome to Week 2!**: Business relevance This structure of a conversation, as a list of messages, is fundamental to the way we build conversational AI assistants and how they are able to keep the context during a conversation. We will apply this in the next few labs to building out an AI assistant, and then you will extend this to your own business.
- **Week 2 Day 4 — Project - Airline AI Assistant**: Business Applications Hopefully this hardly needs to be stated! You now have the ability to give actions to your LLMs. This Airline Assistant can now do more than answer questions - it could interact with booking APIs to make bookings!
- **Week 2 Day 3 — Day 3 - Conversational AI - aka Chatbot!**: Business Applications Conversational Assistants are of course a hugely common use case for Gen AI, and the latest frontier models are remarkably good at nuanced conversation. And Gradio makes it easy to have a user interface. Another crucial skill we covered is how to use prompting to provide context, information and examples. Consider how you could apply an AI Assistant to your business, and make


## Related Topics

- [Prompt Engineering](../02-core-concepts/prompt-engineering.md)
- [Development Tools](dev-tools.md)
- [Model Selection & Comparison](../02-core-concepts/model-selection.md)
- [Tokenization & Context](../02-core-concepts/tokenization.md)
- [Multi-Model Orchestration](../03-integration/multi-model.md)
- [Embeddings](../04-rag/embeddings.md)

## Week References

- [Week 1](../../week-summaries/week1.md)
- [Week 2](../../week-summaries/week2.md)

