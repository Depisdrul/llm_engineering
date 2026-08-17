---
title: Prompt Engineering
topics: ['prompt-engineering']
difficulty: intermediate
weeks: [1, 2, 3, 4, 5]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# Prompt Engineering

## Overview

!!! abstract "Primary reference: [LLM Foundations (Weeks 1-3)](../../reference/llm-foundations.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

System/user prompts, conversation history, streaming, prompt caching

Covered in week(s) 1, 2, 3, 4, 5: 20 notebook(s) with content in the repo, 5 Colab stub(s).

## Core Concepts

### Week 1 Day 5 — A full business solution

- Use a call to gpt-5-nano to read the links on a webpage, and respond in structured JSON
- Second step: make the brochure!

### Week 2 Day 1 — Welcome to Week 2!

- Frontier Model APIs
- Setting up your keys - OPTIONAL!
- Adding API keys to your .env file
- Training vs Inference time scaling
- Testing out the best models on the planet
- A spicy challenge to test the competitive spirit
- Going local
- Gemini and Anthropic Client Library

### Week 1 Day 1 — YOUR FIRST LAB

- Also, be sure to read README.md!
- Answers to the most common questions
- Your first Frontier LLM Project
- If you're new to working in "Notebooks" (also known as Labs or Jupyter Lab)
- I am here to help
- More troubleshooting
- If this is old hat!
- If necessary, install Cursor Extensions

### Week 2 Day 2 — Gradio Day!

- User Interface time!
- Adding authentication
- Forcing dark mode

### Week 2 Day 3 — Day 3 - Conversational AI - aka Chatbot!

- The job of this function

## Implementation Patterns

**function** — Week 5 Day 5 — Let's go PRO!

```python
def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    print(order)
    return [chunks[i - 1] for i in order]
```

**function** — Week 1 Day 5 — A full business solution

```python
def get_brochure_user_prompt(company_name, url):
    user_prompt = f"""
You are looking at a company called: {company_name}
Here are the contents of its landing page and other relevant pages;
use this information to build a short brochure of the company in markdown without code blocks.\n\n
"""
    user_prompt += fetch_page_and_all_relevant_links(url)
    user_prompt = user_prompt[:5_000] # Truncate if more than 5,000 characters
    return user_prompt
```

**function** — Week 1 Day 5 — A full business solution

```python
def get_links_user_prompt(url):
    user_prompt = f"""
Here is the list of links on the website {url} -
Please decide which of these are relevant web links for a brochure about the company,
respond with the full https URL in JSON format.
Do not include Terms of Service, Privacy, email links.

Links (some might be relative links):

"""
    links = fetch_website_links(url)
    user_prompt += "\n".join(links)
    return user_prompt
```

**function** — Week 5 Day 5 — Let's go PRO!

```python
def rewrite_query(question, history=None):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    if history is None:
        history = []
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention the company name unless it's a general question about the company.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content
```


## Business Applications

- **Week 1 Day 5 — A full business solution**: Business applications In this exercise we extended the Day 1 code to make multiple LLM calls, and generate a document. This is perhaps the first example of Agentic AI design patterns, as we combined multiple calls to LLMs. This will feature more in Week 2, and then we will return to Agentic AI in a big way in Week 8 when we build a fully autonomous Agent solution. Generating content in this way is
- **Week 2 Day 1 — Welcome to Week 2!**: Business relevance This structure of a conversation, as a list of messages, is fundamental to the way we build conversational AI assistants and how they are able to keep the context during a conversation. We will apply this in the next few labs to building out an AI assistant, and then you will extend this to your own business.
- **Week 2 Day 3 — Day 3 - Conversational AI - aka Chatbot!**: Business Applications Conversational Assistants are of course a hugely common use case for Gen AI, and the latest frontier models are remarkably good at nuanced conversation. And Gradio makes it easy to have a user interface. Another crucial skill we covered is how to use prompting to provide context, information and examples. Consider how you could apply an AI Assistant to your business, and make


## Related Topics

- [LLM APIs & Integration](../01-foundations/llm-apis.md)
- [Development Tools](../01-foundations/dev-tools.md)
- [Model Selection & Comparison](model-selection.md)
- [Tokenization & Context](tokenization.md)
- [Multi-Model Orchestration](../03-integration/multi-model.md)
- [RAG Systems](../04-rag/rag-systems.md)
- [Embeddings](../04-rag/embeddings.md)
- [Vector Databases](../04-rag/vector-databases.md)

## Week References

- [Week 1](../../week-summaries/week1.md)
- [Week 2](../../week-summaries/week2.md)
- [Week 3](../../week-summaries/week3.md)
- [Week 4](../../week-summaries/week4.md)
- [Week 5](../../week-summaries/week5.md)

