---
title: Model Selection & Comparison
topics: ['model-selection']
difficulty: intermediate
weeks: [2, 3]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# Model Selection & Comparison

## Overview

!!! abstract "Primary reference: [Model Selection & Benchmarks (Week 4)](../../reference/model-selection-benchmarks.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

Model capabilities, pricing, reasoning vs chat, temperature

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

### Week 2 Day 3 — Day 3 - Conversational AI - aka Chatbot!

- The job of this function

### Week 2 Day 5 — Project - Airline AI Assistant

- A bit more about what Gradio actually does
- Price alert: each time I generate an image it costs about 3 cents - don't go crazy with images!
- The 3 types of Gradio UI

### Week 2 Day 4 — Project - Airline AI Assistant

- Tools
- Getting OpenAI to use our Tool
- Exercise

### Week 2 Day 2 — Gradio Day!

- User Interface time!
- Adding authentication
- Forcing dark mode

### Week 2 — Special Extra!!

- This uses OpenRouter.ai so that we easily access the latest models
- In Week 4 we will have more scientific ways to compare models

## Implementation Patterns

**function** — Week 2 Day 5 — Project - Airline AI Assistant

```python
# Callbacks (along with the chat() function above)

def put_message_in_chatbot(message, history):
        return "", history + [{"role":"user", "content":message}]

# UI definition

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500, interactive=False)
    with gr.Row():
        audio_output = gr.Audio(autoplay=True)
    with gr.Row():
        message = gr.Textbox(label="Chat with our AI Assistant:")

# Hooking up events to callbacks

    message.submit(put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]).then(
        chat, inputs=chatbot, outputs=[chatbot, audio_output, image_output]
    )

ui.launch(inbrowser=True, auth=("ed", "bananas"))
```

> Output

```text
* Running on local URL:  http://127.0.0.1:7862
* To create a public link, set `share=True` in `launch()`.
<IPython.core.display.HTML object>
:
```

**api_call** — Week 2 Day 5 — Project - Airline AI Assistant

```python

def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

gr.ChatInterface(fn=chat, type="messages").launch()
```

> Output

```text
* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
<IPython.core.display.HTML object>
```

**api_call** — Week 2 Day 4 — Project - Airline AI Assistant

```python
def chat(message, history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

gr.ChatInterface(fn=chat, type="messages").launch()
```

> Output

```text
* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
<IPython.core.display.HTML object>
Traceback (most recent call last):
  File "c:\Users\Pazzucconibt\REPO\llm_engineering\.venv\Lib\site-packages\httpx\_transports\default.py", line 101, in map_httpcore_exceptions
    yield
  File "c:\Users\Pazzucconibt\REPO\llm_engineering\.venv\Lib\site-packages\httpx\_transports\default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\Pazzucconibt\REPO\llm_engineering\.venv\Lib\site-packages\httpcore\_sync\connection_pool.py", line 256, in handle_request
    raise exc from None
  File "c:\Users\Pazzucconibt\REPO\llm_engineering\.venv\Lib\site-packages\httpcore\_sync\
... [truncated]
```

**code_example** — Week 2 Day 1 — Welcome to Week 2!

```python
# Let's make a conversation between GPT-4.1-mini and Claude-haiku-4.5
# We're using cheap versions of models so the costs will be minimal

gpt_model = "gpt-4.1-mini"
claude_model = "claude-haiku-4-5"

gpt_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

claude_system = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."

gpt_messages = ["Hi there"]
claude_messages = ["Hi"]
```


## Business Applications

- **Week 2 Day 1 — Welcome to Week 2!**: Business relevance This structure of a conversation, as a list of messages, is fundamental to the way we build conversational AI assistants and how they are able to keep the context during a conversation. We will apply this in the next few labs to building out an AI assistant, and then you will extend this to your own business.
- **Week 2 Day 3 — Day 3 - Conversational AI - aka Chatbot!**: Business Applications Conversational Assistants are of course a hugely common use case for Gen AI, and the latest frontier models are remarkably good at nuanced conversation. And Gradio makes it easy to have a user interface. Another crucial skill we covered is how to use prompting to provide context, information and examples. Consider how you could apply an AI Assistant to your business, and make
- **Week 2 Day 4 — Project - Airline AI Assistant**: Business Applications Hopefully this hardly needs to be stated! You now have the ability to give actions to your LLMs. This Airline Assistant can now do more than answer questions - it could interact with booking APIs to make bookings!


## Related Topics

- [LLM APIs & Integration](../01-foundations/llm-apis.md)
- [Prompt Engineering](prompt-engineering.md)
- [Tokenization & Context](tokenization.md)
- [Multi-Model Orchestration](../03-integration/multi-model.md)
- [Embeddings](../04-rag/embeddings.md)
- [GPU Computing](../07-gpu/gpu-computing.md)

## Week References

- [Week 2](../../week-summaries/week2.md)
- [Week 3](../../week-summaries/week3.md)

