---
title: GPU Computing
topics: ['gpu-computing']
difficulty: intermediate
weeks: [3, 6, 7]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# GPU Computing

## Overview

!!! abstract "Primary reference: [Training & Fine-Tuning (Weeks 6-7)](../../reference/training-and-finetuning.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

CUDA, model loading, inference optimization, Colab GPU

Covered in week(s) 3, 6, 7: 10 notebook(s) with content in the repo, 8 Colab stub(s).

## Core Concepts

### Week 6 Day 5 — "THE PRICE IS RIGHT" Capstone Project

- DAY 5: Fine-tuning a Frontier Model

### Week 6 — Week 6 Optional Extra - Deep Neural Network

- If you want to train this yourself
- Or just download the file deepneuralnetwork.pth here

## Implementation Patterns

**api_call** — Week 6 Day 5 — "THE PRICE IS RIGHT" Capstone Project

```python
# The inference function


def gpt_4__1_nano_fine_tuned(item):
    response = openai.chat.completions.create(
        model=fine_tuned_model_name,
        messages=test_messages_for(item),
        max_tokens=7
    )
    return response.choices[0].message.content
```

**function** — Week 6 — Week 6 Optional Extra - Deep Neural Network

```python
def deep_neural_network(item):
    return runner.inference(item)

evaluate(deep_neural_network, test)
```

**function** — Week 6 — Week 6 Optional Extra - Deep Neural Network

```python
def deep_neural_network(item):
    return runner.inference(item)

evaluate(deep_neural_network, test)
```




## Related Topics

- [Prompt Engineering](../02-core-concepts/prompt-engineering.md)
- [Model Selection & Comparison](../02-core-concepts/model-selection.md)
- [Multi-Model Orchestration](../03-integration/multi-model.md)
- [Fine-Tuning & Training](../06-training/fine-tuning.md)
- [Dataset Preparation](../06-training/dataset-prep.md)
- [Evaluation & Metrics](../06-training/evaluation.md)

## Week References

- [Week 3](../../week-summaries/week3.md)
- [Week 6](../../week-summaries/week6.md)
- [Week 7](../../week-summaries/week7.md)

