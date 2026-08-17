---
title: Dataset Preparation
topics: ['dataset-prep']
difficulty: intermediate
weeks: [6]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# Dataset Preparation

## Overview

!!! abstract "Primary reference: [Training & Fine-Tuning (Weeks 6-7)](../../reference/training-and-finetuning.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

Data curation, preprocessing, stratified sampling, deduplication

Covered in week(s) 6: 8 notebook(s) with content in the repo.

## Core Concepts

### Week 6 Day 1 — "THE PRICE IS RIGHT" Capstone Project

- The Big Project begins!!
- DAY 1: Data Curation
- Load our dataset
- Sidenote

### Week 6 Day 4 — "THE PRICE IS RIGHT" Capstone Project

- DAY 4: Neural Networks and LLMs
- There is a different kind of Neural Network we could consider

### Week 6 Day 2 — "THE PRICE IS RIGHT" Capstone Project

- DAY 2: Data Pre-processing
- For this lab
- I've put exactly this logic into a Batch class
- COSTS
- Push the final dataset to the hub

### Week 6 Day 3 — "THE PRICE IS RIGHT" Capstone Project

- DAY 3: Evaluation, Baselines, Traditional ML
- Random Forest model
- Introducing XGBoost

### Week 6 Day 5 — "THE PRICE IS RIGHT" Capstone Project

- DAY 5: Fine-tuning a Frontier Model

### Week 6 — Week 6 Optional Extra - Deep Neural Network

- If you want to train this yourself
- Or just download the file deepneuralnetwork.pth here

## Implementation Patterns

**code_example** — Week 6 Day 1 — "THE PRICE IS RIGHT" Capstone Project

```python
items = []
for dataset_name in dataset_names:
    loader = ItemLoader(dataset_name)
    items.extend(loader.load())
```

**code_example** — Week 6 Day 4 — "THE PRICE IS RIGHT" Capstone Project

```python
# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X.toarray())
y_train_tensor = torch.FloatTensor(y).unsqueeze(1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_tensor, y_train_tensor, test_size=0.01, random_state=42)

# Create the loader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model
input_size = X_train_tensor.shape[1]
model = NeuralNetwork(input_size)
```

**setup** — Week 6 Day 1 — "THE PRICE IS RIGHT" Capstone Project

```python
# imports

import os
import random

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from pricer.items import Item
from pricer.parser import parse
from tqdm.notebook import tqdm

load_dotenv(override=True)
```

**code_example** — Week 6 Day 1 — "THE PRICE IS RIGHT" Capstone Project

```python
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Appliances", split="full", trust_remote_code=True)
```


## Business Applications

- **Week 6 Day 1 — "THE PRICE IS RIGHT" Capstone Project**: Business value of Data Curation Data Curation can be considered the less glamorous work of a Data Scientist. I say that's nonsense! This is where the science happens - what could be more glamorous than that?! R&D with your dataset can often have a greater impact on performance than the fashionable 'hyper-parameter optimization' that we do later. So: prepare for Quality Time with Data Quality.
- **Week 6 Day 2 — "THE PRICE IS RIGHT" Capstone Project**: Business value of Data Pre-processing / Re-writing LLMs have made it simple to do something that was considered impossible only a few years ago. This approach can be applied to almost any business vertical, and it's similar to the advanced techniques we used on Week 5.
- **Week 6 Day 3 — "THE PRICE IS RIGHT" Capstone Project**: Business applications Traditional ML isn't just useful for learning the history; it's still heavily used in industry today, particularly for tasks where there are clearly identifiable features. It's worth spending time exploring the algorithms and experimenting here. See if you can beat my numbers with Traditional ML! I ran the Random Forest for the entire 800,000 training dataset. It took about 1


## Related Topics

- [Fine-Tuning & Training](fine-tuning.md)
- [Evaluation & Metrics](evaluation.md)
- [GPU Computing](../07-gpu/gpu-computing.md)

## Week References

- [Week 6](../../week-summaries/week6.md)

