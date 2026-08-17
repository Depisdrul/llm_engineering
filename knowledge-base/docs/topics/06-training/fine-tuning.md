---
title: Fine-Tuning & Training
topics: ['fine-tuning']
difficulty: intermediate
weeks: [6, 7]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# Fine-Tuning & Training

## Overview

!!! abstract "Primary reference: [Training & Fine-Tuning (Weeks 6-7)](../../reference/training-and-finetuning.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

QLoRA, PEFT, LoRA, dataset preparation, training loops

Covered in week(s) 6, 7: 10 notebook(s) with content in the repo, 3 Colab stub(s).

## Core Concepts

### Week 6 Day 3 — "THE PRICE IS RIGHT" Capstone Project

- DAY 3: Evaluation, Baselines, Traditional ML
- Random Forest model
- Introducing XGBoost

### Week 6 Day 1 — "THE PRICE IS RIGHT" Capstone Project

- The Big Project begins!!
- DAY 1: Data Curation
- Load our dataset
- Sidenote

### Week 6 Day 4 — "THE PRICE IS RIGHT" Capstone Project

- DAY 4: Neural Networks and LLMs
- There is a different kind of Neural Network we could consider

### Week 6 Day 5 — "THE PRICE IS RIGHT" Capstone Project

- DAY 5: Fine-tuning a Frontier Model

### Week 6 — Week 6 Optional Extra - Deep Neural Network

- If you want to train this yourself
- Or just download the file deepneuralnetwork.pth here

## Implementation Patterns

**function** — Week 6 Day 3 — "THE PRICE IS RIGHT" Capstone Project

```python
# That was fun!
# We can do better - here's another rather trivial model

training_prices = [item.price for item in train]
training_average = sum(training_prices) / len(training_prices)
print(training_average)

def constant_pricer(item):
    return training_average
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

**code_example** — Week 6 Day 3 — "THE PRICE IS RIGHT" Capstone Project

```python
username = "ed-donner"
dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"

train, val, test = Item.from_hub(dataset)

print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")
```

**code_example** — Week 6 Day 4 — "THE PRICE IS RIGHT" Capstone Project

```python
username = "ed-donner"
dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"

train, val, test = Item.from_hub(dataset)

print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")
```


## Business Applications

- **Week 6 Day 3 — "THE PRICE IS RIGHT" Capstone Project**: Business applications Traditional ML isn't just useful for learning the history; it's still heavily used in industry today, particularly for tasks where there are clearly identifiable features. It's worth spending time exploring the algorithms and experimenting here. See if you can beat my numbers with Traditional ML! I ran the Random Forest for the entire 800,000 training dataset. It took about 1
- **Week 6 Day 1 — "THE PRICE IS RIGHT" Capstone Project**: Business value of Data Curation Data Curation can be considered the less glamorous work of a Data Scientist. I say that's nonsense! This is where the science happens - what could be more glamorous than that?! R&D with your dataset can often have a greater impact on performance than the fashionable 'hyper-parameter optimization' that we do later. So: prepare for Quality Time with Data Quality.


## Related Topics

- [Dataset Preparation](dataset-prep.md)
- [Evaluation & Metrics](evaluation.md)
- [GPU Computing](../07-gpu/gpu-computing.md)

## Week References

- [Week 6](../../week-summaries/week6.md)
- [Week 7](../../week-summaries/week7.md)

