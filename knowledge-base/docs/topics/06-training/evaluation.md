---
title: Evaluation & Metrics
topics: ['evaluation']
difficulty: intermediate
weeks: [6, 7]
last_updated: 2026-08-17
source: LLM Engineering Course - Ed Donner
course_url: https://edwarddonner.com/
---

# Evaluation & Metrics

## Overview

!!! abstract "Primary reference: [Training & Fine-Tuning (Weeks 6-7)](../../reference/training-and-finetuning.md)"

    That page is written from primary sources and supersedes anything below it.
    This page maps the concept back to the notebooks that demonstrate it.

Model evaluation, baselines, metrics, benchmarking

Covered in week(s) 6, 7: 10 notebook(s) with content in the repo, 3 Colab stub(s).

## Core Concepts

### Week 6 Day 3 — "THE PRICE IS RIGHT" Capstone Project

- DAY 3: Evaluation, Baselines, Traditional ML
- Random Forest model
- Introducing XGBoost

## Implementation Patterns

**imports** — Week 6 Day 3 — "THE PRICE IS RIGHT" Capstone Project

```python
import random

import numpy as np
import pandas as pd
from pricer.evaluator import evaluate
from pricer.items import Item
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```


## Business Applications

- **Week 6 Day 3 — "THE PRICE IS RIGHT" Capstone Project**: Business applications Traditional ML isn't just useful for learning the history; it's still heavily used in industry today, particularly for tasks where there are clearly identifiable features. It's worth spending time exploring the algorithms and experimenting here. See if you can beat my numbers with Traditional ML! I ran the Random Forest for the entire 800,000 training dataset. It took about 1


## Related Topics

- [Fine-Tuning & Training](fine-tuning.md)
- [Dataset Preparation](dataset-prep.md)
- [GPU Computing](../07-gpu/gpu-computing.md)

## Week References

- [Week 6](../../week-summaries/week6.md)
- [Week 7](../../week-summaries/week7.md)

