# Tuned adaptive noise vs fixed anticorrelated noise in Adam for a bi-encoder ranker

## Motivation
The paper "Evaluating Noisy Optimization in Finetuning LMs for Neural Ranking" (2026) studies injecting noise into model parameters during Adam fine-tuning of neural rankers, including anticorrelated noise strategies. This experiment is a small, CPU-only stand-in for that idea: it asks whether the noise level should be tuned and scaled adaptively instead of being fixed by hand.

## Question
Can a validation-tuned, adaptively scaled noise level in Adam beat a hand-chosen fixed anticorrelated-noise Adam at training a small bi-encoder ranker, measured by nDCG@10 on held-out queries?

## Dataset
20 Newsgroups from scikit-learn (`fetch_20newsgroups`), represented with TF-IDF + TruncatedSVD (128-d) as a stand-in for a frozen encoder. A document is a query; same-category documents are relevant. 2,000 training documents, a separate validation set, and a fixed held-out test pool. This is not the paper's dataset.

## Baseline vs improved
- **Baseline:** linear bi-encoder trained with an in-batch contrastive loss and Adam plus anticorrelated parameter noise (differences of consecutive Gaussian perturbations) at a hand-chosen fixed noise level. This is a simplified stand-in, not the paper's method.
- **Improved:** the same model, loss and step budget, but the noise scale is found by a small Gaussian-process Bayesian search on validation nDCG@10 and is scaled per step by the gradient norm and a decay schedule.
- Plain Adam without noise is also reported for context. Secondary metrics: MRR and recall@10. Seeds 42, 43, 44.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> D Vollmers, A Sharma, AC Ngonga Ngomo (2026). *Evaluating Noisy Optimization in Finetuning LMs for Neural Ranking*. International Conference on. https://link.springer.com/chapter/10.1007/978-3-032-29532-3_17

Paper author: Prof. Axel-Cyrille Ngonga Ngomo, Universität Paderborn (axel.ngonga@uni-paderborn.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
