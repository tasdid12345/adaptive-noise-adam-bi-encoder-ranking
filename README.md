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

## Results (latest run)

On 20 Newsgroups (TF-IDF+SVD-128 features), nDCG@10 on held-out queries was 0.522 for a fixed-noise anticorrelated-perturbation Adam (a simplified stand-in for your noise-injected Adam) versus 0.524 for my validation-tuned adaptive-noise variant, a small gain within seed noise.

| Metric | Value |
|---|---|
| baseline_score | 0.5225 |
| improved_score | 0.5241 |
| baseline_score_std | 0.0004432 |
| improved_score_std | 0.001058 |
| no_noise_score | 0.5226 |
| no_noise_score_std | 0.0008448 |
| improvement | 0.001664 |
| baseline_mrr | 0.6426 |
| baseline_recall_at_10 | 0.03388 |
| improved_mrr | 0.6466 |
| improved_recall_at_10 | 0.03398 |
| no_noise_mrr | 0.6421 |
| no_noise_recall_at_10 | 0.0339 |
| tuned_noise_scale_s | 21.96 |
| best_val_ndcg | 0.5275 |
| fixed_sigma | 0.001 |
| n_seeds | 3 |

Run on 2026-10-12. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Frozen-encoder stand-in (TF-IDF+SVD) and a linear model, not a transformer; relevance = same category. Seeds 42-44 (3 seeds), not 5. Validation set is disjoint from the 2,000 training docs and from the test pool; the noise-scale search used its own seeds (1,2) and only validation data. The baseline sigma is hand-chosen and untuned, so the improved model also receives a tuning budget the baseline does not; the training step budget is identical. Test queries are the first 1,000 docs of a 3,000-doc test pool, candidates are the pool. Differences may be within seed noise.

## Reference

This project is motivated by the following paper:

> D Vollmers, A Sharma, AC Ngonga Ngomo (2026). *Evaluating Noisy Optimization in Finetuning LMs for Neural Ranking*. International Conference on. https://link.springer.com/chapter/10.1007/978-3-032-29532-3_17

Paper author: Prof. Axel-Cyrille Ngonga Ngomo, Universität Paderborn (axel.ngonga@uni-paderborn.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
