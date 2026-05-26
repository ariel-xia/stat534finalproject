# Part 1 — Variance-biased feature selection: whole dataset (biased CV)

**Paper:** "On the cross-validation bias due to unsupervised pre-processing"  
Moscovich & Rosset, JRSS-B 2022  
**Original code reference:** [`variable_selected_linear_regression_realdata.py`](https://github.com/mosco/unsupervised-preprocessing/blob/master/variable_selected_linear_regression_realdata.py)

---

## What this experiment does

Reproduces the **"Use whole dataset"** condition of the variance-biased feature
selection experiment (Section 4.2, Figure 2 of the paper) on the real
Superconductivity dataset.

**Key question:** When the top-M features are selected by empirical variance
computed on the *entire* working dataset (train + validation combined), does
the cross-validated error accurately estimate the true generalization error?

**Answer (confirmed by simulation):**  
No — `e_val < e_gen` consistently.  
The CV error is overoptimistic because the feature selector has already "seen"
the validation samples before any cross-validation fold is evaluated.

---

## File structure

```
whole_dataset_feature_selection_replication/
├── part1_biased_cv_whole_dataset.py   ← main simulation script
├── README.md
├── figures/
│   ├── part1_whole_dataset_M10.png    ← results for M = 10 selected features
│   └── part1_whole_dataset_M30.png    ← results for M = 30 selected features
└── results/
    └── part1_whole_dataset_results.csv ← all numerical results
```

---

## How to run

```bash
# Place train.csv (superconductivity dataset) in this folder, then:
python part1_biased_cv_whole_dataset.py
```

Dataset: UCI Superconductivity dataset (Hamidieh, 2018)  
21,263 samples · 81 features · target = critical temperature (K)

---

## CV schemes compared

| Label | Description | Validation size m |
|---|---|---|
| Solid line | K2-fold CV | m = n (half/half split) |
| Dashed line | LOO-CV | m = 1 (leave-one-out) |

Both schemes use the **biased** feature selector — fitted on the full working
set (train + val combined) before any fold is evaluated.

---

## Key results

- Blue `e_val` is consistently **below** orange `e_gen` → CV is overoptimistic
- Bias is largest at small n and shrinks as n grows
- LOO shows larger bias than K2-fold at small n (each left-out sample was
  fully included in feature selection, so the leak is stronger)
- M = 10 shows larger bias than M = 30 (stronger selection pressure)
