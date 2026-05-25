# Split-First Feature Selection Replication

This folder contains a minimal adaptation of the original Section 4 / Figure 1 code from the paper repo.

Original files used as the template:

- `unsupervised-preprocessing/variable_selected_linear_regression.py`
- `unsupervised-preprocessing/simulations_framework.py`

The goal is to keep the paper's main example unchanged except for one preprocessing fit line.

## What Stays The Same

The script keeps the Figure 1 synthetic setup:

- `beta_j ~ N(0, 1)`
- covariates generated with the repo's original `standard_t(df)` code
- Gaussian panels use the repo convention `df=1000000`
- t-distribution panels use `df=4`
- first `M` columns are multiplied by `C`
- `sigma^2 = eta * ((p - M) + C^2 M)` with `eta=1`
- `TopKVarianceVariableSelector`
- `LinearRegression(fit_intercept=False)`
- validation MSE and holdout/generalization MSE
- Figure 1 grids:
  - `p=100, M=5, C=5, K=10, n=20..60`
  - `p=1000, M=10, C=10, K=100, n=200..600`
  - both `m=n` and `m=1`

## Original Leaky Behavior

The original paper/repo behavior computes empirical variances on the combined train and validation covariates:

```python
Xtrainval = np.vstack((Xtrain, Xvalidation))
transformation.fit(Xtrainval)
```

This matches `simulations_framework.py`, where the original code fits the transformation on `Xtrainval`.

## Corrected Split-First Behavior

The corrected behavior changes only the preprocessing fit line:

```python
transformation.fit(Xtrain)
```

The selector is then applied to training, validation, and holdout data exactly as in the original MSE logic.

## Quick Test

From the repo root:

```bash
python split_first_feature_selection_replication/compare_split_first_feature_selection.py --quick --reps 100
```

## Full Run

From the repo root:

```bash
python split_first_feature_selection_replication/compare_split_first_feature_selection.py
```

The default is 1000 repetitions per parameter setting.

## Outputs

CSV results:

```text
split_first_feature_selection_replication/results/split_first_feature_selection_results.csv
```

Figures:

```text
split_first_feature_selection_replication/figures/
```
