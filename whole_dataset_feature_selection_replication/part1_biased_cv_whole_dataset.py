"""
Part (1) – Variance-biased feature selection: using the WHOLE dataset (biased CV)

Paper: "On the cross-validation bias due to unsupervised pre-processing"
       Amit Moscovich & Saharon Rosset
       Journal of the Royal Statistical Society Series B, 84(4):1474-1514, 2022
       https://doi.org/10.1111/rssb.12537

Original code reference:
    https://github.com/mosco/unsupervised-preprocessing/blob/master/variable_selected_linear_regression_realdata.py

Research question
-----------------
When the top-M features are selected by empirical variance on the *whole* working
dataset (train + validation combined), does the cross-validated error accurately
estimate the true generalization error?

Key result (reproduces Figure 2 of the paper)
---------------------------------------------
    e_val  <  e_gen   (CV error is systematically overoptimistic)

The bias occurs because the feature selector has already "seen" the validation
samples when computing variances, so the CV error is artificially low.

Dataset
-------
UCI Superconductivity dataset (Hamidieh, 2018)
21,263 samples · 81 numeric features · target = critical temperature (K)
Place train.csv in the same directory as this script before running.

Outputs
-------
results/part1_whole_dataset_results.csv   – all numerical results
figures/part1_whole_dataset_M10.png       – plot for M = 10 selected features
figures/part1_whole_dataset_M30.png       – plot for M = 30 selected features
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'results'
FIGURES_DIR = SCRIPT_DIR / 'figures'
DATA_PATH   = SCRIPT_DIR / 'train.csv'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

# ── data helpers ──────────────────────────────────────────────────────────────

def load_superconductivity(path: Path):
    """Load superconductivity dataset; return (X, Y) as float arrays."""
    df  = pd.read_csv(path)
    arr = df.to_numpy(dtype=float)
    X   = arr[:, :-1]   # 81 features
    Y   = arr[:, -1]    # critical_temp (target)
    return X, Y


def normalize_by_std(X: np.ndarray) -> np.ndarray:
    """Normalize each feature column by its standard deviation."""
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0   # guard against constant columns
    return X / stds


# ── top-M variance feature selector ──────────────────────────────────────────
# Mirrors TopKVarianceVariableSelector in variable_selected_linear_regression.py

def top_k_variance_indices(X: np.ndarray, M: int) -> np.ndarray:
    """Return indices of the M columns with the largest empirical variance."""
    return np.argsort(X.var(axis=0))[-M:]


# ── single-trial simulations ──────────────────────────────────────────────────

def trial_K2(X, Y, n, M, n_holdout, rng) -> tuple | None:
    """
    One Monte-Carlo trial — K2-fold CV (m = n) with BIASED feature selection.

    Protocol (mirrors simulate_validation_vs_holdout_mse in simulations_framework.py)
    ---------------------------------------------------------------------------------
    1. Draw 2n working + n_holdout samples without replacement.
    2. Select top-M features by variance on ALL 2n working samples  ← BIASED
       (selector sees both the train fold and the validation fold).
    3. 2-fold CV:
         Fold A – train on first n, validate on second n  → err_a
         Fold B – train on second n, validate on first n  → err_b
    4. e_val = mean(err_a, err_b)
    5. e_gen = train on first n, evaluate on holdout (true out-of-sample MSE).

    Returns (e_val, e_gen) or None if dataset has too few rows.
    """
    if 2 * n + n_holdout > len(Y):
        return None

    idx       = rng.choice(len(Y), 2 * n + n_holdout, replace=False)
    work_idx  = idx[:2 * n]
    hold_idx  = idx[2 * n:]

    X_w, Y_w  = X[work_idx], Y[work_idx]
    X_h, Y_h  = X[hold_idx], Y[hold_idx]

    # BIASED: fit selector on full working set (train + val combined)
    sel   = top_k_variance_indices(X_w, M)
    X_w_s = X_w[:, sel]
    X_h_s = X_h[:, sel]

    # 2-fold CV
    reg_a = LinearRegression().fit(X_w_s[:n], Y_w[:n])
    err_a = mean_squared_error(Y_w[n:],  reg_a.predict(X_w_s[n:]))
    reg_b = LinearRegression().fit(X_w_s[n:], Y_w[n:])
    err_b = mean_squared_error(Y_w[:n],  reg_b.predict(X_w_s[:n]))
    e_val = (err_a + err_b) / 2.0

    # Generalization error (train on first n, test on holdout)
    reg_g = LinearRegression().fit(X_w_s[:n], Y_w[:n])
    e_gen = mean_squared_error(Y_h, reg_g.predict(X_h_s))

    return (e_val, e_gen)


def trial_LOO(X, Y, n, M, n_holdout, rng) -> tuple | None:
    """
    One Monte-Carlo trial — LOO-CV (m = 1) with BIASED feature selection.

    Protocol
    --------
    1. Draw n working + n_holdout samples without replacement.
    2. Select top-M features on all n working samples  ← BIASED
       (each left-out validation sample was already used in feature selection).
    3. Leave-one-out over all n working samples → n squared errors.
    4. e_val = mean of n squared LOO errors.
    5. e_gen = train on all n, evaluate on holdout.

    Returns (e_val, e_gen) or None if dataset has too few rows.
    """
    if n + n_holdout > len(Y):
        return None

    idx       = rng.choice(len(Y), n + n_holdout, replace=False)
    work_idx  = idx[:n]
    hold_idx  = idx[n:]

    X_w, Y_w  = X[work_idx], Y[work_idx]
    X_h, Y_h  = X[hold_idx], Y[hold_idx]

    # BIASED: fit selector on full working set
    sel   = top_k_variance_indices(X_w, M)
    X_w_s = X_w[:, sel]
    X_h_s = X_h[:, sel]

    # LOO CV
    mask   = np.ones(n, dtype=bool)
    errors = np.empty(n)
    for i in range(n):
        mask[i]   = False
        reg       = LinearRegression().fit(X_w_s[mask], Y_w[mask])
        errors[i] = (Y_w[i] - reg.predict(X_w_s[i:i+1])[0]) ** 2
        mask[i]   = True
    e_val = errors.mean()

    # Generalization error (train on all n, test on holdout)
    reg_g = LinearRegression().fit(X_w_s, Y_w)
    e_gen = mean_squared_error(Y_h, reg_g.predict(X_h_s))

    return (e_val, e_gen)


# ── Monte-Carlo experiment ────────────────────────────────────────────────────

def run_experiment(X, Y, n_range, M, cv_type, reps, seed=RANDOM_SEED):
    """
    Run Monte-Carlo trials for each n in n_range.
    n_holdout = n (same convention as original paper).
    Returns a list of result-row dicts.
    """
    rng      = np.random.default_rng(seed)
    trial_fn = trial_K2 if cv_type == 'K2' else trial_LOO
    rows     = []

    for n in tqdm(list(n_range), desc=f'M={M} cv={cv_type}'):
        n_holdout      = n
        e_vals, e_gens = [], []

        for _ in range(reps):
            out = trial_fn(X, Y, n, M, n_holdout, rng)
            if out is not None:
                e_vals.append(out[0])
                e_gens.append(out[1])

        if e_vals:
            rows.append({
                'M':                   M,
                'cv_type':             cv_type,
                'n_validation_label':  'm=n' if cv_type == 'K2' else 'm=1',
                'n_train':             n,
                'n_validation':        n if cv_type == 'K2' else 1,
                'n_holdout':           n,
                'reps':                len(e_vals),
                'mean_validation_mse': float(np.mean(e_vals)),
                'mean_holdout_mse':    float(np.mean(e_gens)),
                'std_validation_mse':  float(np.std(e_vals)),
                'std_holdout_mse':     float(np.std(e_gens)),
            })

    return rows


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(df_results: pd.DataFrame, M: int, output_path: Path):
    """
    Plot e_val and e_gen vs n for both K2-fold and LOO.
    Mirrors plot_test_vs_validation_set in variable_selected_linear_regression_realdata.py.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    subset  = df_results[df_results['M'] == M]

    style = {'K2': '-', 'LOO': '--'}
    label = {'K2': r'$(m=n)$', 'LOO': r'$(m=1,\ \mathrm{LOO})$'}

    for cv in ['K2', 'LOO']:
        rows = subset[subset['cv_type'] == cv].sort_values('n_train')
        ax.plot(rows['n_train'], rows['mean_validation_mse'],
                color='C0', linestyle=style[cv], linewidth=1.8,
                label=fr'$e_{{\mathrm{{val}}}}$ {label[cv]}')
        ax.plot(rows['n_train'], rows['mean_holdout_mse'],
                color='C1', linestyle=style[cv], linewidth=1.8,
                label=fr'$e_{{\mathrm{{gen}}}}$ {label[cv]}')

    ax.set_xlabel('$n$  (number of training samples)', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title(
        f'Biased CV — whole-dataset feature selection  (M = {M})\n'
        r'Blue $e_{\mathrm{val}}$ < Orange $e_{\mathrm{gen}}$  →  CV is overoptimistic',
        fontsize=10,
    )
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.35)
    ns = subset['n_train'].sort_values()
    ax.set_xlim(ns.iloc[0], ns.iloc[-1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'  Saved → {output_path}')
    plt.close()


def print_table(df_results: pd.DataFrame, M: int):
    subset = df_results[df_results['M'] == M]
    k2  = subset[subset['cv_type'] == 'K2'].set_index('n_train')
    loo = subset[subset['cv_type'] == 'LOO'].set_index('n_train')
    ns  = sorted(set(k2.index) | set(loo.index))

    print(f'\nM = {M}')
    print(f'{"n":>5} │ {"K2  e_val":>12} {"K2  e_gen":>12} │ '
          f'{"LOO e_val":>12} {"LOO e_gen":>12} │ {"bias K2":>10} {"bias LOO":>10}')
    print('─' * 85)
    for n in ns:
        k  = k2.loc[n]  if n in k2.index  else None
        lo = loo.loc[n] if n in loo.index else None
        bk = (k['mean_holdout_mse']  - k['mean_validation_mse'])  if k  is not None else float('nan')
        bl = (lo['mean_holdout_mse'] - lo['mean_validation_mse']) if lo is not None else float('nan')
        print(f'{n:>5} │ '
              f'{k["mean_validation_mse"]:>12.2f} {k["mean_holdout_mse"]:>12.2f} │ '
              f'{lo["mean_validation_mse"]:>12.2f} {lo["mean_holdout_mse"]:>12.2f} │ '
              f'{bk:>10.2f} {bl:>10.2f}')


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    REPS = 200   # increase to 500 for publication-quality results

    print('Loading and normalizing superconductivity dataset …')
    X, Y = load_superconductivity(DATA_PATH)
    print(f'  X shape : {X.shape}   Y shape : {Y.shape}')
    X = normalize_by_std(X)

    all_rows = []

    # ── Experiment A : M = 10 ─────────────────────────────────────────────────
    print('\n── Experiment A : M = 10 ───────────────────────────────────────────')
    all_rows += run_experiment(X, Y, range(20, 65, 5),   M=10, cv_type='K2',  reps=REPS)
    all_rows += run_experiment(X, Y, range(20, 65, 5),   M=10, cv_type='LOO', reps=REPS)

    # ── Experiment B : M = 30 ─────────────────────────────────────────────────
    print('\n── Experiment B : M = 30 ───────────────────────────────────────────')
    all_rows += run_experiment(X, Y, range(60, 130, 10), M=30, cv_type='K2',  reps=REPS)
    all_rows += run_experiment(X, Y, range(60, 130, 10), M=30, cv_type='LOO', reps=REPS)

    # ── save CSV ──────────────────────────────────────────────────────────────
    df_results = pd.DataFrame(all_rows)
    csv_path   = RESULTS_DIR / 'part1_whole_dataset_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f'\n  Results saved → {csv_path}')

    # ── print tables ──────────────────────────────────────────────────────────
    print_table(df_results, M=10)
    print_table(df_results, M=30)

    # ── save figures ──────────────────────────────────────────────────────────
    print()
    plot_results(df_results, M=10, output_path=FIGURES_DIR / 'part1_whole_dataset_M10.png')
    plot_results(df_results, M=30, output_path=FIGURES_DIR / 'part1_whole_dataset_M30.png')

    print('\nDone.')
