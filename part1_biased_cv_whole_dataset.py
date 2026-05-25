"""
Part (1) – Variance-biased feature selection: using the WHOLE dataset (biased CV)

Paper: "On the cross-validation bias due to unsupervised pre-processing"
       Amit Moscovich & Saharon Rosset
       Journal of the Royal Statistical Society Series B, 84(4):1474-1514, 2022
       https://doi.org/10.1111/rssb.12537

Research question
-----------------
Does selecting the top-M features by empirical variance on the *whole* working
dataset (training + validation combined) cause the cross-validated error to be
a biased (overoptimistic) estimate of the true generalization error?

Key result we aim to reproduce (Figure 2 of the paper)
-------------------------------------------------------
When feature selection sees the validation fold (biased preprocessing):
    e_val  <  e_gen
The cross-validated error is systematically *lower* than the true generalization
error, and the gap is larger for K2-fold (m=n) than for LOO (m=1).

Dataset
-------
UCI Superconductivity dataset – 21 263 samples, 81 numeric features,
target = critical temperature (critical_temp, in Kelvin).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for saving to file
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ── constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED  = 42
DATA_PATH    = os.path.join(os.path.dirname(__file__), 'train (1).csv')
OUTPUT_DIR   = os.path.dirname(__file__)

# ── data helpers ──────────────────────────────────────────────────────────────

def load_superconductivity(path: str):
    df  = pd.read_csv(path)
    arr = df.to_numpy(dtype=float)
    X   = arr[:, :-1]   # 81 features
    Y   = arr[:, -1]    # critical_temp
    return X, Y


def normalize_by_std(X: np.ndarray) -> np.ndarray:
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0   # guard against constant columns
    return X / stds


# ── top-M variance feature selector ──────────────────────────────────────────

def top_k_variance_indices(X: np.ndarray, M: int) -> np.ndarray:
    """Indices of the M columns with the largest empirical variance."""
    return np.argsort(X.var(axis=0))[-M:]


# ── single-trial simulations ──────────────────────────────────────────────────

def trial_K2(X, Y, n, M, n_holdout, rng) -> tuple | None:
    """
    One Monte-Carlo trial of K2-fold CV (m = n) with BIASED feature selection.

    Setup
    -----
    • Draw 2n working samples + n_holdout holdout samples (no overlap).
    • Select top-M features by variance computed on all 2n working samples
      (biased: the variance estimate "sees" both folds).
    • 2-fold CV: fold-1 trains on first n, validates on second n; fold-2 swaps.
    • e_val = mean CV MSE across both folds.
    • e_gen = train on first n, test on n_holdout (true out-of-sample error).

    Returns (e_val, e_gen) or None if dataset is too small.
    """
    if 2 * n + n_holdout > len(Y):
        return None

    idx        = rng.choice(len(Y), 2 * n + n_holdout, replace=False)
    work_idx   = idx[:2 * n]
    hold_idx   = idx[2 * n:]

    X_w, Y_w   = X[work_idx], Y[work_idx]
    X_h, Y_h   = X[hold_idx], Y[hold_idx]

    # ── BIASED: select features using all 2n working samples ─────────────────
    sel    = top_k_variance_indices(X_w, M)
    X_w_s  = X_w[:, sel]
    X_h_s  = X_h[:, sel]

    # ── 2-fold cross-validation ───────────────────────────────────────────────
    # fold A: train on first half, validate on second half
    reg_a  = LinearRegression().fit(X_w_s[:n], Y_w[:n])
    err_a  = mean_squared_error(Y_w[n:], reg_a.predict(X_w_s[n:]))

    # fold B: train on second half, validate on first half
    reg_b  = LinearRegression().fit(X_w_s[n:], Y_w[n:])
    err_b  = mean_squared_error(Y_w[:n], reg_b.predict(X_w_s[:n]))

    e_val  = (err_a + err_b) / 2.0

    # ── generalization error: train on first n, evaluate on holdout ───────────
    reg_g  = LinearRegression().fit(X_w_s[:n], Y_w[:n])
    e_gen  = mean_squared_error(Y_h, reg_g.predict(X_h_s))

    return (e_val, e_gen)


def trial_LOO(X, Y, n, M, n_holdout, rng) -> tuple | None:
    """
    One Monte-Carlo trial of LOO-CV (m = 1) with BIASED feature selection.

    Setup
    -----
    • Draw n working samples + n_holdout holdout samples.
    • Select top-M features on all n working samples (biased).
    • LOO: for each i, train on n-1 samples, validate on sample i.
    • e_val = mean squared LOO error across all n folds.
    • e_gen = train on all n, test on holdout.

    Returns (e_val, e_gen) or None if dataset is too small.
    """
    if n + n_holdout > len(Y):
        return None

    idx        = rng.choice(len(Y), n + n_holdout, replace=False)
    work_idx   = idx[:n]
    hold_idx   = idx[n:]

    X_w, Y_w   = X[work_idx], Y[work_idx]
    X_h, Y_h   = X[hold_idx], Y[hold_idx]

    # ── BIASED: select features using all n working samples ───────────────────
    sel    = top_k_variance_indices(X_w, M)
    X_w_s  = X_w[:, sel]
    X_h_s  = X_h[:, sel]

    # ── LOO cross-validation ──────────────────────────────────────────────────
    mask   = np.ones(n, dtype=bool)
    errors = np.empty(n)
    for i in range(n):
        mask[i]    = False
        reg        = LinearRegression().fit(X_w_s[mask], Y_w[mask])
        errors[i]  = (Y_w[i] - reg.predict(X_w_s[i : i + 1])[0]) ** 2
        mask[i]    = True

    e_val  = errors.mean()

    # ── generalization error ──────────────────────────────────────────────────
    reg_g  = LinearRegression().fit(X_w_s, Y_w)
    e_gen  = mean_squared_error(Y_h, reg_g.predict(X_h_s))

    return (e_val, e_gen)


# ── Monte-Carlo experiment ────────────────────────────────────────────────────

def run_experiment(X, Y, n_range, M, n_holdout_fn, cv_type, reps, seed=RANDOM_SEED):
    """
    Repeat trials for each n in n_range and return mean (e_val, e_gen).

    n_holdout_fn : callable n -> int  (or plain int)
    """
    rng     = np.random.default_rng(seed)
    trial   = trial_K2 if cv_type == 'K2' else trial_LOO
    results = {}

    for n in tqdm(list(n_range), desc=f'M={M} cv={cv_type}'):
        nh   = n_holdout_fn(n) if callable(n_holdout_fn) else n_holdout_fn
        vals, gens = [], []
        for _ in range(reps):
            out = trial(X, Y, n, M, nh, rng)
            if out is not None:
                vals.append(out[0])
                gens.append(out[1])
        if vals:
            results[n] = (float(np.mean(vals)), float(np.mean(gens)))

    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(results_K2, results_LOO, M, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    def _plot(results, linestyle, label_suffix):
        ns     = sorted(results)
        e_vals = [results[n][0] for n in ns]
        e_gens = [results[n][1] for n in ns]
        ax.plot(ns, e_vals, color='C0', linestyle=linestyle, linewidth=1.8,
                label=fr'$e_\mathrm{{val}}$ {label_suffix}')
        ax.plot(ns, e_gens, color='C1', linestyle=linestyle, linewidth=1.8,
                label=fr'$e_\mathrm{{gen}}$ {label_suffix}')

    _plot(results_K2, '-',  r'$(m=n)$')
    _plot(results_LOO, '--', r'$(m=1,\ \mathrm{LOO})$')

    ax.set_xlabel('$n$  (number of training samples)', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title(
        f'Biased CV (whole-dataset feature selection)  —  M = {M} selected features\n'
        r'Blue $e_\mathrm{val}$ < Orange $e_\mathrm{gen}$  →  CV is overoptimistic',
        fontsize=10
    )
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.35)
    ns_all = sorted(set(results_K2) | set(results_LOO))
    ax.set_xlim(ns_all[0], ns_all[-1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f'  Saved → {output_path}')
    plt.close()


def print_table(results_K2, results_LOO, M):
    ns = sorted(set(results_K2) | set(results_LOO))
    print(f'\nM = {M}')
    print(f'{"n":>5} │ {"K2  e_val":>12} {"K2  e_gen":>12} │ '
          f'{"LOO e_val":>12} {"LOO e_gen":>12} │ {"bias K2":>10} {"bias LOO":>10}')
    print('─' * 85)
    for n in ns:
        k2  = results_K2.get(n,  (float('nan'), float('nan')))
        loo = results_LOO.get(n, (float('nan'), float('nan')))
        bias_K2  = k2[1]  - k2[0]
        bias_LOO = loo[1] - loo[0]
        print(f'{n:>5} │ {k2[0]:>12.2f} {k2[1]:>12.2f} │ '
              f'{loo[0]:>12.2f} {loo[1]:>12.2f} │ {bias_K2:>10.2f} {bias_LOO:>10.2f}')


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── hyper-parameters ──────────────────────────────────────────────────────
    # Increase REPS to 500 for publication-quality results (takes ~10–20 min).
    REPS = 200

    print('Loading and normalizing superconductivity dataset …')
    X, Y = load_superconductivity(DATA_PATH)
    print(f'  X shape : {X.shape}   Y shape : {Y.shape}')
    X = normalize_by_std(X)

    # ── Experiment A : M = 10 ─────────────────────────────────────────────────
    print('\n── Experiment A : M = 10 ───────────────────────────────────────────')
    M_A        = 10
    n_range_A  = range(20, 65, 5)       # as in original paper

    res_K2_A  = run_experiment(X, Y, n_range_A, M_A,
                               n_holdout_fn=lambda n: n,   # holdout size = n
                               cv_type='K2', reps=REPS)
    res_LOO_A = run_experiment(X, Y, n_range_A, M_A,
                               n_holdout_fn=lambda n: n,
                               cv_type='LOO', reps=REPS)
    print_table(res_K2_A, res_LOO_A, M_A)
    plot_results(res_K2_A, res_LOO_A, M_A,
                 os.path.join(OUTPUT_DIR, 'part1_whole_dataset_M10.png'))

    # ── Experiment B : M = 30 ─────────────────────────────────────────────────
    print('\n── Experiment B : M = 30 ───────────────────────────────────────────')
    M_B        = 30
    n_range_B  = range(60, 130, 10)     # larger n needed when M is larger

    res_K2_B  = run_experiment(X, Y, n_range_B, M_B,
                               n_holdout_fn=lambda n: n,
                               cv_type='K2', reps=REPS)
    res_LOO_B = run_experiment(X, Y, n_range_B, M_B,
                               n_holdout_fn=lambda n: n,
                               cv_type='LOO', reps=REPS)
    print_table(res_K2_B, res_LOO_B, M_B)
    plot_results(res_K2_B, res_LOO_B, M_B,
                 os.path.join(OUTPUT_DIR, 'part1_whole_dataset_M30.png'))

    print('\nDone. Figures saved to the Final Project folder.')
