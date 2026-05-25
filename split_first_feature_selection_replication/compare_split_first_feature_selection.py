#!/usr/bin/env python3
"""
Minimal split-first replication of Section 4 / Figure 1 from:
    "On the cross-validation bias due to unsupervised pre-processing"

This file is intentionally close to the original repo code in:
    unsupervised-preprocessing/variable_selected_linear_regression.py
    unsupervised-preprocessing/simulations_framework.py

The only experimental change is the line that fits the variance selector:
    original:    transformation.fit(Xtrainval)
    split-first: transformation.fit(Xtrain)

All outputs stay under split_first_feature_selection_replication/.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import namedtuple
from pathlib import Path

REPLICATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPLICATION_DIR.parent
RESULTS_DIR = REPLICATION_DIR / "results"
FIGURES_DIR = REPLICATION_DIR / "figures"
RESULTS_PATH = RESULTS_DIR / "split_first_feature_selection_results.csv"

# Keep Matplotlib cache files inside the one dedicated replication folder.
os.environ.setdefault("MPLCONFIGDIR", str(REPLICATION_DIR / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import normal, standard_t
from sklearn.linear_model import LinearRegression


RANDOM_SEED = 7
NOISE_MULTIPLIER = 1

# Copied name and fields from variable_selected_linear_regression.py.
ParamsSparseLinearRegression = namedtuple(
    "ParamsSparseLinearRegression",
    "n_train n_validation n_holdout D df K_strong_columns strong_column_multiplier K noise_variance",
)


def top_k_indices(a, k):
    # Copied from variable_selected_linear_regression.py.
    assert 1 <= k <= len(a)
    return np.argpartition(a, -k)[-k:]


class TopKVarianceVariableSelector:
    # Copied from variable_selected_linear_regression.py.
    def __init__(self, K):
        self.K = K
        self._is_fitted = False

    def fit(self, X):
        self._variances = np.var(X, axis=0)
        self._selected_variables = top_k_indices(self._variances, self.K)
        # Original used np.bool, which no longer exists in recent NumPy.
        self._mask = np.zeros(len(self._variances), bool)
        self._mask[self._selected_variables] = True
        self._is_fitted = True
        return self

    def transform(self, X):
        assert self._is_fitted
        return X[:, self._selected_variables]


class DatagenSparseDesignLinReg:
    # Copied from variable_selected_linear_regression.py.
    def __init__(
        self,
        dimension,
        t_distribution_df,
        K_strong_columns,
        strong_column_multiplier,
        noise_variance,
    ):
        self.dimension = dimension
        self.t_distribution_df = t_distribution_df
        self.K_strong_columns = K_strong_columns
        self.strong_column_multiplier = strong_column_multiplier
        assert noise_variance >= 0
        self.noise_variance = noise_variance

        self.beta = normal(size=(self.dimension, 1))

    def generate(self, n):
        X = standard_t(self.t_distribution_df, size=(n, self.dimension))
        X[:, : self.K_strong_columns] *= self.strong_column_multiplier
        Y = (X @ self.beta).reshape(n)
        noise = normal(scale=self.noise_variance**0.5, size=n)

        return (X, Y + noise)


def compute_noise_variance(D, K_strong_columns, strong_column_multiplier, noise_multiplier):
    # Copied from variable_selected_linear_regression.py.
    return noise_multiplier * (
        K_strong_columns * (strong_column_multiplier**2)
        + (D - K_strong_columns)
    )


def mean_squared_error(Y_true, Y_predicted):
    # Copied from simulations_framework.py.
    return np.mean((Y_true - Y_predicted) ** 2)


def pairs_average_and_std(pairs):
    # Copied from simulations_framework.py.
    arr = np.array([pair for pair in pairs if pair is not None])
    assert arr.ndim == 2
    assert arr.shape[1] == 2
    pair_averages = np.mean(arr, axis=0)
    pair_stds = np.std(arr, axis=0)
    return np.hstack((pair_averages, pair_stds))


def split_generated_data(X, Y, n_train, n_validation):
    # Same split logic as simulations_framework.py.
    Xtrain = X[:n_train]
    Ytrain = Y[:n_train]
    Xvalidation = X[n_train : n_train + n_validation]
    Yvalidation = Y[n_train : n_train + n_validation]
    Xholdout = X[n_train + n_validation :]
    Yholdout = Y[n_train + n_validation :]
    return Xtrain, Ytrain, Xvalidation, Yvalidation, Xholdout, Yholdout


def validation_vs_holdout_mse_original(
    Xtrain,
    Ytrain,
    Xvalidation,
    Yvalidation,
    Xholdout,
    Yholdout,
    transformation,
    predictor,
):
    """
    Original paper behavior.

    This is the core of simulations_framework.simulate_validation_vs_holdout_mse,
    after data generation and splitting. The leakage line is unchanged.
    """
    Xtrainval = np.vstack((Xtrain, Xvalidation))

    # ORIGINAL PAPER / REPO LINE:
    # validation covariates are used when selecting high-variance features.
    transformation.fit(Xtrainval)

    predictor.fit(transformation.transform(Xtrain), Ytrain)
    Yvalidation_pred = predictor.predict(transformation.transform(Xvalidation))
    validation_mse = mean_squared_error(Yvalidation, Yvalidation_pred)
    Yholdout_pred = predictor.predict(transformation.transform(Xholdout))
    holdout_mse = mean_squared_error(Yholdout, Yholdout_pred)

    return (validation_mse, holdout_mse)


def validation_vs_holdout_mse_split_first(
    Xtrain,
    Ytrain,
    Xvalidation,
    Yvalidation,
    Xholdout,
    Yholdout,
    transformation,
    predictor,
):
    """
    Corrected split-first behavior.

    This is intentionally identical to validation_vs_holdout_mse_original,
    except for the single marked fit line below.
    """
    # SPLIT-FIRST CHANGE:
    # fit preprocessing on training covariates only, before validation/holdout use.
    transformation.fit(Xtrain)

    predictor.fit(transformation.transform(Xtrain), Ytrain)
    Yvalidation_pred = predictor.predict(transformation.transform(Xvalidation))
    validation_mse = mean_squared_error(Yvalidation, Yvalidation_pred)
    Yholdout_pred = predictor.predict(transformation.transform(Xholdout))
    holdout_mse = mean_squared_error(Yholdout, Yholdout_pred)

    return (validation_mse, holdout_mse)


def simulate_both_pipelines(params):
    """
    Generate one Section 4 dataset and evaluate both preprocessing choices.

    The same generated train/validation/holdout split is used for both pipelines;
    this changes only variance-selector fitting, not the data-generating setup.
    """
    gen = DatagenSparseDesignLinReg(
        params.D,
        params.df,
        params.K_strong_columns,
        params.strong_column_multiplier,
        params.noise_variance,
    )
    (X, Y) = gen.generate(params.n_train + params.n_validation + params.n_holdout)
    split = split_generated_data(X, Y, params.n_train, params.n_validation)

    original = validation_vs_holdout_mse_original(
        *split,
        transformation=TopKVarianceVariableSelector(params.K),
        predictor=LinearRegression(fit_intercept=False),
    )
    split_first = validation_vs_holdout_mse_split_first(
        *split,
        transformation=TopKVarianceVariableSelector(params.K),
        predictor=LinearRegression(fit_intercept=False),
    )
    return {"original_leaky": original, "split_first": split_first}


def run_repetitions(params, reps, seed):
    results = {"original_leaky": [], "split_first": []}
    seed_generator = np.random.default_rng(seed)

    for _ in range(reps):
        # The original framework seeds NumPy before each mapper call. We do the
        # same with ordinary NumPy only, avoiding the repo's unavailable mkl_random.
        np.random.seed(int(seed_generator.integers(0, 2**32 - 1)))
        pair = simulate_both_pipelines(params)
        results["original_leaky"].append(pair["original_leaky"])
        results["split_first"].append(pair["split_first"])

    return {
        pipeline: pairs_average_and_std(values)
        for pipeline, values in results.items()
    }


def distribution_name(df):
    if df == 4:
        return "t4"
    if df == 1000000:
        return "normal"
    return f"t{df}"


def setting_name(D):
    return "low_dimensional" if D == 100 else "high_dimensional"


def validation_label(n_train, n_validation):
    return "m=n" if n_validation == n_train else "m=1"


def filename_validation_label(n_train, n_validation):
    return "m_eq_n" if n_validation == n_train else "m_eq_1"


def build_rows(params, reps, seed):
    summaries = run_repetitions(params, reps, seed)
    rows = []
    for pipeline, summary in summaries.items():
        rows.append(
            {
                "distribution": distribution_name(params.df),
                "setting": setting_name(params.D),
                "n_train": params.n_train,
                "n_validation": params.n_validation,
                "n_holdout": params.n_holdout,
                "p": params.D,
                "M": params.K_strong_columns,
                "C": params.strong_column_multiplier,
                "K": params.K,
                "eta": NOISE_MULTIPLIER,
                "reps": reps,
                "pipeline": pipeline,
                "mean_validation_mse": summary[0],
                "mean_holdout_mse": summary[1],
                "std_validation_mse": summary[2],
                "std_holdout_mse": summary[3],
            }
        )
    return rows


def use_original_plot_style_if_possible():
    style_path = REPO_ROOT / "unsupervised-preprocessing" / "latex-paper.mplstyle"
    if style_path.exists() and shutil.which("latex"):
        plt.style.use(str(style_path))


def plot_test_vs_validation_set(df, D, df_distribution, K_strong_columns, strong_column_multiplier, K):
    """
    Plot with the same organization as variable_selected_linear_regression.py:
    one figure compares validation/generalization curves for m=n and m=1.
    """
    use_original_plot_style_if_possible()
    plt.ioff()
    plt.figure(figsize=(7.5, 4.8))
    ax = plt.axes()
    ax.yaxis.grid(True)

    labels = {
        "original_leaky": "original leaky",
        "split_first": "split-first",
    }
    colors = {
        "original_leaky": ("C0", "C1"),
        "split_first": ("C2", "C3"),
    }

    subset = df[
        (df["p"] == D)
        & (df["M"] == K_strong_columns)
        & (df["C"] == strong_column_multiplier)
        & (df["K"] == K)
        & (df["distribution"] == distribution_name(df_distribution))
    ]

    for pipeline in ["original_leaky", "split_first"]:
        for n_validation_rule, linestyle in [("m=n", "-"), ("m=1", "--")]:
            rows = subset[
                (subset["pipeline"] == pipeline)
                & (
                    subset.apply(
                        lambda row: validation_label(
                            int(row["n_train"]), int(row["n_validation"])
                        ),
                        axis=1,
                    )
                    == n_validation_rule
                )
            ].sort_values("n_train")
            if rows.empty:
                continue

            validation_color, holdout_color = colors[pipeline]
            plt.plot(
                rows["n_train"],
                rows["mean_validation_mse"],
                linestyle,
                color=validation_color,
                linewidth=1.5,
                label=f"{labels[pipeline]} validation ({n_validation_rule})",
            )
            plt.plot(
                rows["n_train"],
                rows["mean_holdout_mse"],
                linestyle,
                color=holdout_color,
                linewidth=1.5,
                label=f"{labels[pipeline]} generalization ({n_validation_rule})",
            )

    plt.xlabel("$n$")
    plt.ylabel("MSE")
    plt.xlim([subset["n_train"].min(), subset["n_train"].max()])
    plt.legend(loc="best")
    plt.tight_layout()

    output_name = (
        f"variance_filtered_linear_regression_split_first_"
        f"D{D}_df{df_distribution}_Kstrong{K_strong_columns}_"
        f"multiplier{strong_column_multiplier}_K{K}_"
        f"noisemul{NOISE_MULTIPLIER:.2f}.png"
    )
    figure_path = FIGURES_DIR / output_name.replace(".", "_").replace("_png", ".png")
    print("Saving figure to", figure_path)
    plt.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close()


def precalc(simulation_function, n_range, D, df, K_strong_columns, strong_column_multiplier, K, noise_multiplier, reps, seed):
    """
    Adapted from variable_selected_linear_regression.precalc.

    The parameter construction is the same as the original. The reducer now
    writes CSV rows instead of repo pickles, and simulation_function returns
    both original-leaky and split-first results.
    """
    noise_variance = compute_noise_variance(
        D, K_strong_columns, strong_column_multiplier, noise_multiplier
    )

    params_K2 = [
        ParamsSparseLinearRegression(
            n, n, n, D, df, K_strong_columns, strong_column_multiplier, K, noise_variance
        )
        for n in n_range
    ]

    params_LOO = [
        ParamsSparseLinearRegression(
            n, 1, n, D, df, K_strong_columns, strong_column_multiplier, K, noise_variance
        )
        for n in n_range
    ]

    rows = []
    job_index = 0
    for job_params in [params_K2, params_LOO]:
        for params in job_params:
            print(
                "Running "
                f"D={params.D}, df={params.df}, n={params.n_train}, "
                f"m={params.n_validation}, K={params.K}, reps={reps}"
            )
            rows.extend(simulation_function(params, reps, seed + 1000003 * job_index))
            job_index += 1
    return rows


def precalc_all(reps_lowdim, reps_highdim, quick, seed):
    """
    Adapted from variable_selected_linear_regression.precalc_all.

    Figure 1 parameter settings are unchanged:
      D=100,   df=4 and df=1000000, Kstrong=5,  multiplier=5,  K=10
      D=1000,  df=4 and df=1000000, Kstrong=10, multiplier=10, K=100
    """
    low_range = range(20, 65, 5)
    high_range = range(200, 650, 50)
    if quick:
        low_range = [20, 40, 60]
        high_range = [200, 400, 600]

    rows = []
    rows.extend(precalc(build_rows, low_range, 100, 4, 5, 5, 10, NOISE_MULTIPLIER, reps_lowdim, seed))
    rows.extend(precalc(build_rows, low_range, 100, 1000000, 5, 5, 10, NOISE_MULTIPLIER, reps_lowdim, seed))
    rows.extend(precalc(build_rows, high_range, 1000, 4, 10, 10, 100, NOISE_MULTIPLIER, reps_highdim, seed))
    rows.extend(precalc(build_rows, high_range, 1000, 1000000, 10, 10, 100, NOISE_MULTIPLIER, reps_highdim, seed))
    return pd.DataFrame(rows)


def plot_all(df):
    # Same four Figure 1 panels, now with original-leaky vs split-first overlays.
    plot_test_vs_validation_set(df, D=100, df_distribution=4, K_strong_columns=5, strong_column_multiplier=5, K=10)
    plot_test_vs_validation_set(df, D=100, df_distribution=1000000, K_strong_columns=5, strong_column_multiplier=5, K=10)
    plot_test_vs_validation_set(df, D=1000, df_distribution=4, K_strong_columns=10, strong_column_multiplier=10, K=100)
    plot_test_vs_validation_set(df, D=1000, df_distribution=1000000, K_strong_columns=10, strong_column_multiplier=10, K=100)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal split-first adaptation of the paper's Section 4 / Figure 1 simulation."
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="Repetitions per n and validation-size setting. Default: 1000, or 20 with --quick.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use three n values per panel and default to 20 repetitions.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    reps = args.reps if args.reps is not None else (20 if args.quick else 1000)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = precalc_all(
        reps_lowdim=reps,
        reps_highdim=reps,
        quick=args.quick,
        seed=args.seed,
    )
    df.to_csv(RESULTS_PATH, index=False)
    print("Saving results to", RESULTS_PATH)

    plot_all(df)
    print("Done.")


if __name__ == "__main__":
    main()
