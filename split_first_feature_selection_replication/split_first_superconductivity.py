"""
Corrected split-first variance feature selection for the superconductivity data.

This reproduces the real-data setup from Section 4 of:
    "On the cross-validation bias due to unsupervised pre-processing"
    Amit Moscovich and Saharon Rosset.

The important correction is that the top-M variance columns are selected inside
each training fold only. Validation and holdout arrays are then sliced with the
columns selected from that training fold.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "unsupervised-preprocessing" / "superconductivity" / "train.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"


@dataclass(frozen=True)
class TrialResult:
    validation_mse: float
    generalization_mse: float
    null_validation_mse: float
    null_generalization_mse: float


def load_superconductivity(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the paper-provided train.csv with 81 features and critical_temp."""
    arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    if arr.shape != (21263, 82):
        raise ValueError(f"Expected superconductivity data shape (21263, 82), got {arr.shape}")

    x = arr[:, :81]
    y = arr[:, 81]
    return x, y


def normalize_by_full_dataset_std(x: np.ndarray) -> np.ndarray:
    """
    Match the paper's real-data script, which normalizes by full-data std first.

    The experiment here changes only the variance-based feature-selection step:
    feature variances are estimated after splitting, on the training fold.
    """
    stds = np.std(x, axis=0)
    stds[stds == 0.0] = 1.0
    return x / stds


def top_m_variance_indices(x_train: np.ndarray, m_selected_features: int) -> np.ndarray:
    """Column indices of the M largest empirical variances in a training fold."""
    if not 1 <= m_selected_features <= x_train.shape[1]:
        raise ValueError(
            f"M must be in [1, {x_train.shape[1]}], got {m_selected_features}"
        )

    variances = np.var(x_train, axis=0)
    selected = np.argpartition(variances, -m_selected_features)[-m_selected_features:]
    return selected[np.argsort(variances[selected])[::-1]]


def fit_ols_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """Fit OLS with an intercept using numpy.linalg.lstsq and predict x_eval."""
    x_train_design = np.column_stack([np.ones(x_train.shape[0]), x_train])
    x_eval_design = np.column_stack([np.ones(x_eval.shape[0]), x_eval])
    coef, *_ = np.linalg.lstsq(x_train_design, y_train, rcond=None)
    return x_eval_design @ coef


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def selected_ols_mse(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    m_selected_features: int,
) -> float:
    selected = top_m_variance_indices(x_train, m_selected_features)
    y_pred = fit_ols_predict(x_train[:, selected], y_train, x_eval[:, selected])
    return mse(y_eval, y_pred)


def trial_k2(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    m_selected_features: int,
    n_holdout: int,
    rng: np.random.Generator,
) -> TrialResult | None:
    """
    One K2 trial with split-first feature selection.

    Draw 2n working rows and n_holdout holdout rows. For each fold, select
    features using only that fold's n training rows. For generalization, use
    the first fold's training rows and its selected columns, matching the
    validation-vs-holdout setup in the paper framework.
    """
    if 2 * n + n_holdout > len(y):
        return None

    idx = rng.choice(len(y), 2 * n + n_holdout, replace=False)
    work_idx = idx[: 2 * n]
    holdout_idx = idx[2 * n :]

    x_work, y_work = x[work_idx], y[work_idx]
    x_holdout, y_holdout = x[holdout_idx], y[holdout_idx]

    x_a, y_a = x_work[:n], y_work[:n]
    x_b, y_b = x_work[n:], y_work[n:]

    err_a = selected_ols_mse(x_a, y_a, x_b, y_b, m_selected_features)
    err_b = selected_ols_mse(x_b, y_b, x_a, y_a, m_selected_features)
    validation_mse = (err_a + err_b) / 2.0

    generalization_mse = selected_ols_mse(
        x_a, y_a, x_holdout, y_holdout, m_selected_features
    )

    mean_a = float(np.mean(y_a))
    null_validation = (mse(y_b, np.full_like(y_b, mean_a)) + mse(y_a, np.full_like(y_a, float(np.mean(y_b))))) / 2.0
    null_generalization = mse(y_holdout, np.full_like(y_holdout, mean_a))

    return TrialResult(
        validation_mse=validation_mse,
        generalization_mse=generalization_mse,
        null_validation_mse=null_validation,
        null_generalization_mse=null_generalization,
    )


def trial_loo(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    m_selected_features: int,
    n_holdout: int,
    rng: np.random.Generator,
) -> TrialResult | None:
    """
    One LOO trial with split-first feature selection.

    For every leave-one-out fold, top-M variance features are selected on the
    n-1 training rows only. For generalization, train on all n working rows and
    select features on those n rows; there is no validation row in that final fit.
    """
    if n + n_holdout > len(y):
        return None

    idx = rng.choice(len(y), n + n_holdout, replace=False)
    work_idx = idx[:n]
    holdout_idx = idx[n:]

    x_work, y_work = x[work_idx], y[work_idx]
    x_holdout, y_holdout = x[holdout_idx], y[holdout_idx]

    fold_errors = np.empty(n)
    null_fold_errors = np.empty(n)
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        mask[i] = False
        x_train, y_train = x_work[mask], y_work[mask]
        x_validation = x_work[i : i + 1]
        y_validation = y_work[i : i + 1]
        fold_errors[i] = selected_ols_mse(
            x_train, y_train, x_validation, y_validation, m_selected_features
        )
        null_fold_errors[i] = mse(y_validation, np.array([float(np.mean(y_train))]))
        mask[i] = True

    validation_mse = float(np.mean(fold_errors))
    generalization_mse = selected_ols_mse(
        x_work, y_work, x_holdout, y_holdout, m_selected_features
    )

    mean_work = float(np.mean(y_work))
    null_generalization = mse(y_holdout, np.full_like(y_holdout, mean_work))

    return TrialResult(
        validation_mse=validation_mse,
        generalization_mse=generalization_mse,
        null_validation_mse=float(np.mean(null_fold_errors)),
        null_generalization_mse=null_generalization,
    )


def run_experiment(
    x: np.ndarray,
    y: np.ndarray,
    n_values: range,
    m_selected_features: int,
    cv_type: str,
    reps: int,
    seed: int,
) -> dict[int, tuple[float, float, float, float]]:
    trial = trial_k2 if cv_type == "K2" else trial_loo
    rng = np.random.default_rng(seed)
    results: dict[int, tuple[float, float, float, float]] = {}

    for n in n_values:
        rows: list[TrialResult] = []
        for _ in range(reps):
            result = trial(
                x,
                y,
                n=n,
                m_selected_features=m_selected_features,
                n_holdout=n,
                rng=rng,
            )
            if result is not None:
                rows.append(result)

        if not rows:
            continue

        results[n] = (
            float(np.mean([r.validation_mse for r in rows])),
            float(np.mean([r.generalization_mse for r in rows])),
            float(np.mean([r.null_validation_mse for r in rows])),
            float(np.mean([r.null_generalization_mse for r in rows])),
        )

    return results


def plot_results(
    results_k2: dict[int, tuple[float, float, float, float]],
    results_loo: dict[int, tuple[float, float, float, float]],
    m_selected_features: int,
    output_path: Path,
    include_null: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    def draw(results: dict[int, tuple[float, float, float, float]], linestyle: str, suffix: str) -> None:
        n_values = sorted(results)
        validation = [results[n][0] for n in n_values]
        generalization = [results[n][1] for n in n_values]
        ax.plot(
            n_values,
            validation,
            color="C0",
            linestyle=linestyle,
            linewidth=1.8,
            label=rf"corrected $e_{{val}}$ {suffix}",
        )
        ax.plot(
            n_values,
            generalization,
            color="C1",
            linestyle=linestyle,
            linewidth=1.8,
            label=rf"corrected $e_{{gen}}$ {suffix}",
        )

    draw(results_k2, "-", "(m=n)")
    draw(results_loo, "--", "(m=1)")

    if include_null:
        n_values = sorted(results_k2)
        null_values = [(results_k2[n][2] + results_k2[n][3]) / 2.0 for n in n_values]
        ax.plot(n_values, null_values, color="black", linestyle=":", linewidth=1.2, label="null model reference")

    all_n = sorted(set(results_k2) | set(results_loo))
    ax.set_xlabel("$n$ (number of training samples)")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title(
        f"Split-first variance feature selection, M = {m_selected_features} selected features"
    )
    ax.text(
        0.01,
        0.99,
        "$M$ = selected features; $m$ = validation sample size; $n$ = training sample size",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 3},
    )
    ax.set_xlim(all_n[0], all_n[-1])
    ax.yaxis.grid(True, alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_results_csv(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    fieldnames = [
        "M_selected_features",
        "cv_type",
        "validation_sample_size",
        "n_train",
        "validation_mse",
        "generalization_mse",
        "null_validation_mse",
        "null_generalization_mse",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_result_rows(
    csv_rows: list[dict[str, object]],
    m_selected_features: int,
    cv_type: str,
    validation_sample_size: str,
    results: dict[int, tuple[float, float, float, float]],
) -> None:
    for n in sorted(results):
        validation, generalization, null_validation, null_generalization = results[n]
        csv_rows.append(
            {
                "M_selected_features": m_selected_features,
                "cv_type": cv_type,
                "validation_sample_size": validation_sample_size,
                "n_train": n,
                "validation_mse": validation,
                "generalization_mse": generalization,
                "null_validation_mse": null_validation,
                "null_generalization_mse": null_generalization,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run corrected split-first variance feature selection on superconductivity data."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to superconductivity train.csv.")
    parser.add_argument("--reps", type=int, default=200, help="Monte Carlo repetitions per n and CV type.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Base random seed.")
    parser.add_argument("--no-null", action="store_true", help="Do not draw the optional null-model reference line.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading superconductivity data from {args.data}")
    x, y = load_superconductivity(args.data)
    x = normalize_by_full_dataset_std(x)
    print(f"X shape: {x.shape}; y shape: {y.shape}")

    experiment_specs = [
        (10, range(20, 65, 5)),
        (30, range(60, 130, 10)),
    ]
    csv_rows: list[dict[str, object]] = []

    for m_selected_features, n_values in experiment_specs:
        print(
            "Running split-first corrected experiment for "
            f"M={m_selected_features} selected features"
        )
        results_k2 = run_experiment(
            x,
            y,
            n_values,
            m_selected_features,
            "K2",
            args.reps,
            args.seed + m_selected_features,
        )
        results_loo = run_experiment(
            x,
            y,
            n_values,
            m_selected_features,
            "LOO",
            args.reps,
            args.seed + 1000 + m_selected_features,
        )

        append_result_rows(csv_rows, m_selected_features, "K2", "m=n", results_k2)
        append_result_rows(csv_rows, m_selected_features, "LOO", "m=1", results_loo)

        figure_path = FIGURE_DIR / f"split_first_superconductivity_M{m_selected_features}.png"
        plot_results(
            results_k2,
            results_loo,
            m_selected_features,
            figure_path,
            include_null=not args.no_null,
        )
        print(f"Saved {figure_path}")

    results_path = RESULTS_DIR / "split_first_superconductivity_results.csv"
    write_results_csv(csv_rows, results_path)
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
