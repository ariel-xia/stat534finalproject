# Split-first variance feature selection replication

This folder contains a corrected version of the superconductivity experiment from
Section 4 of the paper's code.

The original biased pipeline selects the top-variance features on the full
training-plus-validation sample before validation. This implementation splits
first:

1. draw the same superconductivity working and holdout samples,
2. compute feature variances only on the current training fold,
3. select the top `M` features from that training fold,
4. apply those same selected columns to the validation or holdout data,
5. fit ordinary least squares and record validation and generalization MSE.

The script produces exactly two final figures by default:

- `figures/split_first_superconductivity_M10.png`
- `figures/split_first_superconductivity_M30.png`

It also writes a tidy CSV of the plotted values to
`results/split_first_superconductivity_results.csv`.

## Run

Use a Python environment with `numpy` and `matplotlib` installed. The paper repo
README recommends Anaconda.

```bash
python3 split_first_feature_selection_replication/split_first_superconductivity.py
```

Use `--reps` for a quicker or more stable run. The default is `200`, matching
the local biased replication script.

```bash
python3 split_first_feature_selection_replication/split_first_superconductivity.py --reps 50
```

The default data path is the paper-provided dataset:

```text
unsupervised-preprocessing/superconductivity/train.csv
```

## Paper notation used in the figures

- `M`: number of selected features, fixed at `10` or `30`.
- `n`: training sample size, shown on the x-axis.
- `m`: validation sample size. Solid lines use `m=n`; dashed lines use `m=1`.
