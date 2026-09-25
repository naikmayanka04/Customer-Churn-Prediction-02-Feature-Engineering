"""Class-imbalance analysis and resampling helpers.

`imbalanced-learn` (SMOTE etc.) is not used here so the project has no extra
dependency beyond scikit-learn; random oversampling with replacement is a
simple, well-understood alternative and is explained as such in the notebook.

IMPORTANT: any resampling must be applied to TRAINING data only, after the
train/test split (and, in cross-validation, only to each training fold).
Resampling before the split, or resampling the whole training set before CV
splits it further, would duplicate the same row into both the training and
validation/test data and leak information.
"""
from __future__ import annotations

import pandas as pd


def class_balance(y: pd.Series) -> pd.DataFrame:
    """Counts and share of each class label."""
    counts = y.value_counts().sort_index()
    share = y.value_counts(normalize=True).sort_index()
    return pd.DataFrame({"count": counts, "share": share.round(3)})


def random_oversample(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Duplicate minority-class rows (sampling with replacement) until classes are balanced.

    Must only ever be called on a TRAINING split/fold, never on the full
    dataset or on validation/test data.
    """
    counts = y.value_counts()
    majority_label, majority_n = counts.idxmax(), counts.max()

    X_parts, y_parts = [X], [y]
    for label, n in counts.items():
        if label == majority_label:
            continue
        deficit = majority_n - n
        if deficit <= 0:
            continue
        idx = y[y == label].index
        extra_idx = pd.Series(idx).sample(n=deficit, replace=True, random_state=random_state)
        X_parts.append(X.loc[extra_idx])
        y_parts.append(y.loc[extra_idx])

    combined = pd.concat(X_parts, axis=0)
    combined_y = pd.concat(y_parts, axis=0)
    order = combined.sample(frac=1, random_state=random_state).index
    return combined.loc[order].reset_index(drop=True), combined_y.loc[order].reset_index(drop=True)
