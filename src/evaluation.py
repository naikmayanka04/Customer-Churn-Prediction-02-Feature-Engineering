"""Evaluation helpers for the binary churn classifier (positive class = churn)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import CV_FOLDS, RANDOM_STATE

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def cross_validate_on_train(model, X_train, y_train) -> pd.DataFrame:
    """Stratified k-fold CV on the TRAINING set only.

    The preprocessing pipeline is re-fitted inside every fold, so validation
    folds never influence imputation, scaling, or encoding.
    Returns a table with mean and std per metric.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=None)
    rows = {
        name: {"mean": scores[f"test_{name}"].mean(), "std": scores[f"test_{name}"].std()}
        for name in CV_SCORING
    }
    return pd.DataFrame(rows).T


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def evaluate_on(model, X, y, threshold: float = 0.5) -> tuple[dict, np.ndarray, np.ndarray]:
    """Score a FITTED model. Returns (metrics, predicted_labels, churn_probabilities)."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return compute_metrics(y, pred, proba), pred, proba


def cross_validate_resampled(build_model_fn, X, y, resample_fn=None, n_splits: int = CV_FOLDS,
                              random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Stratified k-fold CV with an optional resampling step applied INSIDE each fold.

    Unlike `cross_validate_on_train`, this does not use scikit-learn's
    `cross_validate` (whose Pipeline cannot resample rows), so it is used
    specifically to evaluate a resampling strategy (e.g. random oversampling)
    without leaking duplicated rows across the fold boundary: for every fold,
    `resample_fn` sees only that fold's training rows, and the validation
    rows are always the original, un-resampled data.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        if resample_fn is not None:
            X_tr, y_tr = resample_fn(X_tr, y_tr)
        model = build_model_fn(X_tr)
        model.fit(X_tr, y_tr)
        metrics, _, _ = evaluate_on(model, X_val, y_val)
        rows.append(metrics)
    df = pd.DataFrame(rows)
    return df.agg(["mean", "std"]).T


def plot_confusion_matrix(y_true, y_pred, title="Confusion matrix", save_to: Path | None = None):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Stayed", "Churned"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def plot_roc(y_true, y_proba, title="ROC curve", save_to: Path | None = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="Logistic regression")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig
