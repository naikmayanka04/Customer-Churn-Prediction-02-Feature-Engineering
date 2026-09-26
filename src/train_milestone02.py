"""Milestone 02 — Feature Engineering: scaling/encoding (reused from Milestone 01)
plus detecting and addressing class imbalance.

Usage (from the repository root):
    python -m src.train_milestone02

Compares three strategies via cross-validation on the TRAINING set only:
  1. Plain logistic regression (Milestone 01 baseline, no imbalance handling)
  2. Logistic regression with class_weight="balanced"
  3. Logistic regression trained on a randomly oversampled training fold

The strategy with the best training-CV F1 is then fitted once on the full
training set and evaluated once on the held-out test set.
"""
import json

import matplotlib

matplotlib.use("Agg")

from sklearn.model_selection import train_test_split

from .config import FIGURES_DIR, RANDOM_STATE, TEST_SIZE, ROOT
from .data_preparation import clean, load_raw, make_xy
from .evaluation import (
    cross_validate_on_train,
    cross_validate_resampled,
    evaluate_on,
    plot_confusion_matrix,
    plot_roc,
)
from .imbalance import class_balance, random_oversample
from .preprocessing import build_logistic_baseline, build_logistic_class_weighted

MILESTONE02_METRICS_PATH = ROOT / "reports" / "milestone02_metrics.json"


def main() -> None:
    # 1-2. Reuse Milestone 01's loading, cleaning and X/y construction
    raw = load_raw()
    df, _ = clean(raw)
    X, y = make_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 3. Confirm the imbalance
    balance = class_balance(y_train)
    print("Class balance (train):\n", balance)

    # 4. Compare strategies with cross-validation on the training set only
    cv_plain = cross_validate_on_train(build_logistic_baseline(X_train), X_train, y_train)
    cv_weighted = cross_validate_on_train(build_logistic_class_weighted(X_train), X_train, y_train)
    cv_oversampled = cross_validate_resampled(
        build_logistic_baseline, X_train, y_train, resample_fn=random_oversample
    )

    comparison = {
        "plain": cv_plain["mean"].to_dict(),
        "class_weighted": cv_weighted["mean"].to_dict(),
        "oversampled": cv_oversampled["mean"].to_dict(),
    }
    print("\nCross-validation comparison (train only, mean scores):")
    for name, scores in comparison.items():
        print(f"  {name:15s} " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))

    # 5. Pick the strategy with the best F1 on training CV
    best_name = max(comparison, key=lambda k: comparison[k]["f1"])
    print(f"\nSelected strategy: {best_name}")

    builders = {
        "plain": (build_logistic_baseline, None),
        "class_weighted": (build_logistic_class_weighted, None),
        "oversampled": (build_logistic_baseline, random_oversample),
    }
    build_fn, resample_fn = builders[best_name]

    X_fit, y_fit = (resample_fn(X_train, y_train) if resample_fn else (X_train, y_train))
    final_model = build_fn(X_fit)
    final_model.fit(X_fit, y_fit)

    # 6. Evaluate once on the held-out test set
    test_metrics, y_pred, y_proba = evaluate_on(final_model, X_test, y_test)
    print("\nTest metrics (selected strategy):", {k: round(v, 4) for k, v in test_metrics.items()})

    results = {
        "random_state": RANDOM_STATE,
        "class_balance_train": balance["count"].to_dict(),
        "cv_comparison_train": comparison,
        "selected_strategy": best_name,
        "test_metrics_selected_strategy": {k: round(float(v), 4) for k, v in test_metrics.items()},
    }
    MILESTONE02_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MILESTONE02_METRICS_PATH.write_text(json.dumps(results, indent=2))

    plot_confusion_matrix(
        y_test, y_pred, f"Milestone 02 ({best_name}): confusion matrix (test)",
        FIGURES_DIR / "milestone02_confusion_matrix.png",
    )
    plot_roc(
        y_test, y_proba, f"Milestone 02 ({best_name}): ROC curve (test)",
        FIGURES_DIR / "milestone02_roc_curve.png",
    )
    print(f"\nSaved metrics to {MILESTONE02_METRICS_PATH}")


if __name__ == "__main__":
    main()
