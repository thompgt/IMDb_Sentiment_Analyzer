"""Train, select and evaluate the IMDb sentiment model.

Methodology, and why it differs from the original notebook:

* **De-duplicate before splitting.** Identical reviews straddling the split
  leak test data into training.
* **Stratify the split.** On a subsample the class ratio otherwise drifts.
* **Select on cross-validation over the *training* set only.** The original
  compared Naive Bayes against Logistic Regression on the test set and read the
  winner's score off that same set, which is selection bias -- the number is
  optimistic by construction. Here the test set is touched exactly once, at the
  very end, by the single model that CV already picked.
* **Print a majority-class baseline.** "88% accuracy" means nothing without
  knowing that guessing the majority class scores ~50%.

Run ``python -m sentiment.train --help`` for options.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from sentiment.data import CLASS_NAMES, load_reviews
from sentiment.model import (
    DEFAULT_MODEL_PATH,
    ModelBundle,
    build_pipeline,
    save_bundle,
    top_coefficients,
)

#: Candidate models compared by cross-validation on the training split only.
SEARCH_SPACE: list[dict[str, list[Any]]] = [
    {
        "clf": [build_pipeline("logreg").named_steps["clf"]],
        "clf__C": [0.5, 1.0, 4.0],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2],
    },
    {
        "clf": [build_pipeline("nb").named_steps["clf"]],
        "clf__alpha": [0.1, 0.5, 1.0],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2],
    },
]


def _describe(estimator: Any) -> str:
    return type(estimator).__name__


def error_analysis(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba_pos: np.ndarray,
    n: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Most confidently wrong predictions in each direction.

    This is the highest-value missing section in the original notebook: a
    confusion matrix tells you *how many* you got wrong, these tell you *why*.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    def _collect(mask: np.ndarray, confidence: np.ndarray) -> list[dict[str, Any]]:
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(-confidence[idx])][:n]
        return [
            {
                "confidence": float(confidence[i]),
                "true": CLASS_NAMES[int(y_true[i])],
                "predicted": CLASS_NAMES[int(y_pred[i])],
                "review": texts[i][:400],
            }
            for i in idx
        ]

    return {
        # predicted positive, actually negative -> confident in proba_pos
        "false_positives": _collect((y_pred == 1) & (y_true == 0), proba_pos),
        # predicted negative, actually positive -> confident in 1 - proba_pos
        "false_negatives": _collect((y_pred == 0) & (y_true == 1), 1.0 - proba_pos),
    }


def train(
    *,
    csv_path: str | None = None,
    sample_size: int | None = None,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
    model_path: Path | None = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> ModelBundle:
    started = time.perf_counter()
    log = print if verbose else (lambda *a, **k: None)

    df = load_reviews(csv_path, sample_size=sample_size, random_state=random_state)
    log(f"Loaded {len(df):,} unique reviews "
        f"({df['label'].mean():.1%} positive) from {csv_path or 'data/imdb_reviews.csv'}")

    X = df["review"].tolist()
    y = df["label"].to_numpy()

    # --- Split once. The test half is quarantined until the final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    log(f"Train {len(X_train):,} / test {len(X_test):,} (stratified)")

    # --- Baseline: what does guessing the majority class get you?
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_acc = float(accuracy_score(y_test, dummy.predict(X_test)))
    log(f"Majority-class baseline accuracy: {baseline_acc:.4f}")

    # --- Model selection: cross-validated, training data only.
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        build_pipeline("logreg"),
        param_grid=SEARCH_SPACE,
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=1 if verbose else 0,
    )
    log(f"Grid-searching {cv_folds}-fold CV on the training split...")
    search.fit(X_train, y_train)

    best = search.best_estimator_
    log(f"Best CV ROC-AUC {search.best_score_:.4f} with "
        f"{_describe(best.named_steps['clf'])} {search.best_params_}")

    cv_ranking = [
        {
            "classifier": _describe(p.get("clf", best.named_steps["clf"])),
            "params": {k: str(v) for k, v in p.items() if k != "clf"},
            "mean_cv_roc_auc": float(m),
        }
        for p, m in zip(
            search.cv_results_["params"], search.cv_results_["mean_test_score"]
        )
    ]
    cv_ranking.sort(key=lambda r: -r["mean_cv_roc_auc"])

    # --- Single, final touch of the test set.
    y_pred = best.predict(X_test)
    proba_pos = best.predict_proba(X_test)[:, 1]
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, proba_pos)),
        "baseline_accuracy": baseline_acc,
        "best_cv_roc_auc": float(search.best_score_),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, output_dict=True
        ),
        "cv_ranking": cv_ranking[:10],
    }

    log("\n=== Held-out test set (evaluated once) ===")
    log(f"Accuracy {metrics['test_accuracy']:.4f}  "
        f"F1 {metrics['test_f1']:.4f}  ROC-AUC {metrics['test_roc_auc']:.4f}  "
        f"(baseline {baseline_acc:.4f})")
    log(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    coefs = top_coefficients(best)
    if coefs["positive"]:
        log("Top positive features: " + ", ".join(w for w, _ in coefs["positive"][:10]))
        log("Top negative features: " + ", ".join(w for w, _ in coefs["negative"][:10]))

    errors = error_analysis(X_test, y_test, y_pred, proba_pos)
    log(f"\nMost confident mistakes: {len(errors['false_positives'])} false positives, "
        f"{len(errors['false_negatives'])} false negatives (see model metadata)")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_reviews": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": float(df["label"].mean()),
        "random_state": random_state,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "classifier": _describe(best.named_steps["clf"]),
        "class_names": CLASS_NAMES,
        "top_features": coefs,
        "error_analysis": errors,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "train_seconds": round(time.perf_counter() - started, 1),
    }

    bundle = ModelBundle(pipeline=best, metrics=metrics, metadata=metadata)
    saved = save_bundle(bundle, model_path)
    log(f"\nSaved model bundle to {saved} ({saved.stat().st_size / 1e6:.1f} MB)")

    report_path = saved.with_suffix(".metrics.json")
    report_path.write_text(bundle.as_json(), encoding="utf-8")
    log(f"Saved metrics report to {report_path}")

    return bundle


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sentiment.train",
        description="Train and persist the IMDb TF-IDF sentiment model.",
    )
    p.add_argument("--csv", dest="csv_path", default=None,
                   help="review CSV (default: data/imdb_reviews.csv or $IMDB_CSV_PATH)")
    p.add_argument("--sample-size", type=int, default=None,
                   help="stratified subsample for a fast run (default: full corpus)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--model-path", type=Path, default=None,
                   help=f"output .joblib bundle (default: {DEFAULT_MODEL_PATH})")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        bundle = train(
            csv_path=args.csv_path,
            sample_size=args.sample_size,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            model_path=args.model_path,
            n_jobs=args.n_jobs,
            verbose=not args.quiet,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.quiet:
        print(json.dumps(bundle.metrics.get("test_accuracy")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
