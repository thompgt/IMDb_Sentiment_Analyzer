"""Build, persist and load the sentiment pipeline.

The whole model -- vectoriser *and* classifier -- is a single
:class:`sklearn.pipeline.Pipeline`, so ``joblib.dump``/``load`` round-trips one
artifact and the Streamlit app can never accidentally pair a fitted classifier
with an unfitted vectoriser.  Cleaning runs inside the vectoriser via
``preprocessor=clean_text``, so callers hand in raw review text.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sentiment.data import CLASS_NAMES
from sentiment.preprocessing import clean_text

DEFAULT_MODEL_PATH = Path("models") / "sentiment_pipeline.joblib"


@dataclass
class ModelBundle:
    """A fitted pipeline plus the metrics/metadata earned when fitting it."""

    pipeline: Pipeline
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.pipeline.predict(list(texts))

    def predict_one(self, text: str) -> tuple[str, float]:
        """Return ``(label_name, probability_of_that_label)`` for one review."""
        proba = self.predict_proba([text])[0]
        idx = int(np.argmax(proba))
        return CLASS_NAMES[idx], float(proba[idx])

    def as_json(self) -> str:
        return json.dumps(
            {"metrics": self.metrics, "metadata": self.metadata},
            indent=2,
            default=str,
        )


def build_pipeline(
    classifier: str = "logreg",
    *,
    max_features: int | None = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    C: float = 1.0,
    alpha: float = 1.0,
) -> Pipeline:
    """TF-IDF (uni+bigram, negation-preserving cleaning) -> linear classifier."""
    if classifier == "logreg":
        clf = LogisticRegression(max_iter=2000, C=C, solver="liblinear")
    elif classifier == "nb":
        clf = MultinomialNB(alpha=alpha)
    else:
        raise ValueError(f"unknown classifier {classifier!r}; use 'logreg' or 'nb'")

    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def save_bundle(bundle: ModelBundle, path: str | os.PathLike[str] | None = None) -> Path:
    target = Path(path) if path is not None else DEFAULT_MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": bundle.pipeline,
            "metrics": bundle.metrics,
            "metadata": bundle.metadata,
        },
        target,
        compress=3,
    )
    return target


def resolve_model_path(path: str | os.PathLike[str] | None = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("SENTIMENT_MODEL_PATH")
    if env:
        return Path(env)
    return DEFAULT_MODEL_PATH


class ModelNotFoundError(FileNotFoundError):
    """Raised with an actionable message when no trained model exists yet."""


def load_bundle(path: str | os.PathLike[str] | None = None) -> ModelBundle:
    """Load a persisted :class:`ModelBundle`.

    Raises :class:`ModelNotFoundError` with the exact command to run, instead of
    letting a downstream ``predict`` blow up on ``None``.
    """
    target = resolve_model_path(path)
    if not target.exists():
        raise ModelNotFoundError(
            f"No trained model at {target}. Train one first:\n"
            f"    python download_data.py && python -m sentiment.train\n"
            f"(or set SENTIMENT_MODEL_PATH to an existing .joblib bundle)"
        )
    payload = joblib.load(target)
    if isinstance(payload, Pipeline):  # tolerate a bare pipeline dump
        return ModelBundle(pipeline=payload)
    return ModelBundle(
        pipeline=payload["pipeline"],
        metrics=payload.get("metrics", {}),
        metadata=payload.get("metadata", {}),
    )


def top_coefficients(pipeline: Pipeline, n: int = 15) -> dict[str, list[tuple[str, float]]]:
    """Strongest positive/negative TF-IDF features of a linear model."""
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return {"positive": [], "negative": []}
    names = np.asarray(vectorizer.get_feature_names_out())
    coefs = np.ravel(clf.coef_)
    order = np.argsort(coefs)
    return {
        "negative": [(str(names[i]), float(coefs[i])) for i in order[:n]],
        "positive": [(str(names[i]), float(coefs[i])) for i in order[-n:][::-1]],
    }


__all__ = [
    "DEFAULT_MODEL_PATH",
    "ModelBundle",
    "ModelNotFoundError",
    "build_pipeline",
    "load_bundle",
    "resolve_model_path",
    "save_bundle",
    "top_coefficients",
]
