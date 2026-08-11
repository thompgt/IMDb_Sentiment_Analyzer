import numpy as np
import pytest

from sentiment.model import (
    ModelBundle,
    ModelNotFoundError,
    build_pipeline,
    load_bundle,
    resolve_model_path,
    save_bundle,
    top_coefficients,
)


@pytest.fixture
def fitted_bundle(toy_reviews):
    pipeline = build_pipeline("logreg", min_df=1)
    labels = (toy_reviews["sentiment"] == "positive").astype(int)
    pipeline.fit(toy_reviews["review"].tolist(), labels)
    return ModelBundle(pipeline=pipeline, metrics={"test_accuracy": 0.9})


def test_pipeline_bundles_vectorizer_and_classifier():
    pipe = build_pipeline("logreg")
    assert list(pipe.named_steps) == ["tfidf", "clf"]


def test_pipeline_uses_bigrams_so_negation_is_representable():
    assert build_pipeline().named_steps["tfidf"].ngram_range == (1, 2)


def test_unknown_classifier_rejected():
    with pytest.raises(ValueError, match="unknown classifier"):
        build_pipeline("magic")


def test_pipeline_accepts_raw_text_and_learns_something(fitted_bundle):
    label, conf = fitted_bundle.predict_one("an absolutely brilliant and wonderful film")
    assert label == "positive"
    assert 0.5 < conf <= 1.0
    label, conf = fitted_bundle.predict_one("a boring mess with terrible acting")
    assert label == "negative"


def test_predict_proba_rows_sum_to_one(fitted_bundle):
    proba = fitted_bundle.predict_proba(["great film", "awful film"])
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_html_input_is_handled_end_to_end(fitted_bundle):
    """Cleaning lives inside the pipeline, so callers pass raw review text."""
    label, _ = fitted_bundle.predict_one("<br />an absolutely brilliant film!<br/>")
    assert label == "positive"


def test_save_and_load_round_trip(fitted_bundle, tmp_path):
    path = save_bundle(fitted_bundle, tmp_path / "m.joblib")
    assert path.exists()
    loaded = load_bundle(path)
    assert loaded.metrics["test_accuracy"] == 0.9
    assert loaded.predict_one("an absolutely brilliant film")[0] == "positive"


def test_loading_a_missing_model_is_actionable(tmp_path):
    with pytest.raises(ModelNotFoundError, match="sentiment.train"):
        load_bundle(tmp_path / "absent.joblib")


def test_model_path_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTIMENT_MODEL_PATH", str(tmp_path / "custom.joblib"))
    assert resolve_model_path().name == "custom.joblib"


def test_top_coefficients_are_signed_and_ordered(fitted_bundle):
    coefs = top_coefficients(fitted_bundle.pipeline, n=5)
    assert len(coefs["positive"]) == 5
    assert all(a[1] >= b[1] for a, b in zip(coefs["positive"], coefs["positive"][1:]))
    assert all(a[1] <= b[1] for a, b in zip(coefs["negative"], coefs["negative"][1:]))
    assert coefs["positive"][0][1] > coefs["negative"][0][1]


def test_bundle_metrics_serialise_to_json(fitted_bundle):
    assert '"test_accuracy": 0.9' in fitted_bundle.as_json()
