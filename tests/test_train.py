import json

import numpy as np
import pytest

from sentiment.model import load_bundle
from sentiment.train import error_analysis, main, train


@pytest.fixture(scope="module")
def _grid():
    """Shrink the search space so the smoke test stays fast."""
    from sentiment import train as train_mod

    original = train_mod.SEARCH_SPACE
    train_mod.SEARCH_SPACE = [
        {
            "clf__C": [1.0],
            "tfidf__ngram_range": [(1, 2)],
            "tfidf__min_df": [1],
        }
    ]
    yield
    train_mod.SEARCH_SPACE = original


@pytest.fixture(scope="module")
def trained(tmp_path_factory, toy_csv, _grid):
    tmp = tmp_path_factory.mktemp("train")
    bundle = train(
        csv_path=toy_csv,
        cv_folds=3,
        model_path=tmp / "model.joblib",
        n_jobs=1,
        verbose=False,
    )
    return bundle, tmp


def test_training_beats_the_majority_class_baseline(trained):
    bundle, _ = trained
    assert bundle.metrics["test_accuracy"] > bundle.metrics["baseline_accuracy"]


def test_baseline_is_recorded_at_all(trained):
    bundle, _ = trained
    assert 0.0 < bundle.metrics["baseline_accuracy"] <= 1.0


def test_split_is_stratified(trained):
    """A stratified 80/20 split keeps the 50/50 class ratio intact."""
    bundle, _ = trained
    assert bundle.metadata["positive_rate"] == pytest.approx(0.5, abs=0.01)
    assert bundle.metadata["n_train"] + bundle.metadata["n_test"] == bundle.metadata["n_reviews"]


def test_model_selection_used_cross_validation_not_the_test_set(trained):
    bundle, _ = trained
    assert "best_cv_roc_auc" in bundle.metrics
    assert bundle.metrics["cv_ranking"], "CV ranking should record every candidate"


def test_metrics_and_metadata_are_persisted(trained):
    bundle, tmp = trained
    reloaded = load_bundle(tmp / "model.joblib")
    assert reloaded.metrics["test_accuracy"] == bundle.metrics["test_accuracy"]
    assert reloaded.metadata["classifier"]
    assert reloaded.metadata["class_names"] == ["negative", "positive"]


def test_metrics_json_report_is_written(trained):
    _, tmp = trained
    report = json.loads((tmp / "model.metrics.json").read_text(encoding="utf-8"))
    assert "metrics" in report and "metadata" in report


def test_top_features_are_stored_for_the_app(trained):
    bundle, _ = trained
    assert bundle.metadata["top_features"]["positive"]


def test_confusion_matrix_shape(trained):
    bundle, _ = trained
    assert np.array(bundle.metrics["confusion_matrix"]).shape == (2, 2)


def test_error_analysis_ranks_by_confidence():
    texts = ["a", "b", "c", "d"]
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    proba_pos = np.array([0.6, 0.99, 0.4, 0.01])
    out = error_analysis(texts, y_true, y_pred, proba_pos, n=2)
    assert [e["review"] for e in out["false_positives"]] == ["b", "a"]
    assert [e["review"] for e in out["false_negatives"]] == ["d", "c"]
    assert out["false_positives"][0]["true"] == "negative"
    assert out["false_positives"][0]["predicted"] == "positive"


def test_cli_reports_missing_dataset_without_traceback(tmp_path, capsys):
    code = main(["--csv", str(tmp_path / "missing.csv")])
    assert code == 1
    assert "download_data.py" in capsys.readouterr().err
