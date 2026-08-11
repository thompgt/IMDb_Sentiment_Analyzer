import pandas as pd
import pytest

from sentiment.data import (
    CLASS_NAMES,
    DatasetNotFoundError,
    load_reviews,
    prepare_frame,
    resolve_csv_path,
)


def test_load_reviews_normalises_columns(toy_csv):
    df = load_reviews(toy_csv)
    assert list(df.columns) == ["review", "label"]
    assert set(df["label"].unique()) == {0, 1}
    assert df["review"].map(type).eq(str).all()


def test_missing_dataset_raises_actionable_error(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="download_data.py"):
        load_reviews(tmp_path / "nope.csv")


def test_duplicates_are_dropped_before_any_split():
    """Leakage guard: identical reviews must not survive to the split."""
    df = pd.DataFrame(
        {
            "review": ["same text", "same text", "same text", "other text"],
            "sentiment": ["positive", "positive", "positive", "negative"],
        }
    )
    assert len(prepare_frame(df)) == 2
    assert len(prepare_frame(df, drop_duplicates=False)) == 4


def test_blank_and_null_rows_are_dropped():
    df = pd.DataFrame(
        {
            "review": ["good movie", "", None, "bad movie"],
            "sentiment": ["positive", "positive", "negative", "negative"],
        }
    )
    out = prepare_frame(df)
    assert sorted(out["review"]) == ["bad movie", "good movie"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        (["positive", "negative"], [1, 0]),
        (["pos", "neg"], [1, 0]),
        ([1, 0], [1, 0]),
        (["1", "0"], [1, 0]),
        (["Positive", "NEGATIVE"], [1, 0]),
    ],
)
def test_label_encodings_are_all_accepted(raw, expected):
    df = pd.DataFrame({"review": ["a good one", "a bad one"], "sentiment": raw})
    assert prepare_frame(df)["label"].tolist() == expected


def test_label_column_aliases():
    df = pd.DataFrame({"review": ["x good", "y bad"], "label": [1, 0]})
    assert prepare_frame(df)["label"].tolist() == [1, 0]


def test_missing_columns_raise_value_error():
    with pytest.raises(ValueError, match="review"):
        prepare_frame(pd.DataFrame({"text": ["a"], "sentiment": ["positive"]}))
    with pytest.raises(ValueError, match="sentiment"):
        prepare_frame(pd.DataFrame({"review": ["a"]}))


def test_sample_size_preserves_class_balance(toy_reviews):
    out = prepare_frame(toy_reviews, sample_size=40)
    assert len(out) == 40
    assert out["label"].mean() == pytest.approx(0.5, abs=0.05)


def test_env_var_overrides_default_path(monkeypatch):
    monkeypatch.setenv("IMDB_CSV_PATH", "/somewhere/else.csv")
    assert str(resolve_csv_path()).endswith("else.csv")
    # explicit argument still wins
    assert str(resolve_csv_path("explicit.csv")) == "explicit.csv"


def test_class_names_match_label_encoding():
    assert CLASS_NAMES == ["negative", "positive"]
