"""Loading and de-duplicating the IMDb review corpus.

There is exactly **one** supported data pipeline: ``python download_data.py``
fetches the Stanford aclImdb v1 tarball (no credentials required) and writes
``data/imdb_reviews.csv``.  The old kagglehub path is gone -- it silently
required Kaggle API credentials that nothing in the repo told you to set up.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

#: Written by ``download_data.py``.
DEFAULT_CSV_PATH = Path("data") / "imdb_reviews.csv"

#: Canonical label encoding used everywhere: 1 = positive, 0 = negative.
LABEL_MAP = {
    "positive": 1,
    "negative": 0,
    "pos": 1,
    "neg": 0,
    1: 1,
    0: 0,
    "1": 1,
    "0": 0,
}

CLASS_NAMES = ["negative", "positive"]


class DatasetNotFoundError(FileNotFoundError):
    """Raised with an actionable message when the corpus CSV is missing."""


def resolve_csv_path(csv_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the dataset location, honouring ``$IMDB_CSV_PATH``."""
    if csv_path is not None:
        return Path(csv_path)
    env = os.environ.get("IMDB_CSV_PATH")
    if env:
        return Path(env)
    return DEFAULT_CSV_PATH


def load_reviews(
    csv_path: str | os.PathLike[str] | None = None,
    *,
    sample_size: int | None = None,
    random_state: int = 42,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Return a ``DataFrame`` with ``review`` (str) and ``label`` (0/1) columns.

    De-duplication happens **before** any sampling or splitting.  The raw corpus
    contains a few hundred byte-identical reviews; if they survive into the
    split, the same text can land in both train and test and the reported
    accuracy is inflated by leakage.
    """
    path = resolve_csv_path(csv_path)
    if not path.exists():
        raise DatasetNotFoundError(
            f"Dataset not found at {path}. Generate it first:\n"
            f"    python download_data.py\n"
            f"(or point IMDB_CSV_PATH at an existing review CSV)"
        )

    df = pd.read_csv(path)
    return prepare_frame(
        df,
        sample_size=sample_size,
        random_state=random_state,
        drop_duplicates=drop_duplicates,
    )


def prepare_frame(
    df: pd.DataFrame,
    *,
    sample_size: int | None = None,
    random_state: int = 42,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Normalise columns, drop dupes/NaNs, optionally subsample."""
    if "review" not in df.columns:
        raise ValueError(f"expected a 'review' column, got {list(df.columns)}")

    label_col = next(
        (c for c in ("sentiment", "label", "target") if c in df.columns), None
    )
    if label_col is None:
        raise ValueError(f"expected a 'sentiment' or 'label' column, got {list(df.columns)}")

    out = df[["review", label_col]].copy()
    out.columns = ["review", "label"]

    out["review"] = out["review"].astype(str).str.strip()
    out["label"] = out["label"].map(lambda v: LABEL_MAP.get(v, LABEL_MAP.get(str(v).strip().lower())))

    out = out.dropna(subset=["review", "label"])
    out = out[out["review"].str.len() > 0]
    out["label"] = out["label"].astype(int)

    if drop_duplicates:
        out = out.drop_duplicates(subset="review", keep="first")

    out = out.reset_index(drop=True)

    if sample_size is not None and sample_size < len(out):
        # Stratified subsample so the class ratio survives the downsizing.
        from sklearn.model_selection import train_test_split

        out, _ = train_test_split(
            out,
            train_size=sample_size,
            random_state=random_state,
            stratify=out["label"],
        )
        out = out.reset_index(drop=True)

    return out
