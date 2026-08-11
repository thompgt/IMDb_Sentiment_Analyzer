import random

import pandas as pd
import pytest

POSITIVE_SEEDS = [
    "an absolutely brilliant film with a wonderful cast",
    "i loved every minute of this masterpiece",
    "beautifully shot and genuinely moving",
    "a delightful comedy that had me laughing throughout",
    "superb acting and a gripping story",
]
NEGATIVE_SEEDS = [
    "a boring mess with terrible acting",
    "i hated this dull and pointless movie",
    "awful script and wooden performances",
    "a complete waste of two hours",
    "painfully bad, the worst thing i have seen",
]

FILLER = [
    "the director clearly had a vision",
    "the soundtrack was present",
    "it runs about two hours",
    "there is a plot twist",
    "the cinema was half empty",
]


@pytest.fixture(scope="session")
def toy_reviews() -> pd.DataFrame:
    """A small, linearly separable corpus good enough to smoke-test training."""
    rng = random.Random(0)
    rows = []
    for i in range(120):
        pos = POSITIVE_SEEDS[i % len(POSITIVE_SEEDS)]
        neg = NEGATIVE_SEEDS[i % len(NEGATIVE_SEEDS)]
        filler = rng.choice(FILLER)
        rows.append({"review": f"{pos} {filler} number {i}", "sentiment": "positive"})
        rows.append({"review": f"{neg} {filler} number {i}", "sentiment": "negative"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def toy_csv(tmp_path_factory, toy_reviews) -> str:
    path = tmp_path_factory.mktemp("corpus") / "imdb_reviews.csv"
    toy_reviews.to_csv(path, index=False)
    return str(path)
