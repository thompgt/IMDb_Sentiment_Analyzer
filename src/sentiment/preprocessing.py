"""Text cleaning for the IMDb sentiment pipeline.

The one thing worth understanding here: a generic English stopword list is
actively harmful for sentiment analysis because it deletes negations.  NLTK's
and scikit-learn's lists both contain ``no``, ``not``, ``nor``, ``never`` and
friends, so ``"not good"`` and ``"good"`` collapse to the identical token
``good``.  That puts a hard ceiling on accuracy no matter how good the
classifier is.

We therefore start from scikit-learn's frozen English stopword list (no
download, fully deterministic, already a dependency) and subtract every
negation and contraction-negation token before using it.  Combined with the
bigram TF-IDF used in :mod:`sentiment.train`, ``not good`` survives as its own
feature.
"""

from __future__ import annotations

import re
from typing import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

#: Tokens that flip or scope sentiment.  These must never be stripped.
NEGATION_WORDS: frozenset[str] = frozenset(
    {
        "no",
        "not",
        "nor",
        "none",
        "never",
        "neither",
        "nothing",
        "nobody",
        "nowhere",
        "cannot",
        "cant",
        "wont",
        "dont",
        "doesnt",
        "didnt",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "hasnt",
        "havent",
        "hadnt",
        "shouldnt",
        "wouldnt",
        "couldnt",
        "aint",
        "without",
        "against",
        "but",
        "however",
        "although",
        "though",
        "very",
        "too",
        "only",
        "least",
        "less",
    }
)

#: scikit-learn's English stopword list with every negation put back.
ENGLISH_STOPWORDS: frozenset[str] = frozenset(ENGLISH_STOP_WORDS) - NEGATION_WORDS

_HTML_TAG_RE = re.compile(r"<[^>]*>")
_NON_ALPHA_RE = re.compile(r"[^a-z']+")
_APOSTROPHE_RE = re.compile(r"'")

_LEMMATIZER = None


def _get_lemmatizer():
    """Lazily build an NLTK WordNet lemmatizer.

    Optional: lemmatisation is off by default because it needs an NLTK corpus
    download, and it buys very little on top of bigram TF-IDF.  Raises a clear
    error rather than silently degrading if the corpus is missing.
    """
    global _LEMMATIZER
    if _LEMMATIZER is None:
        try:
            from nltk.stem import WordNetLemmatizer

            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("tests")  # forces the corpus lookup
        except Exception as exc:  # pragma: no cover - depends on local NLTK data
            raise RuntimeError(
                "lemmatize=True needs NLTK plus the 'wordnet' and 'omw-1.4' "
                "corpora. Install with `pip install nltk` and run "
                "`python -m nltk.downloader wordnet omw-1.4`, or leave "
                "lemmatisation disabled (the default)."
            ) from exc
        _LEMMATIZER = lemmatizer
    return _LEMMATIZER


def clean_text(
    text: str,
    *,
    remove_stopwords: bool = True,
    lemmatize: bool = False,
    stopwords: Iterable[str] | None = None,
) -> str:
    """Normalise a raw review into a whitespace-joined token string.

    Steps: strip HTML, lowercase, drop non-alphabetic characters (contractions
    are joined rather than split, so ``don't`` becomes ``dont`` and stays in
    :data:`NEGATION_WORDS`), then optionally drop stopwords and lemmatise.

    Non-string input (``None``, ``NaN``) returns an empty string so the function
    is safe to hand straight to ``TfidfVectorizer(preprocessor=...)``.
    """
    if not isinstance(text, str):
        return ""

    text = _HTML_TAG_RE.sub(" ", text)
    text = text.lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _APOSTROPHE_RE.sub("", text)

    words = text.split()

    if remove_stopwords:
        stop = ENGLISH_STOPWORDS if stopwords is None else frozenset(stopwords)
        words = [w for w in words if w not in stop]

    if lemmatize:
        lemmatizer = _get_lemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)
