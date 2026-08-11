"""Reusable, importable implementation of the IMDb sentiment pipeline.

Everything the notebook and the Streamlit app need lives here so there is a
single source of truth for preprocessing, training and inference.
"""

from sentiment.preprocessing import ENGLISH_STOPWORDS, NEGATION_WORDS, clean_text

__all__ = [
    "ENGLISH_STOPWORDS",
    "NEGATION_WORDS",
    "clean_text",
    "__version__",
]

__version__ = "0.2.0"
