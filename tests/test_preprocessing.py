import pytest

from sentiment.preprocessing import (
    ENGLISH_STOPWORDS,
    NEGATION_WORDS,
    clean_text,
)


def test_strips_html_tags():
    assert "br" not in clean_text("A great film.<br /><br />Really.").split()
    assert "great" in clean_text("A great film.<br /><br />Really.").split()


def test_lowercases_and_drops_punctuation_and_digits():
    assert clean_text("AMAZING!!! 10/10 movie") == "amazing movie"


@pytest.mark.parametrize("word", sorted(NEGATION_WORDS))
def test_no_negation_is_ever_a_stopword(word):
    """The whole point: a sentiment model must keep its negations."""
    assert word not in ENGLISH_STOPWORDS


def test_negation_survives_cleaning():
    """Regression test for the original bug: 'not good' collapsed to 'good'."""
    assert clean_text("this was not good") != clean_text("this was good")
    assert "not" in clean_text("this was not good").split()


def test_contractions_become_negation_tokens():
    cleaned = clean_text("I didn't like it, it wasn't good").split()
    assert "didnt" in cleaned
    assert "wasnt" in cleaned


def test_ordinary_stopwords_are_still_removed():
    assert clean_text("the movie was a masterpiece") == "movie masterpiece"


def test_stopword_removal_can_be_disabled():
    assert clean_text("the movie", remove_stopwords=False) == "the movie"


@pytest.mark.parametrize("value", [None, float("nan"), 123, [], {}])
def test_non_string_input_returns_empty_string(value):
    assert clean_text(value) == ""


def test_empty_and_whitespace_input():
    assert clean_text("") == ""
    assert clean_text("   \n\t ") == ""
    assert clean_text("<br/>") == ""


def test_custom_stopword_set_is_honoured():
    assert clean_text("alpha beta gamma", stopwords={"beta"}) == "alpha gamma"


def test_clean_text_is_idempotent():
    once = clean_text("The film was NOT great -- honestly!")
    assert clean_text(once) == once
