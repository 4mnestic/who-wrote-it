import numpy as np
import pytest

from wld.features import FUNCTION_WORD, FUNCTION_WORDS, WORD_LENGTH, chunk_rates, rates


def test_function_word_list_is_the_classic_70():
    """The Mosteller–Wallace list has 70 distinct words, including 'upon'."""
    assert len(FUNCTION_WORDS) == len(set(FUNCTION_WORDS)) == 70
    assert "upon" in FUNCTION_WORDS


def test_function_word_encoding_ignores_other_words():
    """Function words map to their column; everything else is ignored (-1)."""
    codes = FUNCTION_WORD.encode(np.array(["upon", "elephant", "the"]))
    assert codes.tolist() == [FUNCTION_WORDS.index("upon"), -1, FUNCTION_WORDS.index("the")]


def test_word_length_encoding_pools_long_words():
    """A 3-letter word goes to column 2; a 20-letter word to the last column."""
    codes = WORD_LENGTH.encode(np.array(["cat", "a" * 20]))
    assert codes.tolist() == [2, WORD_LENGTH.n_features - 1]


def test_rates_are_shares_of_all_words():
    """Rates divide by total words, so ignored words still count in the denominator."""
    r = rates(np.array([0, 0, -1, 1]), 3)
    assert r.tolist() == pytest.approx([0.5, 0.25, 0])


def test_rates_of_empty_passage_are_zero():
    """An empty passage yields zeros, not a division error."""
    assert rates(np.array([], dtype=int), 2).tolist() == [0, 0]


def test_chunk_rates_match_rates_per_chunk_and_drop_remainder():
    """Chunked rates equal `rates` on each chunk; leftover words are dropped."""
    codes = np.array([0, 1, -1, 1, 1, 1, 0])
    out = chunk_rates(codes, 2, chunk_size=3)
    assert out.shape == (2, 2)
    assert out[0] == pytest.approx(rates(codes[:3], 2))
    assert out[1] == pytest.approx(rates(codes[3:6], 2))


def test_chunk_rates_too_short():
    """A text shorter than one chunk yields zero rows."""
    assert chunk_rates(np.array([0, 1]), 2, chunk_size=10).shape == (0, 2)
