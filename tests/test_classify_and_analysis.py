import json

import numpy as np
import pytest

import wld.analysis as analysis
from wld.classify import chunk_features, leave_one_book_out


def test_chunk_features_drop_remainder_and_normalize():
    """25 words in chunks of 10 gives 2 chunks whose bin proportions sum to 1."""
    f = chunk_features(np.arange(1, 26), 10)
    assert f.shape == (2, 13)
    assert f.sum(axis=1) == pytest.approx([1, 1])


def test_chunk_features_too_short():
    """A text shorter than one chunk yields zero rows, not an error."""
    assert chunk_features(np.array([1, 2]), 10).shape == (0, 13)


def test_leave_one_book_out_separable(synthetic_books):
    """With clearly different authors, held-out books are classified correctly."""
    r = leave_one_book_out(synthetic_books, 500)
    assert r["balanced_accuracy"] > 0.95
    assert r["books_correct_by_vote"] == 6
    assert r["chance"] == pytest.approx(1 / 3)
    assert {p["title"] for p in r["per_book"]} == {b.title for b in synthetic_books}


def test_main_writes_all_outputs(tmp_path, monkeypatch, synthetic_books):
    """The pipeline runs end to end and writes every table, the summary and the figures."""
    monkeypatch.setattr(analysis, "load_books", lambda: synthetic_books)
    monkeypatch.setattr(analysis, "CHUNK_SIZES", [250, 500])
    monkeypatch.setattr(analysis, "NULL_CHUNK_SIZE", 500)
    monkeypatch.setattr(analysis, "NULL_DRAWS", 3)
    summary = analysis.main(tmp_path)

    res = tmp_path / "results"
    for name in ["books.csv", "author_distributions.csv", "chi_square_tests.csv",
                 "classification_by_chunk_size.csv", "classification_per_book_500.csv",
                 "confusion_matrix_500.csv", "summary.json"]:
        assert (res / name).exists(), name
    for name in ["word_length_by_author.png", "mean_length_by_book.png",
                 "classifier_accuracy.png"]:
        assert (tmp_path / "figures" / name).exists(), name
    assert json.loads((res / "summary.json").read_text()) == summary
    assert summary["permutation_test"]["n_permutations"] == 90
    assert summary["classifier_null"]["observed"] > 0.95
