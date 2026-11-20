import json

import numpy as np
import pytest

import wld.analysis as analysis
from wld.classify import (
    leave_one_book_out,
    leave_one_out_documents,
    predict_documents,
    train_test_books,
)
from wld.corpus import Book
from wld.features import FUNCTION_WORD, WORD_LENGTH


@pytest.mark.parametrize("fs", [FUNCTION_WORD, WORD_LENGTH])
def test_leave_one_book_out_separable(synthetic_books, fs):
    """With clearly different authors, every held-out book is classified correctly."""
    r = leave_one_book_out(synthetic_books, 500, fs)
    assert r["balanced_accuracy"] > 0.95
    assert r["books_correct_by_vote"] == 6
    assert {p["title"] for p in r["per_book"]} == {b.title for b in synthetic_books}


def test_train_test_books_reports_class_shares(synthetic_books):
    """Each test book gets the share of its chunks assigned to each training class."""
    train, test = synthetic_books[::2], synthetic_books[1::2]
    rows = train_test_books(train, test, 500, FUNCTION_WORD, label=lambda b: b.translator)
    assert [r["title"] for r in rows] == [b.title for b in test]
    for r, b in zip(rows, test):
        assert r[b.translator] > 0.95
        assert sum(r[c] for c in ("TA", "TB", "TC")) == pytest.approx(1)


def _papers(synthetic_books):
    """Cut synthetic books A and B into 1,000-word 'papers'."""
    out = []
    for b in synthetic_books[:4]:
        for i in range(3):
            out.append(Book(f"{b.title}-{i}", b.author, "", b.words[i * 1000:(i + 1) * 1000]))
    return out


def test_document_level_validation_and_prediction(synthetic_books):
    """Leave-one-document-out finds the authors; P(author) is high for their unseen text."""
    papers = _papers(synthetic_books)
    r = leave_one_out_documents(papers, FUNCTION_WORD)
    assert (r["correct"], r["n"], r["misclassified"]) == (12, 12, "")
    p = predict_documents(papers, [synthetic_books[0], synthetic_books[2]], FUNCTION_WORD, "A")
    assert p[0] > 0.9 and p[1] < 0.1


def test_main_writes_all_outputs(tmp_path, monkeypatch, synthetic_books):
    """The pipeline runs end to end and writes every table, the summary and the figures."""
    main_books, extra = synthetic_books, [
        Book("X", "A", "TB", synthetic_books[2].words), Book("Y", "B", "TA", synthetic_books[0].words)]
    papers = _papers(synthetic_books)
    papers = ([Book(p.title, {"A": "Hamilton", "B": "Madison"}[p.author], "", p.words) for p in papers]
              + [Book("No. 49", "Disputed", "", synthetic_books[1].words[:1000])])
    monkeypatch.setattr(analysis, "load_books", lambda s="main": main_books if s == "main" else extra)
    monkeypatch.setattr(analysis, "load_federalist", lambda: papers)
    monkeypatch.setattr(analysis, "CHUNK_SIZES", [250, 500])
    monkeypatch.setattr(analysis, "REF_CHUNK", 500)
    monkeypatch.setattr(analysis, "NULL_DRAWS", 3)
    monkeypatch.setattr(analysis, "TRANSLATOR_TRAIN", ["A0", "B0"])
    monkeypatch.setattr(analysis, "TRANSLATOR_TEST", ["X", "Y"])
    summary = analysis.main(tmp_path)

    res = tmp_path / "results"
    for name in ["accuracy_by_passage_length.csv", "per_book_500_words.csv", "books.csv",
                 "word_length_distributions.csv", "chi_square_tests.csv",
                 "federalist_validation.csv", "federalist_validation_papers.csv",
                 "federalist_disputed.csv",
                 "federalist_marker_words.csv", "translator_check.csv", "summary.json"]:
        assert (res / name).exists(), name
    for name in ["accuracy_by_passage_length.png", "federalist_disputed.png",
                 "federalist_upon.png", "word_length_by_author.png", "mean_length_by_book.png"]:
        assert (tmp_path / "figures" / name).exists(), name
    assert json.loads((res / "summary.json").read_text()) == summary
    assert summary["word_length_permutation_test"]["n_permutations"] == 90
    assert np.isfinite(summary["word_length_classifier_null"]["p"])
