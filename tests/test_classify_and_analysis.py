import pytest

import wld.analysis as analysis
from wld.classify import leave_one_book_out, leave_one_out_documents, predict_documents
from wld.corpus import Book
from wld.features import FUNCTION_WORD, WORD_LENGTH


@pytest.mark.parametrize("fs", [FUNCTION_WORD, WORD_LENGTH])
def test_leave_one_book_out_separable(synthetic_books, fs):
    """With clearly different authors, every held-out book is classified correctly."""
    r = leave_one_book_out(synthetic_books, 500, fs)
    assert r["balanced_accuracy"] > 0.95
    assert r["books_correct_by_vote"] == 6
    assert {p["title"] for p in r["per_book"]} == {b.title for b in synthetic_books}


def _papers(synthetic_books, names=None):
    """Cut synthetic books A and B into 1,000-word 'papers'."""
    names = names or {"A": "A", "B": "B"}
    return [Book(f"{b.title}-{i}", names[b.author], "", b.words[i * 1000:(i + 1) * 1000])
            for b in synthetic_books[:4] for i in range(3)]


def test_document_level_validation_and_prediction(synthetic_books):
    """Leave-one-document-out finds the authors; P(author) is high for their unseen text."""
    papers = _papers(synthetic_books)
    r = leave_one_out_documents(papers, FUNCTION_WORD)
    assert (r["correct"], r["n"], r["misclassified"]) == (12, 12, "")
    p = predict_documents(papers, [synthetic_books[0], synthetic_books[2]], FUNCTION_WORD, "A")
    assert p[0] > 0.9 and p[1] < 0.1


def test_main_writes_all_outputs(tmp_path, monkeypatch, synthetic_books):
    """The pipeline runs end to end and writes every table and figure."""
    papers = _papers(synthetic_books, {"A": "Hamilton", "B": "Madison"}) + [
        Book("No. 49", "Disputed", "", synthetic_books[1].words[:1000])]
    monkeypatch.setattr(analysis, "load_books", lambda: synthetic_books)
    monkeypatch.setattr(analysis, "load_federalist", lambda: papers)
    monkeypatch.setattr(analysis, "CHUNK_SIZES", [250, 500])
    monkeypatch.setattr(analysis, "REF_CHUNK", 500)
    analysis.main(tmp_path)

    for name in ["accuracy_by_passage_length.csv", "per_book_500_words.csv", "books.csv",
                 "chi_square_tests.csv", "federalist_validation.csv",
                 "federalist_validation_papers.csv", "federalist_disputed.csv",
                 "federalist_marker_words.csv"]:
        assert (tmp_path / "results" / name).exists(), name
    for name in ["accuracy_by_passage_length.png", "federalist_disputed.png",
                 "word_length_by_author.png"]:
        assert (tmp_path / "figures" / name).exists(), name
