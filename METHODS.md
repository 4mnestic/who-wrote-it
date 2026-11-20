# Methods and full results

Every number here comes from `uv run python -m wld.analysis` and is saved in
[`results/`](results/). Accuracy means **balanced accuracy**, the average of per-author hit
rates.

## Data

- **Novels:** four each by Austen, Dickens and Tolstoy, 2,401,328 words
  ([`results/books.csv`](results/books.csv)). [`data/books.csv`](data/books.csv) records
  where each narrative starts and ends. The code keeps only that span (dropping editors'
  prefaces, contents pages and licenses) and removes illustration captions and chapter
  headings.
- **Federalist Papers:** 51 Hamilton, 14 Madison and 12 disputed papers, attributed as in
  Mosteller & Wallace (the source file's own labels credit the disputed papers to Madison,
  so they are ignored). Each paper keeps the text from its salutation to the "PUBLIUS"
  signature.
- **Words:** runs of letters, lowercased. A word's length is its letter count.

## Features and model

- **Word length:** share of words with 1, 2, …, 12 and 13+ letters.
- **Function words:** rate of each of the 70 words Mosteller & Wallace used. They are
  topic-neutral, so they track habit rather than subject.
- **Model:** standardized logistic regression with classes weighted equally. The same
  model is used for both feature sets, so the comparison is about the signal.

## Novels: accuracy on unseen books

Each book is cut into passages of *n* words. A model trained on the other 11 books predicts
the held-out book's passages, rotating through all 12
([`results/accuracy_by_passage_length.csv`](results/accuracy_by_passage_length.csv)).

| Passage length (words) | 100 | 250 | 500 | 1,000 | 2,500 | 5,000 |
|---|---:|---:|---:|---:|---:|---:|
| Function words | 64% | 77% | 85% | 90% | 97% | 99% |
| Word length | 42% | 44% | 47% | 50% | 57% | 62% |

With function words, a majority vote over each book's passages names the right author for
all 12 books at every length. With word length it misses 1–3. At 1,000 words those misses
are *Great Expectations* (labeled Austen) and *War and Peace* (labeled Dickens)
([`results/per_book_1000_words.csv`](results/per_book_1000_words.csv)).

## Word length: significant, but tiny

![Word-length distribution by author](figures/word_length_by_author.png)

With millions of words, any difference passes a significance test, so the effect size
(Cramér's V, where 0 means identical distributions) is what matters
([`results/chi_square_tests.csv`](results/chi_square_tests.csv)):

| Comparison | χ² (df) | p | Cramér's V |
|---|---:|---:|---:|
| Between the three authors | 8,055 (24) | < 10⁻³⁰⁰ | 0.041 |
| Among Austen's books | 878 (36) | 5 × 10⁻¹⁶¹ | 0.025 |
| Among Dickens's books | 3,942 (36) | < 10⁻³⁰⁰ | 0.040 |
| Among Tolstoy's books | 3,404 (36) | < 10⁻³⁰⁰ | 0.032 |

The gap between authors is no bigger than the gap among Dickens's own novels.

## Federalist Papers

**Validation**, holding out one known paper at a time
([`results/federalist_validation.csv`](results/federalist_validation.csv)):

| Features | Balanced accuracy | Correct |
|---|---:|---:|
| Function words | 99% | 64 / 65 |
| Word length | 65% | 48 / 65 |

The function-word model's only miss is No. 13, the shortest paper (960 words), at
P(Madison) = 0.52 ([`results/federalist_validation_papers.csv`](results/federalist_validation_papers.csv)).

**Disputed papers**, model trained on all 65 known papers
([`results/federalist_disputed.csv`](results/federalist_disputed.csv)):

| Paper | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 62 | 63 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P(Madison), function words | .95 | .98 | 1.00 | .97 | 1.00 | .94 | **.41** | .94 | .98 | .93 | .97 | .98 |

No. 55 was also among the weakest cases in Mosteller & Wallace's study. The probabilities
are uncalibrated, so read them as relative confidence, not exact odds.

**Marker words**, per 1,000 words
([`results/federalist_marker_words.csv`](results/federalist_marker_words.csv)):

| | upon | whilst | while | enough |
|---|---:|---:|---:|---:|
| Hamilton | 3.34 | 0.01 | 0.32 | 0.30 |
| Madison | 0.18 | 0.31 | 0.00 | 0.00 |
| Disputed | 0.13 | 0.37 | 0.00 | 0.00 |
