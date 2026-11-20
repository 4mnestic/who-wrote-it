# Who wrote it? Identifying authors from their writing habits

[![tests](https://github.com/4mnestic/word-length-distribution-goodness-of-fit/actions/workflows/ci.yml/badge.svg)](https://github.com/4mnestic/word-length-distribution-goodness-of-fit/actions/workflows/ci.yml)

In 1887, T. C. Mendenhall claimed every writer has a telltale pattern of word lengths. I tested
that on 2.4 million words from Austen, Dickens and Tolstoy, found a signal that works far
better, and used it on the disputed Federalist Papers.

## Results

- **Word length doesn't identify authors.** On 1,000-word passages it picks the right author
  50% of the time, against 33% for a random guess. A chi-square test (SciPy) says the
  authors' word-length distributions differ significantly, but the effect size (Cramér's
  V = 0.04) is no bigger than the differences among one author's own books. With millions
  of words, "significant" doesn't mean "useful."
- **Function words do.** I cut each book into passages and turned each passage into 70
  numbers: how often it uses each of 70 common words (*the, upon, would, whilst*…). A
  logistic regression (scikit-learn) trained on those numbers names the right author 90% of
  the time at 1,000 words and 99% at 5,000, and attributes all 12 novels correctly.
- **The Federalist Papers.** The same model, after correctly attributing 64 of 65 papers of
  known authorship, gives each disputed paper a probability of being Madison's. 11 of the 12
  come out at least 93% Madison, matching the historical consensus. No. 55 is too close to
  call.

![Accuracy by passage length](figures/accuracy_by_passage_length.png)

## The Federalist Papers

Hamilton and Madison both claimed authorship of 12 of these 1787–88 essays. Their word
lengths are nearly identical, so word length can't separate them (grey). Function words can
(blue):

![Disputed Federalist Papers](figures/federalist_disputed.png)

The clearest single clue is *upon*. Hamilton uses it 3.3 times per 1,000 words, Madison
0.18 times, and the disputed papers 0.13 times.

## How it was tested

- **No data leakage.** Every score comes from text the model never saw. For the novels, the
  model trains on 11 books and is tested on the 12th, rotating through all 12. For the
  Federalist Papers, it holds out one paper at a time. A random split would let the model
  recognize a book's characters instead of its author's style.
- **Class imbalance handled.** Hamilton wrote 51 of the 65 known papers, so always guessing
  "Hamilton" would be right 78% of the time. The model weights both authors equally, and
  results are reported as balanced accuracy (the average of each author's hit rate), which
  that trick can't inflate.
- **Cleaning done in code, not by hand.** A small table records where each novel's story
  starts and ends. The code keeps only that span and uses regular expressions to strip
  editors' prefaces, illustration captions and chapter headings, so the cleaning is
  repeatable.

Full tables are in [METHODS.md](METHODS.md); code is in [`wld/`](wld/).

## Limitations

- Only three novelists, and a two-way choice for the Federalist Papers. Choosing among many
  authors is harder.
- The Tolstoy novels are English translations, so their "author" signal partly reflects the
  translators.

## Reproduce

One command rebuilds every table (pandas) and chart (matplotlib) from the raw texts.
Dependencies are locked with uv, and automated tests (pytest) check the cleaning, features
and statistics on every push via GitHub Actions.

```bash
uv sync
uv run python -m wld.analysis   # rewrites results/ and figures/
uv run pytest
```

## Background

This project began as a term paper for LING 388, which I later extended with
[Claude](https://claude.ai).

## References

- Mendenhall, T. C. (1887). The characteristic curves of composition. *Science*, 9(214), 237–249.
- Mosteller, F., & Wallace, D. L. (1964). *Inference and Disputed Authorship: The Federalist*.
  Addison-Wesley.
