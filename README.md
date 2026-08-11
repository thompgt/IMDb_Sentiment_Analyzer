# SentimentSense: IMDb Movie Review Classifier 🎬

Classify IMDb movie reviews as positive or negative with TF-IDF features and a
linear model — end to end, from download to a live demo, with the methodology
written down and the numbers reproducible.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

---

## Quickstart

Three commands from a clean clone:

```bash
pip install -e ".[app,data]"   # install the sentiment package + demo + downloader
python download_data.py        # fetch the corpus -> data/imdb_reviews.csv (~84 MB download)
python -m sentiment.train      # fit, evaluate, persist -> models/sentiment_pipeline.joblib
streamlit run app.py           # interactive demo on http://localhost:8501
```

Want to try it without the full corpus? `python -m sentiment.train --sample-size 5000`
trains on a stratified subsample in well under a minute.

### …or with Docker, touching nothing on your machine

```bash
docker compose run --rm download        # -> ./data/imdb_reviews.csv
docker compose run --rm train           # -> ./models/sentiment_pipeline.joblib
docker compose up demo                  # -> http://localhost:8601
docker compose run --rm test            # pytest inside the image
docker compose --profile notebook up notebook   # JupyterLab on http://localhost:8602
```

`data/` and `models/` are bind-mounted, so artifacts survive container restarts
and are visible from the host.

---

## Repository layout

```
src/sentiment/          the actual implementation — importable, tested, shared
  preprocessing.py      clean_text + the negation-preserving stopword set
  data.py               loading, label normalisation, de-duplication
  model.py              pipeline construction, joblib persistence, feature weights
  train.py              CLI: grid search, honest evaluation, error analysis
tests/                  pytest suite (no network, no dataset required)
notebooks/
  sentiment_analysis.ipynb              narrative walkthrough over src/sentiment
  sentiment_analysis_transformers.ipynb zero-shot DistilBERT-SST2 baseline
app.py                  Streamlit demo — loads the trained joblib bundle
download_data.py        the one supported data pipeline (Stanford aclImdb v1)
Dockerfile              python:3.12-slim image for demo / training / tests
docker-compose.yml      demo, download, train, test and notebook services
.github/workflows/ci.yml  tests, repo hygiene, end-to-end smoke test, docker build
```

Nothing important lives only in a notebook. Preprocessing, training and
inference are all in `src/sentiment/`, so the notebook, the test suite and the
Streamlit app run the same code and cannot drift apart.

---

## Requirements files

Split by purpose so you never install PyTorch to run a unit test:

| File | Contents | Install with |
| --- | --- | --- |
| `requirements.txt` | core: numpy, pandas, scikit-learn, scipy, joblib | `pip install -e .` |
| `requirements-app.txt` | core + streamlit, plotly | `pip install -e ".[app]"` |
| `requirements-data.txt` | core + requests, tqdm (for `download_data.py`) | `pip install -e ".[data]"` |
| `requirements-notebook.txt` | core + matplotlib, seaborn, wordcloud, nltk, jupyter | `pip install -e ".[notebook]"` |
| `requirements-transformers.txt` | torch, transformers, datasets (heavy, optional) | `pip install -r requirements-transformers.txt` |
| `requirements-dev.txt` | pytest, nbstripout | `pip install -e ".[dev]"` |

All pinned. The `pyproject.toml` extras carry the same dependencies as version
ranges for library-style installs; the `requirements*.txt` files are the exact,
reproducible pins.

---

## Data

`python download_data.py` is the **only** supported pipeline. It fetches the
[Stanford aclImdb v1](https://ai.stanford.edu/~amaas/data/sentiment/) tarball —
no credentials required — verifies its SHA-256, extracts it safely
(`tarfile.extractall(filter="data")`), and writes `data/imdb_reviews.csv`
(50,000 labelled reviews).

Everything it produces lands in `data/`, which is git-ignored. Point elsewhere
with `--output`, or `export IMDB_CSV_PATH=/path/to/reviews.csv` to train from
your own CSV (needs `review` and `sentiment`/`label` columns).

---

## Method, and what it fixes

The reported accuracy of the original version of this project was inflated by
construction. Four things changed:

**De-duplicate before splitting.** The raw corpus contains byte-identical
reviews. Sampling straight from it means the same text can land in both train
and test — the model gets scored on rows it memorised. `load_reviews()` drops
duplicates before anything else touches the data.

**Keep the negations.** NLTK's and scikit-learn's English stopword lists both
contain `no`, `not`, `nor`, `never`. Strip them from a *sentiment* task and
`"not good"` and `"good"` reduce to the same token. `ENGLISH_STOPWORDS` in
`preprocessing.py` is scikit-learn's list minus every negation and negated
contraction, and the vectoriser uses uni+bigrams so `not good` survives as its
own feature.

**Select on cross-validation, over the training split only.** Comparing Naive
Bayes against Logistic Regression on the test set and then reporting the
winner's test score is selection bias — the number is optimistic by
construction. `sentiment.train` grid-searches with 5-fold stratified CV on the
training data and touches the test set exactly once, at the end, with the model
CV already chose.

**Stratify, and print a baseline.** The split is stratified so the class ratio
holds, and every run prints a `DummyClassifier(strategy="most_frequent")` score
so "88% accuracy" can be read against the ~50% you get for free.

Every training run also writes an **error analysis** into the model bundle: the
most confidently wrong predictions in each direction, plus the strongest signed
TF-IDF weights. Section 7 of the notebook renders them.

---

## The Streamlit app runs the real model

`app.py` loads `models/sentiment_pipeline.joblib` and shows genuine
`predict_proba` output, alongside the test accuracy, ROC-AUC and training date
read out of the saved bundle. If no model has been trained it says so and stops
— there is deliberately **no** keyword-matching fallback, because a demo that
silently degrades while quoting a confidence score is worse than one that
refuses to run.

Point it at a different artifact with `SENTIMENT_MODEL_PATH=/path/to/x.joblib`.

---

## Development

```bash
pip install -e ".[app,data,notebook,dev]"
pytest                       # the full suite; needs neither network nor dataset
```

CI (`.github/workflows/ci.yml`) runs on every push and PR:

- `pytest` on Python 3.10–3.13 (Linux) and 3.12 (Windows)
- repo hygiene: requirements files are valid UTF-8 and actually resolve, no
  tracked file exceeds 5 MB, no notebook carries embedded outputs
- an end-to-end smoke test: train → persist → reload → predict, then boot the
  Streamlit app and hit its health endpoint
- a Docker image build, then `pytest` inside the container

### Notebook outputs

Notebooks are committed **stripped**, and CI enforces it — embedded base64 PNGs
turn every re-run into an unreviewable binary diff (the original notebook was
899 KB of them). Install the hook once:

```bash
pip install nbstripout && nbstripout --install
```

---

## Known follow-up for the repo owner

An 84 MB tarball and ~9,900 extracted corpus files used to be tracked in git.
They are now untracked and ignored, so clones stop *growing* — but the blobs are
still in history, so a fresh `git clone` still transfers them. Reclaiming that
space needs a history rewrite (`git filter-repo --path aclImdb --path
aclImdb_v1.tar.gz --invert-paths`) followed by a force-push, which invalidates
every existing clone and open PR. That is the owner's call, deliberately left
undone here.

---

## License

MIT — see [LICENSE](LICENSE).
