# News Popularity Prediction

Predicting whether a Mashable news article will be popular on social media, from nothing but its raw HTML.

> Kaggle-style competition (DataLab Cup 1), Deep Learning course, National Tsing Hua University, Oct 2026

**Final result:** 0.615 cross-validated ROC AUC · 0.602 on the public leaderboard

## The task

- **Data:** 27,643 training articles and 11,847 test articles, each a raw HTML page from Mashable
- **Label:** popular (1) or not (−1), with the two classes roughly balanced
- **Metric:** ROC AUC

## Approach

```
raw HTML ──► field-by-field parsing ──► features ──► per-field logistic regressions ──┐
                                           │                                          │ out-of-fold predictions
                                           └──► metadata, time context, SVD ──► LightGBM (10 seeds) ──► prediction
```

1. **Parsing.** Instead of flattening each page into one string, BeautifulSoup extracts the title, author, publisher, publish time, channel, topics and body separately, plus counts of images, videos, links and paragraphs and title-style signals.
2. **Text features.** Lowercasing, NLTK stop-word removal and Porter stemming, then a separate TF-IDF (unigrams + bigrams) for each field. BoW, TF-IDF and feature hashing were compared first.
3. **Stacking.** A logistic regression is trained on each group of sparse features; its out-of-fold predictions become inputs to a LightGBM model alongside 110+ dense features.
4. **Final model.** LightGBM with fixed boosting rounds, averaged over 10 random seeds.

## Results

| Model | 5-fold CV AUC |
|---|---|
| Bag of words, raw text | 0.516 |
| TF-IDF, raw text | 0.539 |
| TF-IDF uni+bigrams, cleaned text | 0.549 |
| Logistic regression, categorical fields only (no text) | 0.564 |
| Logistic regression, all sparse features | 0.576 |
| LightGBM, metadata + time context + SVD | 0.611 |
| **LightGBM + stacked logistic regressions** | **0.615** |

| Submission | CV AUC | Public LB |
|---|---|---|
| v1: LightGBM + stacking | 0.607 (optimistic) | 0.588 |
| v3: + time-context features, fixed rounds, 10 seeds | 0.614 | 0.600 |
| **v4: + channel / author / topic time context** | **0.615** | **0.602** |

## What I learned

- **When and where an article is published matters more than what it says.** The best text-only model reached 0.549 AUC; publish time, channel and author alone beat it without reading a single word.
- **Weekends win.** 60–63% of weekend articles are popular versus 45–47% on weekdays.
- **Attention is a competition.** The more articles published in the same window, the less likely each one is to be popular. Time-context features (posting volume, recent popularity of nearby articles, and the same idea per channel, author and topic) gave the largest single gain, +0.009 AUC.
- **Topics are time-sensitive.** A topic's popularity in the surrounding week predicts better (0.568 AUC alone) than its all-time popularity (0.546).
- **Features beat tuning.** Seed averaging and hyperparameter changes together added about 0.001 AUC.

### Pitfalls I fixed

- **Label leakage in target encoding.** Author and topic popularity rates are computed out-of-fold, and every stacking layer uses the same folds.
- **Optimistic validation.** Early stopping on each validation fold inflated CV AUC by 0.003–0.004, so the final model trains for a fixed number of rounds.

The notebook also logs every experiment that did *not* help (sentiment scores, k-NN on TF-IDF, category-cross encodings, a full HTML-structure scan, and more), with the CV AUC of each.

## Repository layout

```
news_popularity.ipynb   Full report: every step, experiment and ablation, with outputs
train.py                The final pipeline as one script; writes submission.csv
requirements.txt
```

## Running it

The competition data is not included. Put `train.csv` and `test.csv` in `./data`, then:

```bash
pip install -r requirements.txt
python train.py            # trains the final model and writes submission.csv
```

or open `news_popularity.ipynb` to follow the full analysis. The first run parses all the HTML and caches it to `data/parsed_cache.pkl`, so later runs are much faster.

The notebook's explanations are written in Traditional Chinese; code comments are in English.
