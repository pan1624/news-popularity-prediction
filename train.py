"""DataLab Cup 1: Predicting News Popularity — 產生 submission.csv 的程式碼
作者：潘昱翔 (Yu-Xiang Pan)
最終模型：LightGBM（平均 10 個 random seed），特徵包含 HTML 欄位、時間、時間脈絡、target encoding、SVD，
          以及 stacking 的 logistic regression 文字預測值。5-fold CV AUC 0.6153，public leaderboard 0.60221。
需要套件：numpy, pandas, scipy, scikit-learn, lightgbm, beautifulsoup4, lxml, nltk, matplotlib
"""

# ======================================================================
# 0. 環境設定
# ======================================================================
import os, re, time, pickle, warnings
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')
SEED, N_FOLDS = 42, 5
RUN_BASELINES = False   # section 2 comparison (BoW / TF-IDF / Hashing); set False to save time

DATA_DIR = Path(os.environ.get('DATA_DIR', 'data'))
if not (DATA_DIR / 'train.csv').exists():
    DATA_DIR = Path('.')
assert (DATA_DIR / 'train.csv').exists(), 'Put train.csv / test.csv in ./data or next to this notebook'
CACHE = DATA_DIR / 'parsed_cache.pkl'

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
y = (train['Popularity'] == 1).astype(int).values
print(train.shape, test.shape)
print(train['Popularity'].value_counts())
print(train.head())

# ======================================================================
# 1. 資料前處理
# ======================================================================
def text_of(tag):
    return tag.get_text(' ', strip=True) if tag is not None else ''

def parse_article(html):
    soup = BeautifulSoup(html, 'lxml')
    title = text_of(soup.find('h1'))
    author = text_of(soup.find(class_='author_name') or soup.find(class_='byline'))
    author = re.sub(r'^\s*by\s+', '', author, flags=re.I).strip().lower() or 'unknown'
    info = soup.find(class_='article-info') or soup
    pub = info.find('a', href=re.compile(r'/publishers/'))
    publisher = pub['href'].strip('/').split('/')[-1] if pub is not None else 'mashable'
    t = soup.find('time')
    article = soup.find('article')
    channel = (article.get('data-channel') if article is not None else None) or 'unknown'
    footer = soup.find('footer')
    topics = [a.get_text(strip=True).lower() for a in footer.find_all('a')] if footer is not None else []
    body = soup.find('section', class_='article-content') or soup.find('body') or soup
    content = body.get_text(' ', strip=True)
    words = content.split()
    title_words = title.split()
    return dict(
        title=title, author=author, channel=channel.strip().lower(), publisher=publisher,
        has_author_image=int(info.find(class_='author_image') is not None),
        has_author_geo=int(info.find(class_='author_geo') is not None),
        datetime=t.get('datetime') if t is not None else None,
        topics='|'.join(topics), content=content,
        # body structure
        n_words=len(words),
        n_unique_words=len({w.lower() for w in words}),
        n_chars=len(content),
        avg_word_len=np.mean([len(w) for w in words]) if words else 0.0,
        n_paragraphs=len(body.find_all('p')),
        n_links=len(body.find_all('a')),
        n_images=len(body.find_all('img')),
        n_videos=len(body.find_all(['iframe', 'video'])),
        n_blockquotes=len(body.find_all('blockquote')),
        n_lists=len(body.find_all(['ul', 'ol'])),
        n_bold=len(body.find_all(['strong', 'b'])),
        n_headings=len(body.find_all(['h2', 'h3', 'h4'])),
        n_figures=len(soup.find_all('figure')),
        n_topics=len(topics),
        n_question=content.count('?'),
        n_exclaim=content.count('!'),
        # title style
        title_n_words=len(title_words),
        title_n_chars=len(title),
        title_has_question=int('?' in title),
        title_has_exclaim=int('!' in title),
        title_has_colon=int(':' in title),
        title_has_digit=int(bool(re.search(r'\d', title))),
        title_upper_ratio=sum(w[:1].isupper() for w in title_words) / max(len(title_words), 1),
    )

if CACHE.exists():
    parsed_tr, parsed_te = pickle.load(open(CACHE, 'rb'))
else:
    t0 = time.time()
    parsed_tr = pd.DataFrame([parse_article(h) for h in train['Page content']])
    parsed_te = pd.DataFrame([parse_article(h) for h in test['Page content']])
    pickle.dump((parsed_tr, parsed_te), open(CACHE, 'wb'))
    print(f'parsed in {time.time() - t0:.0f}s')

print(parsed_tr[['title', 'author', 'publisher', 'channel', 'datetime', 'topics', 'n_words', 'n_images']].head())

# ======================================================================
# 1.2 時間特徵
# ======================================================================
def time_features(df):
    dt = pd.to_datetime(df['datetime'], utc=True, errors='coerce', format='mixed')
    out = pd.DataFrame(index=df.index)
    out['year'] = dt.dt.year
    out['month'] = dt.dt.month
    out['weekday'] = dt.dt.weekday
    out['hour'] = dt.dt.hour
    out['is_weekend'] = (dt.dt.weekday >= 5).astype(float).where(dt.notna())
    out['days'] = (dt - pd.Timestamp('2013-01-01', tz='UTC')).dt.total_seconds() / 86400
    return out

time_tr, time_te = time_features(parsed_tr), time_features(parsed_te)
print('missing datetime ratio:', round(time_tr['days'].isna().mean(), 4))
print(pd.Series(y).groupby(time_tr['weekday'].values).mean().rename('popular rate by weekday (0=Mon)').to_frame().T)

# ======================================================================
# 1.3 文字清理：移除 stop words 與 word stemming
# ======================================================================
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    STOP = set(stopwords.words('english'))
except Exception:
    STOP = set(ENGLISH_STOP_WORDS)

porter = PorterStemmer()
TOKEN_RE = re.compile(r'[a-z][a-z\']+')

@lru_cache(maxsize=None)
def stem(w):
    return porter.stem(w)

def clean(text):
    return ' '.join(stem(w) for w in TOKEN_RE.findall(text.lower()) if w not in STOP)

t0 = time.time()
for p in (parsed_tr, parsed_te):
    p['title_clean'] = p['title'].map(clean)
    p['content_clean'] = p['content'].map(clean)
print(f'cleaned in {time.time() - t0:.0f}s')
print(parsed_tr['title'][0])
print(parsed_tr['title_clean'][0])

# ======================================================================
# 2. 文字轉向量：BoW vs. TF-IDF vs. feature hashing
# ======================================================================
folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
results = {}

if RUN_BASELINES:
    raw_text = parsed_tr['title'] + ' ' + parsed_tr['content']
    clean_text = parsed_tr['title_clean'] + ' ' + parsed_tr['content_clean']
    setups = {
        'BoW, raw text':               (CountVectorizer(min_df=3), raw_text),
        'TF-IDF, raw text':            (TfidfVectorizer(min_df=3, sublinear_tf=True), raw_text),
        'TF-IDF, cleaned':             (TfidfVectorizer(min_df=3, sublinear_tf=True), clean_text),
        'TF-IDF (1,2)-gram, cleaned':  (TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True), clean_text),
        'Hashing 2^20, cleaned':       (HashingVectorizer(n_features=2**20, alternate_sign=False), clean_text),
    }
    for name, (vec, text) in setups.items():
        pipe = Pipeline([('vect', vec), ('clf', LogisticRegression(solver='liblinear', max_iter=1000))])
        s = cross_val_score(pipe, text, y, cv=folds, scoring='roc_auc')
        results[f'[text only] {name}'] = s.mean()
        print(f'{name:30s} AUC = {s.mean():.4f} (+/-{s.std():.4f})')

# ======================================================================
# 2.1 依欄位分開的 TF-IDF
# ======================================================================
def fit_vec(vec, tr, te):
    X = vec.fit_transform(pd.concat([tr, te]))
    return X[:len(tr)].tocsr(), X[len(tr):].tocsr()

def split_bar(s):
    return [t for t in s.split('|') if t]

Xc_tr, Xc_te = fit_vec(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=300_000,
                                       sublinear_tf=True, dtype=np.float32),
                       parsed_tr['content_clean'], parsed_te['content_clean'])
Xt_tr, Xt_te = fit_vec(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32),
                       parsed_tr['title_clean'], parsed_te['title_clean'])
Xp_tr, Xp_te = fit_vec(TfidfVectorizer(tokenizer=split_bar, lowercase=False, token_pattern=None,
                                       min_df=2, sublinear_tf=True, dtype=np.float32),
                       parsed_tr['topics'], parsed_te['topics'])

def cat_str(p, t):
    return ('ch=' + p['channel'] + '|au=' + p['author'] + '|pub=' + p['publisher']
            + '|wd=' + t['weekday'].astype(str) + '|hr=' + t['hour'].astype(str))

Xm_tr, Xm_te = fit_vec(CountVectorizer(tokenizer=split_bar, lowercase=False, token_pattern=None,
                                       binary=True, min_df=2, dtype=np.float32),
                       cat_str(parsed_tr, time_tr), cat_str(parsed_te, time_te))

for name, X in [('content', Xc_tr), ('title', Xt_tr), ('topics', Xp_tr), ('categorical', Xm_tr)]:
    print(f'{name:12s} {X.shape}')

# ======================================================================
# 3. 建立分類器
# ======================================================================
def cv_lr(Xtr, Xte, Cs=(0.1, 0.3, 1.0, 3.0)):
    best = None
    for C in Cs:
        oof, pred = np.zeros(Xtr.shape[0]), np.zeros(Xte.shape[0])
        for tr, va in folds:
            m = LogisticRegression(C=C, solver='liblinear', max_iter=2000)
            m.fit(Xtr[tr], y[tr])
            oof[va] = m.predict_proba(Xtr[va])[:, 1]
            pred += m.predict_proba(Xte)[:, 1] / N_FOLDS
        auc = roc_auc_score(y, oof)
        print(f'   C={C:<5} AUC={auc:.4f}')
        if best is None or auc > best[0]:
            best = (auc, C, oof, pred)
    return best

blocks = {
    'content':     (Xc_tr, Xc_te),
    'title':       (Xt_tr, Xt_te),
    'topics':      (Xp_tr, Xp_te),
    'categorical': (Xm_tr, Xm_te),
    'all':         (sparse.hstack([Xc_tr, Xt_tr, Xp_tr, Xm_tr]).tocsr(),
                    sparse.hstack([Xc_te, Xt_te, Xp_te, Xm_te]).tocsr()),
}
lr_oof, lr_pred = {}, {}
for name, (Xtr, Xte) in blocks.items():
    print(f'LR [{name}]')
    auc, C, lr_oof[name], lr_pred[name] = cv_lr(Xtr, Xte)
    results[f'LR {name}'] = auc
print(pd.Series(results).round(4))

# ======================================================================
# 3.2 Model B：以 metadata 與文字分數為特徵的 LightGBM
# ======================================================================
def oof_target_encode(tr_col, te_col, alpha=20):
    prior = y.mean()
    def enc(idx):
        s = pd.DataFrame({'k': tr_col.values[idx], 'y': y[idx]}).groupby('k')['y'].agg(['sum', 'count'])
        return (s['sum'] + alpha * prior) / (s['count'] + alpha)
    oof = np.full(len(tr_col), prior)
    for tr, va in folds:
        oof[va] = tr_col.iloc[va].map(enc(tr)).fillna(prior).values
    return oof, te_col.map(enc(np.arange(len(tr_col)))).fillna(prior).values

meta_cols = [c for c in parsed_tr.columns if c.startswith(('n_', 'title_', 'avg_', 'has_')) and c != 'title_clean']

def dense(p, t):
    d = pd.concat([p[meta_cols], t], axis=1)
    d['unique_ratio'] = p['n_unique_words'] / p['n_words'].clip(lower=1)
    d['images_per_word'] = p['n_images'] / p['n_words'].clip(lower=1)
    d['links_per_word'] = p['n_links'] / p['n_words'].clip(lower=1)
    return d

D_tr, D_te = dense(parsed_tr, time_tr), dense(parsed_te, time_te)

channels = pd.Categorical(pd.concat([parsed_tr['channel'], parsed_te['channel']])).categories
D_tr['channel'] = pd.Categorical(parsed_tr['channel'], categories=channels)
D_te['channel'] = pd.Categorical(parsed_te['channel'], categories=channels)

author_freq = pd.concat([parsed_tr['author'], parsed_te['author']]).value_counts()
D_tr['author_freq'] = parsed_tr['author'].map(author_freq).values
D_te['author_freq'] = parsed_te['author'].map(author_freq).values
D_tr['author_te'], D_te['author_te'] = oof_target_encode(parsed_tr['author'], parsed_te['author'])
D_tr['publisher_te'], D_te['publisher_te'] = oof_target_encode(parsed_tr['publisher'], parsed_te['publisher'])

# topic target encoding: each article has several topics -> aggregate their (out-of-fold) popularity rates
def topic_stats(topic_lists, enc, cnt):
    out = np.full((len(topic_lists), 5), np.nan)
    for i, ts in enumerate(topic_lists):
        v = [enc[t] for t in ts if t in enc]
        if v:
            out[i] = (np.mean(v), np.max(v), np.min(v), np.average(v, weights=[cnt[t] for t in ts if t in enc]), len(v))
        else:
            out[i, 4] = 0
    return out

def fit_topic_enc(idx, alpha=10):
    s = pd.DataFrame({'t': T_tr.iloc[idx].values, 'y': y[idx]}).explode('t').dropna()
    g = s.groupby('t')['y'].agg(['sum', 'count'])
    return ((g['sum'] + alpha * y.mean()) / (g['count'] + alpha)).to_dict(), g['count'].to_dict()

T_tr, T_te = parsed_tr['topics'].map(split_bar), parsed_te['topics'].map(split_bar)
TS_tr = np.zeros((len(y), 5))
for tr, va in folds:
    TS_tr[va] = topic_stats(T_tr.iloc[va], *fit_topic_enc(tr))
TS_te = topic_stats(T_te, *fit_topic_enc(np.arange(len(y))))
for j, c in enumerate(['topic_te_mean', 'topic_te_max', 'topic_te_min', 'topic_te_wmean', 'topic_known']):
    D_tr[c], D_te[c] = TS_tr[:, j], TS_te[:, j]

svd = TruncatedSVD(min(50, Xc_tr.shape[1] - 1), random_state=SEED).fit(sparse.vstack([Xc_tr, Xc_te]))
for i, (a, b) in enumerate(zip(svd.transform(Xc_tr).T, svd.transform(Xc_te).T)):
    D_tr[f'svd_{i}'], D_te[f'svd_{i}'] = a, b

# time context: how crowded the publishing slot was, and how popular nearby articles were
days_tr, days_te = time_tr['days'].values, time_te['days'].values
all_days = np.sort(np.concatenate([days_tr, days_te]))   # publish times only, no labels
for w in [0.125, 0.25, 0.5, 1, 3, 7]:                    # +-3 hours ... +-7 days
    for d, days in ((D_tr, days_tr), (D_te, days_te)):
        d[f'vol_pm{w}d'] = np.searchsorted(all_days, days + w, 'right') - np.searchsorted(all_days, days - w, 'left')

def window_rate(fit_idx, q_days, w=1.0, alpha=20):
    order = fit_idx[np.argsort(days_tr[fit_idx])]
    d, cy = days_tr[order], np.concatenate([[0], np.cumsum(y[order])])
    lo, hi = np.searchsorted(d, q_days - w, 'left'), np.searchsorted(d, q_days + w, 'right')
    return (cy[hi] - cy[lo] + alpha * y.mean()) / (hi - lo + alpha)

D_tr['rate_pm1d'] = 0.0
for tr, va in folds:   # out-of-fold, like the other target encodings
    D_tr.loc[va, 'rate_pm1d'] = window_rate(tr, days_tr[va])
D_te['rate_pm1d'] = window_rate(np.arange(len(y)), days_te)

# group-level time context: the same idea per channel, author and topic
n_tr = len(y)
days_all = np.concatenate([days_tr, days_te])
y_all = np.concatenate([y, np.zeros(len(days_te))])          # test labels are never read (test rows are never fit rows)
tr_rows, te_rows = np.arange(n_tr), np.arange(n_tr, len(days_all))

def window_rate_by_key(keys, d, yy, fit, qry, w, alpha=10):
    # smoothed popularity rate of fit rows that share the key and were published within +-w days
    out = np.full(len(qry), y.mean())
    f = pd.DataFrame({'k': keys[fit], 'd': d[fit], 'y': yy[fit]}).sort_values('d')
    look = {k: (g['d'].values, np.r_[0, g['y'].cumsum().values]) for k, g in f.groupby('k')}
    q = pd.DataFrame({'k': keys[qry], 'd': d[qry], 'pos': np.arange(len(qry))})
    for k, g in q.groupby('k'):
        if k in look:
            fd, cy = look[k]
            lo = np.searchsorted(fd, g['d'].values - w, 'left')
            hi = np.searchsorted(fd, g['d'].values + w, 'right')
            out[g['pos'].values] = (cy[hi] - cy[lo] + alpha * y.mean()) / (hi - lo + alpha)
    return out

def volume_by_key(keys, w):
    out = np.zeros(len(keys))
    for k, g in pd.DataFrame({'k': keys, 'd': days_all}).groupby('k'):
        d = np.sort(g['d'].values)
        out[g.index] = np.searchsorted(d, g['d'].values + w, 'right') - np.searchsorted(d, g['d'].values - w, 'left')
    return out

def add_oof(name, fn):
    # fn(fit_rows, query_rows) -> values; out-of-fold for train, all train rows for test
    D_tr[name] = 0.0
    for tr, va in folds:
        D_tr.loc[va, name] = fn(tr, va)
    D_te[name] = fn(tr_rows, te_rows)

def add_unsupervised(name, values):
    D_tr[name], D_te[name] = values[:n_tr], values[n_tr:]

chan_all = np.concatenate([parsed_tr['channel'].values, parsed_te['channel'].values])
auth_all = np.concatenate([parsed_tr['author'].values, parsed_te['author'].values])
for w in (3, 7, 30):
    add_oof(f'ch_rate_pm{w}d', lambda f, q, w=w: window_rate_by_key(chan_all, days_all, y_all, f, q, w))
for w in (0.5, 3):
    add_unsupervised(f'ch_vol_pm{w}d', volume_by_key(chan_all, w))
for w in (30, 90):
    add_oof(f'au_rate_pm{w}d', lambda f, q, w=w: window_rate_by_key(auth_all, days_all, y_all, f, q, w))
add_unsupervised('au_vol_pm7d', volume_by_key(auth_all, 7))
a = pd.DataFrame({'a': auth_all, 'd': days_all}).sort_values(['a', 'd'])
add_unsupervised('au_gap_prev', a.groupby('a')['d'].diff().sort_index().values)        # days since the author's previous article
add_unsupervised('au_gap_next', (-a.groupby('a')['d'].diff(-1)).sort_index().values)   # days until the author's next article

# topic trend: popularity of each topic within +-7 days, aggregated over the article's topics
topics_all = list(T_tr) + list(T_te)
ex_row = np.repeat(np.arange(len(days_all)), [len(t) for t in topics_all])
ex_key = np.array([t for ts in topics_all for t in ts], dtype=object)
def topic_trend(fit_rows, q_rows, w=7):
    fit, qry = np.where(np.isin(ex_row, fit_rows))[0], np.where(np.isin(ex_row, q_rows))[0]
    v = window_rate_by_key(ex_key, days_all[ex_row], y_all[ex_row], fit, qry, w)
    return pd.DataFrame({'row': ex_row[qry], 'v': v}).groupby('row')['v'].agg(['mean', 'max']).reindex(q_rows).values
D_tr['topic_trend_mean'], D_tr['topic_trend_max'] = np.nan, np.nan
for tr, va in folds:
    D_tr.loc[va, ['topic_trend_mean', 'topic_trend_max']] = topic_trend(tr, va)
D_te[['topic_trend_mean', 'topic_trend_max']] = topic_trend(tr_rows, te_rows)

for name in lr_oof:
    D_tr[f'lr_{name}'], D_te[f'lr_{name}'] = lr_oof[name], lr_pred[name]
print(D_tr.shape)

# ======================================================================
# 3.2 Model B：以 metadata 與文字分數為特徵的 LightGBM
# ======================================================================
LGB_PARAMS = dict(objective='binary', learning_rate=0.02, num_leaves=31, min_child_samples=50,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
LGB_ROUNDS = 800       # fixed: CV AUC is flat between 200 and 1200 rounds (see 3.3)
SEEDS = tuple(range(10))   # averaging 10 seeds of the same model (see 3.3)

def cv_lgb(Xtr, Xte, rounds=LGB_ROUNDS, seeds=SEEDS):
    oof, pred = np.zeros(len(Xtr)), np.zeros(len(Xte))
    imp = pd.Series(0.0, index=Xtr.columns)
    for seed in seeds:
        params = {**LGB_PARAMS, 'seed': seed}
        for tr, va in folds:
            m = lgb.train(params, lgb.Dataset(Xtr.iloc[tr], y[tr]), rounds)
            oof[va] += m.predict(Xtr.iloc[va]) / len(seeds)
        m = lgb.train(params, lgb.Dataset(Xtr, y), rounds)   # refit on all training data for the test set
        pred += m.predict(Xte) / len(seeds)
        imp += pd.Series(m.feature_importance('gain'), index=Xtr.columns)
    return roc_auc_score(y, oof), oof, pred, imp.sort_values(ascending=False)

no_stack = [c for c in D_tr.columns if not c.startswith('lr_')]
results['LGBM metadata + SVD (no stacking)'] = cv_lgb(D_tr[no_stack], D_te[no_stack], seeds=(0,))[0]
results['LGBM + stacked LR'], lgb_oof, lgb_pred, importance = cv_lgb(D_tr, D_te)
print(pd.Series(results).round(4))

# ======================================================================
# 3.4 特徵重要度（LightGBM gain）
# ======================================================================
importance.head(25).iloc[::-1].plot.barh(figsize=(7, 7), title='Top 25 features by gain')
import matplotlib.pyplot as plt; plt.tight_layout(); plt.savefig('feature_importance.png', dpi=150)

# ======================================================================
# 3.5 最終預測與上傳檔案
# ======================================================================
submission = pd.DataFrame({'Id': test['Id'], 'Popularity': lgb_pred})
submission.to_csv('submission.csv', index=False)
print(submission.head())

# ======================================================================
# 4. 結果總表（5-fold CV AUC）
# ======================================================================
print(pd.Series(results, name='CV AUC').round(4).to_frame())

print('wrote submission.csv')
