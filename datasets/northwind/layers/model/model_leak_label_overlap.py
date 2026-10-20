import pandas as pd
import numpy as np

EVENTS = "orders"
TIME = "order_date"
KEY = "customer_id"
VALUE = 'freight'
H = 90
STEP = 30
NTRAIN = 4
ELIG = 365


def _feats(db, ids, t):
    idx = pd.Index(ids)
    ev = db[EVENTS]
    ev = ev[(ev[TIME] <= t) & ev[KEY].isin(idx)]
    rec = ev[ev[TIME] > t - pd.Timedelta(days=H)]
    f = pd.DataFrame(index=idx)
    f["n_all"] = ev.groupby(KEY).size().reindex(idx).fillna(0).astype(float)
    f["n_recent"] = rec.groupby(KEY).size().reindex(idx).fillna(0).astype(float)
    last = ev.groupby(KEY)[TIME].max().reindex(idx)
    f["days_since"] = (t - last).dt.days.astype(float).fillna(9999.0)
    if VALUE is not None:
        v = pd.to_numeric(rec[VALUE], errors="coerce")
        f["v_recent"] = v.groupby(rec[KEY]).sum().reindex(idx).fillna(0).astype(float)
    return f


def _count(db, ids, lo, hi, closed_left=False):
    idx = pd.Index(ids)
    ev = db[EVENTS]
    left = (ev[TIME] >= lo) if closed_left else (ev[TIME] > lo)
    w = ev[left & (ev[TIME] <= hi) & ev[KEY].isin(idx)]

    return w.groupby(KEY).size().reindex(idx).fillna(0)


def _eligible(db, t):
    ev = db[EVENTS]
    ev = ev[(ev[TIME] <= t) & (ev[TIME] > t - pd.Timedelta(days=ELIG))]
    return sorted(ev[KEY].dropna().unique().tolist(), key=str)


def _label(db, ids, t):
    c = _count(db, ids, t, t + pd.Timedelta(days=H))
    return (c > 0).astype(float)


def _ridge(X, y, Xq, lam=1.0):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.hstack([(X - mu) / sd, np.ones((X.shape[0], 1))])
    Zq = np.hstack([(Xq - mu) / sd, np.ones((Xq.shape[0], 1))])
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)
    return Zq @ w


def _train(db, ids, t, offsets, extra=None):
    X, y = [], []
    for off in offsets:
        ti = t - pd.Timedelta(days=off)
        f = _feats(db, ids, ti)
        if extra is not None:
            f = f.join(extra)
        X.append(f.to_numpy(float))
        y.append(_label(db, ids, ti).to_numpy(float))
    return np.vstack(X), np.concatenate(y)

def fit_predict(db, entity_ids, seed_time):
    # Training seeds every STEP days up to t: the labels of the latest
    # examples cover (t_i, t_i + h], which reaches past t (no embargo).
    t = pd.Timestamp(seed_time)
    ids = list(entity_ids)
    X, y = _train(db, ids, t, [STEP * (k + 1) for k in range(NTRAIN)])
    q = _feats(db, ids, t)
    return pd.DataFrame({"score": _ridge(X, y, q.to_numpy(float))}, index=pd.Index(ids))
