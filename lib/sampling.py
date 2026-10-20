"""Entity sampling at a prediction time: the one rule every PITBench check uses.

An entity is eligible at t if it has at least one event in (t - eligible_days, t]
of the dataset's event table -- information available at t only, so sampling
itself cannot leak. Up to n eligible entities are drawn with a fixed seed.

Before this rule entities were drawn once from the whole id column, so most of
them were inactive at a given t (retired F1 drivers, dormant customers): labels
were trivially 0 and harm was not measurable (F1 AUC ~0.99).
"""
import numpy as np
import pandas as pd


def eligible_pool(db, t, events, time, key, eligible_days):
    t = pd.Timestamp(t)
    ev = db[events]
    m = (ev[time] <= t) & (ev[time] > t - pd.Timedelta(days=eligible_days))
    return np.sort(ev.loc[m, key].dropna().unique())


def sample_at(db, t, entity, n, seed=0):
    """entity = {"events", "time", "key", "eligible_days"} (datasets/<ds>/layers/expected.json)."""
    pool = eligible_pool(db, t, entity["events"], entity["time"], entity["key"], entity["eligible_days"])
    if len(pool) <= n:
        return list(pool)
    return list(np.random.RandomState(seed).choice(pool, size=n, replace=False))


def ids_by_seed(db, seeds, entity, n, seed=0):
    return {s: sample_at(db, s, entity, n, seed) for s in seeds}
