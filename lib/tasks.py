"""Prediction tasks of PITBench: label functions, one or two per dataset.

A label for (entity, t) is computed from the FULL database over the window
(t, t + horizon] -- by definition it reads the future, so label code lives
here and is never passed to pit-check. Every task is declared in the
dataset's dataset.json ("tasks") with the same name.

All functions: (db, entity_ids, seed_time) -> pd.Series indexed by entity_ids.
"""
import pandas as pd


def _window(df, col, t, days):
    t = pd.Timestamp(t)
    return df[(df[col] > t) & (df[col] <= t + pd.Timedelta(days=days))]


def _count(df, key, idx):
    return df.groupby(key).size().reindex(idx).fillna(0)


def olist_product_sold_next_90d(db, ids, t):
    w = _window(db["order_items"], "ts", t, 90)
    return (_count(w, "product_id", pd.Index(ids)) > 0).astype(int)


def f1_driver_podium_next_60d(db, ids, t):
    w = _window(db["race_results"], "available_at", t, 60)
    pod = w[w["position_number"] <= 3]
    return (_count(pod, "driver_id", pd.Index(ids)) > 0).astype(int)


def f1_driver_dnf_next_60d(db, ids, t):
    w = _window(db["race_results"], "available_at", t, 60)
    dnf = w[w["position_text"].astype(str).isin(["DNF", "DNS", "DSQ", "NC"])]
    return (_count(dnf, "driver_id", pd.Index(ids)) > 0).astype(int)


def pkdd99_account_low_activity_next_90d(db, ids, t):
    w = _window(db["trans"], "date", t, 90)
    return (_count(w, "account_id", pd.Index(ids)) < 13).astype(int)


def adventureworks_product_sold_next_90d(db, ids, t):
    w = _window(db["sales_order_detail"], "ts", t, 90)
    return (_count(w, "product_id", pd.Index(ids)) > 0).astype(int)


def northwind_customer_orders_next_90d(db, ids, t):
    w = _window(db["orders"], "order_date", t, 90)
    return (_count(w, "customer_id", pd.Index(ids)) > 0).astype(int)


def sakila_customer_rentals_next_14d(db, ids, t):
    w = _window(db["rental"], "rental_date", t, 14)
    return _count(w, "customer_id", pd.Index(ids))


def movietweetings_user_active_next_180d(db, ids, t):
    w = _window(db["ratings"], "ts", t, 180)
    return (_count(w, "user_id", pd.Index(ids)) > 0).astype(int)


def tpch_customer_orders_next_180d(db, ids, t):
    w = _window(db["orders"], "order_date", t, 180)
    return (_count(w, "customer_id", pd.Index(ids)) > 0).astype(int)


LABELS = {name: fn for name, fn in globals().items() if callable(fn) and not name.startswith("_")
          and name not in {"pd"}}
