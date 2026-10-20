import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="product_id")
    items = db["order_items"]
    items = items[items["ts"] <= t]
    recent = items[items["ts"] > t - pd.Timedelta(days=90)]
    out = pd.DataFrame(index=idx)
    out["n_sold_90d"] = recent.groupby("product_id").size().reindex(idx).fillna(0)
    out["avg_price"] = items.groupby("product_id")["price"].mean().reindex(idx)
    rev = db["reviews"]
    rev = rev[rev["review_creation_date"] <= t]
    joined = items[["order_id", "product_id"]].merge(rev[["order_id", "review_score"]], on="order_id")
    out["avg_review"] = joined.groupby("product_id")["review_score"].mean().reindex(idx)
    return out
