import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # The corpus' dominant mechanism: items filtered, reviews joined unfiltered.
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="product_id")
    items = db["order_items"]
    items = items[items["ts"] <= t]
    out = pd.DataFrame(index=idx)
    out["avg_price"] = items.groupby("product_id")["price"].mean().reindex(idx)
    joined = items[["order_id", "product_id"]].merge(db["reviews"][["order_id", "review_score"]], on="order_id")
    out["avg_review"] = joined.groupby("product_id")["review_score"].mean().reindex(idx)
    return out
