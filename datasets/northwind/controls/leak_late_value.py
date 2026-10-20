import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="customer_id")
    o = db["orders"]
    o = o[o["order_date"] <= t]
    delay = (o["shipped_date"] - o["order_date"]).dt.days
    out = pd.DataFrame(index=idx)
    out["avg_ship_delay"] = delay.groupby(o["customer_id"]).mean().reindex(idx)
    return out
