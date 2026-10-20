import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    idx = pd.Index(entity_ids, name="customer_id")
    o = db["orders"]
    out = pd.DataFrame(index=idx)
    out["lifetime_spend"] = o.groupby("customer_id")["total_price"].sum().reindex(idx).fillna(0)
    return out
