import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    idx = pd.Index(entity_ids, name="customer_id")
    o = db["orders"]
    out = pd.DataFrame(index=idx)
    out["n_orders"] = o.groupby("customer_id").size().reindex(idx).fillna(0)
    return out
