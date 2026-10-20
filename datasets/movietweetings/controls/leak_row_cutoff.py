import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    idx = pd.Index(entity_ids, name="user_id")
    g = db["ratings"].groupby("user_id")
    out = pd.DataFrame(index=idx)
    out["n_ratings"] = g.size().reindex(idx).fillna(0)
    return out
