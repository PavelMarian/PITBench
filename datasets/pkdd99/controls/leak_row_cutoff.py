import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # Lifetime transaction statistics, no cutoff.
    idx = pd.Index(entity_ids, name="account_id")
    g = db["trans"].groupby("account_id")
    out = pd.DataFrame(index=idx)
    out["n_trans"] = g.size().reindex(idx).fillna(0)
    out["mean_amount"] = g["amount"].mean().reindex(idx)
    return out
