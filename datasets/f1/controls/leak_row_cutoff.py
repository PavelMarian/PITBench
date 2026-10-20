import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # Career points from race_results with no time filter at all.
    idx = pd.Index(entity_ids, name="driver_id")
    res = db["race_results"]
    out = pd.DataFrame(index=idx)
    out["career_points"] = res.groupby("driver_id")["points"].sum().reindex(idx).fillna(0)
    return out
