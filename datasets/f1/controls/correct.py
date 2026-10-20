import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="driver_id")
    res = db["race_results"]
    res = res[(res["available_at"] <= t) & (res["available_at"] > t - pd.Timedelta(days=365))]
    out = pd.DataFrame(index=idx)
    g = res.groupby("driver_id")
    out["races_365d"] = g.size().reindex(idx).fillna(0)
    out["points_365d"] = g["points"].sum().reindex(idx).fillna(0)
    out["avg_finish_365d"] = g["position_number"].mean().reindex(idx)
    q = db["qualifying_results"]
    q = q[(q["available_at"] <= t) & (q["available_at"] > t - pd.Timedelta(days=365))]
    out["avg_quali_365d"] = q.groupby("driver_id")["position_number"].mean().reindex(idx)
    return out
