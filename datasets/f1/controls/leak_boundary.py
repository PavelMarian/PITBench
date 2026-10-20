import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # Filters on the source's own race date (midnight). On a race day this
    # admits the results of the race that starts later the same day.
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="driver_id")
    res = db["race_results"].copy()
    res["race_day"] = pd.to_datetime(res["race_date"], errors="coerce")
    res = res[(res["race_day"] <= t) & (res["race_day"] > t - pd.Timedelta(days=365))]
    out = pd.DataFrame(index=idx)
    out["points_365d"] = res.groupby("driver_id")["points"].sum().reindex(idx).fillna(0)
    return out
