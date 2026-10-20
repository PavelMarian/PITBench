import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="customer_id")
    r = db["rental"]
    r = r[r["rental_date"] <= t]
    p = db["payment"]
    p = p[p["payment_date"] <= t]
    out = pd.DataFrame(index=idx)
    out["rentals_30d"] = r[r["rental_date"] > t - pd.Timedelta(days=30)].groupby("customer_id").size().reindex(idx).fillna(0)
    out["rentals_total"] = r.groupby("customer_id").size().reindex(idx).fillna(0)
    out["paid_total"] = p.groupby("customer_id")["amount"].sum().reindex(idx).fillna(0)
    return out
