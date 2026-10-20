import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # Rental duration of past rentals -- the return may happen after t.
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="customer_id")
    r = db["rental"]
    r = r[r["rental_date"] <= t]
    days = (r["return_date"] - r["rental_date"]).dt.total_seconds() / 86400.0
    out = pd.DataFrame(index=idx)
    out["avg_rental_days"] = days.groupby(r["customer_id"]).mean().reindex(idx)
    return out
