import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # Lines of past orders, late-delivery share computed from receipt dates
    # that may lie after t.
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="customer_id")
    o = db["orders"]
    o = o[o["order_date"] <= t]
    li = db["lineitem"]
    li = li[li["ts"] <= t]
    days_late = (li["receipt_date"] - pd.to_datetime(li["commit_date"], errors="coerce")).dt.days
    j = pd.DataFrame({"order_id": li["order_id"], "days_late": days_late}).merge(o[["order_id", "customer_id"]], on="order_id")
    out = pd.DataFrame(index=idx)
    out["avg_days_late"] = j.groupby("customer_id")["days_late"].mean().reindex(idx)
    return out
