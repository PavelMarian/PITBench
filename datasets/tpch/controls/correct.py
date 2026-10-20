import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="customer_id")
    o = db["orders"]
    o = o[o["order_date"] <= t]
    recent = o[o["order_date"] > t - pd.Timedelta(days=365)]
    out = pd.DataFrame(index=idx)
    out["n_orders_365d"] = recent.groupby("customer_id").size().reindex(idx).fillna(0)
    out["spend_365d"] = recent.groupby("customer_id")["total_price"].sum().reindex(idx).fillna(0)
    li = db["lineitem"]
    li = li[(li["ts"] <= t) & (li["receipt_date"] <= t)]
    late = (li["receipt_date"] > pd.to_datetime(li["commit_date"], errors="coerce")).astype(float)
    j = pd.DataFrame({"order_id": li["order_id"], "late": late}).merge(o[["order_id", "customer_id"]], on="order_id")
    out["late_share_received"] = j.groupby("customer_id")["late"].mean().reindex(idx)
    return out
