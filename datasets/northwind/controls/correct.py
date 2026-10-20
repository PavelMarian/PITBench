import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="customer_id")
    o = db["orders"]
    o = o[o["order_date"] <= t]
    d = db["order_details"]
    d = d[d["ts"] <= t]
    d = d.assign(revenue=d["unit_price"] * d["quantity"] * (1 - d["discount"]))
    rev = d.merge(o[["order_id", "customer_id"]], on="order_id").groupby("customer_id")["revenue"].sum()
    out = pd.DataFrame(index=idx)
    out["n_orders"] = o.groupby("customer_id").size().reindex(idx).fillna(0)
    out["revenue"] = rev.reindex(idx).fillna(0)
    out["avg_freight"] = o.groupby("customer_id")["freight"].mean().reindex(idx)
    return out
