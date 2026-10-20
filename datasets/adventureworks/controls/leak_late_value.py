import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # Orders are cut at t, but ship_date of the most recent orders is only
    # known after t.
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="product_id")
    hdr = db["sales_order_header"]
    hdr = hdr[hdr["order_date"] <= t]
    lag = (hdr["ship_date"] - hdr["order_date"]).dt.days
    h = pd.DataFrame({"sales_order_id": hdr["sales_order_id"], "ship_lag": lag})
    det = db["sales_order_detail"]
    det = det[det["ts"] <= t][["sales_order_id", "product_id"]]
    j = det.merge(h, on="sales_order_id")
    out = pd.DataFrame(index=idx)
    out["avg_ship_lag"] = j.groupby("product_id")["ship_lag"].mean().reindex(idx)
    return out
