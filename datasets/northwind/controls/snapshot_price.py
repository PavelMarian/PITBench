import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # Past orders valued at the products' CURRENT unit price, not the price
    # charged at the time (order_details.unit_price).
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="customer_id")
    o = db["orders"]
    o = o[o["order_date"] <= t]
    d = db["order_details"]
    d = d[d["ts"] <= t][["order_id", "product_id", "quantity"]]
    p = db["products"][["product_id", "unit_price"]]
    d = d.merge(p, on="product_id").merge(o[["order_id", "customer_id"]], on="order_id")
    d = d.assign(value=d["unit_price"] * d["quantity"])
    out = pd.DataFrame(index=idx)
    out["value_at_list_price"] = d.groupby("customer_id")["value"].sum().reindex(idx).fillna(0)
    return out
