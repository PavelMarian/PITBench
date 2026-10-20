import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # Share of a product's past orders that are 'delivered' -- read from the
    # final status column (a snapshot), not from the status as of t.
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="product_id")
    items = db["order_items"]
    items = items[items["ts"] <= t]
    orders = db["orders"]
    orders = orders[orders["order_purchase_timestamp"] <= t]
    o = pd.DataFrame({"order_id": orders["order_id"],
                      "delivered": (orders["order_status"] == "delivered").astype(float)})
    joined = items[["order_id", "product_id"]].merge(o, on="order_id")
    out = pd.DataFrame(index=idx)
    out["share_delivered"] = joined.groupby("product_id")["delivered"].mean().reindex(idx)
    return out
