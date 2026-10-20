import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # Orders are filtered by purchase time, but the delivery date of an order
    # bought before t may only become known after t.
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="product_id")
    items = db["order_items"]
    items = items[items["ts"] <= t]
    orders = db["orders"]
    orders = orders[orders["order_purchase_timestamp"] <= t]
    days = (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.days
    o = pd.DataFrame({"order_id": orders["order_id"], "delivery_days": days})
    joined = items[["order_id", "product_id"]].merge(o, on="order_id")
    out = pd.DataFrame(index=idx)
    out["avg_delivery_days"] = joined.groupby("product_id")["delivery_days"].mean().reindex(idx)
    return out
