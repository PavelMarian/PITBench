import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # Same feature as snapshot_status, computed from the derived event table:
    # an order counts as delivered only if its 'delivered' event is <= t.
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="product_id")
    items = db["order_items"]
    items = items[items["ts"] <= t]
    ev = db["order_status_events"]
    ev = ev[(ev["ts"] <= t) & (ev["status"] == "delivered")]
    items = items.assign(delivered=items["order_id"].isin(ev["order_id"]).astype(float))
    out = pd.DataFrame(index=idx)
    out["share_delivered"] = items.groupby("product_id")["delivered"].mean().reindex(idx)
    return out
