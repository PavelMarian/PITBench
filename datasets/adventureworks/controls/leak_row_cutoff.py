import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    idx = pd.Index(entity_ids, name="product_id")
    det = db["sales_order_detail"]
    out = pd.DataFrame(index=idx)
    out["qty_total"] = det.groupby("product_id")["order_qty"].sum().reindex(idx).fillna(0)
    return out
