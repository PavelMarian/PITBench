import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="product_id")
    det = db["sales_order_detail"]
    det = det[det["ts"] <= t]
    recent = det[det["ts"] > t - pd.Timedelta(days=180)]
    out = pd.DataFrame(index=idx)
    out["qty_180d"] = recent.groupby("product_id")["order_qty"].sum().reindex(idx).fillna(0)
    out["avg_unit_price_paid"] = det.groupby("product_id")["unit_price"].mean().reindex(idx)
    h = db["product_list_price_history"]
    h = h[(h["start_date"] <= t) & (h["end_date"].isna() | (h["end_date"] >= t))]
    out["list_price_asof"] = h.groupby("product_id")["list_price"].last().reindex(idx)
    return out
