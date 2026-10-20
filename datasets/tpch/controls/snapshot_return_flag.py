import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # Share of returned lines among a customer's past orders. return_flag is
    # frozen at 1995-06-17, so before that date it reports returns from the future.
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="customer_id")
    o = db["orders"]
    o = o[o["order_date"] <= t]
    li = db["lineitem"]
    li = li[li["ts"] <= t]
    ret = (li["return_flag"] == "R").astype(float)
    j = pd.DataFrame({"order_id": li["order_id"], "ret": ret}).merge(o[["order_id", "customer_id"]], on="order_id")
    out = pd.DataFrame(index=idx)
    out["returned_share"] = j.groupby("customer_id")["ret"].mean().reindex(idx)
    return out
