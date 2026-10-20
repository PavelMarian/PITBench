import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # Standing orders carry no date: the table is the state at the end of the data.
    idx = pd.Index(entity_ids, name="account_id")
    so = db["standing_orders"]
    out = pd.DataFrame(index=idx)
    out["standing_order_amount"] = so.groupby("account_id")["amount"].sum().reindex(idx).fillna(0)
    return out
