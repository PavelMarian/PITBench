import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # The product's CURRENT list price instead of the price valid at t.
    idx = pd.Index(entity_ids, name="product_id")
    p = db["product"].set_index("product_id")
    out = pd.DataFrame(index=idx)
    out["list_price"] = p["list_price"].reindex(idx)
    return out
