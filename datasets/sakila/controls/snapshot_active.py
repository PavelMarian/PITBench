import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # 'active' is the customer's status at the end of the data.
    idx = pd.Index(entity_ids, name="customer_id")
    c = db["customer"].set_index("customer_id")
    out = pd.DataFrame(index=idx)
    out["active"] = c["active"].reindex(idx)
    return out
