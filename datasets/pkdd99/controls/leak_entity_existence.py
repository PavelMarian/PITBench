import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # Static district attributes attached to every account -- including
    # accounts that are only opened after t (they do not exist yet at t).
    idx = pd.Index(entity_ids, name="account_id")
    acc = db["account"].set_index("account_id")
    dist = db["district"].set_index("district_id")
    out = pd.DataFrame(index=idx)
    out["district_salary"] = acc["district_id"].map(dist["average_salary"]).reindex(idx)
    return out
