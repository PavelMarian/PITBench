import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # District statistics for 1996, used at prediction times inside 1996.
    idx = pd.Index(entity_ids, name="account_id")
    acc = db["account"]
    acc = acc[acc["open_date"] <= pd.Timestamp(seed_time)].set_index("account_id")
    dist = db["district"].set_index("district_id")
    out = pd.DataFrame(index=idx)
    out["district_unemployment_1996"] = acc["district_id"].map(dist["unemployment_1996"]).reindex(idx)
    return out
