import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # Loans are correctly limited to those granted before t, but their status
    # is the status at the END of the data (A/B finished, C/D running).
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="account_id")
    loans = db["loan"]
    loans = loans[loans["granted_date"] <= t]
    bad = loans["status"].isin(["B", "D"]).astype(float)
    out = pd.DataFrame(index=idx)
    out["has_bad_loan"] = bad.groupby(loans["account_id"]).max().reindex(idx).fillna(0)
    return out
