import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="account_id")
    tr = db["trans"]
    tr = tr[tr["date"] <= t]
    recent = tr[tr["date"] > t - pd.Timedelta(days=90)]
    out = pd.DataFrame(index=idx)
    g = recent.groupby("account_id")
    out["n_trans_90d"] = g.size().reindex(idx).fillna(0)
    out["amount_90d"] = g["amount"].sum().reindex(idx).fillna(0)
    last = tr.sort_values(["date", "trans_id"]).groupby("account_id")["balance"].last()
    out["last_balance"] = last.reindex(idx)
    loans = db["loan"]
    loans = loans[loans["granted_date"] <= t]
    out["loan_amount"] = loans.groupby("account_id")["amount"].sum().reindex(idx).fillna(0)
    acc = db["account"]
    acc = acc[acc["open_date"] <= pd.Timestamp(seed_time)].set_index("account_id")
    dist = db["district"].set_index("district_id")
    out["district_unemployment_1995"] = acc["district_id"].map(dist["unemployment_1995"]).reindex(idx)
    return out
