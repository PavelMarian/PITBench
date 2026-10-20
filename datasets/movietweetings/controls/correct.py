import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="user_id")
    r = db["ratings"]
    r = r[r["ts"] <= t]
    pop = r.groupby("movie_id").size()
    mine = r[r["user_id"].isin(idx)]
    mine = mine.assign(pop=mine["movie_id"].map(pop))
    g = mine.groupby("user_id")
    out = pd.DataFrame(index=idx)
    out["n_ratings"] = g.size().reindex(idx).fillna(0)
    out["mean_rating"] = g["rating"].mean().reindex(idx)
    out["mean_movie_popularity"] = g["pop"].mean().reindex(idx)
    return out
