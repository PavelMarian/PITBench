import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # The user's own ratings are cut at t, but movie popularity -- a property
    # of the neighbouring entity -- is computed over ALL ratings.
    t = pd.Timestamp(seed_time)
    idx = pd.Index(entity_ids, name="user_id")
    allr = db["ratings"]
    pop = allr.groupby("movie_id").size()
    mine = allr[(allr["ts"] <= t) & allr["user_id"].isin(idx)]
    mine = mine.assign(pop=mine["movie_id"].map(pop))
    out = pd.DataFrame(index=idx)
    out["mean_movie_popularity"] = mine.groupby("user_id")["pop"].mean().reindex(idx)
    return out
