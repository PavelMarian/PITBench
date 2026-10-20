import pandas as pd
import numpy as np


def get_features(db, entity_ids, seed_time):
    # Career totals stored on the driver row: computed at dump time over the
    # whole history, including every race after t.
    idx = pd.Index(entity_ids, name="driver_id")
    d = db["drivers"].set_index("driver_id")
    out = pd.DataFrame(index=idx)
    out["total_race_wins"] = d["total_race_wins"].reindex(idx)
    out["total_podiums"] = d["total_podiums"].reindex(idx)
    return out
