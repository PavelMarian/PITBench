"""PITBench extension to pit-check: the snapshot channel.

pit-check v1 has two levels: witness (truncate rows by their time) and canary
(perturb future rows and late-filled date columns). Neither can see a value
that is stored as its END-OF-DATA state in a row that legitimately existed
before t -- a final order status, a current list price, a career total, a
balance. PITBench declares such columns in the availability map under an
extra key that pit-check v1 ignores:

    "snapshot_cols": {"<table>": {"<column>": <as_of>}}

``as_of`` says from when the stored value is valid:
    null            -- only at the end of the data: perturbed at every t;
    "<column>"      -- a datetime column of the same table: perturbed in rows
                       where that column is missing or > t;
    "YYYY-MM-DD"    -- a fixed date (e.g. TPC-H CURRENTDATE): perturbed in
                       every row when t is before that date.

The channel reruns the candidate on a copy where those values are perturbed
(numbers shifted, strings/bools replaced by a sentinel, dates pushed past
the horizon) and reports the output columns that moved. Like canary, a hit
means the program read a value that is not knowable at t; unlike canary it
does not depend on the row being in the future.
"""
import pandas as pd

NUM_SHIFT = 7919.0
SENTINEL = "__snapshot_sentinel__"
DATE_TARGET = pd.Timestamp("2199-01-01")


def _mask(df, as_of, t):
    if as_of is None:
        return pd.Series(True, index=df.index)
    if as_of in df.columns:
        v = pd.to_datetime(df[as_of], errors="coerce")
        return v.isna() | (v > t)
    return pd.Series(pd.Timestamp(as_of) > t, index=df.index)


def perturb_snapshot(db: dict, seed_time, snapshot_cols: dict) -> dict:
    t = pd.Timestamp(seed_time)
    out = dict(db)
    for table, spec in (snapshot_cols or {}).items():
        if table not in db:
            raise ValueError(f"snapshot_cols names table {table!r} not in db")
        df = db[table].copy()
        for col, as_of in spec.items():
            if col not in df.columns:
                raise ValueError(f"snapshot_cols names column {table}.{col} that does not exist")
            m = _mask(df, as_of, t)
            if not m.any():
                continue
            s = df[col]
            if pd.api.types.is_bool_dtype(s) or not (pd.api.types.is_numeric_dtype(s)
                                                     or pd.api.types.is_datetime64_any_dtype(s)):
                s = s.astype(object)
                s[m] = SENTINEL
            elif pd.api.types.is_datetime64_any_dtype(s):
                s = s.copy()
                s[m] = DATE_TARGET
            else:
                s = s.astype("float64")
                s[m] = s[m] + NUM_SHIFT
            df[col] = s
        out[table] = df
    return out


def snapshot_diff(fn, db, entity_ids, seed_time, snapshot_cols, frames_equal, call_with_timeout, timeout=60):
    """Columns of fn's output that change when snapshot values unknowable at t are perturbed."""
    t = pd.Timestamp(seed_time)
    full = call_with_timeout(fn, db, entity_ids, t, timeout=timeout)
    pert = call_with_timeout(fn, perturb_snapshot(db, t, snapshot_cols), entity_ids, t, timeout=timeout)
    if full is None or pert is None or full.shape != pert.shape or list(full.columns) != list(pert.columns):
        return ["__shape__"] if not (full is None and pert is None) else []
    return [c for c in full.columns if not frames_equal(full[[c]], pert[[c]])]
