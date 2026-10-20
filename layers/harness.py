"""PITBench layer harness: one invariant, a different view of the database per layer.

Invariant (every layer): the layer's output for prediction time t does not
change when its input is replaced by what was knowable at t.

    layer      program interface (pandas/numpy only, pit-check sandbox)     views compared
    -------    -----------------------------------------------------------  -----------------------------------
    task       select_entities(db, seed_time) -> list of entity ids         selection: D vs D|t, canary_t, snapshot_t
               make_label(db, entity_ids, seed_time) -> DataFrame[label]    label: D vs D|(t+h), canary_(t+h),
                                                                             snapshot_(t+h), tie_t (rows at exactly t removed),
                                                                             tie_inject_t (each entity's latest past event moved
                                                                             to exactly t: still past, so a correct label is unchanged)
    features   get_features(db, entity_ids, seed_time)                      pit_check.py itself (witness, canary) + snapshot
    model      fit_predict(db, entity_ids, seed_time) -> DataFrame[score]   D vs D|t, canary_t, snapshot_t
    context    build_context(db, entity_ids, seed_time) -> DataFrame        D vs D|t, canary_t, snapshot_t
               predict(db, entity_ids, seed_time) -> DataFrame[score]       D vs D|t
    agent      fit_predict(...) as above                                     D vs anonymize(D) (ids hashed, dates shifted)

Entity ids are passed either as one list or as {seed: list} -- the benchmark samples
eligible entities per prediction time (lib/sampling.py).

Programs are loaded through pit_check.load_program (same AST filter and
restricted builtins as the feature checker); a program defining several
functions is loaded once per function.

Harm (model, context, agent): AUC of the scores computed on D ("backtest") vs on
D|t ("what the pipeline would have produced at t"), both against the true
labels of (t, t+h]. Their difference is the inflation a backtest reports.
"""
import hashlib

import numpy as np
import pandas as pd

from ext.snapshot_channel import perturb_snapshot


class Layers:
    def __init__(self, pit_check, core, db, amap, amap_raw, timeout=300):
        self.pc, self.core, self.db = pit_check, core, db
        self.time_cols = amap["time_cols"]
        self.amap = amap
        self.snapshot_cols = amap_raw.get("snapshot_cols", {})
        self.timeout = timeout

    # ------------------------------------------------------------ loading --
    def load(self, code, name):
        return self.pc.load_program(code + f"\n\nget_features = {name}\n", pd, np)

    # -------------------------------------------------------------- views --
    def trunc(self, t):
        return self.core.truncate(self.db, t, self.time_cols)

    def canary(self, t):
        return self.core.perturb_canary(self.db, t, self.time_cols,
                                        self_avail_cols=self.amap["self_availability_cols"],
                                        gatekeeper_cols=self.amap["gatekeeper_cols"],
                                        key_hints=self.amap["key_hints"])

    def snapshot(self, t):
        return perturb_snapshot(self.db, t, self.snapshot_cols)

    def tie(self, t):
        t = pd.Timestamp(t)
        out = {}
        for name, df in self.db.items():
            tc = self.time_cols.get(name)
            out[name] = df if (tc is None or tc not in df.columns) else df[df[tc] != t]
        return out

    def tie_inject(self, t, ids, key):
        """Move, for every entity in ids, its latest event (<= t) to exactly t, in every time-mapped
        table that has the entity key. The event stays in the past and no row is added or removed, so
        a label on (t, t+h] -- and one that also counts the whole past -- must not change; one defined
        on [t, t+h] does. (Adding a copy instead would change the past and flag past-counting labels.)"""
        t = pd.Timestamp(t)
        idx = pd.Index(ids)
        out = dict(self.db)
        for name, tc in self.time_cols.items():
            df = self.db.get(name)
            if df is None or key not in df.columns or tc not in df.columns:
                continue
            past = df[(df[tc] <= t) & df[key].isin(idx)]
            if past.empty:
                continue
            moved = past.sort_values(tc, kind="stable").groupby(key, sort=False).tail(1).index
            df = df.copy()
            df.loc[moved, tc] = t
            out[name] = df
        return out

    def call(self, fn, db, *args):
        return self.core.call_with_timeout(fn, db, *args, timeout=self.timeout)

    def diff(self, a, b):
        if a is None or b is None or not hasattr(a, "shape") or not hasattr(b, "shape"):
            return [] if (a is None and b is None) else ["__shape__"]
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            return ["__shape__"]
        return [c for c in a.columns if not self.core.frames_equal(a[[c]], b[[c]])]

    # ------------------------------------------------------------- layers --
    def check_task(self, code, ids, seeds, horizon_days, key=None):
        return self.check_task_fn(self.load(code, "select_entities"), self.load(code, "make_label"),
                                  ids, seeds, horizon_days, key=key)

    def check_task_fn(self, sel, lab, ids, seeds, horizon_days, key=None):
        rows = []
        for s in seeds:
            t = pd.Timestamp(s)
            ids_s = _ids(ids, s)
            th = t + pd.Timedelta(days=horizon_days)
            base_sel = sorted(map(str, self.call(sel, self.db, t)))
            r = {"seed": s}
            for view, d in (("witness", self.trunc(t)), ("canary", self.canary(t)), ("snapshot", self.snapshot(t))):
                r[f"selection_{view}"] = sorted(map(str, self.call(sel, d, t))) != base_sel
            base = self.call(lab, self.db, ids_s, t)
            views = [("witness", self.trunc(th)), ("canary", self.canary(th)),
                     ("snapshot", self.snapshot(th)), ("tie", self.tie(t))]
            if key is not None:
                views.append(("tie_inject", self.tie_inject(t, ids_s, key)))
            for view, d in views:
                r[f"label_{view}"] = bool(self.diff(base, self.call(lab, d, ids_s, t)))
            r["label_prevalence"] = _prevalence(base)
            rows.append(r)
        return rows

    def check_scores(self, code, name, ids, seeds, y_fn=None):
        return self.check_scores_fn(self.load(code, name), ids, seeds, y_fn=y_fn)

    def check_scores_fn(self, fn, ids, seeds, y_fn=None, score_col="score"):
        rows = []
        for s in seeds:
            t = pd.Timestamp(s)
            ids_s = _ids(ids, s)
            full = self.call(fn, self.db, ids_s, t)
            tr = self.call(fn, self.trunc(t), ids_s, t)
            r = {"seed": s,
                 "witness": bool(self.diff(full, tr)),
                 "canary": bool(self.diff(full, self.call(fn, self.canary(t), ids_s, t))),
                 "snapshot": bool(self.diff(full, self.call(fn, self.snapshot(t), ids_s, t)))}
            if y_fn is not None and score_col in getattr(full, "columns", []):
                y = y_fn(self.db, ids_s, t)
                r["label_prevalence"] = _prevalence(y)
                r["auc_backtest"] = auc(y, full[score_col].reindex(y.index))
                r["auc_deploy"] = auc(y, tr[score_col].reindex(y.index))
            rows.append(r)
        return rows

    def check_agent(self, code, ids, seeds, date_cols, shift_days=2555, salt="pitbench", y_fn=None):
        return self.check_agent_fn(self.load(code, "fit_predict"), ids, seeds, shift_days, salt, y_fn)

    def check_agent_fn(self, fn, ids_all, seeds, shift_days=2555, salt="pitbench", y_fn=None, score_col="score"):
        anon_db, idmap = anonymize(self.db, shift_days, salt)
        rows = []
        for s in seeds:
            t = pd.Timestamp(s)
            ids = _ids(ids_all, s)
            a = self.call(fn, self.db, ids, t)
            ids_a = [idmap(i) for i in ids]
            b = self.call(fn, anon_db, ids_a, t + pd.Timedelta(days=shift_days))
            b = b.reindex(pd.Index(ids_a)).copy()
            b.index = pd.Index(list(ids))           # map anonymised ids back, same order
            a = a.reindex(pd.Index(list(ids)))
            same = np.allclose(a[score_col].to_numpy(float), b[score_col].to_numpy(float), rtol=1e-6, atol=1e-9,
                               equal_nan=True)
            r = {"seed": s, "identity_or_calendar_dependence": not same}
            if y_fn is not None:
                y = y_fn(self.db, ids, t)
                r["auc_original"] = auc(y, a[score_col].reindex(y.index))
                r["auc_anonymized"] = auc(y, b[score_col].reindex(y.index))
            rows.append(r)
        return rows


# ---------------------------------------------------------------- helpers --
def _ids(ids, seed):
    return list(ids[seed]) if isinstance(ids, dict) else list(ids)


def _prevalence(lab):
    v = lab["label"] if hasattr(lab, "columns") and "label" in lab.columns else lab
    v = pd.Series(v).astype(float)
    return None if v.empty else round(float((v > 0).mean()), 4)


def auc(y, s):
    y = pd.Series(y).astype(float)
    s = pd.Series(s).astype(float)
    m = y.notna() & s.notna()
    y, s = y[m], s[m]
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return None
    r = s.rank(method="average")
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def anonymize(db, shift_days, salt):
    """Hash every *_id value (consistently across tables) and shift every datetime column."""
    shift = pd.Timedelta(days=shift_days)
    cache = {}

    def h(v):
        if pd.isna(v):
            return v
        k = str(v)
        if k not in cache:
            cache[k] = "a" + hashlib.sha1((salt + k).encode()).hexdigest()[:12]
        return cache[k]

    out = {}
    for name, df in db.items():
        df = df.copy()
        for c in df.columns:
            if c.endswith("_id"):
                uniq = pd.unique(df[c])
                mp = {u: h(u) for u in uniq}
                df[c] = df[c].map(mp)
            elif pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c] + shift
        out[name] = df
    return out, h
