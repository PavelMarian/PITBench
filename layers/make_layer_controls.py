#!/usr/bin/env python3
"""Generate the layer controls of every dataset from one set of templates.

    python -I layers/make_layer_controls.py

Writes datasets/<ds>/layers/<layer>/<control>.py and datasets/<ds>/layers/expected.json.
Each control is a self-contained pandas/numpy program (it must pass the
pit-check sandbox); dataset specifics are substituted from SPECS below.
"""
import json
import sys
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# event table, its time column, entity key, a numeric value (or None), task horizon,
# step between training seeds, number of training seeds, label expression on the
# window count `c`, calendar date for the calendar-prior control, whether event
# the tie view is expected to fire (True), or is not exercised (None: no events
# exactly at t, or events at t never flip the binary label -- measured, see
# README), optional snapshot label, ELIG = eligibility window in days (an entity is
# predicted for at t only if it had an event in (t - ELIG, t]; uses data <= t only),
# LFILT = optional filter on the rows of the label window (Python expression on `w`).
SPECS = {
 "olist": dict(ELIG=180, LFILT=None, E="order_items", T="ts", K="product_id", V="price", H=90, STEP=30, NTRAIN=4, LABEL="c > 0",
               CAL="2018-03-01", DAY=None,
               SNAP='''    o = db["orders"]
    it = db["order_items"][["order_id", "product_id"]]
    j = it.merge(o[["order_id", "order_status"]], on="order_id")
    ok = (j["order_status"] == "delivered").astype(float).groupby(j["product_id"]).min()
    return pd.DataFrame({"label": ok.reindex(pd.Index(entity_ids)).fillna(0)})'''),
 "f1": dict(ELIG=365, LFILT='w["position_number"] <= 3', E="race_results", T="available_at", K="driver_id", V="points", H=60, STEP=20, NTRAIN=4, LABEL="c > 0",
            CAL="2021-01-01", DAY=None,
            SNAP='''    d = db["drivers"].set_index("driver_id")
    won = (d["total_race_wins"] > 0).astype(float)
    return pd.DataFrame({"label": won.reindex(pd.Index(entity_ids)).fillna(0)})'''),
 "pkdd99": dict(ELIG=180, LFILT=None, E="trans", T="date", K="account_id", V="amount", H=90, STEP=30, NTRAIN=4, LABEL="c < 13",
                CAL="1997-06-01", DAY=None,
                SNAP='''    loans = db["loan"]
    bad = loans["status"].isin(["B", "D"]).astype(float).groupby(loans["account_id"]).max()
    return pd.DataFrame({"label": bad.reindex(pd.Index(entity_ids)).fillna(0)})'''),
 "adventureworks": dict(ELIG=180, LFILT=None, E="sales_order_detail", T="ts", K="product_id", V="order_qty", H=90, STEP=30, NTRAIN=4,
                        LABEL="c > 0", CAL="2024-01-01", DAY=None, SNAP=None),
 "northwind": dict(ELIG=365, LFILT=None, E="orders", T="order_date", K="customer_id", V="freight", H=90, STEP=30, NTRAIN=4, LABEL="c > 0",
                   CAL="1997-09-01", DAY=None, SNAP=None),
 "sakila": dict(ELIG=30, LFILT=None, E="rental", T="rental_date", K="customer_id", V=None, H=14, STEP=7, NTRAIN=4, LABEL="c > 0",
                CAL="2005-07-10", DAY=None,
                SNAP='''    c = db["customer"].set_index("customer_id")
    churned = (c["active"] == 0).astype(float)
    return pd.DataFrame({"label": churned.reindex(pd.Index(entity_ids)).fillna(0)})'''),
 "movietweetings": dict(ELIG=365, LFILT=None, E="ratings", T="ts", K="user_id", V="rating", H=180, STEP=60, NTRAIN=4, LABEL="c > 0",
                        CAL="2018-01-01", DAY=None, SNAP=None),
 "tpch": dict(ELIG=365, LFILT=None, E="orders", T="order_date", K="customer_id", V="total_price", H=180, STEP=60, NTRAIN=4, LABEL="c > 0",
              CAL="1995-06-17", DAY=True,
              SNAP='''    li = db["lineitem"][["order_id", "return_flag"]]
    o = db["orders"][["order_id", "customer_id"]]
    j = li.merge(o, on="order_id")
    ret = (j["return_flag"] == "R").astype(float).groupby(j["customer_id"]).max()
    return pd.DataFrame({"label": ret.reindex(pd.Index(entity_ids)).fillna(0)})'''),
}

COMMON = '''import pandas as pd
import numpy as np

EVENTS = "$E"
TIME = "$T"
KEY = "$K"
VALUE = $V
H = $H
STEP = $STEP
NTRAIN = $NTRAIN
ELIG = $ELIG


def _feats(db, ids, t):
    idx = pd.Index(ids)
    ev = db[EVENTS]
    ev = ev[(ev[TIME] <= t) & ev[KEY].isin(idx)]
    rec = ev[ev[TIME] > t - pd.Timedelta(days=H)]
    f = pd.DataFrame(index=idx)
    f["n_all"] = ev.groupby(KEY).size().reindex(idx).fillna(0).astype(float)
    f["n_recent"] = rec.groupby(KEY).size().reindex(idx).fillna(0).astype(float)
    last = ev.groupby(KEY)[TIME].max().reindex(idx)
    f["days_since"] = (t - last).dt.days.astype(float).fillna(9999.0)
    if VALUE is not None:
        v = pd.to_numeric(rec[VALUE], errors="coerce")
        f["v_recent"] = v.groupby(rec[KEY]).sum().reindex(idx).fillna(0).astype(float)
    return f


def _count(db, ids, lo, hi, closed_left=False):
    idx = pd.Index(ids)
    ev = db[EVENTS]
    left = (ev[TIME] >= lo) if closed_left else (ev[TIME] > lo)
    w = ev[left & (ev[TIME] <= hi) & ev[KEY].isin(idx)]
$LFILT_LINE
    return w.groupby(KEY).size().reindex(idx).fillna(0)


def _eligible(db, t):
    ev = db[EVENTS]
    ev = ev[(ev[TIME] <= t) & (ev[TIME] > t - pd.Timedelta(days=ELIG))]
    return sorted(ev[KEY].dropna().unique().tolist(), key=str)


def _label(db, ids, t):
    c = _count(db, ids, t, t + pd.Timedelta(days=H))
    return ($LABEL).astype(float)


def _ridge(X, y, Xq, lam=1.0):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.hstack([(X - mu) / sd, np.ones((X.shape[0], 1))])
    Zq = np.hstack([(Xq - mu) / sd, np.ones((Xq.shape[0], 1))])
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)
    return Zq @ w


def _train(db, ids, t, offsets, extra=None):
    X, y = [], []
    for off in offsets:
        ti = t - pd.Timedelta(days=off)
        f = _feats(db, ids, ti)
        if extra is not None:
            f = f.join(extra)
        X.append(f.to_numpy(float))
        y.append(_label(db, ids, ti).to_numpy(float))
    return np.vstack(X), np.concatenate(y)
'''

# ---------------------------------------------------------------- task layer
TASK = {
 "task_correct": ('''
def select_entities(db, seed_time):
    return _eligible(db, pd.Timestamp(seed_time))


def make_label(db, entity_ids, seed_time):
    return pd.DataFrame({"label": _label(db, entity_ids, pd.Timestamp(seed_time))})
''', "correct: entities seen by t, label on (t, t+h]"),
 "task_leak_selection_future": ('''
def select_entities(db, seed_time):
    # Survivorship: every entity that ever appears in the event table.
    ev = db[EVENTS]
    return sorted(ev[KEY].dropna().unique().tolist(), key=str)


def make_label(db, entity_ids, seed_time):
    return pd.DataFrame({"label": _label(db, entity_ids, pd.Timestamp(seed_time))})
''', "entity selection uses the whole history"),
 "task_leak_label_no_cap": ('''
def select_entities(db, seed_time):
    return _eligible(db, pd.Timestamp(seed_time))


def make_label(db, entity_ids, seed_time):
    # No upper bound on the label window: everything after t counts.
    t = pd.Timestamp(seed_time)
    c = _count(db, entity_ids, t, pd.Timestamp("2262-01-01"))
    return pd.DataFrame({"label": ($LABEL).astype(float)})
''', "label window without the horizon cap"),
 "task_leak_label_boundary": ('''
def select_entities(db, seed_time):
    return _eligible(db, pd.Timestamp(seed_time))


def make_label(db, entity_ids, seed_time):
    # [t, t+h] instead of (t, t+h]: an event stamped exactly t is both a
    # feature (<= t) and a label.
    t = pd.Timestamp(seed_time)
    c = _count(db, entity_ids, t, t + pd.Timedelta(days=H), closed_left=True)
    return pd.DataFrame({"label": ($LABEL).astype(float)})
''', "label window closed on the left"),
}
TASK["task_correct_count"] = ('''
def select_entities(db, seed_time):
    return _eligible(db, pd.Timestamp(seed_time))


def make_label(db, entity_ids, seed_time):
    t = pd.Timestamp(seed_time)
    return pd.DataFrame({"label": _count(db, entity_ids, t, t + pd.Timedelta(days=H)).astype(float)})
''', "count label on (t, t+h]")
TASK["task_leak_label_boundary_count"] = ('''
def select_entities(db, seed_time):
    return _eligible(db, pd.Timestamp(seed_time))


def make_label(db, entity_ids, seed_time):
    # count label on [t, t+h]: an event stamped exactly t is counted
    t = pd.Timestamp(seed_time)
    c = _count(db, entity_ids, t, t + pd.Timedelta(days=H), closed_left=True)
    return pd.DataFrame({"label": c.astype(float)})
''', "count label window closed on the left")

# all four functions, point-in-time correct: the clean reference for agent checks
# and the seed program for mutation testing (mutate/)
REFERENCE = '''
def select_entities(db, seed_time):
    return _eligible(db, pd.Timestamp(seed_time))


def make_label(db, entity_ids, seed_time):
    return pd.DataFrame({"label": _label(db, entity_ids, pd.Timestamp(seed_time))})


def get_features(db, entity_ids, seed_time):
    return _feats(db, list(entity_ids), pd.Timestamp(seed_time))


def fit_predict(db, entity_ids, seed_time):
    t = pd.Timestamp(seed_time)
    ids = list(entity_ids)
    X, y = _train(db, ids, t, [H + STEP * k for k in range(NTRAIN)])
    q = _feats(db, ids, t)
    return pd.DataFrame({"score": _ridge(X, y, q.to_numpy(float))}, index=pd.Index(ids))
'''

TASK_SNAP = ('''
def select_entities(db, seed_time):
    return _eligible(db, pd.Timestamp(seed_time))


def make_label(db, entity_ids, seed_time):
    # Target defined by an end-of-data state, not by events in (t, t+h].
$SNAP
''', "label defined by a snapshot")

# --------------------------------------------------------------- model layer
MODEL = {
 "model_correct": ('''
def fit_predict(db, entity_ids, seed_time):
    t = pd.Timestamp(seed_time)
    ids = list(entity_ids)
    X, y = _train(db, ids, t, [H + STEP * k for k in range(NTRAIN)])
    q = _feats(db, ids, t)
    return pd.DataFrame({"score": _ridge(X, y, q.to_numpy(float))}, index=pd.Index(ids))
''', "training seeds t_i with t_i + h <= t"),
 "model_leak_label_overlap": ('''
def fit_predict(db, entity_ids, seed_time):
    # Training seeds every STEP days up to t: the labels of the latest
    # examples cover (t_i, t_i + h], which reaches past t (no embargo).
    t = pd.Timestamp(seed_time)
    ids = list(entity_ids)
    X, y = _train(db, ids, t, [STEP * (k + 1) for k in range(NTRAIN)])
    q = _feats(db, ids, t)
    return pd.DataFrame({"score": _ridge(X, y, q.to_numpy(float))}, index=pd.Index(ids))
''', "training labels overlap the period after t"),
 "model_leak_future_target_encoding": ('''
def fit_predict(db, entity_ids, seed_time):
    # Entity target encoding averaged over label windows of the whole data.
    t = pd.Timestamp(seed_time)
    ids = list(entity_ids)
    te = sum(_label(db, ids, t + pd.Timedelta(days=STEP * k)) for k in range(NTRAIN)) / NTRAIN
    te = te.rename("target_encoding")
    X, y = _train(db, ids, t, [H + STEP * k for k in range(NTRAIN)], extra=te)
    q = _feats(db, ids, t).join(te)
    return pd.DataFrame({"score": _ridge(X, y, q.to_numpy(float))}, index=pd.Index(ids))
''', "target encoding fitted on future labels"),
 "model_leak_nontemporal_sampling": ('''
def fit_predict(db, entity_ids, seed_time):
    # Neighbour aggregation inside the model ignores time (the GNN analogue:
    # neighbour sampling without a temporal constraint).
    t = pd.Timestamp(seed_time)
    ids = list(entity_ids)
    ev = db[EVENTS]
    deg = ev[ev[KEY].isin(pd.Index(ids))].groupby(KEY).size().reindex(pd.Index(ids)).fillna(0)
    deg = deg.astype(float).rename("neighbour_degree")
    X, y = _train(db, ids, t, [H + STEP * k for k in range(NTRAIN)], extra=deg)
    q = _feats(db, ids, t).join(deg)
    return pd.DataFrame({"score": _ridge(X, y, q.to_numpy(float))}, index=pd.Index(ids))
''', "neighbour sampling without a time constraint"),
}

# ------------------------------------------------------------- context layer
ICL_BODY = '''
def build_context(db, entity_ids, seed_time):
    t = pd.Timestamp(seed_time)
    ids = list(entity_ids)
    parts = []
    for off in $OFFSETS:
        ti = t - pd.Timedelta(days=off)
        f = _feats(db, ids, ti)
        f["label"] = _label(db, ids, ti)
        f["offset_days"] = float(off)
        parts.append(f.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


def predict(db, entity_ids, seed_time):
    t = pd.Timestamp(seed_time)
    ids = list(entity_ids)
    ctx = build_context(db, ids, t)
    X = ctx.drop(columns=["label", "offset_days"]).to_numpy(float)
    y = ctx["label"].to_numpy(float)
    q = _feats(db, ids, t).to_numpy(float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    Zq = (q - mu) / sd
    d = ((Zq[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)
    k = min(15, Z.shape[0])
    nn = np.argsort(d, axis=1, kind="stable")[:, :k]
    return pd.DataFrame({"score": y[nn].mean(axis=1)}, index=pd.Index(ids))
'''
ICL = {
 "icl_correct": (ICL_BODY.replace("$OFFSETS", "[H + STEP * k for k in range(NTRAIN)]"),
                 "context examples whose labels are known by t"),
 "icl_leak_future_labels": (ICL_BODY.replace("$OFFSETS", "[STEP * (k + 1) for k in range(NTRAIN)]"),
                            "context examples whose label windows reach past t"),
 "icl_leak_retrieval_all": (ICL_BODY.replace("$OFFSETS",
                            "[H + STEP * k for k in range(NTRAIN)] + [-STEP * (k + 1) for k in range(NTRAIN)]"),
                            "retrieval over the whole table, including examples after t"),
}

# --------------------------------------------------------------- agent layer
AGENT_CORRECT = MODEL["model_correct"][0]
AGENT = {
 "agent_correct": (AGENT_CORRECT, "uses only relative time and no entity identity"),
 "agent_leak_known_entities": (AGENT_CORRECT.replace("def fit_predict", "def _base") + '''

KNOWN = $KNOWN


def fit_predict(db, entity_ids, seed_time):
    # Memorised outcomes: a prior on entities the "agent" knows did well later.
    out = _base(db, entity_ids, seed_time)
    boost = pd.Series([1.0 if str(e) in KNOWN else 0.0 for e in out.index], index=out.index)
    return pd.DataFrame({"score": out["score"] + boost}, index=out.index)
''', "prior on memorised entities (pretraining-knowledge proxy)"),
 "agent_leak_calendar_prior": (AGENT_CORRECT.replace("def fit_predict", "def _base") + '''

REGIME_CHANGE = pd.Timestamp("$CAL")


def fit_predict(db, entity_ids, seed_time):
    # Known regime change at an absolute date (e.g. a crisis the model read about).
    t = pd.Timestamp(seed_time)
    out = _base(db, entity_ids, t)
    f = _feats(db, list(entity_ids), t)
    after = 1.0 if t >= REGIME_CHANGE else 0.0
    return pd.DataFrame({"score": out["score"] + after * 0.1 * f["n_recent"]}, index=out.index)
''', "prior tied to an absolute calendar date"),
}


def known_entities(ds, spec):
    """Among the entities sampled at the benchmark's seeds (lib/sampling.py), the 10 most
    active in the second half of the data: what a model that had read about them would 'know'."""
    import pandas as pd
    from lib.sampling import sample_at
    man = json.loads((ROOT / "datasets" / ds / "dataset.json").read_text())
    df = pd.read_csv(ROOT / "data" / ds / f"{spec['E']}.csv", usecols=[spec["K"], spec["T"]])
    df[spec["T"]] = pd.to_datetime(df[spec["T"]], errors="coerce")
    ent = {"events": spec["E"], "time": spec["T"], "key": spec["K"], "eligible_days": spec["ELIG"]}
    sampled = set()
    for s in man["seeds"]["dev"] + man["seeds"]["held_out"]:
        sampled |= set(sample_at({spec["E"]: df}, s, ent, man["entity"]["n_entities"]))
    late = df[(df[spec["T"]] >= df[spec["T"]].quantile(0.5)) & df[spec["K"]].isin(sampled)]
    return [str(x) for x in late[spec["K"]].value_counts().head(10).index]


def render(text, spec, known=None):
    v = repr(spec["V"])
    return Template(COMMON + text).safe_substitute(E=spec["E"], T=spec["T"], K=spec["K"], V=v, H=spec["H"],
                                                   STEP=spec["STEP"], NTRAIN=spec["NTRAIN"], LABEL=spec["LABEL"],
                                                   CAL=spec["CAL"], SNAP=spec.get("SNAP") or "",
                                                   KNOWN=repr(set(known or [])), ELIG=spec["ELIG"],
                                                   LFILT_LINE=(f"    w = w[{spec['LFILT']}]" if spec.get("LFILT")
                                                               else ""))


def main():
    for ds, spec in SPECS.items():
        base = ROOT / "datasets" / ds / "layers"
        expected = {"horizon_days": spec["H"], "label_task": None, "controls": {},
                    "entity": {"events": spec["E"], "time": spec["T"], "key": spec["K"],
                               "eligible_days": spec["ELIG"]}}
        known = known_entities(ds, spec)

        def put(layer, name, text, what, exp):
            d = base / layer
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{name}.py").write_text(render(text, spec, known), encoding="utf-8")
            expected["controls"][f"{layer}/{name}.py"] = {"layer": layer, "what": what, "expected": exp}

        F = False
        ALLF = {k: F for k in ["selection_witness", "selection_canary", "selection_snapshot", "label_witness",
                               "label_canary", "label_snapshot", "label_tie", "label_tie_inject"]}
        put("task", "task_correct", *TASK["task_correct"], ALLF)
        put("task", "task_correct_count", *TASK["task_correct_count"], ALLF)
        put("task", "task_leak_label_boundary_count", *TASK["task_leak_label_boundary_count"],
            {"label_tie_inject": True})
        put("task", "task_leak_selection_future", *TASK["task_leak_selection_future"], {"selection_witness": True})
        put("task", "task_leak_label_no_cap", *TASK["task_leak_label_no_cap"], {"label_witness": True})
        put("task", "task_leak_label_boundary", *TASK["task_leak_label_boundary"],
            {"label_tie": spec["DAY"], "label_tie_inject": True})
        if spec.get("SNAP"):
            put("task", "task_leak_label_snapshot", *TASK_SNAP, {"label_snapshot": True})
        put("model", "model_correct", *MODEL["model_correct"], {"witness": F, "canary": F, "snapshot": F})
        for n in ["model_leak_label_overlap", "model_leak_future_target_encoding", "model_leak_nontemporal_sampling"]:
            put("model", n, *MODEL[n], {"witness": True})
        put("context", "icl_correct", *ICL["icl_correct"],
            {"context_witness": F, "context_canary": F, "context_snapshot": F, "predict_witness": F})
        for n in ["icl_leak_future_labels", "icl_leak_retrieval_all"]:
            put("context", n, *ICL[n], {"context_witness": True})
        put("agent", "agent_correct", *AGENT["agent_correct"], {"identity_or_calendar_dependence": F})
        for n in ["agent_leak_known_entities", "agent_leak_calendar_prior"]:
            put("agent", n, *AGENT[n], {"identity_or_calendar_dependence": True})
        expected["known_entities"] = known
        (base / "reference_pipeline.py").write_text(render(REFERENCE, spec, known), encoding="utf-8")
        (base / "expected.json").write_text(json.dumps(expected, indent=1, ensure_ascii=False) + "\n")
        print(ds, len(expected["controls"]), "controls")


if __name__ == "__main__":
    main()
