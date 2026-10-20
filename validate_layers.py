#!/usr/bin/env python3
"""Validate the layer controls (task, model, context, agent) of PITBench datasets.

    python -I validate_layers.py --skill-scripts <repo>/.claude/skills/pit-repair/scripts [ds ...]

The feature layer is validated by validate.py (pit_check.py itself). For the
other layers every control in datasets/<ds>/layers/ is run through
layers/harness.py, its channels are compared with layers/expected.json, and
for model / context / agent controls the harm is measured:

    inflation = AUC(scores computed on D) - AUC(scores computed on D|t)

both against the true labels of (t, t+h] from the dataset's task_correct.
Writes results_layers/<ds>.json and results_layers/summary.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from layers.harness import Layers  # noqa: E402
from lib.common import DATA, DATASETS, read_json, write_json  # noqa: E402
from lib.sampling import ids_by_seed  # noqa: E402

# A task is informative for harm only if, at most seeds, the label is neither almost
# always 0 nor almost always 1, and the correct model is neither useless nor trivially perfect.
GATE = {"prevalence": (0.05, 0.95), "min_seeds_ok": 4, "auc": (0.55, 0.95)}

ALL = ["olist", "f1", "pkdd99", "adventureworks", "northwind", "sakila", "movietweetings", "tpch"]
OUT = ROOT / "results_layers"


def parse_dates(pit_check, amap_path):
    amap = pit_check.load_availability_map(amap_path)
    pd_map = pit_check.parse_dates_from_map(amap)
    for t, cols in read_json(amap_path).get("date_cols", {}).items():
        pd_map.setdefault(t, [])
        pd_map[t] += [c for c in cols if c not in pd_map[t]]
    return amap, pd_map


def any_flag(rows, key):
    vals = [r.get(key) for r in rows]
    return any(bool(v) for v in vals)


def agrees(expected, observed):
    return all(v is None or bool(v) == observed.get(k) for k, v in expected.items())


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def validate(ds, pit_check, core):
    t0 = time.time()
    man = read_json(DATASETS / ds / "dataset.json")
    amap_path = DATASETS / ds / man["availability_map"]
    amap, pmap = parse_dates(pit_check, amap_path)
    db = core.load_db(DATA / ds, parse_dates=pmap)
    L = Layers(pit_check, core, db, amap, read_json(amap_path))
    seeds = man["seeds"]["dev"] + man["seeds"]["held_out"]
    exp = read_json(DATASETS / ds / "layers" / "expected.json")
    H = exp["horizon_days"]
    ids = ids_by_seed(db, seeds, exp["entity"], man["entity"]["n_entities"])
    key = exp["entity"]["key"]
    lab_fn = L.load((DATASETS / ds / "layers" / "task" / "task_correct.py").read_text(), "make_label")

    def y_fn(db_, ids_, t):
        return L.call(lab_fn, db_, ids_, t)["label"]

    results = {}
    for rel, spec in exp["controls"].items():
        code = (DATASETS / ds / "layers" / rel).read_text(encoding="utf-8")
        layer = spec["layer"]
        rec = {"layer": layer, "what": spec["what"], "expected": spec["expected"]}
        try:
            if layer == "task":
                rows = L.check_task(code, ids, seeds, H, key=key)
                keys = ["selection_witness", "selection_canary", "selection_snapshot",
                        "label_witness", "label_canary", "label_snapshot", "label_tie", "label_tie_inject"]
                obs = {k: any_flag(rows, k) for k in keys}
            elif layer == "model":
                rows = L.check_scores(code, "fit_predict", ids, seeds, y_fn=y_fn)
                obs = {k: any_flag(rows, k) for k in ("witness", "canary", "snapshot")}
            elif layer == "context":
                rows_c = L.check_scores(code, "build_context", ids, seeds)
                rows_p = L.check_scores(code, "predict", ids, seeds, y_fn=y_fn)
                obs = {f"context_{k}": any_flag(rows_c, k) for k in ("witness", "canary", "snapshot")}
                obs["predict_witness"] = any_flag(rows_p, "witness")
                rows = {"context": rows_c, "predict": rows_p}
            elif layer == "agent":
                rows = L.check_agent(code, ids, seeds, read_json(amap_path).get("date_cols", {}), y_fn=y_fn)
                obs = {"identity_or_calendar_dependence": any_flag(rows, "identity_or_calendar_dependence")}
            rec["observed"] = obs
            rec["per_seed"] = rows
            score_rows = rows["predict"] if layer == "context" else rows
            if layer in ("model", "context"):
                rec["auc_backtest"] = mean([r.get("auc_backtest") for r in score_rows])
                rec["auc_deploy"] = mean([r.get("auc_deploy") for r in score_rows])
                if rec["auc_backtest"] is not None and rec["auc_deploy"] is not None:
                    rec["inflation"] = round(rec["auc_backtest"] - rec["auc_deploy"], 4)
            if layer == "agent":
                rec["auc_original"] = mean([r.get("auc_original") for r in rows])
                rec["auc_anonymized"] = mean([r.get("auc_anonymized") for r in rows])
            rec["agrees"] = agrees(spec["expected"], obs)
        except Exception as e:  # a control that does not run is a failure of the benchmark
            rec.update({"error": f"{type(e).__name__}: {e}"[:400], "agrees": False})
        results[rel] = rec
    # the reference pipeline (all four functions, correct) must be clean on every layer
    from agent.check_runs import check_one
    ref = check_one(DATASETS / ds / "layers" / "reference_pipeline.py", L, ids, seeds, H, y_fn, pit_check, key=key)
    results["reference_pipeline.py"] = {
        "layer": "all", "what": "reference pipeline (correct)", "expected": {"clean": True},
        "observed": {"clean": ref.get("status") == "OK" and not any(ref.get("fired", {}).values())
                     and not ref.get("layer_errors")},
        "fired": ref.get("fired"), "layer_errors": ref.get("layer_errors"), "error": ref.get("error")}
    results["reference_pipeline.py"]["agrees"] = results["reference_pipeline.py"]["observed"]["clean"]
    if results["reference_pipeline.py"]["error"] is None:
        results["reference_pipeline.py"].pop("error")
    gate = difficulty_gate(results)
    res = {"dataset": ds, "n_entities_by_seed": {s: len(v) for s, v in ids.items()}, "seeds": seeds,
           "horizon_days": H, "difficulty": gate, "controls": results,
           "all_agree": all(r["agrees"] for r in results.values()), "seconds": round(time.time() - t0, 1)}
    OUT.mkdir(exist_ok=True)
    write_json(res, OUT / f"{ds}.json")
    return res


def difficulty_gate(results):
    """Is the dataset's task informative for harm? Uses task_correct (label prevalence per seed)
    and model_correct (AUC of the correct model)."""
    tc = results.get("task/task_correct.py", {}).get("per_seed", [])
    prev = {r["seed"]: r.get("label_prevalence") for r in tc}
    lo, hi = GATE["prevalence"]
    ok_seeds = [s for s, v in prev.items() if v is not None and lo <= v <= hi]
    mc = results.get("model/model_correct.py", {})
    a = mc.get("auc_backtest")
    alo, ahi = GATE["auc"]
    informative = len(ok_seeds) >= GATE["min_seeds_ok"] and a is not None and alo <= a <= ahi
    return {"label_prevalence_by_seed": prev, "seeds_in_range": len(ok_seeds), "correct_model_auc": a,
            "informative_for_harm": informative, "gate": GATE}


def fmt(v):
    return "—" if v is None else (f"{v:+.3f}" if isinstance(v, float) else str(v))


def summary_md(results):
    lines = ["| dataset | control | channels fired | AUC backtest | AUC at t (deploy) | inflation | agrees |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        for name, c in r["controls"].items():
            fired = ", ".join(k for k, v in c.get("observed", {}).items() if v) or ("ERROR" if "error" in c else "—")
            if c["layer"] == "agent":
                b, d = c.get("auc_original"), c.get("auc_anonymized")
                infl = round(b - d, 4) if (b is not None and d is not None) else None
            else:
                b, d, infl = c.get("auc_backtest"), c.get("auc_deploy"), c.get("inflation")
            lines.append(f"| {r['dataset']} | {name} | {fired} | {b if b is not None else '—'} | "
                         f"{d if d is not None else '—'} | {fmt(infl)} | {'yes' if c['agrees'] else '**NO**'} |")
    return "\n".join(lines)


def tables(results):
    """difficulty.md (the gate) and harm.md (gap backtest - at t; medians over informative tasks only)."""
    info = {r["dataset"]: r["difficulty"]["informative_for_harm"] for r in results}
    d = ["| dataset | entities per seed | label prevalence (min-max) | correct-model AUC | informative for harm |",
         "|---|---|---|---|---|"]
    for r in results:
        g = r["difficulty"]
        pv = [v for v in g["label_prevalence_by_seed"].values() if v is not None]
        n = list(r["n_entities_by_seed"].values())
        d.append(f"| {r['dataset']} | {min(n)}-{max(n)} | {min(pv):.2f}-{max(pv):.2f} | {g['correct_model_auc']} | "
                 f"{'yes' if g['informative_for_harm'] else 'no'} |")
    (OUT / "difficulty.md").write_text("\n".join(d) + "\n")
    names = [r["dataset"] for r in results]
    gaps = {}
    for r in results:
        for name, c in r["controls"].items():
            if c["layer"] in ("model", "context") and c.get("inflation") is not None:
                gaps.setdefault(name, {})[r["dataset"]] = c["inflation"]
    h = ["n/i = not informative for harm (difficulty gate); medians use informative tasks only", "",
         "| control | " + " | ".join(n + ("" if info[n] else " (n/i)") for n in names) +
         " | median gap | abs(gap) > 0.02 |", "|---|" + "---|" * (len(names) + 2)]
    for name, g in gaps.items():
        inf = [g[n] for n in names if info[n] and g.get(n) is not None]
        h.append(f"| {name} | " + " | ".join("—" if g.get(n) is None else f"{g[n]:+.3f}" for n in names) +
                 f" | {float(np.median(inf)):+.3f} | {sum(abs(v) > 0.02 for v in inf)}/{len(inf)} |")
    (OUT / "harm.md").write_text("\n".join(h) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill-scripts", required=True)
    ap.add_argument("--tables-only", action="store_true", help="rebuild summary/difficulty/harm from saved results")
    ap.add_argument("datasets", nargs="*")
    a = ap.parse_args()
    if a.tables_only:
        res = [read_json(OUT / f"{d}.json") for d in ALL]
        (OUT / "summary.md").write_text(summary_md(res) + "\n")
        tables(res)
        return
    sys.path.insert(0, a.skill_scripts)
    import checker_core as core
    import pit_check
    results = []
    for ds in a.datasets or ALL:
        r = validate(ds, pit_check, core)
        bad = {n: c.get("error") or c.get("observed") for n, c in r["controls"].items() if not c["agrees"]}
        print(f"{ds:15s} controls={len(r['controls'])} disagree={bad} {r['seconds']}s", flush=True)
        results.append(r)
    if not a.datasets:
        (OUT / "summary.md").write_text(summary_md(results) + "\n", encoding="utf-8")
        tables(results)
    sys.exit(0 if all(r["all_agree"] for r in results) else 1)


if __name__ == "__main__":
    main()
