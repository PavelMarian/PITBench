#!/usr/bin/env python3
"""Mutation testing of the layer checks on realistic pipelines.

    python -I mutate/run_mutations.py --skill-scripts <.../pit-repair/scripts> --dataset northwind \
        [--pipeline path/to/pipeline.py] [--ops drop_cutoff future_offset ...]

Default pipeline: datasets/<ds>/layers/reference_pipeline.py. Point --pipeline at
agent-written pipelines (agent_runs/...) to mutate code agents actually wrote.

For every mutant:
  * effect   -- does any of the four functions change its output on the full
                database at any seed? If not, the mutant is EQUIVALENT and excluded;
  * verdict  -- the all-layer check of agent/check_runs.py: KILLED if any layer
                fires, SURVIVED otherwise, ERROR if the mutant does not run.
Score per operator = killed / (killed + survived). drop_lower_bound is a
specificity probe (not a PIT violation): there, survivors are the right answer.
With a triage file (default mutate/triage_reference.json for reference pipelines) every
live mutant also gets a ground-truth label "is a PIT violation", decided by reading the
code, and the summary reports recall (violations killed) and false alarms (non-violations
killed) next to the per-operator score.
Writes results_mutations/<ds>__<pipeline>.json and summary.md.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from agent.check_runs import check_one  # noqa: E402
from layers.harness import Layers  # noqa: E402
from layers.sandbox import load_namespace  # noqa: E402
from lib.common import DATA, DATASETS, read_json, write_json  # noqa: E402
from lib.sampling import ids_by_seed  # noqa: E402
from mutate.mutations import DESIGNED_TO_VIOLATE, OPERATORS, mutants  # noqa: E402
from validate_layers import parse_dates  # noqa: E402

FUNCS = ("select_entities", "make_label", "get_features", "fit_predict")


def outputs(ns, L, db, ids, seeds):
    out = {}
    for s in seeds:
        t = pd.Timestamp(s)
        out[s] = {"select_entities": sorted(map(str, L.call(ns["select_entities"], db, t)))}
        for f in FUNCS[1:]:
            out[s][f] = L.call(ns[f], db, ids[s], t)
    return out


def same(a, b, L):
    for s in a:
        if a[s]["select_entities"] != b[s]["select_entities"]:
            return False
        for f in FUNCS[1:]:
            if L.diff(a[s][f], b[s][f]):
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill-scripts", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pipeline", default=None)
    ap.add_argument("--ops", nargs="*", default=OPERATORS)
    ap.add_argument("--triage", default=None, help="JSON with labels [{operator, before, after, pit_violation}]")
    a = ap.parse_args()
    sys.path.insert(0, a.skill_scripts)
    import checker_core as core
    import pit_check

    ds = a.dataset
    pipe = Path(a.pipeline) if a.pipeline else DATASETS / ds / "layers" / "reference_pipeline.py"
    man = read_json(DATASETS / ds / "dataset.json")
    amap_path = DATASETS / ds / man["availability_map"]
    amap, pmap = parse_dates(pit_check, amap_path)
    db = core.load_db(DATA / ds, parse_dates=pmap)
    L = Layers(pit_check, core, db, amap, read_json(amap_path))
    seeds = man["seeds"]["dev"] + man["seeds"]["held_out"]
    exp = read_json(DATASETS / ds / "layers" / "expected.json")
    H, key = exp["horizon_days"], exp["entity"]["key"]
    ids = ids_by_seed(db, seeds, exp["entity"], man["entity"]["n_entities"])
    lab = L.load((DATASETS / ds / "layers" / "task" / "task_correct.py").read_text(), "make_label")

    def y_fn(db_, ids_, t):
        return L.call(lab, db_, ids_, t)["label"]

    triage_path = Path(a.triage) if a.triage else (ROOT / "mutate" / "triage_reference.json" if not a.pipeline else None)
    triage = {}
    if triage_path and triage_path.exists():
        for t in read_json(triage_path)["labels"]:
            triage[(t["operator"], t["before"], t["after"])] = t
    code = pipe.read_text(encoding="utf-8")
    base = outputs(load_namespace(code, pit_check), L, db, ids, seeds)
    out_dir = ROOT / "mutants" / ds / pipe.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (op, line, before, after, mcode) in enumerate(mutants(code)):
        if op not in a.ops:
            continue
        mp = out_dir / f"{k:03d}_{op}.py"
        mp.write_text(f"# mutant of {pipe.name}: {op} at line {line}: `{before}` -> `{after}`\n" + mcode)
        rec = {"id": mp.name, "operator": op, "line": line, "before": before, "after": after,
               "designed_to_violate": DESIGNED_TO_VIOLATE[op]}
        try:
            ns = load_namespace(mcode, pit_check)
            rec["equivalent"] = same(base, outputs(ns, L, db, ids, seeds), L)
        except Exception as e:
            rec.update({"verdict": "ERROR", "error": f"{type(e).__name__}: {e}"[:300]})
            rows.append(rec)
            print(rec["id"], "ERROR", flush=True)
            continue
        if rec["equivalent"]:
            rec["verdict"] = "EQUIVALENT"
        else:
            r = check_one(mp, L, ids, seeds, H, y_fn, pit_check, key=key)
            fired = {k2: v for k2, v in (r.get("fired") or {}).items() if v}
            rec["fired"] = fired
            rec["layer_errors"] = r.get("layer_errors")
            rec["verdict"] = "ERROR" if r.get("status") != "OK" else ("KILLED" if fired else "SURVIVED")
        tl = triage.get((op, before, after))
        if tl is not None and rec["verdict"] in ("KILLED", "SURVIVED"):
            rec["pit_violation"], rec["triage_reason"] = tl["pit_violation"], tl["reason"]
        rows.append(rec)
        print(rec["id"], rec["verdict"], rec.get("fired", ""), flush=True)

    summary = {}
    for op in OPERATORS:
        rs = [r for r in rows if r["operator"] == op]
        if not rs:
            continue
        c = {v: sum(1 for r in rs if r["verdict"] == v) for v in ("KILLED", "SURVIVED", "EQUIVALENT", "ERROR")}
        live = c["KILLED"] + c["SURVIVED"]
        c["score"] = round(c["KILLED"] / live, 3) if live else None
        c["designed_to_violate"] = DESIGNED_TO_VIOLATE[op]
        summary[op] = c
    res_dir = ROOT / "results_mutations"
    res_dir.mkdir(exist_ok=True)
    live = [r for r in rows if r["verdict"] in ("KILLED", "SURVIVED")]
    triaged = [r for r in live if "pit_violation" in r]
    viol = [r for r in triaged if r["pit_violation"]]
    non = [r for r in triaged if not r["pit_violation"]]
    summary["_triage"] = {"live": len(live), "triaged": len(triaged),
                          "violations": len(viol), "violations_killed": sum(r["verdict"] == "KILLED" for r in viol),
                          "non_violations": len(non), "non_violations_killed": sum(r["verdict"] == "KILLED" for r in non)}
    write_json({"dataset": ds, "pipeline": str(pipe), "summary": summary, "mutants": rows},
               res_dir / f"{ds}__{pipe.stem}.json")
    lines = ["| dataset / pipeline | operator | designed to violate | killed | survived | equivalent | error | score |",
             "|---|---|---|---|---|---|---|---|"]
    tri = ["", "| dataset / pipeline | live mutants | triaged | violations killed (recall) | "
               "non-violations killed (false alarms) |", "|---|---|---|---|---|"]
    for f in sorted(res_dir.glob("*.json")):
        r = read_json(f)
        for op, c in r["summary"].items():
            if op.startswith("_"):
                continue
            lines.append(f"| {f.stem} | {op} | {c['designed_to_violate']} | {c['KILLED']} | {c['SURVIVED']} | "
                         f"{c['EQUIVALENT']} | {c['ERROR']} | {c['score']} |")
        t = r["summary"].get("_triage")
        if t:
            tri.append(f"| {f.stem} | {t['live']} | {t['triaged']} | {t['violations_killed']}/{t['violations']} | "
                       f"{t['non_violations_killed']}/{t['non_violations']} |")
    lines += tri
    (res_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
