#!/usr/bin/env python3
"""Execution step of the agent scenario: check every agent-written pipeline on every layer.

    python -I agent/check_runs.py --skill-scripts <.../pit-repair/scripts> [--runs-dir agent_runs]

Refuses to run if an LLM provider key is in the environment (pit-repair's rule:
the step that executes model-written code is a separate, key-free process).

For each agent_runs/<label>/<ds>/run<k>/pipeline.py:
  task      select_entities on D vs D|t / canary / snapshot; make_label on D vs D|(t+h) / canary / snapshot / tie
  features  get_features on D vs D|t (witness), canary, snapshot
  model     fit_predict on D vs D|t, canary, snapshot  + AUC backtest vs AUC at t (deployment)
  agent     fit_predict on D vs anonymize(D)           + AUC original vs anonymized
plus the agent's own report.json (its claimed validation AUC). Every layer that
fires is attributed; a pipeline that does not load or crashes is recorded as
ERROR with the reason. Output: results_agents/<label>.json and summary.md.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from layers.harness import Layers  # noqa: E402
from layers.sandbox import load_namespace  # noqa: E402
from lib.common import DATA, DATASETS, read_json, write_json  # noqa: E402
from lib.sampling import ids_by_seed  # noqa: E402
from validate_layers import parse_dates  # noqa: E402

FORBIDDEN = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def any_flag(rows, k):
    return any(bool(r.get(k)) for r in rows)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def sha(run_dir):
    p = run_dir / "pipeline.py"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def check_one(path, L, ids, seeds, H, y_fn, pit_check, key=None):
    rec = {"pipeline": str(path)}
    try:
        ns = load_namespace(path.read_text(encoding="utf-8"), pit_check)
    except Exception as e:
        return {**rec, "status": "ERROR", "stage": "load", "error": f"{type(e).__name__}: {e}"[:400]}
    have = {f: callable(ns.get(f)) for f in ("select_entities", "make_label", "get_features", "fit_predict")}
    if not any(have.values()):
        return {**rec, "status": "ERROR", "stage": "interface", "error": "none of the four functions defined"}
    # wrapping rule 2 (corpus/README.md): a layer whose functions are missing is reported absent, never filled in
    plan = []
    if have["select_entities"] and have["make_label"]:
        plan.append(("task", lambda: L.check_task_fn(ns["select_entities"], ns["make_label"], ids, seeds, H, key=key)))
    if have["get_features"]:
        plan.append(("features", lambda: L.check_scores_fn(ns["get_features"], ids, seeds)))
    if have["fit_predict"]:
        plan.append(("model", lambda: L.check_scores_fn(ns["fit_predict"], ids, seeds, y_fn=y_fn)))
        plan.append(("agent", lambda: L.check_agent_fn(ns["fit_predict"], ids, seeds, y_fn=y_fn)))
    rec["absent_layers"] = [n for n in ("task", "features", "model", "agent") if n not in dict(plan)]
    layers = {}
    for name, run in plan:
        t0 = time.time()
        try:
            rows = run()
            layers[name] = {"rows": rows, "seconds": round(time.time() - t0, 1)}
        except Exception as e:
            layers[name] = {"error": f"{type(e).__name__}: {e}"[:400], "seconds": round(time.time() - t0, 1)}
    fired = {}
    if "rows" in layers.get("task", {}):
        r = layers["task"]["rows"]
        fired["task"] = [k for k in ("selection_witness", "selection_canary", "selection_snapshot", "label_witness",
                                     "label_canary", "label_snapshot", "label_tie", "label_tie_inject")
                         if any_flag(r, k)]
    for name in ("features", "model"):
        if "rows" in layers.get(name, {}):
            fired[name] = [k for k in ("witness", "canary", "snapshot") if any_flag(layers[name]["rows"], k)]
    if "rows" in layers.get("agent", {}):
        fired["agent"] = ["identity_or_calendar_dependence"] if any_flag(
            layers["agent"]["rows"], "identity_or_calendar_dependence") else []
    rec.update({"status": "OK", "layers": layers, "fired": fired,
                "layer_errors": {k: v["error"] for k, v in layers.items() if "error" in v}})
    if "rows" in layers.get("model", {}):
        rec["auc_backtest"] = mean([r.get("auc_backtest") for r in layers["model"]["rows"]])
        rec["auc_deploy"] = mean([r.get("auc_deploy") for r in layers["model"]["rows"]])
    rep = path.parent / "report.json"
    if rep.exists():
        try:
            rec["agent_report"] = json.loads(rep.read_text())
        except Exception:
            rec["agent_report"] = {"unparseable": rep.read_text()[:300]}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill-scripts", required=True)
    ap.add_argument("--runs-dir", default=str(ROOT / "agent_runs"))
    ap.add_argument("--n-entities", type=int, default=None, help="override dataset.json entity.n_entities")
    ap.add_argument("--shard", default="0/1", help="i/k: check every k-th program starting at i (run k in parallel)")
    ap.add_argument("--summary-only", action="store_true", help="merge shard results and write summary.md")
    a = ap.parse_args()
    si, sk = map(int, a.shard.split("/"))
    if a.summary_only:
        write_summary(ROOT / "results_agents")
        return
    present = [k for k in FORBIDDEN if os.environ.get(k)]
    if present:
        sys.exit(f"check_runs: refusing, provider key in environment: {present}")
    sys.path.insert(0, a.skill_scripts)
    import checker_core as core
    import pit_check

    out_dir = ROOT / "results_agents"
    out_dir.mkdir(exist_ok=True)
    for label_dir in sorted(p for p in Path(a.runs_dir).iterdir() if p.is_dir()):
        suffix = "" if sk == 1 else f".shard{si}of{sk}"
        out_file = out_dir / f"{label_dir.name}{suffix}.json"
        # results are written after every program and an interrupted run resumes where it stopped
        results = read_json(out_file) if out_file.exists() else {}
        for ds_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
            # resume: skip a program only if it was checked and its pipeline.py is unchanged
            # (a run redone after an error gets a new pipeline and is checked again)
            todo = [r for j, r in enumerate(sorted(p for p in ds_dir.iterdir() if p.is_dir()))
                    if j % sk == si and results.get(f"{ds_dir.name}/{r.name}", {}).get("pipeline_sha") != sha(r)]
            if not todo:
                continue
            ds = ds_dir.name
            man = read_json(DATASETS / ds / "dataset.json")
            amap_path = DATASETS / ds / man["availability_map"]
            amap, pmap = parse_dates(pit_check, amap_path)
            db = core.load_db(DATA / ds, parse_dates=pmap)
            L = Layers(pit_check, core, db, amap, read_json(amap_path))
            seeds = man["seeds"]["dev"] + man["seeds"]["held_out"]
            exp = read_json(DATASETS / ds / "layers" / "expected.json")
            H = exp["horizon_days"]
            ids = ids_by_seed(db, seeds, exp["entity"], a.n_entities or man["entity"]["n_entities"])
            lab = L.load((DATASETS / ds / "layers" / "task" / "task_correct.py").read_text(), "make_label")

            def y_fn(db_, ids_, t, lab=lab, L=L):
                return L.call(lab, db_, ids_, t)["label"]

            for run_dir in todo:
                meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
                pipe = run_dir / "pipeline.py"
                rec = (check_one(pipe, L, ids, seeds, H, y_fn, pit_check, key=exp["entity"]["key"]) if pipe.exists()
                       else {"status": "NO_PIPELINE", "error": meta.get("error")})
                rec["meta"] = {k: meta.get(k) for k in ("wall_seconds", "num_turns", "turns", "total_cost_usd",
                                                         "cost", "models", "model", "mode", "error")}
                # natural code (agent, external) counts for prevalence; mutants and controls only for recall
                rec["origin"] = meta.get("origin", "agent")
                rec["pipeline_sha"] = sha(run_dir)
                results[f"{ds}/{run_dir.name}"] = rec
                write_json(results, out_file)
                print(label_dir.name, ds, run_dir.name, rec["status"], rec.get("fired"), rec.get("error", ""),
                      flush=True)
        write_json(results, out_file)
    if sk == 1:
        write_summary(out_dir)


def write_summary(out_dir):
    """One row per agent/source: how many programs ran, which layers leaked, backtest vs deployment AUC.
    Only natural code (origin agent / external) counts toward prevalence; mutants and controls are
    listed for recall but marked."""
    lines = ["| agent / source | origin | counts for prevalence | programs | ran | leaking (any layer) | task | "
             "features | model | identity/calendar | AUC claimed | AUC backtest | AUC at t |",
             "|---|" + "---|" * 12]
    merged = {}
    for f in sorted(out_dir.glob("*.json")):
        merged.setdefault(f.name.split(".shard")[0].removesuffix(".json"), {}).update(read_json(f))
    for stem, r in merged.items():
        f = Path(stem)
        ok = [v for v in r.values() if v.get("status") == "OK"]
        def cnt(layer):
            return sum(1 for v in ok if v.get("fired", {}).get(layer))
        leaking = sum(1 for v in ok if any(v.get("fired", {}).values()))
        claimed = [v.get("agent_report", {}).get("validation_auc") for v in ok]
        claimed = [c for c in claimed if isinstance(c, (int, float))]
        origins = sorted({v.get("origin", "agent") for v in r.values()})
        natural = all(o in ("agent", "external") for o in origins)
        lines.append(f"| {f.stem} | {','.join(origins)} | {'yes' if natural else 'no (recall only)'} | "
                     f"{len(r)} | {len(ok)} | {leaking} | {cnt('task')} | {cnt('features')} | "
                     f"{cnt('model')} | {cnt('agent')} | {mean(claimed)} | "
                     f"{mean([v.get('auc_backtest') for v in ok])} | {mean([v.get('auc_deploy') for v in ok])} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
