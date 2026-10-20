#!/usr/bin/env python3
"""Check real ML frameworks with the layer harness (run with the frameworks venv).

    python -I validate_frameworks.py --skill-scripts <.../pit-repair/scripts> [--only lgbm ft ag gnn]

Frameworks are optional: families whose packages are not installed are skipped
(frameworks/adapters.py: REQUIRES, available()).

For every (framework configuration, dataset):
  1. determinism: two runs on the same database must agree (otherwise a
     differential check is meaningless; PLAN_main_track H2);
  2. views: D vs D|t (witness), canary, snapshot -- at the 6 benchmark seeds;
  3. harm (score-producing configurations): AUC on D vs AUC on D|t against the
     true labels of (t, t+h].
Expected verdicts are in EXPECTED below. Writes results_frameworks/<ds>.json and summary.md.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from frameworks import adapters as A  # noqa: E402
from layers.harness import Layers  # noqa: E402
from layers.make_layer_controls import SPECS  # noqa: E402
from lib.common import DATA, DATASETS, read_json, write_json  # noqa: E402
from lib.sampling import ids_by_seed  # noqa: E402
from validate_layers import parse_dates  # noqa: E402

CLEAN = {"witness": False, "canary": False, "snapshot": False}
LEAK = {"witness": True}
EXPECTED = {
    "ft_d3": CLEAN, "ft_d2": CLEAN, "ft_d1": LEAK,
    "lgbm_correct": CLEAN, "lgbm_label_overlap": LEAK, "lgbm_target_encoding": LEAK,
    "ag_correct": CLEAN, "ag_label_overlap": LEAK,
    "gnn_temporal": CLEAN, "gnn_nontemporal": LEAK,
}
PLAN = {
    "ft": {"datasets": list(A.FT_SCHEMAS), "configs": ["ft_d3", "ft_d2", "ft_d1"]},
    "lgbm": {"datasets": ["olist", "f1", "pkdd99", "adventureworks", "northwind", "sakila", "movietweetings", "tpch"],
             "configs": ["lgbm_correct", "lgbm_label_overlap", "lgbm_target_encoding"]},
    "ag": {"datasets": ["olist", "northwind", "tpch"], "configs": ["ag_correct", "ag_label_overlap"]},
    "gnn": {"datasets": list(A.GNN_GRAPHS), "configs": ["gnn_temporal", "gnn_nontemporal"]},
}


def make(cfg, ds, workdir):
    s = A.Spec(SPECS[ds])
    return {
        "ft_d3": lambda: A.featuretools(ds, "d3"), "ft_d2": lambda: A.featuretools(ds, "d2"),
        "ft_d1": lambda: A.featuretools(ds, "d1"),
        "lgbm_correct": lambda: A.lgbm(s, A.correct_offsets),
        "lgbm_label_overlap": lambda: A.lgbm(s, A.overlap_offsets),
        "lgbm_target_encoding": lambda: A.lgbm(s, A.correct_offsets, target_encoding=True),
        "ag_correct": lambda: A.autogluon(s, A.correct_offsets, workdir),
        "ag_label_overlap": lambda: A.autogluon(s, A.overlap_offsets, workdir),
        "gnn_temporal": lambda: A.gnn(ds, s, temporal=True),
        "gnn_nontemporal": lambda: A.gnn(ds, s, temporal=False),
    }[cfg]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill-scripts", required=True)
    ap.add_argument("--only", nargs="*", default=list(PLAN))
    a = ap.parse_args()
    sys.path.insert(0, a.skill_scripts)
    import checker_core as core
    import pit_check

    out = ROOT / "results_frameworks"
    out.mkdir(exist_ok=True)
    have = A.available()
    skipped = [f for f in a.only if not have[f]]
    if skipped:
        print("skipped (framework not installed):", ", ".join(f"{f} needs {A.REQUIRES[f]}" for f in skipped))
    by_ds = {}
    for fam in [f for f in a.only if have[f]]:
        for ds in PLAN[fam]["datasets"]:
            by_ds.setdefault(ds, []).extend(PLAN[fam]["configs"])
    if not by_ds:
        print("nothing to run: install at least one framework (see frameworks/README.md)")
        return
    workdir = Path(tempfile.mkdtemp(prefix="pitbench_ag_"))
    for ds, cfgs in by_ds.items():
        man = read_json(DATASETS / ds / "dataset.json")
        amap_path = DATASETS / ds / man["availability_map"]
        amap, pmap = parse_dates(pit_check, amap_path)
        db = core.load_db(DATA / ds, parse_dates=pmap)
        L = Layers(pit_check, core, db, amap, read_json(amap_path), timeout=1800)
        seeds = man["seeds"]["dev"] + man["seeds"]["held_out"]
        exp = read_json(DATASETS / ds / "layers" / "expected.json")
        ids = ids_by_seed(db, seeds, exp["entity"], man["entity"]["n_entities"])
        s = A.Spec(SPECS[ds])

        def y_fn(db_, ids_, t, s=s):
            return A.label(db_, s, ids_, t)

        rp = out / f"{ds}.json"
        res = read_json(rp) if rp.exists() else {}
        for cfg in cfgs:
            t0 = time.time()
            fn = make(cfg, ds, workdir)
            rec = {"expected": EXPECTED[cfg]}
            try:
                t = pd.Timestamp(seeds[0])
                a1, a2 = fn(db, ids[seeds[0]], t), fn(db, ids[seeds[0]], t)
                rec["deterministic"] = not L.diff(a1, a2)
                rows = L.check_scores_fn(fn, ids, seeds, y_fn=y_fn if not cfg.startswith("ft_") else None)
                obs = {k: any(r[k] for r in rows) for k in ("witness", "canary", "snapshot")}
                rec.update({"observed": obs, "per_seed": rows,
                            "agrees": rec["deterministic"] and all(obs.get(k) == v for k, v in EXPECTED[cfg].items())})
                bt = [r.get("auc_backtest") for r in rows if r.get("auc_backtest") is not None]
                dp = [r.get("auc_deploy") for r in rows if r.get("auc_deploy") is not None]
                if bt and dp:
                    rec.update({"auc_backtest": round(float(np.mean(bt)), 4), "auc_deploy": round(float(np.mean(dp)), 4)})
                    rec["gap"] = round(rec["auc_backtest"] - rec["auc_deploy"], 4)
            except Exception as e:
                rec.update({"error": f"{type(e).__name__}: {e}"[:500], "agrees": False})
            rec["seconds"] = round(time.time() - t0, 1)
            res[cfg] = rec
            write_json(res, rp)
            print(f"{ds:15s} {cfg:22s} det={rec.get('deterministic')} obs={rec.get('observed')} "
                  f"gap={rec.get('gap')} agrees={rec['agrees']} {rec.get('error', '')} {rec['seconds']}s", flush=True)
    rows = ["| dataset | configuration | deterministic | channels fired | AUC backtest | AUC at t | gap | agrees |",
            "|---|---|---|---|---|---|---|---|"]
    for rp in sorted(out.glob("*.json")):
        r = read_json(rp)
        for cfg, c in r.items():
            fired = ", ".join(k for k, v in c.get("observed", {}).items() if v) or ("ERROR" if "error" in c else "—")
            rows.append(f"| {rp.stem} | {cfg} | {c.get('deterministic')} | {fired} | {c.get('auc_backtest', '—')} | "
                        f"{c.get('auc_deploy', '—')} | {c.get('gap', '—')} | {'yes' if c['agrees'] else '**NO**'} |")
    (out / "summary.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
