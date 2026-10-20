#!/usr/bin/env python3
"""Validate PITBench datasets against the pit-repair checker.

    python -I validate.py --skill-scripts <repo>/.claude/skills/pit-repair/scripts [ds ...]

For every dataset:
  1. map sanity: every table/column named by the availability map exists,
     time channels parse, share of missing times, seeds inside the data range;
  2. power: for each seed, how many sampled entities have history <= t;
  3. tasks: label prevalence per seed (sampled entities);
  4. controls: every program in datasets/<ds>/controls is run through the
     UNMODIFIED pit_check.py (subprocess, fresh state file) and, in-process,
     through the snapshot channel of ext/snapshot_channel.py; verdicts are
     compared with "expected" in dataset.json.

Writes results/<ds>.json and results/validation.md. Exit code 1 if any
control disagrees with its expected verdict.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ext.snapshot_channel import snapshot_diff  # noqa: E402
from lib.common import DATA, DATASETS, read_json, write_json  # noqa: E402
from lib.tasks import LABELS  # noqa: E402

OUT = ROOT / "results"
ALL = ["olist", "f1", "pkdd99", "adventureworks", "northwind", "sakila", "movietweetings", "tpch"]


USE_DATE_COLS = False


def run_pit_check(skill_scripts, code, ds, man, map_path):
    with tempfile.TemporaryDirectory() as tmp:
        entry = ([str(ROOT / "ext" / "pit_check_dates.py"), "--skill-scripts", str(skill_scripts)]
                 if USE_DATE_COLS else [str(Path(skill_scripts) / "pit_check.py")])
        cmd = [sys.executable, "-I", *entry, "--json",
               "--code", str(code), "--db-dir", str(DATA / ds), "--availability-map", str(map_path),
               "--entity-table", man["entity"]["table"], "--entity-column", man["entity"]["column"],
               "--n-entities", str(man["entity"]["n_entities"]),
               "--dev-seeds", ",".join(man["seeds"]["dev"]),
               "--held-out-seeds", ",".join(man["seeds"]["held_out"]),
               "--state", str(Path(tmp) / "pit_state.json"), "--timeout", "120"]
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        out = {"verdict": "HARNESS_ERROR", "stderr": p.stderr[-800:]}
    out["exit_code"] = p.returncode
    return out


def summarize_v1(r):
    seeds = r.get("per_seed", [])
    return {"verdict": r.get("verdict"),
            "witness": any(s["witness"] for s in seeds),
            "canary": any(s["canary"] for s in seeds),
            "witness_columns": r.get("witness_columns", []),
            "canary_columns": r.get("canary_columns", []),
            "per_seed": seeds, **({"error": r["error"]} if "error" in r else {}),
            **({"stderr": r["stderr"]} if "stderr" in r else {})}


def agrees(expected, v1, snap):
    ok = expected["pit_check"]["verdict"] == v1["verdict"]
    for k in ("witness", "canary"):
        e = expected["pit_check"].get(k)
        if e is not None:
            ok = ok and (bool(e) == v1[k])
    if "snapshot_ext" in expected and expected["snapshot_ext"] is not None:
        ok = ok and (bool(expected["snapshot_ext"]) == bool(snap))
    return ok


def map_sanity(db, amap, man):
    issues, info = [], {}
    refs = []
    for t, c in amap["time_cols"].items():
        refs.append((t, c, "time"))
    for kind in ("self_availability_cols", "gatekeeper_cols"):
        for t, cols in amap.get(kind, {}).items():
            refs += [(t, c, kind) for c in cols]
    for t, spec in amap.get("snapshot_cols", {}).items():
        for c, as_of in spec.items():
            refs.append((t, c, "snapshot"))
            if isinstance(as_of, str) and as_of in db.get(t, pd.DataFrame()).columns:
                refs.append((t, as_of, "snapshot_as_of"))
    for t, c, kind in refs:
        if t not in db:
            issues.append(f"{kind}: table {t} missing")
        elif c not in db[t].columns:
            issues.append(f"{kind}: column {t}.{c} missing")
    lo, hi = [], []
    for t, c in amap["time_cols"].items():
        if t in db and c in db[t].columns:
            s = db[t][c]
            if not pd.api.types.is_datetime64_any_dtype(s):
                issues.append(f"time column {t}.{c} did not parse as datetime")
                continue
            info[f"{t}.{c}"] = {"min": str(s.min()), "max": str(s.max()), "missing": round(float(s.isna().mean()), 4)}
            lo.append(s.min()); hi.append(s.max())
    seeds = man["seeds"]["dev"] + man["seeds"]["held_out"]
    for s in seeds:
        if not (min(lo) <= pd.Timestamp(s) <= max(hi)):
            issues.append(f"seed {s} outside data range {min(lo)} .. {max(hi)}")
    return issues, info


def power(db, amap, ids, col, seeds):
    out = {}
    for s in seeds:
        t = pd.Timestamp(s)
        seen = set()
        for tab, tc in amap["time_cols"].items():
            df = db.get(tab)
            if df is not None and col in df.columns and tc in df.columns:
                seen |= set(df.loc[df[tc] <= t, col].dropna().unique())
        out[s] = int(sum(1 for i in ids if i in seen))
    return out


def validate(ds, skill_scripts, pit_check, core):
    t0 = time.time()
    man = read_json(DATASETS / ds / "dataset.json")
    map_path = DATASETS / ds / man["availability_map"]
    amap_raw = read_json(map_path)
    amap = pit_check.load_availability_map(map_path)
    db = core.load_db(DATA / ds, parse_dates=pit_check.parse_dates_from_map(amap))
    issues, ranges = map_sanity(db, {**amap, "snapshot_cols": amap_raw.get("snapshot_cols", {})}, man)
    ent = man["entity"]
    ids = core.sample_entities(db, ent["table"], ent["column"], ent["n_entities"], seed=0)
    seeds = man["seeds"]["dev"] + man["seeds"]["held_out"]
    pw = power(db, amap, ids, ent["column"], seeds)

    labels = {}
    for task in man.get("tasks", []):
        fn = LABELS[task["name"]]
        labels[task["name"]] = {s: round(float(fn(db, ids, s).mean()), 3) for s in seeds}

    import numpy as np
    controls = {}
    for code_path in sorted((DATASETS / ds / "controls").glob("*.py")):
        spec = man["controls"].get(code_path.name)
        if spec is None:
            issues.append(f"control {code_path.name} has no expected verdict in dataset.json")
            continue
        v1 = summarize_v1(run_pit_check(skill_scripts, code_path, ds, man, map_path))
        fn = pit_check.load_program(code_path.read_text(encoding="utf-8"), pd, np)
        snap_cols = {}
        for s in seeds:
            cols = snapshot_diff(fn, db, ids, s, amap_raw.get("snapshot_cols", {}),
                                 core.frames_equal, core.call_with_timeout)
            if cols:
                snap_cols[s] = cols
        rec = {"class": spec["class"], "expected": spec["expected"], "pit_check": v1,
               "snapshot_ext": {"hit": bool(snap_cols), "columns_by_seed": snap_cols}}
        rec["agrees"] = agrees(spec["expected"], v1, snap_cols)
        if "expected_naive_map" in spec:
            naive = DATASETS / ds / "availability_map.naive.json"
            v1n = summarize_v1(run_pit_check(skill_scripts, code_path, ds, man, naive))
            rec["pit_check_naive_map"] = v1n
            rec["agrees_naive_map"] = spec["expected_naive_map"]["pit_check"]["verdict"] == v1n["verdict"]
            rec["agrees"] = rec["agrees"] and rec["agrees_naive_map"]
        controls[code_path.name] = rec
    res = {"dataset": ds, "map_issues": issues, "time_ranges": ranges,
           "n_entities_sampled": int(len(ids)), "entities_with_history": pw,
           "label_prevalence": labels, "controls": controls,
           "all_agree": all(c["agrees"] for c in controls.values()) and not issues,
           "seconds": round(time.time() - t0, 1)}
    OUT.mkdir(exist_ok=True)
    write_json(res, OUT / f"{ds}.json")
    return res


def md_table(results):
    lines = ["| dataset | control | class | expected (pit-check / snapshot ext) | pit-check v1 | snapshot ext | agrees |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        for name, c in r["controls"].items():
            e = c["expected"]
            exp = f'{e["pit_check"]["verdict"]} / {"hit" if e.get("snapshot_ext") else "—"}'
            v = c["pit_check"]
            ch = "+".join(k for k in ("witness", "canary") if v[k]) or "—"
            got = f'{v["verdict"]} ({ch})'
            if "pit_check_naive_map" in c:
                got += f'; naive map: {c["pit_check_naive_map"]["verdict"]}'
            lines.append(f'| {r["dataset"]} | {name} | {c["class"] or "—"} | {exp} | {got} | '
                         f'{"hit" if c["snapshot_ext"]["hit"] else "—"} | {"yes" if c["agrees"] else "**NO**"} |')
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill-scripts", required=True)
    ap.add_argument("--date-cols", action="store_true",
                    help="run pit_check through ext/pit_check_dates.py (map key date_cols)")
    ap.add_argument("datasets", nargs="*")
    a = ap.parse_args()
    global USE_DATE_COLS, OUT
    USE_DATE_COLS = a.date_cols
    OUT = ROOT / ("results_date_cols" if a.date_cols else "results")
    sys.path.insert(0, a.skill_scripts)
    import checker_core as core
    import pit_check
    results = []
    for ds in a.datasets or ALL:
        r = validate(ds, a.skill_scripts, pit_check, core)
        bad = [n for n, c in r["controls"].items() if not c["agrees"]]
        print(f"{ds:15s} controls={len(r['controls'])} disagree={bad} map_issues={r['map_issues']} "
              f"power={r['entities_with_history']} {r['seconds']}s", flush=True)
        results.append(r)
    if not a.datasets:
        (OUT / "validation.md").write_text(md_table(results) + "\n", encoding="utf-8")
    sys.exit(0 if all(r["all_agree"] for r in results) else 1)


if __name__ == "__main__":
    main()
