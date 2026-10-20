"""Re-run the original Olist LEAK corpus (commit a9cd898) on PITBench's olist build."""
import csv, json, re, subprocess, sys, tempfile
from pathlib import Path
C = Path(sys.argv[1]); BENCH = Path(sys.argv[2]); SKILL = sys.argv[3]
labels = {(r["model"], int(r["idx"])): r["mechanism"] for r in csv.DictReader(open(C / "mechanism_labels.csv"))}
recs = [json.loads(l) for l in open(C / "results.jsonl")]
man = json.loads((BENCH / "datasets/olist/dataset.json").read_text())
jobs = []
for r in recs:
    key = (r["model"], int(r["idx"]))
    if r.get("verdict") != "LEAK" or labels.get(key) == "GENERATION_ARTIFACT":
        continue
    code = r["code"]
    jobs.append((key, code))

def run(job):
    key, code = job
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "get_features.py"; f.write_text(code)
        cmd = [sys.executable, "-I", f"{SKILL}/pit_check.py", "--json", "--code", str(f),
               "--db-dir", str(BENCH / "data/olist"), "--availability-map", str(BENCH / "datasets/olist/availability_map.json"),
               "--entity-table", "products", "--entity-column", "product_id", "--n-entities", "15",
               "--dev-seeds", ",".join(man["seeds"]["dev"]), "--held-out-seeds", ",".join(man["seeds"]["held_out"]),
               "--state", str(Path(tmp) / "s.json")]
        p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        v = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        v = {"verdict": "HARNESS_ERROR"}
    return {"model": key[0], "idx": key[1], "mechanism": labels[key], "verdict": v.get("verdict"),
                "witness_columns": v.get("witness_columns"), "error": v.get("error")}
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(4) as ex:
    out = list(ex.map(run, jobs))
(C / "compat_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
from collections import Counter
print(Counter((o["mechanism"], o["verdict"]) for o in out))
