#!/usr/bin/env python3
"""Build data/<ds>/*.csv for every dataset (or the ones named) from raw/<ds>/.

    python -I prepare_all.py                 # all datasets
    python -I prepare_all.py f1 tpch         # a subset

Writes data/<ds>/ and datasets/<ds>/build_info.json (row counts, columns,
sha256 of every raw input file). Raw inputs are never modified.
"""
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib.common import DATA, DATASETS, RAW, raw_manifest, write_json, write_tables  # noqa: E402

ALL = ["olist", "f1", "pkdd99", "adventureworks", "northwind", "sakila", "movietweetings", "tpch"]


def load_module(ds):
    spec = importlib.util.spec_from_file_location(f"prep_{ds}", DATASETS / ds / "prepare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(names):
    for ds in names:
        t0 = time.time()
        tables = load_module(ds).prepare()
        summary = write_tables(tables, DATA / ds)
        info = {"dataset": ds, "tables": summary, "raw_sha256": raw_manifest(RAW / ds),
                "build_seconds": round(time.time() - t0, 1)}
        write_json(info, DATASETS / ds / "build_info.json")
        rows = sum(v["rows"] for v in summary.values())
        print(f"{ds:15s} {len(summary):2d} tables {rows:>9d} rows  {info['build_seconds']}s")


if __name__ == "__main__":
    main(sys.argv[1:] or ALL)
