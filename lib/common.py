"""Shared helpers for PITBench dataset preparation.

Every prepared dataset follows one contract (see README.md, "Dataset format"):

* data/<ds>/<table>.csv   -- one CSV per table, snake_case table and column
                             names, every key column ends in ``_id``, every
                             timestamp written as ISO ``YYYY-MM-DD HH:MM:SS``;
* datasets/<ds>/availability_map.json   -- the pit-check map (time_cols,
                             self_availability_cols, gatekeeper_cols, key_hints)
                             plus the PITBench extension ``snapshot_cols``,
                             which the current pit-check ignores;
* datasets/<ds>/dataset.json            -- manifest: source, entity, seeds,
                             tasks, roughness notes, expected control verdicts.
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
DATA = ROOT / "data"
DATASETS = ROOT / "datasets"

ISO = "%Y-%m-%d %H:%M:%S"


def snake(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return re.sub(r"_+", "_", s).strip("_").lower()


def snake_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: snake(c) for c in df.columns})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_ts(s: pd.Series, **kw) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", **kw)


def write_tables(tables: dict, out_dir: Path) -> dict:
    """Write every table as CSV with ISO timestamps; return a size summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):
        old.unlink()
    summary = {}
    for name, df in tables.items():
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].dt.strftime(ISO)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        summary[name] = {"rows": int(len(df)), "columns": list(df.columns)}
    return summary


def raw_manifest(raw_dir: Path) -> dict:
    return {p.name: sha256(p) for p in sorted(raw_dir.iterdir()) if p.is_file()}


def write_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
