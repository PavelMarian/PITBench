"""Build an agent workspace: the raw tables, a loader, and a task card.

The task card is written the way a data-science ticket is written: business
question, entity, horizon, interface. It never mentions leakage, point-in-time
correctness or availability -- whether the agent handles time correctly is
what the scenario measures. The data dictionary lists tables, columns and row
counts only; the availability map stays with the benchmark.
"""
import json
import shutil
import subprocess
from pathlib import Path

from lib.common import DATA, DATASETS, read_json

PROMPT = """You are a data scientist. Everything you need is in the current directory:
TASK.md (the task and the required interface), data/ (the tables as CSV) and
pitbench_io.py (a loader). Explore the data with Python, then write pipeline.py
implementing the four functions described in TASK.md. Make sure pipeline.py
imports cleanly and that fit_predict runs end to end on the full database for
one of the example prediction times. Finally write report.json with your own
validation estimate as {{"validation_auc": <float>, "how_validated": "<one sentence>"}}.
Use only {libs} in pipeline.py."""

OPTIONAL_LIBS = {"sklearn": "scikit-learn", "lightgbm": "lightgbm"}


def available_libs(python_bin_dir):
    """pandas + numpy always; scikit-learn / lightgbm only if the agent's interpreter has them."""
    code = "import importlib.util as u;print(' '.join(m for m in %r if u.find_spec(m)))" % list(OPTIONAL_LIBS)
    out = subprocess.run([str(Path(python_bin_dir) / "python"), "-c", code], capture_output=True, text=True)
    return ["pandas", "numpy"] + [OPTIONAL_LIBS[m] for m in out.stdout.split()]


def prompt(libs):
    return PROMPT.format(libs=", ".join(libs))

LOADER = '''"""Load the task database: one pandas DataFrame per CSV in data/, date columns parsed."""
from pathlib import Path

import pandas as pd

DATE_COLUMNS = {date_columns}


def load_db(path="data"):
    db = {{}}
    for f in sorted(Path(path).glob("*.csv")):
        cols = DATE_COLUMNS.get(f.stem, [])
        db[f.stem] = pd.read_csv(f, parse_dates=cols, low_memory=False) if cols else pd.read_csv(f, low_memory=False)
    return db
'''

CARD = """# Task: {task_name}

**Business question.** {description}. For every entity in `{entity_table}.{entity_column}`
we want a score at a given prediction time `seed_time` (the moment the prediction is made),
higher = more likely.

**Horizon.** {horizon} days after `seed_time`.

**Database.** {title} ({domain}). Load it with:

```python
from pitbench_io import load_db
db = load_db("data")          # dict: table name -> DataFrame, date columns already parsed
```

{dictionary}

**Example prediction times.** {seeds}

## Required interface (pipeline.py)

```python
def select_entities(db, seed_time) -> list:
    # entities for which a prediction should be made at seed_time

def make_label(db, entity_ids, seed_time) -> pd.DataFrame:
    # index = entity_ids, one column "label" (0/1): the target for (entity, seed_time)

def get_features(db, entity_ids, seed_time) -> pd.DataFrame:
    # index = entity_ids, numeric feature columns

def fit_predict(db, entity_ids, seed_time) -> pd.DataFrame:
    # train a model on past examples and return index = entity_ids, one column "score"
```

`db` is always the dictionary returned by `load_db`. `seed_time` is a `pandas.Timestamp`.
`entity_ids` is a list of ids from `{entity_table}.{entity_column}`. All four functions must be
deterministic (fix random seeds). Allowed libraries: {libs}.
"""

TASKS = {  # dataset -> (task name in lib/tasks.py, description, binary label rule)
    "olist": "olist_product_sold_next_90d",
    "f1": "f1_driver_podium_next_60d",
    "pkdd99": "pkdd99_account_low_activity_next_90d",
    "adventureworks": "adventureworks_product_sold_next_90d",
    "northwind": "northwind_customer_orders_next_90d",
    "sakila": "sakila_customer_rentals_next_14d",
    "movietweetings": "movietweetings_user_active_next_180d",
    "tpch": "tpch_customer_orders_next_180d",
}


def date_columns(ds):
    amap = read_json(DATASETS / ds / "availability_map.json")
    out = {}
    for t, c in amap["time_cols"].items():
        out.setdefault(t, []).append(c)
    for key in ("self_availability_cols", "date_cols"):
        for t, cols in amap.get(key, {}).items():
            out.setdefault(t, [])
            out[t] += [c for c in cols if c not in out[t]]
    return out


def build(ds, ws: Path, libs=("pandas", "numpy")):
    man = read_json(DATASETS / ds / "dataset.json")
    info = read_json(DATASETS / ds / "build_info.json")
    task = next(t for t in man["tasks"] if t["name"] == TASKS[ds])
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "data").mkdir(parents=True)
    for f in sorted((DATA / ds).glob("*.csv")):
        shutil.copy2(f, ws / "data" / f.name)
    (ws / "pitbench_io.py").write_text(LOADER.format(date_columns=json.dumps(date_columns(ds), indent=1)))
    dictionary = "| table | rows | columns |\n|---|---|---|\n" + "\n".join(
        f"| `{t}` | {v['rows']} | {', '.join(v['columns'])} |" for t, v in info["tables"].items())
    card = CARD.format(task_name=task["name"], description=task["description"][0].upper() + task["description"][1:],
                       entity_table=man["entity"]["table"], entity_column=man["entity"]["column"],
                       horizon=task["horizon_days"], title=man["title"], domain=man["domain"],
                       dictionary=dictionary, seeds=", ".join(man["seeds"]["dev"]), libs=", ".join(libs))
    (ws / "TASK.md").write_text(card)
    return {"dataset": ds, "task": task["name"], "workspace": str(ws), "allowed_libs": list(libs)}
