# PITBench — relational databases prepared for point-in-time checking across the ML stack

Eight relational databases, each brought to the input contract of the
`pit-repair` skill (`pit_check.py`): a directory of CSV tables, an
availability map, an entity to sample, prediction times, and prediction
tasks. Each comes with control programs whose verdicts under the checker
are known, and with a list of its temporal "roughness": the places where
the data does not fit point-in-time checking directly, and how each one was
handled.

Nothing here changes the checker. Everything runs against the unmodified
`.claude/skills/pit-repair/scripts/pit_check.py`. Two optional map keys,
both ignored by `pit_check.py`, add what the data needed:

* `snapshot_cols` — the **snapshot channel** (`ext/snapshot_channel.py`),
  for values stored as their end-of-data state;
* `date_cols` — undeclared date columns to parse as datetimes, used through
  the wrapper `ext/pit_check_dates.py` (see "Olist corpus compatibility").

## Layout

```
fetch_raw.sh               download / generate raw inputs into raw/<ds>/
prepare_all.py             raw/<ds>/ -> data/<ds>/*.csv  (+ datasets/<ds>/build_info.json)
validate.py                map sanity, power, label prevalence, control verdicts -> results/
validate_layers.py         task / model / context / agent layer checks + harm -> results_layers/
layers/harness.py          views per layer (truncate, canary, snapshot, tie, anonymize), AUC
layers/make_layer_controls.py   generates datasets/<ds>/layers/ from templates
layers/sandbox.py          pit-check's AST filter with a wider import list (agent pipelines)
frameworks/adapters.py     featuretools / LightGBM / AutoGluon / PyG as layer programs (optional)
validate_frameworks.py     framework configurations through the layer views -> results_frameworks/
agent/run_agents.py        agent scenario, network step (backends: claude, openrouter)
agent/check_runs.py        agent scenario / corpus, execution step: all layers + harm -> results_agents/
lib/sampling.py            entities eligible at t (active in a look-back window), per seed
mutate/                    mutation operators, runner, triage of reference-pipeline mutants -> results_mutations/
corpus/                    external code: layout, wrapping rules, add_program.py, FSE Olist corpus importer
lib/common.py              normalisation helpers
lib/tasks.py               label functions of every task
ext/snapshot_channel.py    the snapshot channel (PITBench extension)
ext/pit_check_dates.py     pit_check wrapper that also parses the map's date_cols
tools/compat_olist_corpus.py  re-run of the original Olist LEAK corpus on this build
datasets/<ds>/
  prepare.py               source-specific adaptation (docstring = what was changed and why)
  availability_map.json    pit-check map (+ snapshot_cols)
  dataset.json             manifest: source, licence, entity, seeds, tasks, roughness, controls
  controls/*.py            get_features programs with expected verdicts
  build_info.json          row counts, columns, sha256 of every raw file used
  layers/<layer>/*.py      layer controls (generated); layers/expected.json
results/                   validation with plain pit_check (per-dataset JSON + validation.md)
results_date_cols/         the same validation through ext/pit_check_dates.py
results/olist_corpus_compat.json   original Olist corpus re-checked on this build
```

Reproduce from scratch:

```bash
PITBULL_REPO=/path/to/pitbull ./fetch_raw.sh
python -I prepare_all.py
python -I validate.py --skill-scripts /path/to/pitbull/.claude/skills/pit-repair/scripts
python -I validate.py --date-cols --skill-scripts /path/to/pitbull/.claude/skills/pit-repair/scripts
```

Tested with Python 3.11, pandas 2.2.3, numpy 2.2.6 — the versions
`pit-repair` is validated on.

## Dataset contract

* `data/<ds>/<table>.csv`: snake_case table and column names. Every key
  column ends in `_id`, so maps can use `key_hints: ["_id"]`; the default
  `"id"` hint also matches columns like `paid` or `valid`. Every timestamp
  is ISO `YYYY-MM-DD HH:MM:SS`.
* Every table whose rows live in time has a time column in the map. Child
  tables without a time of their own (order lines, payments, results)
  get an explicit inherited column (`ts` / `available_at`). The inheritance
  is listed as roughness, never hidden.
* Columns that are not declared in the map load as strings
  (`checker_core.load_db` parses only declared time columns). Every other
  date column — planned dates, birth dates, the raw midnight `race_date` —
  is listed under the map's `date_cols`, so `ext/pit_check_dates.py` loads
  it as a datetime. Under plain `pit_check.py`, feature code that parses
  such a column itself must use `errors="coerce"`: canary replaces strings
  in future rows with a sentinel.
* `dataset.json → entity.n_entities` is set per dataset so the controls have
  power; `pit_check`'s default of 15 is too few for sparse entities.

## Datasets

| dataset | domain | kind | tables / rows | time span | entity (n) | task(s) |
|---|---|---|---|---|---|---|
| `olist` | marketplace | real | 8 / 0.94M | 2016-09 … 2018-10 | product (300) | product sold in next 90d |
| `f1` | sports | real | 9 / 59k | 2000-03 … 2026-10 | driver (60) | podium in next 60d (DNF as second task) |
| `pkdd99` | banking | real | 8 / 1.08M | 1993-01 … 1998-12 | account (200) | < 13 transactions in next 90d |
| `adventureworks` | retail / manufacturing | sample | 11 / 174k | 2022-05 … 2025-06 | product (100) | product sold in next 90d |
| `northwind` | wholesale | sample | 8 / 3.2k | 1996-07 … 1998-05 | customer (91) | customer orders in next 90d |
| `sakila` | rental | sample | 9 / 39k | 2005-05 … 2006-02 | customer (200) | rentals in next 14d (count) |
| `movietweetings` | social web / recsys | real | 4 / 1.12M | 2013-02 … 2021-11 | user (400) | user active in next 180d |
| `tpch` | order entry | synthetic, generated | 8 / 0.43M | 1992-01 … 1998-08 | customer (400) | customer orders in next 180d |

`entity (n)` is the sample size. The feature-layer check (`pit_check.py`) samples once
from the entity column, as the skill does; the layer, agent and corpus checks sample
entities active at each prediction time (`lib/sampling.py`).

Sources and licences are listed in each `dataset.json`. Olist is
CC BY-NC-SA and PKDD'99 is research-use only; check both before any
redistribution.

## Roughness: what does not fit directly, and how it is handled

Classes used across `dataset.json → roughness` and `controls → class`:

| class | meaning | caught by |
|---|---|---|
| `row_cutoff` | rows of a joined event table are not cut at t | witness (v1) |
| `late_value` | a column filled after its row's time (delivery, return, death date, 1996 statistics) | canary (v1); snapshot ext for non-date values |
| `neighbor_path` | a statistic of a neighbouring entity computed on full history | witness (v1) |
| `entity_existence` | an entity used before it exists; existence time missing or wrong | witness when the entity table has a time column; otherwise documented |
| `boundary` | granularity / tie at t (midnight-dated events that happen later that day) | witness, **only** with the right time column in the map and seeds on event days |
| `snapshot` | mutable value stored as its final state (status, current price, balance) | **snapshot ext** (blind spot of v1) |
| `dump_aggregate` | aggregate computed at dump time over all history (career totals, YTD sales) | **snapshot ext** |
| `time_inheritance` | child rows without their own time take the parent's | map choice (documented per dataset) |
| `wrong_time_column` | a planned or future-valued date looks like the row time (due/commit/estimated dates) | map choice |
| `restamp` | load-time or process-time stamps that look like row time (`modified_date`, `last_update`) | map choice / documented |

Per-dataset highlights:

* **olist**: final `order_status` is a snapshot. The derived
  `order_status_events` table gives the status as of t. `payments` and
  `order_items` have no time of their own; `payments.approved_at` is
  offered as the stricter map variant.
* **f1**: the source has a midnight `date` plus an often-missing `time`.
  With `available_at` (race start, or 23:59) in the map, a feature filtering
  `race_date <= t` is caught on race-day seeds. Under
  `availability_map.naive.json` (time = source date) the same leak passes
  CLEAN. This ablation is part of validation. Driver and constructor
  career totals are dump-time aggregates.
* **pkdd99**: `loan.status` (the classic PKDD target) is end-of-data state,
  so the dataset ships it as an *anti-task*. `standing_orders` have no date.
  District 1996 statistics are valid only from 1997-01-01. Dates were
  YYMMDD integers, and the birth date and sex were hidden in `birth_number`.
* **adventureworks**: `product.list_price` and `standard_cost` are current
  values, while the true history sits in `product_*_history`. That gives
  ground truth for the snapshot class. `modified_date` is a load-time or
  ship-time re-stamp.
* **northwind**: `products.unit_price` is current; the price actually
  charged is `order_details.unit_price`.
* **sakila**: `customer.create_date` is 2006-02-14 for everyone, which is
  after all their rentals, so it is an unusable existence time and is
  deliberately not in the map. `last_update` is load time. `payment.amount`
  includes late fees but is stamped at rental time.
* **movietweetings**: only ratings carry time. Movie popularity must be
  computed on ratings ≤ t (`neighbor_path`).
* **tpch**: by the TPC-H specification, `return_flag`, `line_status` and
  `order_status` are frozen at CURRENTDATE = 1995-06-17. That gives a
  snapshot with ground truth by construction: a leak before that date and
  merely stale after it.

## Snapshot channel (extension)

```json
"snapshot_cols": {"<table>": {"<column>": null | "<as_of column>" | "YYYY-MM-DD"}}
```

`as_of` says from when the stored value is valid:

* `null` — only at the end of the data;
* a column name — from that column's value onward, per row;
* a date — a fixed date for the whole table.

At each t the channel reruns the program with values that are not yet
valid perturbed: numbers shifted, strings and bools replaced by a
sentinel. Any output column that moves is a hit. pit-check v1 has no way
to see these values, because the rows themselves exist before t.

## Validation (all datasets)

`validate.py` runs every control through the unmodified `pit_check.py`
(6 seeds: 3 dev + 3 held-out) and through the snapshot channel, and
compares with `dataset.json → expected`. Result:
**34/34 controls agree with their expected verdicts**, both with plain
`pit_check.py` (`results/`) and through the `date_cols` wrapper
(`results_date_cols/`). The per-control
table is in `results/validation.md`; per-seed columns are in
`results/<ds>.json`.

Summary by kind of control:

* `correct*.py` (9 programs): CLEAN on witness, canary and snapshot at every seed.
* `leak_row_cutoff.py` (8): LEAK, witness at 6/6 seeds.
* `leak_late_value.py` (5): LEAK through canary only (witness silent), 6/6 seeds.
* `leak_neighbor_path.py` (movietweetings): LEAK, witness only.
* `leak_entity_existence.py` (pkdd99): LEAK, witness at the 4 seeds where some sampled accounts are not yet opened.
* `leak_boundary.py` (f1): LEAK under the `available_at` map, CLEAN under the naive map.
* `snapshot_*.py` (9): **CLEAN under pit-check v1** (known blind spot). The snapshot channel fires exactly at the seeds where the stored value is not yet valid. For example, TPC-H `return_flag` fires at the 4 seeds before 1995-06-17, AdventureWorks `list_price` at the 4 seeds before 2024-05-29, and PKDD 1996 district statistics only at 1996-07-01.

## Layers: task → features → model → context → agent

The benchmark checks the whole stack of the main-track plan
(`PLAN_main_track.md`, steps H1/H2 and the B1/B3 measurements), not just
feature code. Every layer uses one invariant: **the layer's output for
prediction time t does not change when its input is replaced by what was
knowable at t**. Only the view of the database changes from layer to layer.

| layer | program interface | views compared | controls (per dataset) |
|---|---|---|---|
| task | `select_entities(db, t)`, `make_label(db, ids, t)` | selection: D vs D\|t, canary_t, snapshot_t. label: D vs D\|(t+h), canary_(t+h), snapshot_(t+h), tie_t (rows exactly at t removed), tie-injection_t (each entity's latest past event moved to exactly t) | correct (binary and count label); selection by whole history (survivorship); label without the horizon cap; label window closed at t (binary and count); label from an end-of-data snapshot |
| features | `get_features(db, ids, t)` | `pit_check.py` (witness, canary) + snapshot channel | see "Validation" above |
| model | `fit_predict(db, ids, t)` (builds training set, trains ridge, predicts) | D vs D\|t, canary_t, snapshot_t | correct (train seeds with t_i + h ≤ t); label overlap (no embargo); target encoding on future labels; neighbour sampling without time (GNN analogue) |
| context | `build_context(db, ids, t)` (the serialized context, i.e. the prompt) + `predict` (kNN in-context learner) | D vs D\|t, canary_t, snapshot_t | correct; examples whose labels reach past t; retrieval over the whole table |
| agent | `fit_predict` | D vs anonymize(D): ids hashed, dates shifted 2555 days | correct; prior on memorised entities; prior tied to an absolute date |

All layer programs are pandas/numpy only and load through `pit_check.load_program`,
so they run under the same sandbox as feature code. The context check
compares the serialized context itself. If the context is invariant, any
deterministic in-context learner fed that context is invariant too,
including an LLM at temperature 0. So the check covers LLM in-context
learning without calling an LLM.

Generated by `layers/make_layer_controls.py` (one template per control,
dataset specifics substituted) into `datasets/<ds>/layers/`, with
expectations in `datasets/<ds>/layers/expected.json`. Run with:

```bash
python -I validate_layers.py --skill-scripts /path/to/pitbull/.claude/skills/pit-repair/scripts
```

### Sampling: entities active at t

Entities are drawn **at each prediction time** from those with at least one
event in a look-back window before t (`lib/sampling.py`; the window is
`entity.eligible_days` in `layers/expected.json`). The rule uses only data
≤ t. Earlier versions drew once from the whole id column, so most entities
were inactive at a given t: retired F1 drivers, dormant customers. Their
labels were trivially 0 and harm was not measurable (F1 AUC ≈ 0.99).

### Difficulty gate: which tasks can measure harm

A task counts as informative for harm only if two conditions hold. Its
label prevalence must lie in [0.05, 0.95] at ≥ 4 of 6 seeds. The correct
model's AUC must lie in [0.55, 0.95] (`results_layers/difficulty.md`).

| dataset | entities per seed | label prevalence | correct-model AUC | informative |
|---|---|---|---|---|
| olist | 300 | 0.17–0.28 | 0.65 | yes |
| f1 (podium in next 60 days) | 22–27 | 0.20–0.41 | 0.90 | yes |
| pkdd99 (< 13 transactions in 90 days) | 200 | 0.09–0.32 | 0.94 | yes |
| adventureworks | 60–100 | 0.74–0.98 | 0.76 | yes |
| northwind | 74–87 | 0.62–0.83 | 0.61 | yes |
| sakila | 200 | 0.68–1.00 | 0.54 | **no** |
| movietweetings | 400 | 0.28–0.43 | 0.80 | yes |
| tpch | 400 | 0.66–0.70 | 0.55 | yes |

The gate caught two broken tasks, and both were fixed in the task
definition:

* F1 "any race result in 60 days" became **"podium in 60 days"**, via a label
  row filter (`LFILT`).
* PKDD "< 3 transactions" was never true for an active account and became
  **"< 13 transactions"**, roughly the bottom quarter.

Sakila stays non-informative. The data covers one rental season of about
3.5 months, and within it activity swings from almost nobody to almost
everybody renting in 14 days, so no fixed label threshold is stable. Its
harm numbers are shown but excluded from medians.

### Detection: 141/141 layer controls agree, reference pipelines clean

Across 8 datasets, 141 checks pass:

* 133 generated controls: task 53 (incl. count-label variants), model 32,
  context 24, agent 24;
* 8 reference pipelines (all four functions, correct). Each is silent on
  every layer, view and seed.

Every leaking control fires on its expected channel
(`results_layers/summary.md`).

**Boundary errors are now detected wherever they exist in code.** The
natural tie view only fires when events happen to sit exactly at t *and*
flip the binary label. With binary labels it stayed silent on PKDD and
AdventureWorks. The new **tie-injection view** moves, for each sampled
entity, its latest past event to exactly t. The event stays in the past
and no row is added, so a label on (t, t+h] cannot change (nor can one
that also counts the whole past), while one on [t, t+h] must. An earlier
version added a copy of the event instead; that changed the past and
raised one false alarm in mutation testing (pkdd99, a label counting all
past events). Both the binary and the new count-label boundary controls
fire on all 8 datasets. Detection and harm are thus separated: injection
finds the error, and the natural data says whether it matters (often it
does not).

### Harm: backtest AUC minus AUC at t (informative tasks)

The model or context program is run on D (what a backtest reports) and on
D|t (what it would have produced at t). Both outputs are scored against
the true labels of (t, t+h], and the table reports the gap, as the mean
over 6 seeds. Full table: `results_layers/harm.md`.

| control | median gap (7 informative tasks) | tasks with abs(gap) > 0.02 |
|---|---|---|
| model: label overlap (no embargo) | +0.004 | 2/7 (pkdd +0.10, tpch +0.06) |
| model: target encoding on future labels | +0.131 | 7/7 |
| model: neighbour sampling without time (GNN analogue) | +0.158 | 5/7 |
| context: examples with future labels | +0.042 | 7/7 |
| context: retrieval over the whole table | +0.071 | 6/7 |

Observations, not yet claims:

* **Label overlap**, the most common split error, rarely matters (2 of 7
  tasks). Future target encoding, time-blind neighbour sampling and
  whole-table retrieval matter almost everywhere.
* **Harm depends on the model.** With LightGBM (`results_frameworks/`),
  label overlap gives about +0.03, versus about 0 with ridge here. Harm
  should be reported per model family.
* **Agent-layer controls are synthetic priors.** They test that the
  anonymization check detects identity and calendar dependence; their AUC
  gaps carry no meaning.

### What this does not cover

* **No GNN, no TabPFN or relational foundation model, no LLM calls.** The
  model layer uses a ridge model with a model-internal neighbour
  aggregation as the analogue of non-temporal GNN sampling. The context
  layer uses a kNN learner and checks the context, i.e. the prompt. Plugging
  in a real GNN, TabPFN or an LLM only needs an adapter with the same
  `fit_predict` / `build_context` signature.
* **No RelBench.** The datasets are the eight PITBench databases. RelBench
  (plan step A1) was not reachable from the build environment; adding it
  needs an adapter in `datasets/` like the others.
* **Hand-written violations.** Controls and mutants measure recall and false
  alarms; how often errors occur comes only from natural code (agent runs and
  `corpus/`), see "Natural code" below.

## Real frameworks (optional adapters)

`frameworks/adapters.py` wraps featuretools (DFS with and without
`cutoff_time`), LightGBM / scikit-learn, AutoGluon and PyTorch Geometric
(GraphSAGE over a temporal vs a non-temporal graph) as layer programs. None
of them is a PITBench dependency: each adapter imports its framework when
called, and `validate_frameworks.py` skips families that are not installed.
Configurations, required packages and the validation status are in
`frameworks/README.md`. featuretools (9/9) and LightGBM (24/24) were
validated in an earlier run; AutoGluon and PyG are written but not
validated here.

```bash
pip install lightgbm scikit-learn            # or any subset of the families
python -I validate_frameworks.py --skill-scripts /path/to/pitbull/.claude/skills/pit-repair/scripts
```

## Agent scenario: agents solve the task from raw data

An agent gets a fresh workspace: the tables (`data/*.csv`), a loader
(`pitbench_io.py`) and a task card (`TASK.md`). The card has the business
question, the entity, the horizon, a data dictionary and the required
interface — `select_entities`, `make_label`, `get_features`,
`fit_predict`. It never mentions leakage or availability. The agent writes
`pipeline.py` and `report.json` (its own validation AUC). Every pipeline is
then checked on all layers: task, features, model, and identity/calendar
dependence. The result is "AUC claimed by the agent / AUC backtest / AUC
at t" side by side.

Two backends, one contract (`agent/backends.py`):

* **`claude`** — Claude Code headless (`claude -p`) with its own tools;
* **`openrouter`** — any model on OpenRouter through its OpenAI-compatible
  API, in two modes:
  * `tools`: an agent loop with `list_files`, `read_file`, `write_file`,
    `run_python`;
  * `single`: one completion; `pipeline.py` is taken from the ```python
    block, as in the protocol of the original Olist corpus.

  Several models can be run in one call (`--model a b c`).

```bash
# network step (needs OPENROUTER_API_KEY and network access to openrouter.ai)
python -I agent/run_agents.py --backend openrouter --model z-ai/glm-5.3-flash deepseek/deepseek-v4-flash-0731 \
       --datasets olist tpch pkdd99 northwind --runs 3
python -I agent/run_agents.py --backend claude --datasets olist --runs 3
# execution step (refuses to start if a provider key is in the environment)
env -u OPENROUTER_API_KEY python -I agent/check_runs.py --skill-scripts /path/to/pitbull/.claude/skills/pit-repair/scripts
# -> results_agents/<agent>.json, results_agents/summary.md
```

Safety discipline (inherited from pit-repair):

* The provider key lives only in the network process.
* `run_python` executes model code in a child process with an
  **allow-list environment** (PATH, HOME, locale, TMPDIR): no keys, no cloud
  credentials, no git or proxy configuration. Paths cannot leave the
  workspace.
* The check step is a separate process and refuses to run if a key is
  present.
* The `claude` backend strips cloud and provider credentials before starting
  the CLI. The CLI's own tool shell may still see variables injected by the
  host environment, so prefer the `openrouter` backend, or a host without
  such credentials, for untrusted models.

Agent pipelines run in a sandbox (`layers/sandbox.py`) that applies
pit-check's banned names and methods unchanged and widens only the import
list. The allowed list is pandas and numpy, plus scikit-learn and lightgbm
if the agent's interpreter has them; the task card names exactly the
libraries that are present.

Status: the backends and the check step were tested end to end against a
local OpenAI-compatible stub (tool calls, path confinement, usage
accounting, key refusal). No real agent results are included yet.
`openrouter.ai` was blocked by the build environment's network policy and
no key was configured.

### Finishing the agent runs locally (opencode or Claude Code)

The cloud run of Claude Code stopped at the account's session limit
(`terminal_reason: api_error`, "You've hit your session limit"). It left 15
clean runs: olist 10/10, f1 4/10, northwind 1/10. Every other run is
redone on resume. A run is kept only if its `meta.json` shows a pipeline
and no error.

```bash
# 0. environment: Python with pandas + numpy (+ sklearn/lightgbm if the agent may use them)
python -m venv venv && venv/bin/pip install pandas numpy
# 1. data (skip if you unpacked pitbench_data.zip into data/)
PITBULL_REPO=/path/to/pitbull ./fetch_raw.sh && venv/bin/python -I prepare_all.py
#    f1 and movietweetings are live sources: compare with build_info.json (sha256) if exact data matters

# 2a. opencode: log in once (`opencode auth login`), pick provider/model
venv/bin/python -I agent/run_agents.py --backend opencode --model anthropic/claude-sonnet-5-5 \
    --datasets olist f1 pkdd99 adventureworks northwind sakila movietweetings tpch --runs 10 --parallel 4
# 2b. or Claude Code on your machine: finishes the cloud run in place (same label claude-code)
venv/bin/python -I agent/run_agents.py --backend claude \
    --datasets olist f1 pkdd99 adventureworks northwind sakila movietweetings tpch --runs 10 --parallel 4

# 3. check (separate shell, no provider key in the environment)
env -u ANTHROPIC_API_KEY -u OPENROUTER_API_KEY -u OPENAI_API_KEY \
    venv/bin/python -I agent/check_runs.py --skill-scripts /path/to/pitbull/.claude/skills/pit-repair/scripts --shard 0/4 &
#    ... shards 1/4 2/4 3/4, then
venv/bin/python -I agent/check_runs.py --skill-scripts ... --summary-only    # -> results_agents/summary.md
```

Notes:

* **opencode runs are a different agent.** Their label is
  `opencode_<provider_model>`. Even with the same model, the harness
  differs: tools, system prompt and loop. They are reported in their own
  row and never merged into `claude-code`. To *finish* the `claude-code`
  row, use 2b.
* opencode gets a workspace-local `opencode.json`: edits and shell
  commands are allowed, web fetch is denied. The agent must solve the task
  from `data/`. opencode's shell is not sandboxed, so run it in a
  container or VM, or under a user without access to your files.
* Credentials for other services (cloud, GitHub) are removed from the
  agent's environment. Only the provider key opencode needs is kept
  (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` or `OPENAI_API_KEY`).
  `check_runs.py` refuses to start if any of them is present.
* `run_agents.py` resumes: rerun the same command after an interruption.
  `--redo` reruns everything.

## Natural code vs injected errors

Two questions are kept apart:

* **Recall and false alarms** of the checks come from code whose answer is
  known: generated controls (`datasets/<ds>/layers/`) and mutants of correct
  pipelines (`mutate/`).
* **How often errors occur** comes only from code PITBench did not write:
  agent runs (`agent/`) and the external corpus (`corpus/`).

Every checked program carries an `origin`, and prevalence tables count
only `agent` and `external` (`agent/check_runs.py` → `results_agents/summary.md`).

### Mutation testing (`mutate/`)

`mutate/mutations.py` applies five operators to every matching site of a
correct pipeline, one change per mutant:

* remove the upper time cut;
* move a window after t;
* widen the forward (label) horizon;
* swap `<`/`<=` and `>`/`>=` on time comparisons;
* remove the lower bound — not a PIT violation, a specificity probe.

Mutants that change no output on the full database are discarded as
equivalent. Every remaining mutant was triaged by reading the code: is it a
PIT violation or not (`mutate/triage_reference.json`, with a reason per
case).

Results on the reference pipelines of all eight datasets
(`results_mutations/summary.md`):

| pipeline | live mutants | violations killed (recall) | non-violations killed (false alarms) |
|---|---|---|---|
| olist | 10 | 5/5 | 0/5 |
| f1 | 10 | 5/5 | 0/5 |
| pkdd99 | 15 | 6/6 | 0/9 |
| adventureworks | 13 | 5/5 | 0/8 |
| northwind | 14 | 6/6 | 0/8 |
| sakila | 10 | 5/5 | 0/5 |
| movietweetings | 10 | 5/5 | 0/5 |
| tpch | 16 | 6/6 | 0/10 |
| **total** | **98** | **43/43** | **0/55** |

The 55 survivors all change behaviour but read nothing after t: a longer
look-back, a stricter `< t` cut, a window moved past t while the `<= t`
cut still applies, or a label that also counts past events (a wrong target,
not a leak). So **the checks separate "changed" from "leaks"**, which a
code-review or diff-based detector would not.

The first run of this table had one false alarm (pkdd99: the label with its
lower bound removed, flagged by the tie-injection view). The cause was the
view, which then *added* a copy of a past event and so changed the past.
It now *moves* the event to t; layer validation stays at 141/141 and the
false alarm is gone.

One limitation: a `>` → `>=` shift on the label window is *equivalent on
the data* wherever no event is stamped exactly at a seed (F1, Olist, Sakila,
MovieTweetings, part of AdventureWorks). Such mutants are excluded before
checking, although the tie-injection view would catch them; they are
counted only where real ties exist (PKDD, Northwind, TPC-H).

```bash
python -I mutate/run_mutations.py --skill-scripts .../pit-repair/scripts --dataset northwind        # reference pipeline
python -I mutate/run_mutations.py --skill-scripts ... --dataset olist --pipeline agent_runs/.../pipeline.py  # agent code
```

On agent-written pipelines there is no triage file. Survivors must be
triaged by hand before recall can be reported.

### External corpus (`corpus/`)

The corpus has a fixed layout and **wrapping rules fixed in advance**
(`corpus/README.md`):

* keep the logic and only re-plumb inputs and outputs;
* missing functions stay missing — the layer is reported absent;
* a hard-coded cutoff date becomes `seed_time`, and the change is recorded;
* code that needs a library outside the sandbox is recorded as "not
  wrappable", not rewritten;
* wrapping time is recorded.

Tools: `corpus/add_program.py` registers a wrapped program with its
provenance. `corpus/import_fse_olist_corpus.py` imports the original Olist
corpus of the FSE study from this repository's history: 134 executed
programs written by two code models, 63 recorded as LEAK and 71 as CLEAN
by that study. Candidate sources are the RelBench user-study SQL and the
RelAgent SQL (both in commit a9cd898, but they need RelBench data) and
public Olist/Berka notebooks.

`agent/check_runs.py --runs-dir corpus` checks the corpus through the same
path as agent runs. It accepts `--n-entities` to override the sample size
and `--shard i/k` to split the work across parallel processes, followed by
`--summary-only` to merge.

#### First natural-code result: the FSE Olist corpus, feature layer

All 134 imported programs were checked: 60 entities active per seed, 6
seeds, run in 4 parallel shards (`results_agents/fse_olist_corpus*.json`).
The results are cross-tabulated with the verdict the original study
recorded:

| original verdict | witness (rows after t) | canary only (late value) | snapshot only (final state) | clean | did not run |
|---|---|---|---|---|---|
| LEAK (63) | 60 | — | — | 2 | 1 (`locals()`, as in A4) |
| CLEAN (71) | — | 15 | 20 | 36 | — |

* **All witness-level leaks are found again** (60 of 63). The 2 LEAK
  programs that stay silent do not show their leak on the 60 entities
  sampled here; the original study used 15 random products per seed.
* **15 originally-CLEAN programs read late-filled values** (canary only).
  This matches the 14 canary-only programs that A2 found when it re-checked
  the clean records.
* **20 originally-CLEAN programs read `orders.order_status` as a final
  state** (snapshot only); 11 LEAK programs do too. pit-check v1 cannot see
  this. Counted over the whole corpus, **43 of 133 programs (32%) use an
  end-of-data value**, and the snapshot channel adds 20 programs (+27%) to
  what v1 flags.
* **Snapshot flags need human judgement, like canary flags.** A cancelled
  order has no cancellation time, so its status is treated as unknowable at
  every t. Where a program only uses `delivered`, this is a real leak.
  Where it uses `canceled`, it may or may not be: the time is simply not in
  the data.

## Olist corpus compatibility

The original corpus is the 63 programs with verdict LEAK in
`prestudy/p3_out/baseline/results.jsonl` (commit a9cd898), minus one
GENERATION_ARTIFACT. It was re-checked on this Olist build with the
published setup: 15 products and the 6 published seeds
(`tools/compat_olist_corpus.py`, `results/olist_corpus_compat.json`).

| checker | LEAK | ERROR | CLEAN |
|---|---:|---:|---:|
| plain `pit_check.py` | 50 / 62 | 12 | 0 |
| `ext/pit_check_dates.py` (map `date_cols`) | **59 / 62** | 3 | 0 |

Under plain `pit_check.py`, 9 of the 12 ERRORs have one cause. The checker
parses only the time columns the map declares, so
`order_estimated_delivery_date` (a planned date with no availability
semantics) stays a string, and `delivered - estimated` crashes. The A4
skill run reported only 2 ERRORs in 76, which suggests the current checker parses
dates more narrowly than the setup the published numbers came from (not verified). This
is worth fixing in the skill itself: an optional `date_cols` key in the
map, which is exactly what the wrapper adds.

Two of the remaining 3 ERRORs are the same ones the published A4 run
had: a `KeyError` on `pay_type_*` and a forbidden `locals()`. The third
(`deepseek#76`, "Other Series must have a name") was not investigated.

No leaking program comes out CLEAN on this build.

## Remaining work: natural code for the seven other datasets

Status per dataset (✓ done, — missing):

| dataset | layer controls | reference pipeline | mutants + triage | natural code (prevalence) |
|---|---|---|---|---|
| olist | ✓ | ✓ | ✓ | ✓ FSE corpus, feature layer only (134 programs) |
| f1 | ✓ | ✓ | ✓ | — |
| pkdd99 | ✓ | ✓ | ✓ | — |
| adventureworks | ✓ | ✓ | ✓ | — |
| northwind | ✓ | ✓ | ✓ | — |
| sakila | ✓ | ✓ | ✓ | — (detection only: not informative for harm) |
| movietweetings | ✓ | ✓ | ✓ | — |
| tpch | ✓ | ✓ | ✓ | — |

Detection (controls) and recall / false alarms (mutants) are measured on all
eight. What is missing everywhere except Olist is **natural code**, without
which no prevalence or harm-on-real-code claim can be made; and even Olist
lacks natural code for the task, model and agent layers. Steps, per dataset:

1. **Agent runs (network step).** Workspaces are built from
   `dataset.json` (the task card never mentions leakage).

   ```bash
   # Claude Code headless (needs the `claude` CLI logged in)
   python -I agent/run_agents.py --backend claude --datasets f1 pkdd99 adventureworks northwind movietweetings tpch sakila --runs 10
   # OpenRouter: OPENROUTER_API_KEY as an environment secret, openrouter.ai reachable
   python -I agent/run_agents.py --backend openrouter --model <model-a> <model-b> --datasets ... --runs 10
   ```

   Output: `agent_runs/<agent>/<ds>/run<k>/{pipeline.py, report.json, meta.json}`.
   Suggested size: ≥ 10 runs × ≥ 2 agents × each dataset, so a prevalence
   of 20% has a 95% interval narrower than ±18 points per dataset; pool
   across datasets for per-layer rates.

2. **Execution step (separate, key-free process).**

   ```bash
   env -u OPENROUTER_API_KEY python -I agent/check_runs.py --skill-scripts .../pit-repair/scripts \
       --runs-dir agent_runs --shard 0/4 &   # ... shards 1/4..3/4, then:
   python -I agent/check_runs.py --skill-scripts ... --summary-only
   ```

   Output: per-layer firings, `auc_backtest` vs `auc_deploy` (harm on real
   code) and the agent's claimed AUC, in `results_agents/summary.md`.

3. **Manual check of the verdicts (ground truth for natural code).** For a
   random sample of programs per dataset (all LEAK, ≥ 20 CLEAN), read the
   code and record "PIT violation yes/no, which layer, why", in the same
   format as `mutate/triage_reference.json`, as
   `agent_runs/<agent>/triage.json`. This gives precision and an estimate
   of missed leaks on natural code, not only on mutants.

4. **Mutants of agent code.** Run `mutate/run_mutations.py --dataset <ds>
   --pipeline agent_runs/<agent>/<ds>/run<k>/pipeline.py` on correct
   agent pipelines, and triage the live mutants (`--triage` with a new
   file). Mutation sites are found only where agents use the time
   names in `mutate/mutations.py: TIME_NAMES`; extend that list if agents
   name the cutoff differently, and say so in the results.

5. **External programs (optional, `origin: external`).** Wrap them under
   the rules in `corpus/README.md` and register them with
   `corpus/add_program.py`. Candidate sources: Berka/PKDD'99 churn and
   loan notebooks (pkdd99); Kaggle F1 podium/position notebooks (f1); SQL
   tutorial customer-churn and RFM queries (northwind, sakila,
   adventureworks); MovieTweetings recommender notebooks (movietweetings);
   TPC-H has no natural ML code (agents only).

6. **Update** this README: a natural-code table per dataset like the FSE
   Olist one (programs, ran, leaking per layer, harm), and the
   "Limitations" entries below.

## Limitations

* Availability maps and roughness notes are written by one author and are
  not independently annotated. Treat them like the RelBench maps of the
  FSE plan's A6: they need a second annotator before they can be cited as
  ground truth.
* Controls and mutants show that each class is *checkable*, together with
  recall and false alarms on realistic code. Prevalence needs natural code,
  and the only natural corpus checked so far is the FSE Olist corpus
  (feature layer: 134 programs). The other seven datasets have controls,
  reference pipelines and mutants, but no natural code yet: that needs agent
  runs (`agent/run_agents.py`) or external programs (`corpus/add_program.py`);
  see "Remaining work" above.
* The mutation triage was done by one person, by reading the code. It
  covers the reference pipelines of all eight datasets only; agent-written
  mutants need their own triage.
* Sakila is not informative for harm (one short season, unstable label
  prevalence). Its controls still count for detection.
* Entity counts per seed are capped by `n_entities`. The corpus check of
  the FSE Olist corpus used 60 per seed for speed, where layer validation
  uses 300.
* `f1` (f1db `latest`) and `movietweetings` (`master`) are live sources.
  The exact files used are pinned by sha256 in `build_info.json`;
  re-downloading may give newer data.
* F1 race times are taken as given (`time`, assumed UTC); a missing time
  becomes 23:59 of race day.
* RelBench, Kaggle (Home Credit, Instacart, Elo), UCI, HuggingFace and
  archive.org were not reachable from the build environment. The natural
  next additions are rel-stack (with `PostHistory` as real version history),
  rel-event, Home Credit, and Instacart (relative time only).
