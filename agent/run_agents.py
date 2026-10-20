#!/usr/bin/env python3
"""Network step of the agent scenario: agents solve PITBench tasks from raw data.

    # Claude Code (headless), default model
    python -I agent/run_agents.py --backend claude --datasets olist tpch --runs 3

    # opencode (headless `opencode run`), any provider opencode is logged in to
    python -I agent/run_agents.py --backend opencode --model anthropic/claude-sonnet-5-5 --datasets sakila --runs 10

    # OpenRouter models, agent loop with tools (needs OPENROUTER_API_KEY and network access to openrouter.ai)
    python -I agent/run_agents.py --backend openrouter --model z-ai/glm-5.3-flash qwen/qwen3-coder --datasets olist --runs 3
    # one-shot protocol of the original corpus
    python -I agent/run_agents.py --backend openrouter --model deepseek/deepseek-v4-flash-0731 --mode single ...

Each run gets a fresh workspace agent_runs/<label>/<ds>/run<k>/ (data copy,
TASK.md, pitbench_io.py) and leaves pipeline.py, report.json, a transcript and
meta.json there. Nothing is checked here: checking executes the agent's code
and runs in a separate, key-free process (agent/check_runs.py).
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import backends  # noqa: E402
from agent.workspace import TASKS, available_libs, build, prompt  # noqa: E402


def label_of(a, model):
    if a.backend == "claude":
        return "claude-code" + (f"_{model}" if model else "")
    if a.backend == "opencode":
        return "opencode_" + re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    return "openrouter_" + re.sub(r"[^A-Za-z0-9._-]+", "_", model) + f"_{a.mode}"


def one(a, model, ds, k, libs):
    ws = ROOT / "agent_runs" / label_of(a, model) / ds / f"run{k}"
    # resume: a run whose meta.json records a clean finish is kept; one without meta.json
    # (interrupted) or ending in an error (e.g. a usage limit) is redone from scratch
    if not a.redo and (ws / "meta.json").exists():
        old = json.loads((ws / "meta.json").read_text())
        if old.get("produced_pipeline") and not old.get("is_error") and not old.get("error"):
            return old
    info = build(ds, ws, libs)
    try:
        if a.backend == "claude":
            meta = backends.run_claude(ws, prompt(libs), a.python_bin_dir, model=model, max_turns=a.max_turns)
        elif a.backend == "opencode":
            meta = backends.run_opencode(ws, prompt(libs), a.python_bin_dir, model=model)
        else:
            meta = backends.run_openrouter(ws, prompt(libs), a.python_bin_dir, model=model, mode=a.mode,
                                           max_turns=a.max_turns, temperature=a.temperature)
    except Exception as e:
        meta = {"error": f"{type(e).__name__}: {e}"[:1000]}
    meta.update(info)
    meta["produced_pipeline"] = (ws / "pipeline.py").exists()
    (ws / "meta.json").write_text(json.dumps(meta, indent=1))
    # the data copy is not needed after the run; checks use the benchmark's own data/
    for f in (ws / "data").glob("*.csv"):
        f.unlink()
    print(json.dumps({k: meta.get(k) for k in ("dataset", "produced_pipeline", "wall_seconds", "error")}), flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["claude", "opencode", "openrouter"], required=True)
    ap.add_argument("--model", nargs="+", default=[None], help="one or more model ids (OpenRouter: required)")
    ap.add_argument("--mode", choices=["tools", "single"], default="tools")
    ap.add_argument("--datasets", nargs="+", default=list(TASKS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--redo", action="store_true", help="rerun finished runs too (default: skip runs with meta.json)")
    ap.add_argument("--python-bin-dir", default=str(Path(sys.executable).parent),
                    help="bin/ of the Python the agent's code runs with (default: this interpreter)")
    a = ap.parse_args()
    if a.backend in ("openrouter", "opencode") and a.model == [None]:
        ap.error(f"--model is required for {a.backend} (opencode: provider/model, e.g. anthropic/claude-sonnet-5-5)")
    if a.backend == "openrouter" and not __import__("os").environ.get("OPENROUTER_API_KEY"):
        ap.error("OPENROUTER_API_KEY is not set")
    libs = available_libs(a.python_bin_dir)
    print("agent may use:", ", ".join(libs), flush=True)
    jobs = [(m, ds, k) for m in a.model for ds in a.datasets for k in range(1, a.runs + 1)]
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(lambda j: one(a, *j, libs), jobs))


if __name__ == "__main__":
    main()
