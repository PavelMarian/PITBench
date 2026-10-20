"""Agent backends. Each one turns a workspace (TASK.md, data/, pitbench_io.py) into pipeline.py.

    claude      -- Claude Code headless (`claude -p`), its own tools, its own auth;
    opencode    -- opencode headless (`opencode run`), its own tools, its own auth
                   (whatever provider `opencode auth login` set up);
    openrouter  -- any chat model on OpenRouter (OpenAI-compatible API), either
                   * mode "tools":  an agent loop with four tools
                     (list_files, read_file, write_file, run_python), or
                   * mode "single": one completion, pipeline.py taken from the
                     ```python block (the protocol of the original Olist corpus).

Safety discipline carried over from pit-repair: the API key lives only in this
(network) process. Model-written code is executed by run_python in a child
process whose environment has every *_API_KEY / OPENROUTER* variable removed,
with the workspace as cwd. The later check step (check_runs.py) refuses to
start if a key is present at all.
"""
import json
import os
import re
import ssl
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

OPENROUTER_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1") + "/chat/completions"
# Model-written code runs with an ALLOW-list environment: nothing else leaks in
# (no provider keys, no cloud credentials, no git/proxy configuration).
ALLOW_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "OMP_NUM_THREADS")
# The Claude Code CLI needs its own session variables; everything that looks like
# a credential for something else is removed before it starts.
DENY_FOR_CLI = ("AWS_", "GOOGLE_", "AZURE_", "CLOUDSDK_", "GIT_CONFIG_", "GH_TOKEN", "GITHUB_TOKEN", "OPENROUTER", "OPENAI_",
                "_API_KEY", "SECRET")


def clean_env(python_bin_dir=None):
    env = {k: os.environ[k] for k in ALLOW_ENV if k in os.environ}
    if python_bin_dir:
        env["PATH"] = f"{python_bin_dir}:{env.get('PATH', '/usr/bin:/bin')}"
    return env


# ------------------------------------------------------------------ claude --
def run_claude(ws: Path, prompt: str, python_bin_dir: str, model=None, max_turns=80, timeout=3600):
    cmd = ["claude", "-p", prompt, "--output-format", "json",
           "--allowedTools", "Read,Write,Edit,Bash,Glob,Grep", "--max-turns", str(max_turns)]
    if model:
        cmd += ["--model", model]
    env = {k: v for k, v in os.environ.items() if not any(d in k for d in DENY_FOR_CLI)}
    env["PATH"] = f"{python_bin_dir}:{env.get('PATH', '')}"
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True, timeout=timeout)
    meta = {"backend": "claude", "wall_seconds": round(time.time() - t0, 1), "exit_code": p.returncode}
    try:
        out = json.loads(p.stdout)
        meta.update({k: out.get(k) for k in ("num_turns", "total_cost_usd", "terminal_reason", "is_error")})
        meta["models"] = sorted((out.get("modelUsage") or {}).keys())
        (ws / "transcript_result.txt").write_text(str(out.get("result", "")))
    except Exception:
        meta["stdout_tail"] = p.stdout[-2000:]
        meta["stderr_tail"] = p.stderr[-2000:]
    return meta


# ---------------------------------------------------------------- opencode --
# Workspace-local opencode config: the agent may edit files and run commands in the
# workspace, but not fetch from the web (the task must be solved from data/ alone).
OPENCODE_CONFIG = {"$schema": "https://opencode.ai/config.json",
                   "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"}}


def run_opencode(ws: Path, prompt: str, python_bin_dir: str, model, timeout=3600):
    """`opencode run` in the workspace. Like the claude backend, credentials for other
    services are removed from the environment; opencode keeps its own provider auth
    (~/.local/share/opencode/auth.json or the provider's *_API_KEY, which is allowed here
    because this is the network step -- check_runs.py later refuses any key)."""
    (ws / "opencode.json").write_text(json.dumps(OPENCODE_CONFIG, indent=1))
    cmd = ["opencode", "run", "--model", model, prompt]
    keep_keys = ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")
    env = {k: v for k, v in os.environ.items()
           if k in keep_keys or not any(d in k for d in DENY_FOR_CLI)}
    env["PATH"] = f"{python_bin_dir}:{env.get('PATH', '')}"
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"backend": "opencode", "model": model, "wall_seconds": round(time.time() - t0, 1),
                "is_error": True, "error": f"timeout after {timeout}s"}
    meta = {"backend": "opencode", "model": model, "wall_seconds": round(time.time() - t0, 1),
            "exit_code": p.returncode, "is_error": p.returncode != 0}
    (ws / "transcript_result.txt").write_text(p.stdout[-200000:])
    if p.returncode != 0:
        meta["error"] = p.stderr[-2000:] or p.stdout[-2000:]
    return meta


# -------------------------------------------------------------- openrouter --
TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List files under a directory of the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file of the workspace (first max_chars characters).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                        "max_chars": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a text file in the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_python", "description": "Run a Python script in the workspace (cwd = workspace) and return "
                                             "stdout/stderr. The libraries listed in TASK.md are installed.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"},
                                                        "timeout_seconds": {"type": "integer"}},
                       "required": ["code"]}}},
]


def _inside(ws: Path, rel: str) -> Path:
    p = (ws / rel).resolve()
    if ws.resolve() not in [p, *p.parents]:
        raise ValueError(f"path {rel!r} is outside the workspace")
    return p


def _tool(ws, python_bin_dir, name, args):
    try:
        if name == "list_files":
            base = _inside(ws, args.get("path", "."))
            return "\n".join(str(p.relative_to(ws)) for p in sorted(base.rglob("*"))[:500] if p.is_file())
        if name == "read_file":
            return _inside(ws, args["path"]).read_text(errors="replace")[: int(args.get("max_chars") or 20000)]
        if name == "write_file":
            p = _inside(ws, args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"])
            return f"wrote {p.relative_to(ws)} ({len(args['content'])} chars)"
        if name == "run_python":
            script = ws / "_agent_run.py"
            script.write_text(args["code"])
            r = subprocess.run([str(Path(python_bin_dir) / "python"), str(script.name)], cwd=ws,
                               env=clean_env(python_bin_dir), capture_output=True, text=True,
                               timeout=min(int(args.get("timeout_seconds") or 300), 900))
            return (f"exit={r.returncode}\n--- stdout ---\n{r.stdout[-6000:]}\n--- stderr ---\n{r.stderr[-3000:]}")
        return f"unknown tool {name}"
    except subprocess.TimeoutExpired:
        return "error: timeout"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def _post(payload, api_key, retries=4):
    ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
                                     or ("/root/.ccr/ca-bundle.crt" if Path("/root/.ccr/ca-bundle.crt").exists()
                                         else None))
    data = json.dumps(payload).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(OPENROUTER_URL, data=data, method="POST", headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/PavelMarian/pitbull", "X-Title": "PITBench agent scenario"})
        try:
            with urllib.request.urlopen(req, timeout=600, context=ctx) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f"OpenRouter HTTP {e.code}: {e.read()[:500]!r}")
        except urllib.error.URLError:
            if attempt < retries:
                time.sleep(2 ** (attempt + 1))
                continue
            raise


def run_openrouter(ws: Path, prompt: str, python_bin_dir: str, model: str, mode="tools", max_turns=60,
                   temperature=0.2, max_tokens=16000):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (add it as an environment secret)")
    log = ws / "transcript.jsonl"
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "calls": 0}
    t0 = time.time()
    if mode == "single":
        task = (ws / "TASK.md").read_text()
        schema = (ws / "pitbench_io.py").read_text()
        msgs = [{"role": "user", "content": f"{task}\n\nLoader (pitbench_io.py):\n```python\n{schema}\n```\n\n"
                                            "Return pipeline.py in a single ```python block."}]
        resp = _post({"model": model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens,
                      "usage": {"include": True}}, api_key)
        _account(usage, resp)
        text = resp["choices"][0]["message"].get("content") or ""
        log.write_text(json.dumps({"request": msgs, "response": resp}) + "\n")
        m = re.search(r"```python\n(.*?)```", text, re.S)
        (ws / "pipeline.py").write_text(m.group(1) if m else text)
        return {"backend": "openrouter", "model": model, "mode": mode, "turns": 1, **usage,
                "wall_seconds": round(time.time() - t0, 1)}

    msgs = [{"role": "system", "content": "You are an autonomous data scientist working in a sandboxed directory. "
                                          "Use the tools to inspect files and run Python. Stop when done."},
            {"role": "user", "content": prompt}]
    turns, finish = 0, "max_turns"
    with open(log, "w") as fh:
        while turns < max_turns:
            turns += 1
            resp = _post({"model": model, "messages": msgs, "tools": TOOLS, "temperature": temperature,
                          "max_tokens": max_tokens, "usage": {"include": True}}, api_key)
            _account(usage, resp)
            msg = resp["choices"][0]["message"]
            fh.write(json.dumps({"turn": turns, "message": msg}) + "\n")
            msgs.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
            calls = msg.get("tool_calls") or []
            if not calls:
                finish = "stop"
                break
            for c in calls:
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = _tool(ws, python_bin_dir, c["function"]["name"], args)
                fh.write(json.dumps({"turn": turns, "tool": c["function"]["name"], "args": args,
                                     "result": out[:4000]}) + "\n")
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
    return {"backend": "openrouter", "model": model, "mode": mode, "turns": turns, "finish": finish, **usage,
            "wall_seconds": round(time.time() - t0, 1)}


def _account(usage, resp):
    u = resp.get("usage") or {}
    usage["calls"] += 1
    usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
    usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
    usage["cost"] += float(u.get("cost") or 0.0)
