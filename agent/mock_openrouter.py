"""Scripted OpenAI-compatible endpoint: tests the openrouter backend plumbing offline (not a model).

    python -I agent/mock_openrouter.py 18765 &
    OPENROUTER_BASE_URL=http://127.0.0.1:18765/api/v1 OPENROUTER_API_KEY=test-key python -I agent/run_agents.py --backend openrouter --model mock/x ...
"""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
PIPE = '''import pandas as pd
import numpy as np
def select_entities(db, seed_time):
    return []
def make_label(db, entity_ids, seed_time):
    return pd.DataFrame({"label": 0.0}, index=pd.Index(entity_ids))
def get_features(db, entity_ids, seed_time):
    return pd.DataFrame({"x": 0.0}, index=pd.Index(entity_ids))
def fit_predict(db, entity_ids, seed_time):
    return pd.DataFrame({"score": 0.0}, index=pd.Index(entity_ids))
'''
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n_tool = sum(1 for m in body["messages"] if m["role"] == "tool")
        assert self.headers["Authorization"] == "Bearer test-key"
        if "tools" not in body:
            msg = {"role": "assistant", "content": "```python\n" + PIPE + "```"}
        elif n_tool == 0:
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "pipeline.py", "content": PIPE})}},
                {"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "../../etc/passwd"})}}]}
        elif n_tool == 2:
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c3", "type": "function", "function": {"name": "run_python", "arguments": json.dumps({"code": "import os, lightgbm, pipeline; print('keys:', [k for k in os.environ if 'KEY' in k]); print('ok')"})}}]}
        else:
            msg = {"role": "assistant", "content": "done"}
        out = {"choices": [{"message": msg}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}}
        b = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
