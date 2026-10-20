"""Sandbox for agent-written pipelines: pit-check's AST filter, wider import list.

pit_check.py admits only pandas/numpy. Agents solving a whole task reach for
scikit-learn and LightGBM, so this loader reuses pit-check's banned names and
banned methods unchanged and only widens the import allow-list and the set of
harmless builtins. File, network and process access stay forbidden.
"""
import ast

ALLOWED_IMPORTS = {"pandas", "numpy", "sklearn", "lightgbm", "math", "collections", "itertools",
                   "functools", "warnings", "datetime", "typing", "dataclasses", "statistics", "re"}

EXTRA_BUILTINS = ("print", "hasattr", "map", "filter", "reversed", "iter", "next", "callable",
                  "frozenset", "slice", "divmod", "pow", "repr", "format", "object", "type",
                  "KeyError", "IndexError", "AttributeError", "ZeroDivisionError", "RuntimeError",
                  "StopIteration", "NotImplementedError", "Warning", "UserWarning", "property",
                  "staticmethod", "classmethod", "super", "id", "hash", "chr", "ord", "bytes")


def inspect_code(code, pit_check):
    tree = ast.parse(code)
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"from {node.module} import")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in pit_check.BANNED_NAMES:
                problems.append(f"call {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in pit_check.BANNED_METHODS:
                problems.append(f"method {node.func.attr}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"dunder attribute {node.attr}")
    return sorted(set(problems))


def load_namespace(code, pit_check):
    import builtins

    import numpy as np
    import pandas as pd

    problems = inspect_code(code, pit_check)
    if problems:
        raise ValueError("unsafe code: " + ", ".join(problems))

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(f"import blocked: {name}")
        return __import__(name, globals, locals, fromlist, level)

    allowed = {"__import__": safe_import, "__build_class__": builtins.__build_class__}
    for n in ("len", "range", "min", "max", "sum", "abs", "round", "sorted", "enumerate", "zip", "list",
              "dict", "set", "tuple", "float", "int", "str", "bool", "any", "all", "isinstance",
              "Exception", "ValueError", "TypeError") + EXTRA_BUILTINS:
        allowed[n] = getattr(builtins, n)
    ns = {"pd": pd, "np": np, "__builtins__": allowed, "__name__": "agent_pipeline"}
    exec(compile(code, "<agent pipeline>", "exec"), ns)
    return ns
