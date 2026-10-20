"""Mutation operators for point-in-time errors in pipeline code.

Hand-written controls show that each class of error is *checkable*; mutants of
realistic, correct pipelines (agent-written ones, or datasets/<ds>/layers/
reference_pipeline.py) measure recall on code that looks like what people and
agents write. One mutant = one change at one place.

Operators (each applied to every matching site separately):

  drop_cutoff       X <= T  /  X < T   ->  X <= pd.Timestamp.max     upper time bound removed
                    (and the mirrored T >= X)                       designed to violate PIT
  future_offset     T - pd.Timedelta(...)  ->  T + pd.Timedelta(...) history / training window moved
                                                                     after T: a violation for training seeds,
                                                                     harmless where an upper cut <= t remains
  extend_horizon    T + pd.Timedelta(days=N) -> days=4*N              forward window (label horizon)
                                                                     widened; violates PIT for labels
  boundary_shift    <  <-> <=,  >  <-> >=  on a time comparison      mixed: [t, t+h] labels violate,
                                                                     stricter feature cutoffs do not
  drop_lower_bound  X > T  /  X >= T   ->  X > pd.Timestamp.min      lower bound removed (longer
                                                                     look-back): NOT a PIT violation --
                                                                     a specificity probe, should survive

"T" is a time expression: a name in TIME_NAMES, or an expression containing
one (t + pd.Timedelta(days=H)). Matching is syntactic on the AST; the runner
discards mutants that do not change any output on the full database
(equivalent mutants) before computing a score.
"""
import ast
import copy

TIME_NAMES = {"seed_time", "t", "ti", "t0", "t_i", "lo", "hi", "cutoff", "cut", "at", "now", "as_of",
              "ref_time", "prediction_time", "pred_time", "start", "end", "window_start", "window_end"}
UPPER = (ast.Lt, ast.LtE)
LOWER = (ast.Gt, ast.GtE)
SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt}
OPERATORS = ["drop_cutoff", "future_offset", "extend_horizon", "boundary_shift", "drop_lower_bound"]
# True: every live mutant should be a violation; False: none should be (specificity probe);
# None: depends on the site (decided per mutant by triage, see mutate/triage_reference.json)
DESIGNED_TO_VIOLATE = {"drop_cutoff": True, "future_offset": None, "extend_horizon": True,
                       "boundary_shift": None, "drop_lower_bound": False}


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def is_time(node):
    return bool(_names(node) & TIME_NAMES)


def _is_timedelta(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Timedelta")


def _ts(attr):
    return ast.parse(f"pd.Timestamp.{attr}", mode="eval").body


def sites(tree):
    """Every (operator, node index) a mutation can apply to, in a stable order."""
    out = []
    for i, node in enumerate(ast.walk(tree)):
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op, left, right = node.ops[0], node.left, node.comparators[0]
            # orient as  column OP time
            if is_time(right) and not is_time(left):
                kind = "upper" if isinstance(op, UPPER) else "lower" if isinstance(op, LOWER) else None
            elif is_time(left) and not is_time(right):
                kind = "upper" if isinstance(op, LOWER) else "lower" if isinstance(op, UPPER) else None
            else:
                kind = None
            if kind == "upper":
                out += [("drop_cutoff", i), ("boundary_shift", i)]
            elif kind == "lower":
                out += [("drop_lower_bound", i), ("boundary_shift", i)]
        elif isinstance(node, ast.BinOp) and is_time(node.left) and _is_timedelta(node.right):
            if isinstance(node.op, ast.Sub):
                out.append(("future_offset", i))
            elif isinstance(node.op, ast.Add) and any(k.arg == "days" for k in node.right.keywords):
                out.append(("extend_horizon", i))
    return out


def apply(tree, op, index):
    tree = copy.deepcopy(tree)
    node = list(ast.walk(tree))[index]
    if op in ("drop_cutoff", "drop_lower_bound"):
        attr = "max" if op == "drop_cutoff" else "min"
        if is_time(node.comparators[0]):
            node.comparators[0] = _ts(attr)
        else:
            node.left = _ts(attr)
    elif op == "boundary_shift":
        node.ops[0] = SWAP[type(node.ops[0])]()
    elif op == "future_offset":
        node.op = ast.Add()
    elif op == "extend_horizon":
        for k in node.right.keywords:
            if k.arg == "days":
                k.value = ast.BinOp(left=ast.Constant(4), op=ast.Mult(), right=k.value)
    return ast.fix_missing_locations(tree)


def mutants(code):
    """Yield (operator, line, before, after, mutated_code) for every site."""
    tree = ast.parse(code)
    nodes = list(ast.walk(tree))
    for op, i in sites(tree):
        before = ast.unparse(nodes[i])
        mutated = apply(tree, op, i)
        after = ast.unparse(list(ast.walk(mutated))[i])
        if before == after:
            continue
        yield op, getattr(nodes[i], "lineno", None), before, after, ast.unparse(mutated)
