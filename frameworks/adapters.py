"""Real ML frameworks as layer programs: featuretools, LightGBM, AutoGluon, PyTorch Geometric.

Every adapter is a callable (db, entity_ids, seed_time) -> DataFrame with the
same contract as the generated controls, so layers/harness.py checks it with
the same views (D vs D|t, canary, snapshot) and the same harm measure. These
are trusted benchmark code, so they run outside the pit-check sandbox.

No framework is a dependency of PITBench. Each adapter imports its framework
only when called; `available()` tells which ones are installed, and the
validation skips the rest. Install what you want to check, e.g.
`pip install lightgbm scikit-learn` or `pip install featuretools "setuptools<70"`
(see frameworks/README.md for tested versions).

Each framework appears in a correct configuration and in the configuration a
practitioner gets by default or by a common mistake:

  featuretools   ft_d3   cutoff_time per (entity, t), include_cutoff_time=False
                 ft_d2   cutoff_time, library default include_cutoff_time=True
                 ft_d1   no cutoff_time ("how most notebooks call dfs")
  LightGBM       lgbm_correct            training seeds with t_i + h <= t
                 lgbm_label_overlap      training seeds up to t (no embargo)
                 lgbm_target_encoding    sklearn TargetEncoder on entity id, fitted on all windows
  AutoGluon      ag_correct / ag_label_overlap   fixed hyperparameters (GBM), no bagging -> deterministic
  PyG GNN        gnn_temporal     GraphSAGE, message passing only over edges with time <= t_i
                 gnn_nontemporal  same model, all edges (neighbour sampling without a time constraint)
"""
import importlib.util
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# configuration family -> modules it needs
REQUIRES = {
    "ft": ["featuretools"],
    "lgbm": ["lightgbm", "sklearn"],
    "ag": ["autogluon.tabular"],
    "gnn": ["torch", "torch_geometric"],
}


def _has(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def available():
    """{family: True/False} -- which framework adapters can run in this environment."""
    return {fam: all(_has(m) for m in mods) for fam, mods in REQUIRES.items()}


class Spec:
    def __init__(self, d):
        self.E, self.T, self.K, self.V = d["E"], d["T"], d["K"], d["V"]
        self.H, self.STEP, self.NTRAIN, self.LABEL = d["H"], d["STEP"], d["NTRAIN"], d["LABEL"]
        self.LFILT = d.get("LFILT")


# ------------------------------------------------------------ shared pieces --
def feats(db, s, ids, t):
    idx = pd.Index(ids)
    ev = db[s.E]
    ev = ev[(ev[s.T] <= t) & ev[s.K].isin(idx)]
    rec = ev[ev[s.T] > t - pd.Timedelta(days=s.H)]
    f = pd.DataFrame(index=idx)
    f["n_all"] = ev.groupby(s.K).size().reindex(idx).fillna(0).astype(float)
    f["n_recent"] = rec.groupby(s.K).size().reindex(idx).fillna(0).astype(float)
    last = ev.groupby(s.K)[s.T].max().reindex(idx)
    f["days_since"] = (t - last).dt.days.astype(float).fillna(9999.0)
    if s.V is not None:
        v = pd.to_numeric(rec[s.V], errors="coerce")
        f["v_recent"] = v.groupby(rec[s.K]).sum().reindex(idx).fillna(0).astype(float)
    return f


def label(db, s, ids, t):
    idx = pd.Index(ids)
    ev = db[s.E]
    w = ev[(ev[s.T] > t) & (ev[s.T] <= t + pd.Timedelta(days=s.H)) & ev[s.K].isin(idx)]
    if s.LFILT:
        w = w[eval(s.LFILT, {}, {"w": w})]  # trusted spec expression (layers/make_layer_controls.SPECS)
    c = w.groupby(s.K).size().reindex(idx).fillna(0)
    return eval(s.LABEL, {}, {"c": c}).astype(float)  # trusted spec expression


def train_table(db, s, ids, t, offsets, feat_fn=None):
    feat_fn = feat_fn or (lambda ti: feats(db, s, ids, ti))
    X, y, g = [], [], []
    for off in offsets:
        ti = t - pd.Timedelta(days=off)
        f = feat_fn(ti)
        X.append(f)
        y.append(label(db, s, ids, ti))
        g.append(pd.Series(list(f.index), index=f.index))
    return pd.concat(X, ignore_index=True), pd.concat(y, ignore_index=True), pd.concat(g, ignore_index=True)


def correct_offsets(s):
    return [s.H + s.STEP * k for k in range(s.NTRAIN)]


def overlap_offsets(s):
    return [s.STEP * (k + 1) for k in range(s.NTRAIN)]


# --------------------------------------------------------------- LightGBM --
def _lgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                              subsample=1.0, colsample_bytree=1.0, random_state=0, n_jobs=1,
                              deterministic=True, force_row_wise=True, verbose=-1)


def _fit_score(model, X, y, Xq):
    if y.nunique() < 2:
        return np.full(len(Xq), float(y.iloc[0]) if len(y) else 0.0)
    model.fit(X, y.astype(int))
    return model.predict_proba(Xq)[:, 1]


def lgbm(s, offsets_fn, target_encoding=False):
    def fit_predict(db, entity_ids, seed_time):
        t = pd.Timestamp(seed_time)
        ids = list(entity_ids)
        X, y, g = train_table(db, s, ids, t, offsets_fn(s))
        q = feats(db, s, ids, t)
        if target_encoding:
            from sklearn.preprocessing import TargetEncoder
            # entity-id target encoding fitted on label windows across the whole data
            Xa, ya, ga = train_table(db, s, ids, t, [-s.STEP * k for k in range(s.NTRAIN)] + offsets_fn(s))
            te = TargetEncoder(target_type="binary", random_state=0, cv=2)
            te.fit(ga.astype(str).to_frame(), ya.astype(int))
            X = X.assign(id_te=te.transform(g.astype(str).to_frame())[:, 0])
            q = q.assign(id_te=te.transform(pd.Series([str(i) for i in ids]).to_frame())[:, 0])
        return pd.DataFrame({"score": _fit_score(_lgbm(), X, y, q)}, index=pd.Index(ids))
    return fit_predict


# -------------------------------------------------------------- AutoGluon --
def autogluon(s, offsets_fn, workdir):
    def fit_predict(db, entity_ids, seed_time):
        from autogluon.tabular import TabularPredictor
        t = pd.Timestamp(seed_time)
        ids = list(entity_ids)
        X, y, _ = train_table(db, s, ids, t, offsets_fn(s))
        q = feats(db, s, ids, t)
        if y.nunique() < 2:
            return pd.DataFrame({"score": float(y.iloc[0])}, index=pd.Index(ids))
        tr = X.assign(label=y.astype(int))
        pred = TabularPredictor(label="label", problem_type="binary", eval_metric="roc_auc", verbosity=0,
                                path=str(workdir / f"ag_{abs(hash((tuple(map(str, ids)), str(t), len(tr))))}"))
        pred.fit(tr, hyperparameters={"GBM": {"num_boost_round": 200, "seed": 0, "deterministic": True,
                                              "num_threads": 1}},
                 num_bag_folds=0, num_stack_levels=0, holdout_frac=0.2, ag_args_fit={"num_cpus": 1},
                 fit_weighted_ensemble=False, time_limit=None, raise_on_no_models_fitted=True)
        p = pred.predict_proba(q.reset_index(drop=True))[1].to_numpy()
        return pd.DataFrame({"score": p}, index=pd.Index(ids))
    return fit_predict


# -------------------------------------------------------------- featuretools --
FT_SCHEMAS = {
    # dataset -> (target dataframe, its index, [(child table, child index or None, time col, parent key)], value cols)
    "olist": ("products", "product_id",
              [("order_items", None, "ts", "product_id")]),
    "tpch": ("customer", "customer_id",
             [("orders", "order_id", "order_date", "customer_id")]),
    "pkdd99": ("account", "account_id",
               [("trans", "trans_id", "date", "account_id"), ("loan", "loan_id", "granted_date", "account_id")]),
}


def featuretools(ds, mode):
    """DFS features at (entity, t). mode: d1 no cutoff, d2 cutoff incl. cutoff time (library default), d3 strict."""
    target, tidx, children = FT_SCHEMAS[ds]

    def get_features(db, entity_ids, seed_time):
        import featuretools as ft
        t = pd.Timestamp(seed_time)
        ids = list(entity_ids)
        es = ft.EntitySet(id=ds)
        tgt = db[target][[tidx]].drop_duplicates() if target not in ("account",) else db[target][[tidx]]
        es = es.add_dataframe(dataframe_name=target, dataframe=tgt.copy(), index=tidx)
        for name, idx_col, tcol, key in children:
            df = db[name].copy()
            num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.endswith("_id")
                   and c not in ("payment_sequential", "order_item_id", "line_number")]
            df = df[[c for c in ([idx_col] if idx_col else []) + [key, tcol] + num if c in df.columns]]
            if idx_col is None:
                df = df.reset_index(drop=True).reset_index().rename(columns={"index": f"{name}_row"})
                idx_col = f"{name}_row"
            es = es.add_dataframe(dataframe_name=name, dataframe=df, index=idx_col, time_index=tcol)
            es = es.add_relationship(target, tidx, name, key)
        kw = dict(target_dataframe_name=target, agg_primitives=["count", "sum", "mean", "max"],
                  trans_primitives=[], max_depth=1, verbose=False)
        if mode == "d1":
            fm, _ = ft.dfs(entityset=es, **kw)
            fm = fm.reindex(ids)
        else:
            cut = pd.DataFrame({tidx: ids, "time": t})
            fm, _ = ft.dfs(entityset=es, cutoff_time=cut, cutoff_time_in_index=False,
                           include_cutoff_time=(mode == "d2"), **kw)
            fm = fm.reindex(ids)
        fm = fm.select_dtypes("number").astype(float)
        return fm[sorted(fm.columns)]
    return get_features


# ------------------------------------------------------------- PyG GraphSAGE --
GNN_GRAPHS = {
    # dataset -> (event table, entity key, partner key, partner via (table, join key) or None)
    "movietweetings": ("ratings", "user_id", "movie_id", None),
    "sakila": ("rental", "customer_id", "film_id", ("inventory", "inventory_id")),
    "tpch": ("orders", "customer_id", "part_id", ("lineitem", "order_id")),
}


def _edges(db, ds, s):
    ev_t, ek, pk, via = GNN_GRAPHS[ds]
    ev = db[ev_t]
    if via is None:
        e = ev[[ek, pk, s.T]]
    else:
        vt, vk = via
        e = ev[[ek, vk, s.T]].merge(db[vt][[vk, pk]].drop_duplicates(), on=vk)[[ek, pk, s.T]]
    return e.rename(columns={ek: "src", pk: "dst", s.T: "time"})


def gnn(ds, s, temporal=True, epochs=60, hidden=16):
    def fit_predict(db, entity_ids, seed_time):
        import torch
        from torch_geometric.nn import SAGEConv
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        t = pd.Timestamp(seed_time)
        ids = list(entity_ids)
        edges_all = _edges(db, ds, s)
        partners = pd.Index(sorted(edges_all["dst"].dropna().unique(), key=str))
        n_e, n_p = len(ids), len(partners)
        eidx = {e: i for i, e in enumerate(ids)}
        pidx = {p: n_e + i for i, p in enumerate(partners)}

        class Net(torch.nn.Module):
            def __init__(self, d):
                super().__init__()
                self.c1 = SAGEConv(d, hidden)
                self.c2 = SAGEConv(hidden, hidden)
                self.out = torch.nn.Linear(hidden, 1)

            def forward(self, x, ei):
                h = torch.relu(self.c1(x, ei))
                h = torch.relu(self.c2(h, ei))
                return self.out(h).squeeze(-1)

        def graph(at):
            e = edges_all[edges_all["src"].isin(eidx)]
            if temporal:
                e = e[e["time"] <= at]
            src = torch.tensor([eidx[v] for v in e["src"]], dtype=torch.long)
            dst = torch.tensor([pidx[v] for v in e["dst"]], dtype=torch.long)
            ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]) if len(e) else torch.zeros((2, 0),
                                                                                                         dtype=torch.long)
            f = feats(db, s, ids, at).to_numpy(float)
            f = np.log1p(np.clip(f, 0, None))
            x = np.zeros((n_e + n_p, f.shape[1] + 1))
            x[:n_e, :-1] = f
            x[n_e:, -1] = 1.0                                     # partner-node indicator
            return torch.tensor(x, dtype=torch.float32), ei

        train = [(graph(t - pd.Timedelta(days=o)),
                  torch.tensor(label(db, s, ids, t - pd.Timedelta(days=o)).to_numpy(), dtype=torch.float32))
                 for o in correct_offsets(s)]
        model = Net(train[0][0][0].shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(epochs):
            for (x, ei), y in train:
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x, ei)[:n_e], y)
                loss.backward()
                opt.step()
        x, ei = graph(t)
        with torch.no_grad():
            score = torch.sigmoid(model(x, ei)[:n_e]).numpy().astype(float)
        return pd.DataFrame({"score": score}, index=pd.Index(ids))
    return fit_predict
