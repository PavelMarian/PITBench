# Framework adapters (optional)

`adapters.py` wraps real ML frameworks as layer programs `(db, entity_ids, seed_time) -> DataFrame`
so that `layers/harness.py` checks them with the same views as the generated controls.
None of them is a dependency of PITBench: each adapter imports its framework only when it is
called, `available()` reports what is installed, and `validate_frameworks.py` skips the rest.

| family | configurations | needs |
|---|---|---|
| featuretools | `ft_d3` cutoff + strict, `ft_d2` cutoff + library default, `ft_d1` no cutoff | `featuretools`, `setuptools<70` (woodwork imports `pkg_resources`) |
| LightGBM | `lgbm_correct`, `lgbm_label_overlap`, `lgbm_target_encoding` (sklearn `TargetEncoder`) | `lightgbm`, `scikit-learn` |
| AutoGluon | `ag_correct`, `ag_label_overlap` (fixed GBM hyperparameters, no bagging: deterministic) | `autogluon.tabular` |
| PyTorch Geometric | `gnn_temporal`, `gnn_nontemporal` (GraphSAGE; edges with time <= t_i vs all edges) | `torch`, `torch_geometric` |

Validation status (`results_frameworks/`), from one earlier run with numpy 1.26.4, pandas 2.2.3,
scikit-learn 1.9.1, lightgbm 4.7.0, featuretools 1.31.0:

* featuretools on olist, tpch, pkdd99: 9/9 agree. D1 leaks (witness + canary); D2 and D3 are clean,
  i.e. the off-by-one of `include_cutoff_time` does not reproduce, as in the prestudy.
* LightGBM on all 8 datasets: 24/24 agree, all deterministic. Gap backtest − at t:
  label overlap +0.03 typical (up to +0.09), id target encoding up to +0.28.
* AutoGluon and PyG adapters are written but **not validated** here (the environment was
  rebuilt without heavy dependencies). Run `validate_frameworks.py --only ag gnn` where they
  are installed; the determinism check comes first and must pass.
