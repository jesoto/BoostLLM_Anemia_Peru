"""BoostLLM: rutas raíz-hoja de XGBoost comprimidas por intersección de restricciones."""
from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier


def _tree_tables(model: XGBClassifier) -> pd.DataFrame:
    return model.get_booster().trees_to_dataframe()


def sample_path_constraints(model: XGBClassifier, x: pd.Series, rounds: slice | None = None) -> dict[str, tuple[float, float]]:
    """Intervalos (lo, hi) por variable tras intersectar las rutas de las rondas seleccionadas."""
    t = _tree_tables(model)
    bounds: dict[str, list[float]] = {}
    trees = sorted(t["Tree"].unique())[rounds or slice(None)]
    for tr in trees:
        nodes = t[t["Tree"] == tr].set_index("ID")
        cur = nodes.index[0]
        while nodes.loc[cur, "Feature"] != "Leaf":
            n = nodes.loc[cur]
            f, thr = n["Feature"], n["Split"]
            v = x[f]
            lo, hi = bounds.setdefault(f, [-float("inf"), float("inf")])
            if pd.isna(v):
                cur = n["Missing"]
            elif v < thr:
                bounds[f][1] = min(hi, thr)
                cur = n["Yes"]
            else:
                bounds[f][0] = max(lo, thr)
                cur = n["No"]
    return {f: (lo, hi) for f, (lo, hi) in bounds.items()}


def constraints_to_text(c: dict[str, tuple[float, float]], names: dict[str, str] | None = None) -> str:
    names = names or {}
    parts = []
    for f, (lo, hi) in c.items():
        n = names.get(f, f)
        if lo > -float("inf") and hi < float("inf"):
            parts.append(f"{lo:.1f} ≤ {n} < {hi:.1f}")
        elif hi < float("inf"):
            parts.append(f"{n} < {hi:.1f}")
        else:
            parts.append(f"{n} ≥ {lo:.1f}")
    return "Reglas del modelo de árboles para este caso: " + "; ".join(parts) + "."
