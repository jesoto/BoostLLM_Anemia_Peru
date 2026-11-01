"""Discriminación, calibración, equidad por subgrupo y ADS."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score, brier_score_loss


def sens_spec(y, p, thr=0.5):
    y, pred = np.asarray(y), np.asarray(p) >= thr
    tp, fn = ((y == 1) & pred).sum(), ((y == 1) & ~pred).sum()
    tn, fp = ((y == 0) & ~pred).sum(), ((y == 0) & pred).sum()
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1)


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any())


def summary(y, p, w=None, thr=0.5) -> dict:
    se, sp = sens_spec(y, p, thr)
    return {"auroc": roc_auc_score(y, p, sample_weight=w), "brier": brier_score_loss(y, p),
            "ece": ece(y, p), "sens": se, "spec": sp}


def altitude_group(alt_m: pd.Series) -> pd.Series:
    # Nota: 3000-3500 m queda fuera de los grupos del protocolo original; se etiqueta aparte.
    return pd.cut(alt_m, [-1, 500, 3000, 3500, 1e9], labels=["<500", "500-3000", "3000-3500", ">3500"])


def fairness_table(df: pd.DataFrame, y: str, p: str, group_cols: list[str], thr=0.5) -> pd.DataFrame:
    rows = []
    for g in group_cols:
        for k, sub in df.groupby(g, observed=True):
            if sub[y].nunique() < 2:
                continue
            rows.append({"group": g, "level": k, "n": len(sub), **summary(sub[y], sub[p], thr=thr)})
    return pd.DataFrame(rows)


def attribution_discrepancy(shap_rank: list[str], llm_rank: list[str]) -> float:
    """ADS: tau de Kendall entre el ranking SHAP y el ranking de variables priorizadas por el LLM."""
    feats = [f for f in shap_rank if f in llm_rank]
    tau, _ = kendalltau([shap_rank.index(f) for f in feats], [llm_rank.index(f) for f in feats])
    return float(tau)
