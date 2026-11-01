"""Etiquetas de anemia: histórica vs. recalculada bajo NTS 213-MINSA/DGIESP-2024.

Cortes (Tabla 2) y ajuste por altitud (Tabla 4) verificados contra la Guía OMS 2024
(9789240088542). Falta contrastar con el texto de la NTS 213-MINSA/DGIESP-2024 si difiere de la OMS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CUTOFFS = {(6, 23): 10.5, (24, 59): 11.0}  # g/dL, OMS 2024 Tabla 2


def altitude_adjustment_legacy(alt_m: pd.Series) -> pd.Series:
    """Ajuste CDC 1989/OMS 2001 (g/dL a restar), aplica desde 1000 m s. n. m."""
    a = np.clip(alt_m, 0, None) * 3.3 / 1000.0
    adj = -0.032 * a + 0.022 * a**2
    return pd.Series(np.where(alt_m >= 1000, adj, 0.0), index=alt_m.index).clip(lower=0)


# OMS 2024 (Guideline on haemoglobin cutoffs, Tabla 4), g/L por tramo de 500 m desde 500 m s. n. m.
WHO2024_ADJ_GL = [(500, 4), (1000, 8), (1500, 11), (2000, 14), (2500, 18),
                  (3000, 21), (3500, 25), (4000, 29), (4500, 33)]


def altitude_adjustment_who2024(alt_m: pd.Series, use_equation: bool = False) -> pd.Series:
    """Ajuste OMS 2024 en g/dL (se resta de la Hb observada). Tabla 4 por tramos de 500 m.

    use_equation=True usa la ecuación continua de la guía: (0.0056384*h + 0.0000003*h^2) g/L.
    Sobre 5000 m la tabla no llega: se usa la ecuación (extrapolación, reportar como limitación).
    """
    h = alt_m.astype(float)
    eq = (0.0056384 * h + 0.0000003 * h**2) / 10.0
    if use_equation:
        return eq.where(h >= 500, 0.0)
    step = pd.Series(0.0, index=h.index)
    for lo, adj in WHO2024_ADJ_GL:
        step[h >= lo] = adj / 10.0
    return step.where(h < 5000, eq)


def hb_cutoff(age_months: pd.Series, cutoffs=DEFAULT_CUTOFFS) -> pd.Series:
    out = pd.Series(np.nan, index=age_months.index)
    for (lo, hi), c in cutoffs.items():
        out[age_months.between(lo, hi)] = c
    return out


def recompute_anemia(df: pd.DataFrame, equation: str = "who2024", cutoffs=DEFAULT_CUTOFFS) -> pd.Series:
    """Anemia = (Hb sin ajustar HC53 [g/dL] - ajuste por altitud) < corte por edad.

    HC53 en ENDES viene en g/dL; si viene en décimas (p. ej. 105) divida entre 10 antes.
    """
    fn = {"legacy_cdc": altitude_adjustment_legacy, "who2024": altitude_adjustment_who2024}[equation]
    hb_adj = df["HC53"] - fn(df["HV040"])
    return (hb_adj < hb_cutoff(df["HC1"], cutoffs)).astype(int)


def add_labels(df: pd.DataFrame, equation: str = "who2024") -> pd.DataFrame:
    out = df.copy()
    if "HC57" in out:
        out["anemia_hist"] = (out["HC57"].isin([1, 2, 3])).astype(int)  # 1-3 = leve/moderada/severa
    out["anemia_nts2024"] = recompute_anemia(out, equation)
    return out
