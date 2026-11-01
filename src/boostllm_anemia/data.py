"""Carga de ENDES, cohorte, split y protección anti-fuga."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

# Hemoglobina medida/ajustada y su derivado: jamás pueden ser predictores de anemia.
FORBIDDEN_FEATURES = {
    "HC53", "HW53", "HA53", "HC56", "HW56", "HA56", "HC57", "HW57", "HA57",
    "anemia_hist", "anemia_nts2024",
}


def load_child_module(path: str) -> pd.DataFrame:
    """Lee un .sav/.dta/.csv del módulo de niños (RECH6/REC44 fusionado). Ajustar por ronda."""
    if path.endswith(".sav"):
        return pd.read_spss(path)
    if path.endswith(".dta"):
        return pd.read_stata(path)
    return pd.read_csv(path)


def build_cohort(df: pd.DataFrame, age_range=(6, 59)) -> pd.DataFrame:
    lo, hi = age_range
    out = df[df["HC1"].between(lo, hi)].copy()
    return out.dropna(subset=["HC53", "HV040"])


def assert_no_leakage(features: list[str]) -> None:
    leaked = FORBIDDEN_FEATURES.intersection(features)
    if leaked:
        raise ValueError(f"Fuga de datos: {sorted(leaked)} no pueden ser predictores")


def split(df: pd.DataFrame, label: str, test_size=0.2, seed=42):
    """Partición estratificada única, compartida por todos los modelos."""
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[label])
