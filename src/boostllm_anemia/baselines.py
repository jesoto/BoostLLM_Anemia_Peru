"""Líneas base fuertes: XGBoost (+SHAP) y regresión logística."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from .data import assert_no_leakage


def fit_xgb(train: pd.DataFrame, features: list[str], label: str, seed=42, **kw) -> XGBClassifier:
    assert_no_leakage(features)
    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                          eval_metric="auc", random_state=seed, **kw)
    model.fit(train[features], train[label])
    return model


def fit_logit(train: pd.DataFrame, features: list[str], label: str):
    assert_no_leakage(features)
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    m.fit(train[features], train[label])
    return m


def shap_values(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    import shap
    return shap.TreeExplainer(model).shap_values(X)
