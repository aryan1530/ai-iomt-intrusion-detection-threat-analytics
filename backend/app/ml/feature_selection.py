from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ..config import settings


def mi_scores(X: np.ndarray, y: np.ndarray, names: list[str]) -> pd.Series:
    scores = mutual_info_classif(X, y, random_state=42)
    s = pd.Series(scores, index=names).sort_values(ascending=False)
    return s[s >= settings.mi_threshold]


def _fit_importances(model, X, y, names: list[str]) -> pd.Series:
    model.fit(X, y)
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return pd.Series(0, index=names)
    return pd.Series(imp, index=names).sort_values(ascending=False)


def ensemble_rank(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict[str, pd.Series]:
    y = LabelEncoder().fit_transform(y) if y.dtype == object else y
    models = {
        "random_forest": RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1),
        "adaboost": AdaBoostClassifier(n_estimators=50, random_state=42, algorithm="SAMME"),
        "xgboost": XGBClassifier(n_estimators=80, max_depth=6, n_jobs=-1, random_state=42, eval_metric="mlogloss"),
        "lightgbm": LGBMClassifier(n_estimators=80, random_state=42, verbose=-1),
    }
    ranks = {}
    for k, m in models.items():
        ranks[k] = _fit_importances(m, X, y, names)
    return ranks


def combine_ranks(ranks: dict[str, pd.Series], mode: str | None = None) -> list[str]:
    mode = (mode or settings.feature_mode).lower()
    k = settings.top_k_per_model
    sets = [set(s.head(k).index) for s in ranks.values()]
    if mode == "intersection":
        selected = set.intersection(*sets) if sets else set()
    else:
        selected = set.union(*sets) if sets else set()
    # keep stable order by mean rank
    mean = pd.concat(ranks, axis=1).mean(axis=1)
    return [f for f in mean.sort_values(ascending=False).index if f in selected]


def select_features(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    mi = mi_scores(X, y, names)
    mi_names = list(mi.index)
    idx = [names.index(n) for n in mi_names]
    Xm = X[:, idx]
    ranks = ensemble_rank(Xm, y, mi_names)
    selected = combine_ranks(ranks)
    if not selected:
        selected = mi_names[: min(15, len(mi_names))]
    payload = {
        "mi": mi.to_dict(),
        "ranks": {k: v.to_dict() for k, v in ranks.items()},
        "selected": selected,
        "mode": settings.feature_mode,
        "mi_threshold": settings.mi_threshold,
        "top_k_per_model": settings.top_k_per_model,
    }
    return payload


def save_selection(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
