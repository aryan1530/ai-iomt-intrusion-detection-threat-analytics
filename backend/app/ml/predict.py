from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import settings


class Predictor:
    def __init__(self, dataset: str = "nsl_kdd"):
        self.dataset = dataset
        self.dir = settings.artifacts_dir / dataset
        self.pre = joblib.load(self.dir / "preprocessor.joblib")
        self.bin_model = joblib.load(self.dir / "rf_binary.joblib")
        self.cat_model = joblib.load(self.dir / "rf_multiclass.joblib")
        self.selection = json.loads((self.dir / "feature_selection.json").read_text())
        self.selected = self.selection["selected"]
        self.feature_names = json.loads((self.dir / "feature_names.json").read_text())
        bundle = joblib.load(self.dir / "encoders.joblib")
        self.bin_enc = bundle["bin_enc"]
        self.cat_enc = bundle["cat_enc"]
        ranks = self.selection.get("ranks", {})
        mean: dict[str, float] = {}
        for d in ranks.values():
            for k, v in d.items():
                mean[k] = mean.get(k, 0.0) + float(v)
        n = max(len(ranks), 1)
        self.mean_importance = {k: v / n for k, v in mean.items()}

    def _row(self, features: dict) -> pd.DataFrame:
        raw_names = list(self.pre.feature_names_in_)
        return pd.DataFrame([{c: features.get(c, np.nan) for c in raw_names}])

    def predict_one(self, features: dict) -> dict:
        t0 = time.perf_counter()
        X = self.pre.transform(self._row(features))
        idx = [self.feature_names.index(n) for n in self.selected if n in self.feature_names]
        Xs = X[:, idx]
        pb = self.bin_model.predict_proba(Xs)[0]
        pc = self.cat_model.predict_proba(Xs)[0]
        binary = str(self.bin_enc.inverse_transform([int(np.argmax(pb))])[0])
        category = str(self.cat_enc.inverse_transform([int(np.argmax(pc))])[0])
        expl = sorted(
            [{"feature": f, "importance": float(self.mean_importance.get(f, 0.0))} for f in self.selected],
            key=lambda x: x["importance"],
            reverse=True,
        )[:12]
        return {
            "binary_label": binary,
            "attack_category": category if binary != "normal" else "normal",
            "confidence": float(max(pb)),
            "execution_ms": (time.perf_counter() - t0) * 1000,
            "explanation": expl,
            "selected_features_used": self.selected,
        }


_cache: dict[str, Predictor] = {}


def get_predictor(dataset: str) -> Predictor:
    if dataset not in _cache:
        _cache[dataset] = Predictor(dataset)
    return _cache[dataset]


def clear_cache() -> None:
    _cache.clear()
