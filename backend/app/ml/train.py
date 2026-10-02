from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ..config import settings


def _metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }


def train_rf(X_train, y_train, X_test, y_test, n_estimators: int = 120) -> dict:
    proc = psutil.Process()
    mem0 = proc.memory_info().rss / (1024 * 1024)
    t0 = time.perf_counter()
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = model.predict(X_test)
    pred_s = time.perf_counter() - t1
    mem1 = proc.memory_info().rss / (1024 * 1024)
    m = _metrics(y_test, pred)
    m.update({
        "train_seconds": train_s,
        "predict_seconds": pred_s,
        "memory_mb": max(mem1 - mem0, 0.0),
        "n_features": int(X_train.shape[1]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    })
    return {"model": model, "metrics": m}


def column_index(all_names: list[str], selected: list[str]) -> np.ndarray:
    return np.array([all_names.index(n) for n in selected if n in all_names])


def train_baseline_and_proposed(bundle: dict, selected: list[str]) -> dict:
    names = bundle["feature_names"]
    idx = column_index(names, selected)
    base_bin = train_rf(bundle["X_train"], bundle["yb_train"], bundle["X_test"], bundle["yb_test"])
    prop_bin = train_rf(bundle["X_train"][:, idx], bundle["yb_train"], bundle["X_test"][:, idx], bundle["yb_test"])
    base_cat = train_rf(bundle["X_train"], bundle["yc_train"], bundle["X_test"], bundle["yc_test"])
    prop_cat = train_rf(bundle["X_train"][:, idx], bundle["yc_train"], bundle["X_test"][:, idx], bundle["yc_test"])
    return {
        "baseline_binary": base_bin,
        "proposed_binary": prop_bin,
        "baseline_multiclass": base_cat,
        "proposed_multiclass": prop_cat,
        "selected": selected,
        "all_features": names,
    }


def save_models(result: dict, dataset: str) -> Path:
    d = settings.artifacts_dir / dataset
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["proposed_binary"]["model"], d / "rf_binary.joblib")
    joblib.dump(result["proposed_multiclass"]["model"], d / "rf_multiclass.joblib")
    joblib.dump(result["baseline_binary"]["model"], d / "rf_baseline_binary.joblib")
    metrics = {
        "baseline_binary": result["baseline_binary"]["metrics"],
        "proposed_binary": result["proposed_binary"]["metrics"],
        "baseline_multiclass": result["baseline_multiclass"]["metrics"],
        "proposed_multiclass": result["proposed_multiclass"]["metrics"],
        "selected": result["selected"],
        "n_all_features": len(result["all_features"]),
    }
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return d
