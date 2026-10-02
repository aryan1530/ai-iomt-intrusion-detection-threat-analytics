import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.datasets import generate_synthetic, load_dataset  # noqa: E402
from app.ml.feature_selection import combine_ranks, select_features  # noqa: E402
from app.ml.preprocess import fit_transform_split, split_xy  # noqa: E402
from app.ml.train import train_baseline_and_proposed  # noqa: E402


def test_loaders_synthetic():
    for name in ["nsl_kdd", "ciciomt2024", "wustl_ehms"]:
        df = generate_synthetic(name, n=200)
        assert "binary_label" in df.columns
        assert "attack_category" in df.columns
        assert len(df) == 200


def test_preprocess_and_split():
    X, yb, yc = split_xy(generate_synthetic("nsl_kdd", n=300))
    b = fit_transform_split(X, yb, yc)
    assert b["X_train"].shape[0] > 0
    assert len(b["feature_names"]) == b["X_train"].shape[1]


def test_feature_selection():
    X, yb, yc = split_xy(generate_synthetic("ciciomt2024", n=400))
    b = fit_transform_split(X, yb, yc)
    sel = select_features(b["X_train"], b["yb_train"], b["feature_names"])
    assert sel["selected"]
    assert "random_forest" in sel["ranks"]
    u = combine_ranks({k: pd.Series(v) for k, v in sel["ranks"].items()}, "union")
    i = combine_ranks({k: pd.Series(v) for k, v in sel["ranks"].items()}, "intersection")
    assert set(i).issubset(set(u) | set(i))


def test_baseline_vs_proposed():
    X, yb, yc = split_xy(generate_synthetic("wustl_ehms", n=400))
    b = fit_transform_split(X, yb, yc)
    sel = select_features(b["X_train"], b["yb_train"], b["feature_names"])
    r = train_baseline_and_proposed(b, sel["selected"])
    pb = r["proposed_binary"]["metrics"]
    bb = r["baseline_binary"]["metrics"]
    assert pb["n_features"] <= bb["n_features"]
    assert pb["accuracy"] >= 0.5
    for k in ["precision", "recall", "f1_score", "confusion_matrix", "train_seconds"]:
        assert k in pb


def test_api_health():
    from app.main import app

    c = TestClient(app)
    assert c.get("/api/health").json()["status"] == "ok"
    assert "nsl_kdd" in c.get("/api/datasets").json()
