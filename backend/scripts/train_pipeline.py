"""Train pipeline: load → preprocess → MI → ensemble rank → union/intersection → RF baseline vs proposed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import settings  # noqa: E402
from app.database import ModelMetric, SessionLocal, init_db  # noqa: E402
from app.ml.datasets import LOADERS, load_dataset  # noqa: E402
from app.ml.feature_selection import save_selection, select_features  # noqa: E402
from app.ml.preprocess import fit_transform_split  # noqa: E402
from app.ml.train import save_models, train_baseline_and_proposed  # noqa: E402


def train_one(name: str) -> dict:
    df = load_dataset(name)
    if len(df) > 4000:
        df = df.sample(4000, random_state=42)
    from app.ml.preprocess import split_xy

    X, y_bin, y_cat = split_xy(df)
    bundle = fit_transform_split(X, y_bin, y_cat)
    sel = select_features(bundle["X_train"], bundle["yb_train"], bundle["feature_names"])
    result = train_baseline_and_proposed(bundle, sel["selected"])
    out = settings.artifacts_dir / name
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle["preprocessor"], out / "preprocessor.joblib")
    joblib.dump({"bin_enc": bundle["bin_enc"], "cat_enc": bundle["cat_enc"]}, out / "encoders.joblib")
    (out / "feature_names.json").write_text(json.dumps(bundle["feature_names"], indent=2))
    save_selection(sel, out / "feature_selection.json")
    save_models(result, name)
    comparison = {
        "dataset": name,
        "n_rows": int(len(df)),
        "n_features_full": len(bundle["feature_names"]),
        "n_features_selected": len(sel["selected"]),
        "mode": sel["mode"],
        "baseline_binary": result["baseline_binary"]["metrics"],
        "proposed_binary": result["proposed_binary"]["metrics"],
        "baseline_multiclass": result["baseline_multiclass"]["metrics"],
        "proposed_multiclass": result["proposed_multiclass"]["metrics"],
    }
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2, default=float))
    init_db()
    db = SessionLocal()
    try:
        for key, m in [
            ("baseline_binary", result["baseline_binary"]["metrics"]),
            ("proposed_binary", result["proposed_binary"]["metrics"]),
            ("baseline_multiclass", result["baseline_multiclass"]["metrics"]),
            ("proposed_multiclass", result["proposed_multiclass"]["metrics"]),
        ]:
            db.add(
                ModelMetric(
                    model_name=key,
                    dataset=name,
                    accuracy=m["accuracy"],
                    precision=m["precision"],
                    recall=m["recall"],
                    f1_score=m["f1_score"],
                    train_seconds=m["train_seconds"],
                    predict_seconds=m["predict_seconds"],
                    memory_mb=m["memory_mb"],
                    n_features=m["n_features"],
                    extra_json=json.dumps({"confusion_matrix": m["confusion_matrix"]}),
                )
            )
        db.commit()
    finally:
        db.close()
    return comparison


def main():
    names = sys.argv[1:] or list(LOADERS)
    reports = [train_one(n) for n in names]
    (settings.artifacts_dir / "all_comparisons.json").write_text(json.dumps(reports, indent=2, default=float))
    print(json.dumps(reports, indent=2, default=float))


if __name__ == "__main__":
    main()
