from __future__ import annotations

import json
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from .database import AttackRecord, ModelMetric, Prediction, get_db
from .ml.datasets import list_available
from .ml.predict import get_predictor
from .schemas import BatchPredictRequest, PredictRequest, PredictResponse

router = APIRouter()


def _store(db: Session, dataset: str, result: dict, features: dict) -> None:
    db.add(
        Prediction(
            dataset=dataset,
            binary_label=result["binary_label"],
            attack_category=result["attack_category"],
            confidence=result["confidence"],
            execution_ms=result["execution_ms"],
            features_json=json.dumps(features, default=str),
            explanation_json=json.dumps(result["explanation"]),
        )
    )
    if result["binary_label"] != "normal":
        db.add(AttackRecord(attack_category=result["attack_category"], source="prediction", count=1))
    db.commit()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/datasets")
def datasets():
    return list_available()


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, db: Session = Depends(get_db)):
    try:
        pred = get_predictor(body.dataset)
        result = pred.predict_one(body.features)
    except FileNotFoundError as e:
        raise HTTPException(400, f"Model artifacts missing. Train first. {e}") from e
    except Exception as e:
        raise HTTPException(400, str(e)) from e
    _store(db, body.dataset, result, body.features)
    return result


@router.post("/predict/batch")
def predict_batch(body: BatchPredictRequest, db: Session = Depends(get_db)):
    pred = get_predictor(body.dataset)
    out = []
    for rec in body.records:
        r = pred.predict_one(rec)
        _store(db, body.dataset, r, rec)
        out.append(r)
    return {"count": len(out), "results": out}


@router.post("/predict/csv")
async def predict_csv(dataset: str = "nsl_kdd", file: UploadFile = File(...), db: Session = Depends(get_db)):
    import pandas as pd

    df = pd.read_csv(file.file)
    pred = get_predictor(dataset)
    results = []
    for rec in df.to_dict(orient="records"):
        r = pred.predict_one(rec)
        _store(db, dataset, r, rec)
        results.append(r)
    return {"count": len(results), "results": results}


@router.get("/metrics")
def metrics(db: Session = Depends(get_db)):
    rows = db.query(ModelMetric).order_by(ModelMetric.id.desc()).limit(40).all()
    return [
        {
            "model_name": r.model_name,
            "dataset": r.dataset,
            "accuracy": r.accuracy,
            "precision": r.precision,
            "recall": r.recall,
            "f1_score": r.f1_score,
            "train_seconds": r.train_seconds,
            "predict_seconds": r.predict_seconds,
            "memory_mb": r.memory_mb,
            "n_features": r.n_features,
            "extra": json.loads(r.extra_json or "{}"),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.get("/comparison")
def comparison():
    from .config import settings

    p = settings.artifacts_dir / "all_comparisons.json"
    if p.exists():
        return json.loads(p.read_text())
    items = []
    for d in settings.artifacts_dir.iterdir():
        c = d / "comparison.json"
        if c.exists():
            items.append(json.loads(c.read_text()))
    return items


@router.get("/feature-importance")
def feature_importance(dataset: str = "nsl_kdd"):
    from .config import settings

    p = settings.artifacts_dir / dataset / "feature_selection.json"
    if not p.exists():
        raise HTTPException(404, "Train the model first")
    return json.loads(p.read_text())


@router.get("/attacks/stats")
def attack_stats(db: Session = Depends(get_db)):
    rows = db.query(Prediction.attack_category, func.count(Prediction.id)).group_by(Prediction.attack_category).all()
    return {"distribution": {k: v for k, v in rows}, "total": sum(v for _, v in rows)}


@router.get("/history")
def history(limit: int = 100, db: Session = Depends(get_db)):
    rows = db.query(Prediction).order_by(Prediction.id.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "dataset": r.dataset,
            "binary_label": r.binary_label,
            "attack_category": r.attack_category,
            "confidence": r.confidence,
            "execution_ms": r.execution_ms,
            "explanation": json.loads(r.explanation_json or "[]"),
        }
        for r in rows
    ]


@router.get("/analytics/summary")
def analytics(db: Session = Depends(get_db)):
    total = db.query(func.count(Prediction.id)).scalar() or 0
    mal = db.query(func.count(Prediction.id)).filter(Prediction.binary_label == "malicious").scalar() or 0
    avg_ms = db.query(func.avg(Prediction.execution_ms)).scalar() or 0
    cats = dict(db.query(Prediction.attack_category, func.count(Prediction.id)).group_by(Prediction.attack_category).all())
    return {
        "total_predictions": total,
        "malicious": mal,
        "normal": total - mal,
        "avg_execution_ms": float(avg_ms),
        "categories": cats,
    }
