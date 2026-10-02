import pytest
from fastapi.testclient import TestClient

def test_e2e_predict_api():
    from app.main import app
    from app.config import settings

    if not (settings.artifacts_dir / "nsl_kdd" / "rf_binary.joblib").exists():
        pytest.skip("train first")
    c = TestClient(app)
    r = c.post("/api/predict", json={"dataset": "nsl_kdd", "features": {"count": 210, "serror_rate": 0.92, "src_bytes": 40}})
    assert r.status_code == 200
    body = r.json()
    assert body["binary_label"] in {"normal", "malicious"}
    assert "explanation" in body
    assert c.get("/api/analytics/summary").status_code == 200
    assert c.get("/api/feature-importance?dataset=nsl_kdd").status_code == 200
