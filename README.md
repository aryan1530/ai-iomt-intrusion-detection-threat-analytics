# IoMT Intrusion Detection and Threat Analytics

Software IDS that detects malicious IoMT/network traffic with reduced compute via MI + ensemble feature ranking (RF, AdaBoost, XGBoost, LightGBM) and a Random Forest detector.

Flow: Datasets → Preprocessing → MI → ensemble ranking → Union/Intersection → RF (baseline vs proposed) → Evaluation → FastAPI → DB → React/Plotly dashboard → monitoring.

## Datasets

Place files in:

- `data/raw/nsl_kdd/` — NSL-KDD `KDDTrain+.txt`, `KDDTest+.txt`
- `data/raw/ciciomt2024/` — CICIoMT2024 CSVs
- `data/raw/wustl_ehms/` — WUSTL-EHMS-2020 CSVs

If files are absent, synthetic but correlated traffic is generated so the stack still trains and demos.

## Run

```bash
python backend/scripts/prepare_data_dirs.py
pip install -r backend/requirements.txt
python backend/scripts/train_pipeline.py
# optional: python backend/scripts/train_pipeline.py nsl_kdd
cd backend && python -m uvicorn app.main:app --reload --port 8000
cd frontend && npm install && npm run dev
```

Dashboard: http://127.0.0.1:5173  API docs: http://127.0.0.1:8000/docs

PostgreSQL: set `DATABASE_URL=postgresql://user:pass@host:5432/iomt` or `docker compose up --build`.

## Tests

```bash
pip install -r backend/requirements.txt
pytest tests -q
```

## Feature mode

`FEATURE_MODE=union` (default) or `intersection` in `.env`.
