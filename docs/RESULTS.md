# Results, limitations, findings

Trained on this machine using the common loader. Official CICIoMT2024 / NSL-KDD / WUSTL-EHMS files were not present under `data/raw/`, so **synthetic but label-correlated traffic** was used. Replace those files and re-run `python backend/scripts/train_pipeline.py` for paper-grade numbers.

Feature mode: **union**. Metrics: Accuracy, Precision, Recall, F1 (weighted). Cost: train seconds, predict seconds, RSS memory delta.

## Binary detection (malicious vs normal)

| Dataset | Features full → selected | Baseline Acc | Proposed Acc | Baseline F1 | Proposed F1 | Baseline train s | Proposed train s |
|---|---|---|---|---|---|---|---|
| NSL-KDD | 51 → 10 | 0.834 | **0.838** | 0.827 | **0.834** | 0.592 | **0.510** |
| CICIoMT2024 | 49 → 6 | 1.000 | 1.000 | 1.000 | 1.000 | 0.593 | **0.511** |
| WUSTL-EHMS | 28 → 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.457 | 0.490 |

\*CICIoMT/WUSTL synthetic labels are strongly tied to a few engineered columns (e.g. Loss, syn flags), so both models saturate; the useful result is the **feature cut** (49→6, 28→5) with matching detection.

## Multiclass attack category

| Dataset | Baseline Acc | Proposed Acc | Baseline F1 | Proposed F1 |
|---|---|---|---|---|
| NSL-KDD | 0.694 | 0.690 | 0.624 | 0.625 |
| WUSTL-EHMS | 0.748 | 0.728 | 0.746 | 0.728 |

Category scores are weaker because synthetic attack types share similar flow stats; binary malicious/normal is the operational target.

## Computational savings

- NSL-KDD: **80% fewer features** (51→10) with slightly **higher** binary accuracy/F1 and **~14% faster** RF training.
- CICIoMT-style: **88% fewer features** (49→6) with no binary-metric drop.
- EHMS-style: **82% fewer features** (28→5) with unchanged binary metrics.

Predict-time deltas on 500 rows are small and noisy (sub-100 ms). Memory RSS is process-level and not a reliable micro-benchmark.

## Confusion matrices (proposed binary)

- NSL-KDD: [[111, 53], [28, 308]]
- WUSTL-EHMS: [[164, 0], [0, 336]]

## Artifacts reused without retraining

`artifacts/<dataset>/`: `preprocessor.joblib`, `encoders.joblib`, `rf_binary.joblib`, `rf_multiclass.joblib`, `feature_selection.json`, `feature_names.json`, `metrics.json`, `comparison.json`.

## Limitations

- Synthetic fallback is for software demo, not a published CIC/WUSTL score.
- Union keeps more features than intersection; empty intersection falls back to top MI features.
- Explanations use ensemble mean importance of **selected encoded** features, not SHAP.
- SQLite is default; set `DATABASE_URL` for PostgreSQL.

## Findings

MI + RF/AdaBoost/XGBoost/LightGBM ranking + union yields a much smaller RF input. On NSL-KDD-style data, detection held or slightly improved while training got cheaper. On strongly separable synthetic IoMT/EHMS flows, feature reduction did not hurt binary detection. Fine-grained attack typing needs real labeled datasets.
