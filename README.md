
# AI-Based IoMT Intrusion Detection & Threat Analytics

A software-based **Intrusion Detection System (IDS)** for Internet of Medical Things (IoMT) and network environments that uses machine learning to identify malicious traffic while reducing computational complexity through intelligent feature selection.

The system combines **Mutual Information (MI)** with an ensemble of feature-ranking algorithms — **Random Forest, AdaBoost, XGBoost, and LightGBM** — to identify relevant network traffic features. A Random Forest classifier is then used to detect potentially malicious traffic and provide security analytics through a web-based dashboard.

---

## Overview

IoMT environments generate large volumes of network traffic from connected medical devices and services. An effective intrusion detection system needs to identify malicious activity while maintaining reasonable computational efficiency.

This project addresses this problem through a machine-learning pipeline that:

- Preprocesses network traffic datasets
- Applies **Mutual Information (MI)** for feature selection
- Ranks features using multiple ensemble learning algorithms
- Combines selected features using **Union / Intersection** strategies
- Compares a **baseline Random Forest model** with the proposed reduced-feature model
- Evaluates detection performance and computational cost
- Provides REST APIs through **FastAPI**
- Stores application data using a database
- Presents results through a **React-based dashboard**
- Provides threat analytics and monitoring information

---

## Key Features

### Machine Learning Detection

- Random Forest based intrusion detection
- Binary malicious/normal traffic classification
- Baseline vs reduced-feature model comparison
- Dataset-specific training and evaluation

### Intelligent Feature Selection

The feature-selection pipeline combines multiple approaches:

| Technique | Purpose |
|---|---|
| Mutual Information | Identifies informative features |
| Random Forest | Measures feature importance using tree ensembles |
| AdaBoost | Provides boosting-based feature ranking |
| XGBoost | Provides gradient-boosting feature importance |
| LightGBM | Provides efficient boosting-based ranking |
| Union | Combines features identified by ranking methods |
| Intersection | Uses features consistently selected across methods |

This approach aims to reduce the number of input features while maintaining detection performance.

---

## System Architecture

```text
                 ┌──────────────────────┐
                 │      Datasets        │
                 │                      │
                 │ NSL-KDD              │
                 │ CICIoMT2024          │
                 │ WUSTL-EHMS-2020      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Preprocessing     │
                 │                      │
                 │ Cleaning             │
                 │ Encoding             │
                 │ Transformation       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Mutual Information   │
                 │ Feature Selection    │
                 └──────────┬───────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │   Ensemble Feature Ranking  │
              │                             │
              │ RF │ AdaBoost │ XGBoost    │
              │              │ LightGBM     │
              └──────────────┬──────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Feature Combination  │
                  │                      │
                  │ Union / Intersection │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Random Forest IDS    │
                  │                      │
                  │ Baseline vs Proposed │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │     Evaluation       │
                  │                      │
                  │ Accuracy             │
                  │ Precision            │
                  │ Recall               │
                  │ F1-Score             │
                  │ Confusion Matrix     │
                  │ Training Time        │
                  └──────────┬───────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │          FastAPI            │
              │       REST Backend          │
              └──────────────┬──────────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
             ┌─────────────┐   ┌─────────────┐
             │  Database   │   │    React    │
             │             │   │  Dashboard  │
             └─────────────┘   └──────┬──────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │ Threat Analytics │
                            │ & Monitoring     │
                            └──────────────────┘
