from typing import Any, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    dataset: str = "nsl_kdd"
    features: dict[str, Any]


class BatchPredictRequest(BaseModel):
    dataset: str = "nsl_kdd"
    records: list[dict[str, Any]] = Field(default_factory=list)


class PredictResponse(BaseModel):
    binary_label: str
    attack_category: str
    confidence: float
    execution_ms: float
    explanation: list[dict[str, Any]]
    selected_features_used: list[str]
