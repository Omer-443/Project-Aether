"""
Pydantic request/response models for API validation.
"""
from typing import List, Optional
from pydantic import BaseModel


class PolicyChange(BaseModel):
    """Represents a policy change scenario."""
    policy_id: str
    change_type: str
    severity: float  # 0.0 to 1.0


class InferenceRequest(BaseModel):
    """Request model for inference endpoint."""
    historical_data: List[float]
    policy_changes: Optional[List[PolicyChange]] = None
    time_horizon: int = 12  # months


class InferenceResponse(BaseModel):
    """Response model for inference endpoint."""
    prediction: List[float]
    confidence: float
    policy_impact: Optional[dict] = None


class TrainingRequest(BaseModel):
    """Request model for training endpoint."""
    data_source: str
    epochs: int = 10
    learning_rate: float = 0.001


class TrainingResponse(BaseModel):
    """Response model for training endpoint."""
    job_id: str
    status: str
    message: str
