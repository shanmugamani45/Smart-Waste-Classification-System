from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    status: str
    model_name: str
    is_model_trained: bool
    version: str
    timestamp: str


class PredictionDetails(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    predicted_class: str = Field(..., alias="class")
    confidence: float
    is_model_trained: bool
    all_probabilities: Dict[str, float]
    latency_ms: float


class RecommendationDetails(BaseModel):
    recommended_action: str
    circular_strategy: str
    environmental_priority: str
    confidence_level: str
    confidence_warning: Optional[str] = None
    guidelines: str


class ExplainabilityDetails(BaseModel):
    gradcam_available: bool
    heatmap_overlay: Optional[str] = None  # Base64 data URI


class PredictResponse(BaseModel):
    id: int
    prediction: PredictionDetails
    recommendation: RecommendationDetails
    explainability: ExplainabilityDetails


class HistoryItem(BaseModel):
    id: int
    timestamp: str
    predicted_class: str
    confidence: float
    recommended_action: str
    circular_strategy: str
    environmental_priority: str
    image_path: Optional[str] = None
    user_feedback: Optional[str] = None
    corrected_class: Optional[str] = None


class StatisticsResponse(BaseModel):
    total_predictions: int
    average_confidence: float
    category_distribution: Dict[str, int]
    category_percentages: Dict[str, float]
    action_distribution: Dict[str, int]
    priority_distribution: Dict[str, int]


class FeedbackRequest(BaseModel):
    prediction_id: int
    user_feedback: str  # "correct", "incorrect", "flagged"
    corrected_class: Optional[str] = None


class FeedbackResponse(BaseModel):
    success: bool
    message: str


class ModelInfoResponse(BaseModel):
    active_model: str
    classes: List[str]
    num_classes: int
    confidence_thresholds: Dict[str, float]
    model_comparison: List[Dict[str, Any]]
