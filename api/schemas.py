from pydantic import BaseModel
from typing import Dict, List

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    predictions: Dict[str, float]
    is_cyberbullying: bool
    detected_categories: List[str]
