from __future__ import annotations

import os
from typing import List, Union

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# California Housing feature schema
class InputFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group (10k USD)")
    HouseAge: float = Field(..., description="Median house age in years")
    AveRooms: float = Field(..., description="Average number of rooms")
    AveBedrms: float = Field(..., description="Average number of bedrooms")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average house occupancy")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")

class PredictionResponse(BaseModel):
    predictions: List[float]

app = FastAPI(title="California Housing Price API", version="0.1.0")

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")

# Global artifact
_artifact = None
_feature_names: List[str] = []

@app.on_event("startup")
def load_model():
    global _artifact, _feature_names
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Did you run `python src/train.py`?"
        )
    _artifact = joblib.load(MODEL_PATH)
    _feature_names = _artifact["feature_names"]
    if "model" not in _artifact:
        raise RuntimeError("Invalid artifact: missing 'model' key.")
    print(f"Model loaded. Features: {_feature_names}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "message": "California Housing API. Visit /docs for Swagger.",
        "model_path": MODEL_PATH,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: Union[InputFeatures, List[InputFeatures]]):
    """
    Accepts a single feature dict or a list of dicts.
    """
    if isinstance(payload, list):
        df = pd.DataFrame([p.dict() for p in payload], columns=_feature_names)
    else:
        df = pd.DataFrame([payload.dict()], columns=_feature_names)

    model = _artifact["model"]
    preds = model.predict(df)
    return {"predictions": [float(p) for p in preds]}