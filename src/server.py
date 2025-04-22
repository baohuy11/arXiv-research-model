from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import numpy as np
from typing import List, Dict, Any

class PredictionInput(BaseModel):
    data: List[Dict[str, Any]]

app = FastAPI(title="arXiv Research Model API")

# Load model with error handling
try:
    model_path = os.path.join('models', 'gb_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

@app.get("/")
def read_root():
    return {
        "message": "arXiv Research Model API",
        "status": "operational"
    }

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convert input to numpy array
        input_array = np.array([list(d.values()) for d in input_data.data])
        
        # Make prediction
        prediction = model.predict(input_array)
        
        return {
            "status": "success",
            "prediction": prediction.tolist()
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )

    
