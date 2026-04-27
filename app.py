from fastapi import FastAPI, HTTPException, Path, Query
from typing import Optional, Annotated
from src.utils.utils import load_object
from src.logger.logger import logger
import numpy as np
import pandas as pd 
import time

from src.pipeline.prediction_pipeline import CustomerChurn, PredictionResponse
app = FastAPI(title="Churn Prediction API")
MODEL_VERSION = "1.0.0"

PREPROCESSOR_PATH = r"artifacts\preprocessor.pkl"
MODEL_PATH = r"artifacts\model.pkl"

try:
    preprocessor = load_object(PREPROCESSOR_PATH)
    model = load_object(MODEL_PATH)

    if preprocessor is None or model is None:
        raise Exception("Model or Preprocessor not loaded")

    logger.info("Model and Preprocessor loaded successfully")

except Exception as e:
    logger.error(f"Startup error: {e}")
    preprocessor = None
    model = None

@app.get('/')
def hello():
  return {"message": "Welcome to Churn Prediction API Model."}

@app.get('/health')
def health_check():
  return {
    "status":'OK',
    "version":MODEL_VERSION,
    "model_loaded":model is not None
      }

@app.post('/predict', response_model=PredictionResponse)
def predict_data_point(customer: CustomerChurn):
  start_time = time.time()
  try:
    
    if model is None or preprocessor is None:
      raise HTTPException(status_code=500, detail="Model not found")
    
    
    input_data = pd.DataFrame([customer.model_dump()])

    logger.info(f"Input Data: {input_data.to_dict()}")
    print(input_data.to_dict())
        
    data_scaled = preprocessor.transform(input_data)

    prediction = model.predict(data_scaled)[0]

    
    try:
        probability = model.predict_proba(data_scaled)[0][1]
    except Exception:
        probability = None

    
    churn_label = "Yes" if prediction == 1 else "No"
    
    latency = time.time() - start_time

    response = {
            "prediction": churn_label,
            "probability": float(probability) if probability is not None else None,
            "latency": round(latency * 1000, 2)
        }

    logger.info(f"Prediction: {response}")

    return response
      
  except Exception as e:
    logger.error(f"Error during prediction: {e}")
    raise HTTPException(status_code=500, detail="Error loading model")
  
  


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)