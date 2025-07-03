# -*- coding: utf-8 -*-
import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import pandas as pd

# --- 1. API and Model Configuration ---

class TextInput(BaseModel):
    text: str = Field(..., min_length=10, description="Text to classify (min 10 characters)")

class PredictionOutput(BaseModel):
    predicted_class: str
    probabilities: Dict[str, float]

app = FastAPI(
    title="Scikit-learn DBPedia Classifier API",
    description="An API to classify text into DBPedia L1 categories using a TF-IDF + Logistic Regression model.",
    version="2.0.0" # Version bump to reflect the new model
)

# --- 2. Load Model and Configuration ---

# Define model paths and configuration
# Get the absolute path of the directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the model file
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "tfidf_logreg_pipeline_full.joblib")
# os.path.join handles the path separators correctly for any OS
# ".." moves one directory up from /app/api to /app
LABEL_MAPPING = {
    0: 'Agent', 1: 'Device', 2: 'Event', 3: 'Place', 4: 'Species',
    5: 'SportsSeason', 6: 'TopicalConcept', 7: 'UnitOfWork', 8: 'Work'
}

# Initialize model variable to None for robust error handling
pipeline = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure it's in the 'models' directory.")
    
    # Load the Scikit-learn pipeline from the .joblib file
    pipeline = joblib.load(MODEL_PATH)
    print("INFO: Scikit-learn pipeline loaded successfully.")

except Exception as e:
    print(f"ERROR: Could not load Scikit-learn pipeline. {e}")

# --- 3. Define API Endpoints ---

@app.get("/", summary="Health Check")
def read_root():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "ok", "message": "API is running"}

@app.post("/predict", response_model=PredictionOutput, summary="Classify Text")
def predict(payload: TextInput):
    """
    Takes a text input and returns the predicted DBPedia L1 category.
    The input text is passed as a list to the pipeline as it expects an iterable.
    """
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Model pipeline is not available. Check server logs."
        )

    text_to_classify = [payload.text] # The pipeline expects a list or iterable of texts
    
    # Make prediction
    try:
        # Get the predicted class index (returns an array)
        predicted_index = pipeline.predict(text_to_classify)[0]
        
        # Get the class name from the mapping
        predicted_class = LABEL_MAPPING.get(int(predicted_index), "Unknown")
        
        # Get the probability distribution for all classes (returns an array of arrays)
        probabilities = pipeline.predict_proba(text_to_classify)[0]
        
        # Create a dictionary of class names and their probabilities
        probs_dict = {LABEL_MAPPING[i]: prob for i, prob in enumerate(probabilities)}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {e}"
        )

    return PredictionOutput(predicted_class=predicted_class, probabilities=probs_dict)