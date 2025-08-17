import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import random

app = FastAPI(
    title="Movie Prediction API",
)

try:
    model = joblib.load('sentiment_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file 'sentiment_model.pkl' not found.")
    print("Please run the 'train_model.evaluate.py' script first to generate the model file.")
    model = None


class PredictionInput(BaseModel):
    text: str

@app.get("/health")
def health_check():
    """
    Health Check Endpoint
    Simple endpoint to confirm that the API is running.
    """
    return {"status": "ok", "message": "API is running"}


@app.post("/predict")
def predict(input_data: PredictionInput):
    """
    Prediction Endpoint
    Takes a text input and returns the predicted sentiment of the movie review
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Cannot make predictions."
        )

    review_text = input_data.text
    prediction = model.predict([review_text])[0]

    return {"sentiment": prediction}

@app.post("/predict_proba")
def predict_with_probability(input_data: PredictionInput):
    """
    Prediction with Probability Endpoint
    Takes a text input and returns the predicted sentiment along with its confidence score.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Cannot make predictions."
        )

    review_text = input_data.text
    prediction = model.predict([review_text])[0]
    prediction_proba = model.predict_proba([review_text])[0]

    return {
        "prediction": prediction,
        "negative_probability": prediction_proba[0],
        "positive_probability": prediction_proba[1]
    }

@app.get("/example")
async def training_example():
    """
    Training Example Endpoint
    Returns a random review from the original IMDB training dataset.
    This can be used to test the prediction endpoints
    """
    df = pd.read_csv('IMDB Dataset.csv')
    entry = random.randint(2,len(df))
    return {"review": df.iat[entry,0]}