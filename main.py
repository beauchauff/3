import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List
import random

app = FastAPI(
    title="Movie Prediction API",
)

try:
    model = joblib.load('sentiment_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file 'sentiment_model.pkl' not found.")
    print("Please run the 'train_model.py' script first to generate the model file.")
    model = None


class PredictionInput(BaseModel):
    features: List[str]

@app.get("/health")
def health_check():
    """
    Health Check Endpoint
    Simple endpoint to confirm that the API is running.
    """
    return {"status": "ok"}


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

    review_text = input_data.features
    prediction = model.predict(review_text)

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

    review_text = input_data.features
    prediction = model.predict(review_text)

    probabilities = model.predict_proba(review_text)

    # The probabilities for class 0 and class 1
    prob_class_0 = probabilities[0][0]
    prob_class_1 = probabilities[0][1]

    return {
        "prediction": int(prediction[0]),
        "probability_class_0": f"{prob_class_0:.4f}",
        "probability_class_1": f"{prob_class_1:.4f}"
    }

@app.get("/example")
def training_example():
    """
    Training Example Endpoint
    Returns a random review from the original IMDB training dataset.
    This can be used to test the prediction endpoints
    """
    df = pd.read_csv('IMDB Dataset.csv')
    entry = random.randint(2,len(df))
    return {"review": df.iat[entry,0]}

