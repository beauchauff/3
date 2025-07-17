"""
COMP 4705
Homework Assignment: Building a Sentiment Analysis Web App with Streamlit
Objective: The goal of this assignment is to build a complete, end-to-end machine learning application. You will train a sentiment analysis model on movie review data, save it, and then build an interactive web app with Streamlit that allows a user to input any text and see the predicted sentiment. I have included sufficient hints and comments to assist you with completing this assignment easily. Feel free to email/Discord if you face any confusion.
Due Date: Thursday, 26th June. 11.59 pm MT

Gayla Hess
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

"""Step 1: Get the Data: We will use the Large Movie Review Dataset (IMDB). For simplicity, you can use a pre-processed 
version available on Kaggle. Dataset: IMDB Dataset of 50K Movie Reviews. 
Download the IMDB Dataset.csv file from the link above and place it in your project folder.

Step 2: Create a Training Script: Create a Python script named train_model.py. 
This script will be responsible for loading the data, training the model, and saving it.

Step 3: Load and Preprocess the Data: 
    Use pandas to load the IMDB Dataset.csv file.
    The dataset has two columns: review and sentiment. The sentiment is already conveniently labeled as positive or negative.
"""
df = pd.read_csv('IMDB Dataset.csv')
#print(df.head())

#Split your data into features (the review text) and labels (the sentiment). Let's call them X and y.

x=df["review"]
y=df["sentiment"]
#print(y.head())

#Step 4: Train the model with TfidfVectorizer and MultinomialNB
#Create a pipeline that first transforms the text data using TfidfVectorizer and then feeds it to the MultinomialNB classifier.
pipeline = Pipeline([
    ('text',TfidfVectorizer()),
     ('xText', MultinomialNB())
     ])

#train the pipeline on entire dataset
pipeline.fit(x,y)

#Save the model to a file
joblib.dump(pipeline, 'sentiment_model.pkl')
joblib.dump(y,'target_names.pkl')