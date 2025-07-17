import pandas as pd
import streamlit as st
import joblib

#Borrow function from lab to load sentiment_model.pkl using joblib.load().
@st.cache_data
def load_model():
    """Loads the pre-trained model and target names."""
    model = joblib.load('sentiment_model.pkl')
    target_names = joblib.load('target_names.pkl')
    return model, target_names

#load the model and names
model, target_names = load_model()

#app layout
st.title('Movie Review Sentiment Analyzer')
st.write("Predicts positive or negative reviews of movies")

#create user input interface with text prompt
userReview = st.text_area("Enter a movie review to analyze:")
#defaults to empty string
#https://docs.streamlit.io/develop/api-reference/widgets/st.text_area

#add a button labeled "Analyze"
button = st.button("Analyze", "primary")
# (bool) returns True if the button as clicked on the last run of the app
#https://docs.streamlit.io/develop/api-reference/widgets/st.button
#primary button chosen for emphasis

#Make predictions & display results
#user text must be passed as a list for pipeline being used
if button ==True:
    if len(userReview)>1:
        predSent=model.predict([userReview])
        predSent_prob = model.predict_proba([userReview])

        st.subheader("Prediction:")
        #st.write(f'The predicted review sentiment is: {predSent[0].capitalize()}.')

        #inelegant way to use the tip to show color coding
        ans = predSent[0].capitalize()
        if ans == 'Positive':
            colorAns =st.success(ans)
        else:
            colorAns = st.error(ans)

        st.subheader('Prediction Probabilities:')
        st.write(f'Probability =  {predSent_prob}')

    else:
        st.write("Input review to analyze")