import streamlit as st
import pandas as pd
import string
import re
import numpy as np
import pickle
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def predict_sentiment(input_text):
    loaded_model = pickle.load(open('finalized_model.pkl', 'rb'))
    loaded_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    cleaned_text = clean_text(input_text)
    tv = loaded_vectorizer
    review_tv = tv.transform([cleaned_text])
    model_predict = loaded_model.predict(review_tv)
    print("Predicted sentiment label: ", model_predict)

temp = True

if temp:
    st.title("Single Review Prediction")
    st.subheader('Predict the sentiment of a review provided by customers, whether is positive, negative or neutral.')
    text = st.text_input('Enter the review for which you want to know the sentiment:')
    submitted = st.button('Submit')
    if submitted:
        predict_sentiment(text)
        # st.success("You did it !")
