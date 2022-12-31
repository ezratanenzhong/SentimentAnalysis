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

st.title("Customer Review Analyzer")

st.markdown("The analyzer can identify the sentiment label of reviews ")

with st.sidebar:
    st.subheader('About')
    st.markdown('This is a web application to identify the sentiment label of reviews provided by customers.')
    st.markdown("Code: Github")

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Single Review Prediction', 'Batch Review Prediction'])


def predict_sentiment(input_text):
    loaded_model = pickle.load(open('finalized_model.pkl', 'rb'))
    loaded_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    cleaned_text = clean_text(input_text)
    tv = loaded_vectorizer
    review_tv = tv.transform([cleaned_text])
    model_predict = loaded_model.predict(review_tv)
    print("Predicted sentiment label: ", model_predict)


if app_mode == 'Single Review Prediction':
    st.title("Single Review Prediction")
    st.subheader('Predict the sentiment of a review provided by customers, whether is positive, negative or neutral.')
    text = st.text_input('Enter the review for which you want to know the sentiment:')
    submitted = st.button('Submit')
    if submitted:
        predict_sentiment(text)
        # st.success("You did it !")


def predict_sentiment_batch(review):
    loaded_model = pickle.load(open('finalized_model.pkl', 'rb'))
    loaded_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    label_list = []
    for input_text in review:
        cleaned_text = clean_text(input_text)
        tv = loaded_vectorizer
        text_tv = tv.transform([cleaned_text])
        prediction = loaded_model.predict(text_tv)
        label_list.append(prediction)
    output = pd.DataFrame(data=label_list, columns=['label'])
    return output


if app_mode == 'Batch Review Prediction':
    st.title("Batch Review Prediction")
    st.subheader('Predict the sentiment of multiple reviews, whether is positive, negative or neutral.')
    upload_file = st.file_uploader("Upload a CSV file which contain one column only - the reviews column", type=["csv"])
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        df = df.drop(columns=['Unnamed: 0'])
        df = df.rename(columns={df.columns[0]: 'text'}, inplace=True)
        predict_output = pd.DataFrame(predict_sentiment_batch(list(df['text'])))
        result_df = df.assign(label=predict_output)
        st.subheader('Result')
        st.markdown('Output (first five rows)')
        st.write(result_df.head())

        # Plot distribution of sentiment
        # funnel chart
    else:
        st.warning('Please upload the file in the required format')

