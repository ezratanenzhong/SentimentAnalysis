import streamlit as st

st.set_page_config(page_title='Reviewalyze', initial_sidebar_state="expanded", layout="wide")

st.title("Reviewalyze")
col1, col2 = st.columns(2)

with col1:
    st.markdown('Welcome to **Reviewalyze** - designed to **analyze reviews** and help businesses and individuals **realize** the sentiment of reviews on them.')
    st.markdown('The Sentiment Analyzer can be useful for businesses looking to improve customer satisfaction or for individuals looking to understand the sentiment of a brand or product.')
    st.markdown('**👈 Select a page from the sidebar** to try it out !')
    st.write("- Choose 'Single Review Analysis' if want to predict the sentiment of a review")
    st.write("- Choose 'Batch Review Analysis' if want to predict the sentiment of multiple reviews")

with col2:
    st.image("Image.png")

with st.sidebar:
    st.sidebar.success('Select a page above')
    st.header('About')
    st.markdown(' - This is a web application to identify the sentiment label of reviews provided by customers whether they are positive, negative, or neutral.')
    st.write(' - Simply input a review and the model will predict whether the sentiment of the review is positive, negative, or neutral.')
    st.write(' - The model used in the analyzer is trained with customer reviews on social media using Logistic Regression algorithm')
    st.markdown("Code: https://github.com/ezratanenzhong/SentimentAnalysis")

