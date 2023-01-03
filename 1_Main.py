import streamlit as st

st.title("Customer Review Analyzer")
st.markdown("""
    The analyzer can identify the sentiment label of customer reviews, whether it is positive, negative, or neutral.
    **👈 Select a page from the sidebar** to see what the analyzer can do
    - Choose 'Single Prediction' if want to predict the sentiment of a review
    - Choose 'Batch Prediction' if want to predict the sentiment of reviews in batch   
""")
st.image("Image 1.jpg")

with st.sidebar:
    st.sidebar.success('Select a page above')
    st.subheader('About')
    st.markdown('This is a web application to identify the sentiment label of reviews provided by customers.')
    st.markdown("Code: https://github.com/ezratanenzhong/SentimentAnalysis")

