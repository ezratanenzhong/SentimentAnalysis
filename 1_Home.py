import streamlit as st

st.set_page_config(page_title='Reviewalyze', initial_sidebar_state="expanded", menu_items=None)

st.title("Reviewalyze")
st.markdown("""
    Welcome to Reviewalyze ! The web application is designed to **analyze reviews** and help businesses and individuals **realize** the sentiment of their customer reviews.
    This can be a valuable tool for understanding customer sentiment and improving customer satisfaction.
    Whether you're a business owner looking to gauge the success of your products or services, or an individual looking to understand the sentiment of a particular brand or product, our Sentiment Analyzer has you covered. 
    /n **👈 Select a page from the sidebar** to try it out and get a better understanding of your customer reviews!
    - Choose 'Single Review Analysis' if want to predict the sentiment of a review
    - Choose 'Batch Review Analysis' if want to predict the sentiment of reviews in batch   
""")
st.image("Image 1.jpg")

with st.sidebar:
    st.sidebar.success('Select a page above')
    st.subheader('About')
    st.markdown(' - This is a web application to identify the sentiment label of reviews provided by customers whether they are positive, negative, or neutral.')
    st.write(' - Simply input a review and our model will predict whether the sentiment of the review is positive, negative, or neutral.')
    st.write(' - The model used in the analyzer is trained with customer review on social media using Logistic Regression algorithm')
    st.markdown("Code: https://github.com/ezratanenzhong/SentimentAnalysis")

