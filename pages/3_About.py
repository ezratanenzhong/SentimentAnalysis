import streamlit as st

st.set_page_config(layout='wide')
st.header('About')
st.markdown("""**Reviewalyze** is a web application to analyze reviews provided by customers. The name is inspired by words like review, analyze and realize.
            Simply input a review and the model will predict whether its sentiment is positive, negative, or neutral.
            User can also analyze multiple reviews by uploading them in a CSV file.
            The main purpose of analyzing the reviews is to improve customer satisfaction for businesses or to understand the sentiment of a brand or product.
            The model is trained with data scraped from Walmart's, Target's and JoMalone's Facebook reviews.""")

st.header("User Manual")

st.subheader('Single Review Analysis')
st.write('This page can analyze the sentiment of a single review. Below are the detailed steps:')
st.write(' 1. Click Single Review Analysis page at the sidebar')
st.write(' 2. Enter the review in the text area')
st.write(' 3. Click the Analyze button')
st.write(' 4. The output will show the predicted sentiment label of the review and the class probabilities for the input data points (i.e. the probability that a particular data point falls into the underlying classes).')

st.subheader('Batch Review Analysis')
st.write('This page can analyze the sentiment of multiple reviews. Below are the detailed steps:')
st.write(' 1. Click Single Review Analysis at the sidebar')
st.markdown("""2. Upload a file containing the reviews with the required format: 
- CSV file has a column which contains the reviews
- Column name must be **text**
- Reviews can only be in English""")
st.write("3. Click **Analyze** button. ")
st.write("4. A table with the reviews and their predicted sentiment labels will be displayed. Click **Download data as CSV** button to download the result data.")
st.markdown("""5. At the Visualization section, choose the type of plot to view:
- Click **Bar Chart** button to view the distribution of the sentiment labels
- Click **Wordcloud** button to view the words contain in positive and negative sentiment sentences
- CLick **N-grams** button to view the N-grams words in positive and negative sentiment sentences""")


st.markdown("Any question can email me via ezratan2001@gmail.com")

