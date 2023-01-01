import streamlit as st

st.title("Customer Review Analyzer")

st.markdown("The analyzer can identify the sentiment label of reviews ")

with st.sidebar:
    st.sidebar.success('Select a page above')
    st.subheader('About')
    st.markdown('This is a web application to identify the sentiment label of reviews provided by customers.')
    st.markdown("Code: Github")

