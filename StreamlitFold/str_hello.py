import streamlit as st
import pandas as pd
import seaborn as sns

def app():
    st.title("Introduction 👋")

    st.markdown("""Hepatitis C is a viral infection that causes inflammation of the liver and can lead to serious health
        problems. Machine Learning (ML) methods offer a promising approach to analyzing various features
        and laboratory test values to effectively diagnose Hepatitis C patients and generate laboratory diagnostic
        protocols. In this assignment, I built a complete ML pipeline to classify future patients according to
        the label column using a nested Cross Validation (nCV) pipeline. I compared the performance of five
        classification algorithms and selected the one that achieved the highest average test MCC performance in
        5 trials of nCV. After finding the winner algorithm, I used the whole dataset and simple cross-validation
        with 3 folds to determine the final model to deploy in the field.""")

    st.markdown("""
        ## Welcome to the Hepatitis C Data Exploration & Classification App
        In this app, we will explore a dataset containing information about Hepatitis C patients.
        - **Page 1:** Introduction.
        - **Page 2:**  Overview and Interactive data visualizations to explore the dataset.
    """)
    # st.title('Interactive Data Exploration')

    df = pd.read_csv("Hepatitis_C.csv")  # Load your data

    st.dataframe(df)

    
