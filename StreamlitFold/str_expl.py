import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def app():
    st.title('Data Exploration 🔍')

    st.write('Exploring Hepatitis C data.')

    st.markdown("""## Overview of data:""")

    st.markdown("""Understanding the DataThe given dataset, contains an extensive list of values for 204 different patients (68 of them found to be positive in Hepatitis C), 
                including their age, sex, and various biochemical markers such as ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, and PROT. Additionally, the dataset also includes
                a label that indicates if a patient has hepatitis C or not. To ensure the accuracy of the data, I conducted a search for NaN and null values, but none were found.
                However, it is noted that some individual extreme values were found, which were labeled as positive for Hepatitis C, so I made the decision to keep these values in the
                data set, rather than remove or change them. In order to gain a deeper understanding of the data, the second step was to find the correlations between the various variables.
                Also to illustrate the differences between the distributions of the healthy and deceased patients in figure 1 and figure 6. In addition to this, I was also able to identify
                the seven parameters that had the highest absolute correlation with Hepatis C, which is a critical finding for anyone working in the field. This information is illustrated
                in figure 3.""")

    df = pd.read_csv("Hepatitis_C.csv")
    correlations = df.corr()

    if st.checkbox('Show Boxplots'):
        st.header("Boxplots")
        column4box = st.selectbox('Select column to visualize', df.columns[2:-2], key='boxplot_column')
        plt.figure()
        sns.boxplot(x='Age', y=column4box, data=df)
        st.pyplot(plt)
        # fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))
        # for i, col in enumerate(df.columns):
        #     if not (col == 'Age' or col == 'label' or col == 'Sex'):
        #         ax = axes.flatten()[i - 3]  # Adjust this indexing as needed
        #         df.boxplot(by='Age', column=[col], ax=ax, grid=False)
        #         ax.set_title(col)
        # plt.tight_layout()
        # st.pyplot(fig)

    if st.checkbox('Show Pairplot'):
        st.header("Pairplot")
        pairplot_fig = sns.pairplot(df, hue='label')
        st.pyplot(pairplot_fig)

    if st.checkbox('Show Histogram'):
        st.header("Histogram")
        column = st.selectbox('Select column to visualize', df.columns, key='hist_column')

        # Dynamic visualization based on selected column
        plt.figure()
        sns.histplot(data=df, x=column, kde=True)
        st.pyplot(plt)

    if st.checkbox('Show Correlations'):
        st.header("Correlations")
        corr_plt = plt.figure(figsize=(12, 10))
        sns.heatmap(correlations, cmap='coolwarm', annot=True, fmt='.2f')
        st.pyplot(corr_plt)
    
    st.markdown("""## Data normalization:""")

    st.markdown("""For Data normalization I used the StandardScaler (except the 'Sex' column) and separeted the dataset into the data features and tha target labels.""")
    
    scaled_features = StandardScaler()
    columns_to_scale = df.columns[(df.columns != 'label') & (df.columns != 'Sex')]
    feat_sc = pd.DataFrame(scaled_features.fit_transform(df[columns_to_scale]), columns=columns_to_scale, index=df.index)
    feat_sc['Sex'] = df['Sex']
    feat_sc['label'] = df['label']
    st.dataframe(feat_sc)