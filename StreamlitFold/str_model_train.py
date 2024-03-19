import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import optuna
import sklearn
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from optuna_grid_streamlit import optuna_grid_streamlit
from sklearn.metrics import (roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
                             f1_score, recall_score, precision_score, accuracy_score,
                             average_precision_score, fbeta_score, make_scorer,get_scorer)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


available_metrics = {
        'AUC': make_scorer(roc_auc_score),
        'MCC': make_scorer(matthews_corrcoef),
        'Balanced_accuracy': make_scorer(balanced_accuracy_score),
        'F1': make_scorer(f1_score),
        'F2': make_scorer(fbeta_score, beta=2),
        'Recall': make_scorer(recall_score),
        'Precision': make_scorer(precision_score),
        'Accuracy': make_scorer(accuracy_score),
        'Average_precision': make_scorer(average_precision_score)}
    
available_estimators = {'Logistic Regression': LogisticRegression(),
                        'Decision Tree': DecisionTreeClassifier(), 
                        'Random Forest': RandomForestClassifier(),
                        'K-Nearest Neighbors': KNeighborsClassifier(),
                        'Support Vector Machine': SVC(),
                        'Gaussian Naive Bayes': GaussianNB(),
                        'Linear Discriminant Analysis': LinearDiscriminantAnalysis()}
# @st.cache_data
def opt_final_sel(estimator,data,labels,scorer,cv, direction, n_trials,metrics2calculate):  
    selected_metric = available_metrics[scorer]
    selected_estimator = {est_name: available_estimators[est_name] for est_name in estimator}
    selected_metrics4df = {metric_name: available_metrics[metric_name] for metric_name in metrics2calculate}
    results = pd.DataFrame(list(selected_metrics4df.keys()))
    
    grid = optuna_grid_streamlit['ManualSearch'][estimator[0]]
    def objective(trial):
        clf = grid(trial)
        final_score = cross_val_score(clf, data, labels, scoring=selected_metric, cv=cv).mean()
        return final_score

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_score = study.best_value
    best_estimator = grid(study.best_trial)    
    best_estimator.fit(data, labels)
    scores_dict = {}
    for metric_name, scorer_function in selected_metrics4df.items():
        score = cross_val_score(best_estimator, data, labels, scoring=scorer_function, cv=cv).mean()
        scores_dict[metric_name] = [score]
    
    results = pd.DataFrame.from_dict(scores_dict)
    return results


def app():
    st.title("Model Training and Tunning")

    scaled_features = StandardScaler()
    df = pd.read_csv("Hepatitis_C.csv")
    columns_to_scale = df.columns[(df.columns != 'label') & (df.columns != 'Sex')]
    feat_sc = pd.DataFrame(scaled_features.fit_transform(df[columns_to_scale]), columns=columns_to_scale, index=df.index)
    feat_sc['Sex'] = df['Sex']
    labels = df['label']

    with st.expander("Choose estimator"):
        st.markdown("### Choose an estimator to train:")
        est_names = list(available_estimators.keys())
        chosen_estimator = st.selectbox("Select one estimator:", options=est_names, index=est_names.index('Logistic Regression'))

    with st.expander("Choose metric for evaluation"):
        st.markdown("### Choose a metric for optuna's evaluation:")
        metric_names = list(available_metrics.keys())
        chosen_metric = st.selectbox("Select one metric:", options=metric_names, index=metric_names.index('MCC'))

    with st.expander("Available metrics to calculate"):
        st.markdown("### Choose the metrics to calculate:")
        metrics2calculate = []
        preselected_metrics = ['MCC','AUC','Recall']
        for metric_name in available_metrics.keys():
            if st.checkbox(metric_name, key=f"cb_{metric_name}", value=metric_name in preselected_metrics):
                metrics2calculate.append(metric_name)

    with st.expander("Available Parameters for Cross-Validation"):
        st.markdown("""### Available Parameters for Cross-Validation:""")
        splits = st.slider('Provide the the outer splits', 0, 5, 3)
        trials = st.slider('Provide the the number of trials', 0, 150, 100)
        direction = st.selectbox('Provide the direction of the study', ['maximize', 'minimize'])
    
    with st.expander("Run Cross-Validation"):
        if st.button("Run Cross-Validation"):       
            scores_metrics = opt_final_sel([chosen_estimator], feat_sc, labels, chosen_metric, splits, direction, trials, metrics2calculate)
            st.dataframe(scores_metrics)
