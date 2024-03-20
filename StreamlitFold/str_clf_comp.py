import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
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
from optuna.integration import OptunaSearchCV
import sklearn
from sklearn.model_selection import StratifiedKFold
from optuna_grid_streamlit import optuna_grid_streamlit
import os

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
# @st.cache_resource
# def _optuna_integ(est_name, innercv, trials,chosen_metric, X_train, y_train):
#     selected_metric = available_metrics[chosen_metric]
#     param_distributions = optuna_grid_streamlit["NestedCV"][est_name]
#     estimator = available_estimators[est_name]
#     clf = OptunaSearchCV(
#         estimator=estimator,
#         param_distributions=param_distributions,
#         cv=innercv,
#         n_jobs=1,
#         verbose=0,
#         n_trials=trials,
#         scoring=selected_metric)
#     clf.fit(X_train, y_train)
#     return clf

# @st.cache_resource
def nestedcv(rounds, chosen_metric, inner_splits, outer_splits, est_list, trials, data, labels, metrics2calculate):
    selected_estimators = {est_name: available_estimators[est_name] for est_name in est_list}
    selected_metric = available_metrics[chosen_metric]
    selected_metrics4df = {metric_name: available_metrics[metric_name] for metric_name in metrics2calculate}

    results = pd.DataFrame(columns=['Estimator'] + list(selected_metrics4df.keys()))
    data2append = []

    # Progress bar
    progress_bar = st.progress(0)
    total_steps = rounds * outer_splits * len(est_list)
    current_step = 0

    for round in range(rounds):
        innercv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=round)
        outercv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=round)

        for train_index, test_index in outercv.split(data, labels):
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]  

            for est_name, estimator in selected_estimators.items():
                param_distributions = optuna_grid_streamlit["NestedCV"][est_name]
                clf = OptunaSearchCV(
                    estimator=estimator,
                    param_distributions=param_distributions,
                    cv=innercv,
                    n_jobs=1,
                    verbose=0,
                    n_trials=trials,
                    scoring=selected_metric)
                clf.fit(X_train, y_train)
                # clf = _optuna_integ(est_name,innercv,trials,chosen_metric,X_train,y_train)
                scores_for_current_estimator = {'Estimator': est_name}

                for metric_name, scorer_function in selected_metrics4df.items():
                    score = scorer_function(clf.best_estimator_, X_test, y_test)
                    scores_for_current_estimator[metric_name] = score

                data2append.append(scores_for_current_estimator)
                
                # Update the progress bar after each step
                current_step += 1
                progress_bar.progress(current_step / total_steps)

    results = pd.DataFrame(data2append)
    return results

def app():
    st.title('Estimators Comparison 🧐')

    st.write('Exploring and Compare the different estimators.')

    st.markdown("""## Set the Comparison:""")

    with st.expander("Available estimators"):
        st.markdown("""### Available estimators:""")
        
        est_list = []

        preselected_estimators = ['Random Forest', 'Logistic Regression']

        for estimator_name in available_estimators.keys():
            if st.checkbox(estimator_name, key=f"cb_{estimator_name}", value=estimator_name in preselected_estimators):
                est_list.append(estimator_name)

        # if est_list:  # Check if the list is not empty
        #     markdown_list = "\n".join(f"- {classifier}" for classifier in est_list)
        #     st.markdown("Selected Classifiers:\n" + markdown_list)
        # else:
        #     st.write("No classifiers selected.")
        
### NICE FOR THESIS
        # selected_clf_multiselect = st.multiselect("Review and modify your selections:", 
        #                                       options=[
        #                                           "Logistic Regression", "Decision Tree",
        #                                           "Random Forest", "Gradient Boosting", 
        #                                           "XGBoost", "LightGBM", "CatBoost", 
        #                                           "K-Nearest Neighbors", "Support Vector Machine", 
        #                                           "Gaussian Naive Bayes", "Linear Discriminant Analysis", 
        #                                           "Gaussian Process Classifier"
        #                                       ],
        #                                       default=est_list)

        # st.write("Final Selected Classifiers:", selected_clf_multiselect) 
    with st.expander("Available metrics for evaluation"):
        st.markdown("### Choose a metric for optuna's evaluation:")
        metric_names = list(available_metrics.keys())

        chosen_metric = st.selectbox("Select one metric:", options=metric_names, index=metric_names.index('MCC'))

    with st.expander("Available metrics to calculate"):
        st.markdown("### Choose the metrics to calculate:")
        metrics2calculate = []
        preselected_metrics = ['MCC','AUC']
        for metric_name in available_metrics.keys():
        # Preselect the checkbox if the metric is in the preselected_metrics list
            if st.checkbox(metric_name, key=f"cb_{metric_name}", value=metric_name in preselected_metrics):
                metrics2calculate.append(metric_name)

        # for metric_name in available_metrics.keys():
        # # Preselect the checkbox if the metric is in the preselected_metrics list
        #     if st.checkbox(metric_name, key=f"cb_{metric_name}", value=metric_name in preselected_metrics):
        #         selected_metrics.append(metric_name)

        # st.write("Selected Metrics for Calculation:", selected_metrics)
        # if selected_metrics:  # Check if any metrics were selected
        # # Filter the metrics DataFrame to only include selected metrics
        #     results = pd.DataFrame(columns=selected_metrics)
        #     results_test = pd.DataFrame(columns=selected_metrics)
        #     st.write("Selected Metrics for Calculation:", selected_metrics)
        # else:
        #     st.write("No metrics selected.")

    with st.expander("Available Parameters for Nested Cross-Validation"):
        st.markdown("""### Available Parameters for Nested Cross-Validation:""")

        inner_splits = st.slider('Provide the the inner splits', 0, 5, 3)
        outer_splits = st.slider('Provide the the outer splits', 0, 5, 3)
        trials = st.slider('Provide the the number of trials', 0, 150, 100)
        rounds = st.slider('Provide the the number of rounds', 0, 10, 3)

    scaled_features = StandardScaler()
    df = pd.read_csv("Hepatitis_C.csv")
    columns_to_scale = df.columns[(df.columns != 'label') & (df.columns != 'Sex')]
    feat_sc = pd.DataFrame(scaled_features.fit_transform(df[columns_to_scale]), columns=columns_to_scale, index=df.index)
    feat_sc['Sex'] = df['Sex']
    labels = df['label']
    
    image_dir = "saved_plots"
    os.makedirs(image_dir, exist_ok=True)  # Create the directory if it doesn't exist

    with st.expander("Run Nested Cross-Validation"):
        if st.button("Run Nested Cross-Validation"):
            # Assuming nestedcv is correctly defined and returns a DataFrame
            results = nestedcv(rounds, chosen_metric, inner_splits, outer_splits, est_list, trials, feat_sc, labels, metrics2calculate)
            
            # Clear previous images to avoid displaying old plots
            for filename in os.listdir(image_dir):
                file_path = os.path.join(image_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    st.error(f'Failed to delete {file_path}. Reason: {e}')
            
            # Generate and save new plots
            for metric in results.columns[1:]:  # Skip the first column ('Estimator')
                # plt.figure(figsize=(20, 15))
                sns.boxplot(data=results, x='Estimator', y=metric)
                plt.title(f'Boxplot of {metric} by Estimator')
                plt.xticks(rotation=90)
                plt.ylabel(metric)
                plt.xlabel('Estimator')
                
                # Save the plot as an image
                image_path = os.path.join(image_dir, f'{metric}.png')
                plt.savefig(image_path)
                plt.close()  # Close the figure to free memory
                
            st.dataframe(results)
            
        st.write(f"Parameters for Nested Cross-Validation: rounds = {rounds}, inner_splits = {inner_splits}, outer_splits = {outer_splits}, trials = {trials}\nestimators list = {est_list}, evaluated metric = {chosen_metric}")
        st.write("Plots saved in:", image_dir)
        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            st.image(file_path, caption=filename)
    
    # with st.expander("Run Nested Cross-Validation"):
    #     if st.button("Run Nested Cross-Validation"):
    #         # Use selected_estimators here
    #         results = nestedcv(rounds, chosen_metric, inner_splits, outer_splits, est_list, trials, feat_sc, labels, metrics2calculate)
    #         for metric in results.columns[1:]:  # Skip the first column ('Estimator') and iterate through metrics
    #             plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    #             sns.boxplot(data=results, x='Estimator', y=metric)
    #             plt.title(f'Boxplot of {metric} by Estimator')
    #             plt.xticks(rotation=45)  # Rotate the x labels to avoid overlap
    #             plt.ylabel(metric)
    #             plt.xlabel('Estimator')
    #             st.pyplot(plt)  # Display the plot in Streamlit
    #             plt.clf() 
    #         st.dataframe(results)

