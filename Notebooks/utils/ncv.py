import numpy as np
import pandas as pd
import optuna

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import make_scorer, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, f1_score, recall_score, precision_score, accuracy_score, average_precision_score
from sklearn.metrics import fbeta_score
from .optuna_grid import optuna_grid
import logging

class EstWithNest():
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.estimator = None
        self.name = None
        self.best_params = None
        self.best_score = None
        self.best_est = None
        self.available_clfs = {
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
            'LogisticRegression': LogisticRegression(),
            'XGBClassifier': XGBClassifier(),
            'GaussianNB': GaussianNB(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'SVC': SVC()
        }
        self.scorers = {
            'AUC': make_scorer(roc_auc_score),
            'MCC': make_scorer(matthews_corrcoef),
            'Balanced_accuracy': make_scorer(balanced_accuracy_score),
            'F1': make_scorer(f1_score),
            'Recall': make_scorer(recall_score),
            'Precision': make_scorer(precision_score),
            'Accuracy': make_scorer(accuracy_score),
            'Average_precision': make_scorer(average_precision_score),
            'F2': make_scorer(fbeta_score, beta=2),
        }
        self.metrics = ['MCC', 'Balanced_accuracy', 'F1', 'F2', 'Recall', 'Precision', 'Accuracy', 'AUC', 'Average_precision', 'NPV', 'Iteration', 'Model']

    def nestedcv(self,rounds,scoring,
                 inner_split=3,outer_split=5,
                 estimators=None,trials=100):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        if scoring not in sklearn.metrics.get_scorer_names():
                raise ValueError(
                    f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
        if type(estimators)==list:
             estimators2try = estimators
        elif estimators == None:
             estimators2try = list(self.available_clfs.values())
        else: raise ValueError(
             f'Invalid estimators type. Provide a list with the desired estimators or None.'
        )
        
        results = pd.DataFrame(columns=self.metrics)
        results_test = pd.DataFrame(columns=self.metrics)
        data2append = []
        for round in tqdm(range(rounds)):
            print(f'Round {round+1} out of {rounds}')
            innercv = StratifiedKFold(n_splits=inner_split, shuffle=True, random_state=round)
            outercv = StratifiedKFold(n_splits=outer_split, shuffle=True, random_state=round)
            scoring_eval = get_scorer(scoring)
            j=1
            for train_index, test_index in tqdm(outercv.split(self.data, self.labels), desc='Outer CV', total=outer_split):
                X_train, X_test = self.data.iloc[train_index], self.data.iloc[test_index]
                y_train, y_test = self.labels[train_index], self.labels[test_index]
                for est in estimators2try:
                    self.estimator = est
                    self.name = self.estimator.__class__.__name__
                    clf = optuna.integration.OptunaSearchCV(estimator=self.estimator, scoring=scoring_eval,
                                                    param_distributions=optuna_grid['NestedCV'][self.name],
                                                    cv=innercv, n_jobs=-1, verbose=0, n_trials=trials)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    scores_for_current_estimator = {'Estimator': self.name}
                    for metric in self.scorers.keys():
                        scorer_function = self.scorers[metric]
                        score = scorer_function(clf, X_test, y_test)
                        scores_for_current_estimator[metric] = score
                    data2append.append(scores_for_current_estimator)
                j+=1
        results = pd.DataFrame(data2append)
        return results

    def opt_final_sel(self,estimator,X=None,y=None,scoring='matthews_corrcoef',cv=5, direction='maximize', n_trials=100):
        optuna.logging.set_verbosity(0)
        
        grid = optuna_grid['ManualSearch']
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
            
        if X is None and y is None:
            X = self.data
            y = self.labels
                              
        def objective(trial):
            clf = grid[estimator](trial)
            final_score = cross_val_score(clf, X, y, scoring=scoring, cv=cv).mean()
            return final_score

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials,show_progress_bar=True)
        self.best_params = study.best_params
        self.best_score = study.best_value
        self.best_estimator = grid[estimator](study.best_trial)
        self.name = self.best_estimator.__class__.__name__
        
        self.best_estimator.fit(X, y) #fit in all X,y data

        return self.best_estimator
        
        
