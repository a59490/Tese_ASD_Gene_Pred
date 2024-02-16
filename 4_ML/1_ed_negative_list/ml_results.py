import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer,matthews_corrcoef,recall_score



def cross_val(model, X, Y):
    sensitivity_scorer = make_scorer(recall_score)
    specificity_scorer = make_scorer(recall_score, pos_label=0)
    MCC=make_scorer(matthews_corrcoef)


    scoring = {'AUC': 'roc_auc', 'Accuracy': "accuracy", "f1": "f1",
                        "Recall": "recall", "Precision": "precision","MCC":MCC, "Average Precision": "average_precision",
                        "Sensitivity": sensitivity_scorer, "Specificity": specificity_scorer}

    scores=cross_validate(model, X, Y, scoring=scoring,n_jobs=5,cv=5)

    mean_scores = {metric: values.mean() for metric, values in scores.items()}


    return mean_scores



def model_hyperparameter_tuning(model, param_grid):
    MCC=make_scorer(matthews_corrcoef)
    results_list = []




    for dataset,name in dataset_list:

        x = dataset["3"].copy()
        
        x = x.str.split(expand=True)

        x= x.astype(float)


        y = dataset["4"].copy().astype('category')

        grid=GridSearchCV(model, param_grid, cv=5, scoring=MCC, verbose=1)

        search=grid.fit(x,y)

        best_params=search.best_params_
        best_model=search.best_estimator_

        results=cross_val(best_model,x,y)
        result_entry = {'dataset_name': name, **results, "best_params": best_params}

        results_list.append(result_entry)

    results_df = pd.DataFrame(results_list)
    results_df.set_index('dataset_name', inplace=True)

    with open('model_results.csv', 'a') as f:
        results_df.to_csv(f)


        


cat_1=pd.read_csv('gene_lists/cat_1.csv.gz',compression = 'gzip')
cat_1_sd=pd.read_csv('gene_lists/cat_1_sd.csv.gz',compression = 'gzip')
cat_2=pd.read_csv('gene_lists/cat_1_2.csv.gz',compression = 'gzip')
cat_2_sd=pd.read_csv('gene_lists/cat_1_2_sd.csv.gz',compression = 'gzip')
cat_3=pd.read_csv('gene_lists/cat_1_2_3.csv.gz',compression = 'gzip')
complete=pd.read_csv('gene_lists/complete.csv.gz',compression = 'gzip')

dataset_list=[(cat_1,"Cat_1"),(cat_1_sd,"Cat_1_sd"),(cat_2,"Cat_1_2")
              ,(cat_2_sd,"Cat_1_2_sd"),(cat_3,"Cat_1_2_3"),(complete,"Complete")]


# Logistic Regression
model = LogisticRegression(max_iter=1000,class_weight="balanced")
param_grid = {'C': [0.001, 0.1, 1, 10, 100, 1000], 'penalty': ['l2']}
model_hyperparameter_tuning(model, param_grid)

# Random Forest
model = RandomForestClassifier(class_weight="balanced")
param_grid = {'n_estimators': [1000], 'max_features': [ 'log2'],
              'max_depth': [3, 5, 10], 'min_samples_split': [ 10]}
model_hyperparameter_tuning(model, param_grid)

# SVM
model = SVC(class_weight="balanced")
param_grid = {'C': [1000], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'poly', 'sigmoid']}
model_hyperparameter_tuning(model, param_grid)

# KNN
model = KNeighborsClassifier()
param_grid = {'n_neighbors': [2], 'weights': ['uniform', 'distance']}
model_hyperparameter_tuning(model, param_grid)

#LightGBM
model = LGBMClassifier(class_weight="balanced",n_jobs=1)
param_grid = {'n_estimators': [1000], 'learning_rate': [0.01, 0.05]}
model_hyperparameter_tuning(model, param_grid)

#XGBoost
model = XGBClassifier(class_weight="balanced",n_jobs=1)
param_grid = {'n_estimators': [1000], 'learning_rate': [0.01, 0.05]}
model_hyperparameter_tuning(model, param_grid)