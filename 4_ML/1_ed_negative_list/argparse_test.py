import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

def cross_val(model, X, Y):
    sensitivity_scorer = make_scorer(recall_score)
    specificity_scorer = make_scorer(recall_score, pos_label=0)
    MCC = make_scorer(matthews_corrcoef)

    scoring = {'AUC': 'roc_auc', 'Accuracy': "accuracy", "f1": "f1",
               "Recall": "recall", "Precision": "precision", "MCC": MCC, "Average Precision": "average_precision",
               "Sensitivity": sensitivity_scorer, "Specificity": specificity_scorer}

    scores = cross_validate(model, X, Y, scoring=scoring, cv=5)

    mean_scores = {metric: values.mean() for metric, values in scores.items()}

    return mean_scores

def model_hyperparameter_tuning(model, param_grid, dataset_list, model_name):
    MCC = make_scorer(matthews_corrcoef)
    results_list = []

    for dataset, name in dataset_list:
        x = dataset["3"].copy()
        x = x.str.split(expand=True)
        x = x.astype(float)

        y = dataset["4"].copy().astype('category')

        grid = GridSearchCV(model, param_grid, cv=5, scoring=MCC, verbose=1)

        results = cross_val(grid, x, y)
        result_entry = {'dataset_name': name, **results, "model": model_name}

        results_list.append(result_entry)

    results_df = pd.DataFrame(results_list)
    results_df.set_index('dataset_name', inplace=True)

    with open('model_results.csv', 'a') as f:
        results_df.to_csv(f)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model hyperparameter tuning.')
    parser.add_argument('model', choices=['lr', 'rf', 'svm', 'knn', 'lgbm', 'xgb', 'nb', 'all'],
                        help='Model to perform hyperparameter tuning: lr (Logistic Regression), rf (Random Forest), svm (Support Vector Machine), knn (K-Nearest Neighbors), lgbm (LightGBM), xgb (XGBoost), nb (Naive Bayes), all (to run all models)')
    args = parser.parse_args()

    # Load datasets---------------------------------------------------------------------------------

    values_to_remove = ['ENSG00000142599', 'ENSG00000135636', 'ENSG00000285508']
    dataset_paths = {
        'cat_1': 'gene_lists/cat_1.csv.gz',
        'cat_1_sd': 'gene_lists/cat_1_sd.csv.gz',
        'cat_1_2': 'gene_lists/cat_1_2.csv.gz',
        'cat_1_2_sd': 'gene_lists/cat_1_2_sd.csv.gz',
        'cat_1_2_3': 'gene_lists/cat_1_2_3.csv.gz',
        'complete': 'gene_lists/complete.csv.gz'
    }

    dataset_list = [(pd.read_csv(path, compression='gzip')[~pd.read_csv(path, compression='gzip')['1'].isin(values_to_remove)], name) for name, path in dataset_paths.items()]

    # Define models and parameter grids---------------------------------------------------------------

    model_params = {
        'lr': (LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=10), {'C': [0.001, 0.1, 1, 10, 100, 1000]}),
        'rf': (RandomForestClassifier(class_weight="balanced", n_jobs=10), {'n_estimators': [100, 200, 300, 400, 500, 1000], 'max_features': ['sqrt', 'log2'],
                  'max_depth': [3, 5, 10, 20, 30, 40, 50], 'min_samples_split': [2, 5, 10],
                  'min_samples_leaf': [1, 2, 4]}),
        'svm': (SVC(class_weight="balanced"), {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'poly', 'sigmoid'], 'degree': [3, 4, 5, 6, 7, 8, 9, 10]}),
        'knn': (KNeighborsClassifier(n_jobs=10), {'n_neighbors': [2, 3, 5, 7, 10, 19], 'weights': ['uniform', 'distance'],
                  'metric': ['euclidean', 'manhattan']}),
        'lgbm': (LGBMClassifier(class_weight="balanced", n_jobs=10), {'n_estimators': [100, 200, 300, 1000], 'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
                  'max_depth': [3, 5, 10, 20, 30, 40, 50], "reg_alpha": [0, 0.1, 0.5, 1, 2, 5, 10]}),
        'xgb': (XGBClassifier(class_weight="balanced", n_jobs=10), {'n_estimators': [100, 200, 300, 1000], 'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
                  'max_depth': [3, 5, 10, 20, 30, 40, 50], "booster": ['gbtree', 'gblinear', 'dart']}),
        'nb': (GaussianNB(), {})
    }

    # Model hyperparameter tuning-------------------------------------------------------------------
    if args.model == 'all':
        for model_name, (model, param_grid) in model_params.items():
            model_hyperparameter_tuning(model, param_grid, dataset_list, model_name)

    elif args.model in model_params:
        model, param_grid = model_params[args.model]
        model_hyperparameter_tuning(model, param_grid, dataset_list, args.model)
