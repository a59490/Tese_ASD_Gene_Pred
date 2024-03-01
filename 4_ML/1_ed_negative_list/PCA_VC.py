import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef, recall_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, average_precision_score
from sklearn.model_selection import StratifiedKFold


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# Model hyperparameter tuning-------------------------------------------------------------------
def model_evaluation(model, param_grid, dataset_list, model_name):
    
    sensitivity_scorer = make_scorer(recall_score)
    specificity_scorer = make_scorer(recall_score, pos_label=0)
    MCC = make_scorer(matthews_corrcoef)

    scoring = {'AUC': roc_auc_score,'Accuracy': accuracy_score, "f1": f1_score,
            "Recall": recall_score,"Precision": precision_score, "Average Precision": average_precision_score,
            "Sensitivity": sensitivity_scorer, "Specificity": specificity_scorer, "MCC": MCC}

    # Create the folds
    cat_1=dataset_list[0][0]
    X_fold=cat_1["3"].copy()

    y_fold=cat_1["4"].copy()

    skf = StratifiedKFold(n_splits=5)

    #Iterate through the datasets
    dataset_results = []
    for dataset, name in dataset_list:
        results_list = []

        for i, (train_index, test_index) in enumerate(skf.split(X_fold, y_fold)):
                
            cat_1 = pd.read_csv('gene_lists/cat_1.csv.gz', compression='gzip')

            # Fold filters
            
            test_filter = cat_1["0"].iloc[test_index]

            test_dataset= cat_1[cat_1['0'].isin(test_filter)]
                
            dataset_ed = dataset[~dataset['0'].isin(test_filter)]

            # create X and Y
            X_data = dataset_ed["3"].copy()
            X_data = X_data.str.split(expand=True)
            X_data = X_data.astype(float)

            y_data = dataset_ed["4"].copy().astype('category')

            # PCA train

            pca=PCA(n_components=0.90)

            X_data = pca.fit_transform(X_data)

            #fit model with best param_grid
            grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=MCC, verbose=1, refit=True)

            search = grid.fit(X_data, y_data)

            best_model = search.best_estimator_

            #edit test data
            test_data_x= test_dataset["3"].copy()
            test_data_x = test_data_x.str.split(expand=True)
            test_data_x = test_data_x.astype(float)


            test_data_y = test_dataset["4"].copy().astype('category')

            #PCA test
            test_data_x = pca.transform(test_data_x)

            #make the predictions
            y_pred = search.predict(test_data_x)
            y_true = test_data_y

            #calculate the scores
            scores = {}
            for metric, scorer in scoring.items():
                if metric in ['Sensitivity', 'Specificity', 'MCC']:
                    scores[metric] = scorer(best_model, test_data_x, test_data_y,sample_weight=None)
                else:
                    scores[metric] = scorer(y_true, y_pred)

            results_list.append(scores)

            mean_scores = {metric: (round(np.mean([result[metric] for result in results_list]), 6),
                                    round(np.std([result[metric] for result in results_list]), 6)) for metric in scoring}

        
        mean_scores['model'] = model_name
        mean_scores['dataset_name'] = name  
        
        dataset_results.append(mean_scores)

    results_df = pd.DataFrame(dataset_results)
    results_df.set_index('dataset_name', inplace=True)
    
    with open('PCA_results.csv', 'a') as f:
        results_df.to_csv(f)

# Parse arguments---------------------------------------------------------------------------------          

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
        'lr': (LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=10), {'C': [80]}),
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
            model_evaluation(model, param_grid, dataset_list, model_name)

    elif args.model in model_params:
        model, param_grid = model_params[args.model]
        model_evaluation(model, param_grid, dataset_list, args.model)
