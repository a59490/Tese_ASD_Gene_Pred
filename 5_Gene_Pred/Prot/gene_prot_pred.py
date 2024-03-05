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

from sklearn.linear_model import LogisticRegressionCV


def remover(x):
    x = x.str.replace(' ','')
    x = x.str.replace('\'','')
    x = x.str.replace('[','')
    x = x.str.replace(']','')
    return x

# read data

cat_1_sd = pd.read_csv('cat_1_sd.csv.gz', compression='gzip')

X_data = cat_1_sd["3"].copy()
X_data = X_data.str.split(expand=True,pat=',')
X_data= X_data.apply(remover)
X_data = X_data.astype(float)

y_data = cat_1_sd['4'].copy().astype('category')

all_genes = pd.read_csv('prot_emb.csv.gz', compression='gzip', header=None)
X_test = all_genes[3].copy()
X_test = X_test.str.split(expand=True,pat=',')
X_test = X_test.apply(remover)
X_test = X_test.astype(float)

gene_names = all_genes[[0, 1]].copy()

lr_model = LogisticRegressionCV(cv=5, random_state=0, max_iter=1000, n_jobs=8, scoring='f1', verbose=1).fit(X_data, y_data)

predictions = lr_model.predict_proba(X_test)

predictions_df = pd.DataFrame(predictions, columns=['Probability_Class_0', 'Probability_Class_1'])

result_df = pd.concat([gene_names, predictions_df], axis=1)
result_df = result_df.sort_values(by='Probability_Class_1', ascending=False)

# Save the result
result_df.to_csv("predictions_with_gene_names.csv", index=False)


