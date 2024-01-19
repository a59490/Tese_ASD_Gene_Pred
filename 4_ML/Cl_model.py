import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

X=pd.read_csv('Classification_file.csv')
# drop the last column
X=X[X.columns[-2]]

Y=pd.read_csv('Classification_file.csv')
# keep only the last column
Y = Y[Y.columns[-1]]


lr=LogisticRegression()

from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score

sensitivity_scorer = make_scorer(recall_score)
specificity_scorer = make_scorer(recall_score, pos_label=0)


scoring = {'AUC': 'roc_auc', 'Accuracy': "accuracy", "f1": "f1",
                     "Recall": "recall", "Precision": "precision", "Average Precision": "average_precision","MCC":"mcc",
                    "Sensitivity": sensitivity_scorer, "Specificity": specificity_scorer}

print(cross_validate(lr, X, Y, scoring=scoring))