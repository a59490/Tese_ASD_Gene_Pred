import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import StratifiedKFold


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import cross_val_predict


def remover(x):
    x = x.str.replace(' ','')
    x = x.str.replace('\'','')
    x = x.str.replace('[','')
    x = x.str.replace(']','')
    return x

# read data

cat_1_sd = pd.read_csv('cat_1_sd.csv.gz', compression='gzip')

emb_file = pd.read_csv('prot_emb.csv.gz', compression='gzip', header=None)


# Filter the cat1 genes from the all_genes
cat_1_sd_filter= cat_1_sd["1"].copy()

all_genes = emb_file[~emb_file[1].isin(cat_1_sd_filter)].copy()
all_genes = all_genes.reset_index(drop=True)
gene_names = all_genes[[0, 1]].copy()


# format all_genes

all_genes = all_genes[3].copy()
all_genes = all_genes.str.split(expand=True,pat=',')
all_genes = all_genes.apply(remover)
all_genes = all_genes.astype(float)



# make the folds

skf = StratifiedKFold(n_splits=5)

X_fold=cat_1_sd["3"].copy()

y_fold=cat_1_sd["4"].copy()

predictions_cat1_list = []
test_gene_names = []

for i, (train_index, test_index) in enumerate(skf.split(X_fold, y_fold)):

    # Fold filters
    test_filter = cat_1_sd["0"].iloc[test_index]

    test_dataset= cat_1_sd[cat_1_sd['0'].isin(test_filter)]
                
    train_dataset = cat_1_sd[~cat_1_sd['0'].isin(test_filter)]

    t_gene_names=test_dataset[['0', '1']].copy()

    # create train X and Y
    X_data = train_dataset["3"].copy()
    X_data = X_data.str.split(expand=True,pat=',').apply(remover).astype(float)

    y_data = train_dataset["4"].copy().astype('category')

    # format x_test
    X_test = test_dataset["3"].copy()
    X_test = X_test.str.split(expand=True,pat=',').apply(remover).astype(float)


    # model ----------------------
    model=SVC(class_weight="balanced" ,probability=True)

    # Params ----------------------
    params={'C': [100]}

    grid_model= GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='f1', verbose=1, refit=True)

    grid_model.fit(X_data, y_data)

    predictions_cat1=grid_model.predict_proba(X_test)
    test_gene_names.append(t_gene_names)

    # Append predictions to lists
    predictions_cat1_list.append(predictions_cat1)


# Calculate all preds----------------------------------------------------------------------------------------------
    
cat_1_x = cat_1_sd["3"].copy()
cat_1_x = cat_1_x.str.split(expand=True,pat=',').apply(remover).astype(float)
cat_1_y = cat_1_sd["4"].copy().astype('category')

# model ----------------------
model=SVC(class_weight="balanced" ,probability=True)

# Params ----------------------
params={'C': [100]}

grid_model= GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='f1', verbose=1, refit=True)
search=grid_model.fit(cat_1_x, cat_1_y)

predictions_all = search.predict_proba(all_genes)
    
# Concatenate the predictions and the gene names----------------------------------------------------------------------------------------------
#concat
predictions_cat1_list = np.concatenate(predictions_cat1_list, axis=0)
test_gene_names= np.concatenate(test_gene_names, axis=0)


predictions_cat1_df = pd.DataFrame(predictions_cat1_list, columns=['Probability_Class_0', 'Probability_Class_1'])
test_gene_names_df = pd.DataFrame(test_gene_names, columns=['Gene', 'Ensemble_ID'])

predictions_all_df = pd.DataFrame(predictions_all, columns=['Probability_Class_0', 'Probability_Class_1'])


# concat the  predictions_cat1
cat_1_sd_df = pd.concat([test_gene_names_df, predictions_cat1_df], axis=1)

# concat the  predictions_all
all_genes_df = pd.concat([gene_names, predictions_all_df], axis=1)
all_genes_df.columns = ['Gene', 'Ensemble_ID', 'Probability_Class_0', 'Probability_Class_1']

# concat the  predictions_all with the cat_1_sd
concat_df = pd.concat([cat_1_sd_df, all_genes_df], axis=0, ignore_index=True)
concat_df = concat_df.sort_values(by='Probability_Class_1', ascending=False)


# Save the result
concat_df.to_csv("Results/test.csv", index=False)

# Clean the data
from Csv_clean import *

main("Results/test.csv")

