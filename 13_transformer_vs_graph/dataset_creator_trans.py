import pandas as pd


dataset_paths = {
        'cat_1': 'gene_lists/cat_1.csv.gz',
        'cat_1_sd': 'gene_lists/cat_1_sd.csv.gz',
        'cat_1_2': 'gene_lists/cat_1_2.csv.gz',
        'cat_1_2_sd': 'gene_lists/cat_1_2_sd.csv.gz',
        'cat_1_2_3': 'gene_lists/cat_1_2_3.csv.gz',
        'complete': 'gene_lists/complete.csv.gz'
    }

dataset_list = [(pd.read_csv(path, compression='gzip'), name) for name, path in dataset_paths.items()]

trans_embeddings = pd.read_csv('final_prot_emb(1).csv.gz', compression='gzip', header=None)

for dataset, name in dataset_list:
    name= f"{name}_trans"
    dataset= dataset[["y", "ensb_gene_id"]]
    dataset_ed = pd.merge(dataset, trans_embeddings, left_on='ensb_gene_id', right_on=1, how='inner')
    # export dataset_ed to csv
    dataset_ed.to_csv(f"gene_trans/{name}.csv.gz", compression='gzip', index=False)
    print(f"Exported {name}.csv.gz")