import pandas as pd
def remover(x):
    x = x.str.replace(' ','')
    x = x.str.replace('\'','')
    x = x.str.replace('[','')
    x = x.str.replace(']','')
    return x

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
trans_embeddings.columns = ['syb', 'gene_id', 'protein_id', 'embedding']

for dataset, name in dataset_list:
    name= f"{name}_both"
    dataset= dataset.drop(columns=['syb','ensb_prot_id'])
    dataset_ed = pd.merge(dataset, trans_embeddings, left_on='ensb_gene_id', right_on='gene_id', how='inner')

    x_data = dataset_ed['embedding'].str.split(expand=True,pat=',')
    x_data = x_data.apply(remover)
    x_data = x_data.astype(float)
    dataset_ed.drop(columns=['embedding'], inplace=True)
    final_dataset = pd.concat([dataset_ed, x_data], axis=1)
    print(final_dataset.shape)
    # export dataset_ed to csv
    final_dataset.to_csv(f"gene_both/{name}.csv.gz", compression='gzip', index=False)
    print(f"Exported {name}.csv.gz")