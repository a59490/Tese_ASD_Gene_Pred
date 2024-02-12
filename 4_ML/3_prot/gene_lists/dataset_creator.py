import pandas as pd

def dataset_creator(dataset,emb_file):
    sfari=pd.read_csv(dataset)

    #get the sfari genes with score <= 1,2,3
    sfari_1=sfari[sfari['gene-score'] <= 1].copy()
    sfari_2=sfari[sfari['gene-score'] <= 2].copy()
    sfari_3=sfari[sfari['gene-score'] <= 3].copy()


    #get the syndromic genes with no gene score
    sydromic = pd.read_csv(dataset)
    sydromic = sydromic[pd.isna(sydromic['gene-score'])].copy()

    #get the sfari genes with score <= 1,2,3 and the syndromic genes with no gene score
    sfari_1_sd= pd.concat([sydromic,sfari_1],ignore_index=True)
    sfari_2_sd= pd.concat([sydromic,sfari_2],ignore_index=True)
    sfari_3_sd= pd.concat([sydromic,sfari_3],ignore_index=True)


    positives=((sfari_1,"cat_1"),(sfari_2,"cat_1_2"),(sfari_3,"cat_1_2_3")
               ,(sfari_1_sd,"cat_1_sd"),(sfari_2_sd,"cat_1_2_sd"),(sfari_3_sd,"cat_1_2_3_sd"))


    #get the negative dataset
    negative_dataset=pd.read_csv(emb_file,header=None)
    negative_dataset=negative_dataset[negative_dataset[4] ==0]

    #concatenate the positive and negative datasets
    emb_dataset=pd.read_csv(emb_file,header=None)

    for positive,name in positives:
        positive_dataset=pd.merge(emb_dataset,positive,left_on=1,right_on='ensembl-id')
        final_dataset=pd.concat([negative_dataset,positive_dataset],ignore_index=True)
        final_dataset.to_csv(name+'.csv.gz', index=False, compression='gzip')
    
    emb_dataset.to_csv("complete.csv.gz", index=False, compression='gzip')

dataset_creator('sfari_ed_01_16_2024.csv','dataset_prot.csv')