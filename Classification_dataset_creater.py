import csv
import pandas as pd
import numpy as np

sfari= pd.read_csv("sfari_ed.csv") #open sfari file
ensembl_id_column = sfari["ensembl-id"].to_list() #get ensembl id column

xls = pd.read_excel("Negative_Positive_genes.xlsx", sheet_name=1)
NCBI_id_column = xls["gene id"].to_list() #get NCBI id column
NCBI_id=[str(id) for x in NCBI_id_column]

final_list=[]

with open("emb_file.csv", "r") as f:
    csv_reader = csv.reader(f)
    next(csv_reader) 

    for row in csv_reader:
        if len(row) == 6:
            if row[1] in ensembl_id_column:
                final_list.append(row)
            elif row[3] in NCBI_id:
                final_list.append(row)

#write final list to csv file


with open("final_dataset.csv", "w", newline="") as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerows(final_list)
                