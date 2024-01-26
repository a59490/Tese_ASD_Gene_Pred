import csv
import pandas as pd
import numpy as np

gene_list = pd.read_csv("gene_list_01_16_2024.csv", index_col=0)

final_list=[]

with open("emb_file.csv", "r") as f:
    csv_reader = csv.reader(f)
    next(csv_reader)

    for row in csv_reader:
        if len(row) == 4:
            if row[1] in gene_list.index:
 
               row = row + [gene_list.loc[row[1],"y"]]
               final_list.append(row)

#write final list to csv file
with open("final_dataset.csv", "w", newline="") as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerows(final_list) 
#test
