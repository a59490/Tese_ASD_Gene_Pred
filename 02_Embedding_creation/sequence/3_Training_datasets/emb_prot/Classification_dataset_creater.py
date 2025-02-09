import csv
import gzip
import pandas as pd

gene_list = pd.read_csv("gene_list_krs_clean.csv", index_col=0)

final_list = []

with gzip.open("prot_emb.csv.gz", "rt") as f:
    csv_reader = csv.reader(f)
    next(csv_reader)

    for row in csv_reader:
        if len(row) == 4:
            if row[1] in gene_list.index:

                row[3]=row[3].strip("[]")
                row[3] = [float(value) for value in row[3].split(",")]

                row = row + [gene_list.loc[row[1], "y"]]
                final_list.append(row)

# Write final list to csv file
with open("dataset_prot.csv", "w", newline="") as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerows(final_list)
