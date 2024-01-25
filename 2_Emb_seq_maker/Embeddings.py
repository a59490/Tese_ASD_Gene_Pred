import csv
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

Errors = []

with open("transcript_seq.csv", "r") as f:
    csv_reader = csv.reader(f)
    next(csv_reader)  # Skip the header row

    for row in csv_reader:
        if row[3] != "None" and len(row[3]) <= 7000:
            try:
                dna_seq = row[3]

                inputs = tokenizer(dna_seq, return_tensors='pt')["input_ids"]
                hidden_states = model(inputs)[0]
                # Embedding with mean pooling
                embedding_mean = torch.mean(hidden_states[0], dim=0)
                embedding = embedding_mean.detach().numpy()

                embedding_str = " ".join(map(str, embedding))
                row=row[:3]
                row.append(embedding_str)

                # Open the output file for writing, write the modified row, and close the file
                with open("emb_file.csv", mode='a', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(row)
            except:
                Errors.append(row[0])
                continue
f.close()