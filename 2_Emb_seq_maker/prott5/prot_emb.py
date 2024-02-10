import csv
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)



with open("peptide_seq.csv", "r") as f:
    csv_reader = csv.reader(f)
    next(csv_reader) 
    for row in csv_reader:
            if row[3] != "None":
                try:
                    processed_sequence = " ".join(list(re.sub(r"[UZOB]", "X", row[3])))
                    ids = tokenizer(processed_sequence, add_special_tokens=True, padding="longest", return_tensors="pt").to(device)
                    input_ids = ids['input_ids']
                    attention_mask = ids['attention_mask']

                    with torch.no_grad():
                        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
                    protein_embedding = embedding_repr.last_hidden_state.mean(dim=1)[0]  # shape (1024,)

                    protein_embedding = protein_embedding.tolist()

                    row=row[:3]
                    row.append(protein_embedding)
                    with open("prot_emb.csv", "a", newline='') as file:
                        csv_writer = csv.writer(file)
                        csv_writer.writerow(row)
                except:
                    continue

