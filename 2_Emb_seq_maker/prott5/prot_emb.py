import csv
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

# prepare your protein sequences as a list
protein_sequences = ["PRTEINO", "SEQWENCE"]

# replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
protein_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in protein_sequences]

# List to hold all the protein embeddings
all_embeddings = []

# Iterate over each protein sequence
for sequence in protein_sequences:
    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer(sequence, add_special_tokens=True, padding="longest", return_tensors="pt").to(device)
    input_ids = ids['input_ids']
    attention_mask = ids['attention_mask']

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    # extract residue embeddings and remove padded & special tokens
    per_protein_embedding = embedding_repr.last_hidden_state.mean(dim=1)[0]  # shape (1024,)
    
    # Append the embedding to the list
    all_embeddings.append(per_protein_embedding.tolist())

# Write all the embeddings to a single CSV file
with open("all_protein_embeddings.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(all_embeddings)
