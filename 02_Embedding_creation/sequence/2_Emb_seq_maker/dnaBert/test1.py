from transformers import T5Tokenizer, T5EncoderModel
import torch
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

#model.to(torch.float32) if device==torch.device("cpu")

sequence_examples = ["PRTEINO", "SEQWENCE"]

sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

with torch.no_grad():
    embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

emb_0 = embedding_repr.last_hidden_state[0,:7] # shape (7 x 1024)

emb_1 = embedding_repr.last_hidden_state[1,:8] # shape (8 x 1024)

print(emb_0)
