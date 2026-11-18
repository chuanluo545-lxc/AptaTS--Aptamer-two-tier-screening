import torch
import esm
from esm import pretrained, Alphabet, FastaBatchedDataset
from esm.data import BatchConverter
import csv
from Bio import SeqIO
import numpy as np
import time
from tqdm import tqdm

start_time = time.time()  
model_name = r"esm-aa-35M.pt"  
model, alphabet = pretrained.load_model_and_alphabet(model_name)
batch_converter = alphabet.get_batch_converter()
model.eval() 
input_afp=open('protein.fasta','r')
output_afp=open('protein.csv','a')
a_dict={}
writer=csv.writer(output_afp)
for record in SeqIO.parse(input_afp,'fasta'):
    a_dict[record.id]=str(record.seq)
for seq_id,sequence in a_dict.items():
    data=[(seq_id,sequence)]
    batch_labels,batch_strs,batch_tokens=batch_converter(data)
    with torch.no_grad():
        results=model(batch_tokens,repr_layers=[12],return_contacts=True)
    token_embeddings=results['representations'][12]
    nump_token=torch.mean(token_embeddings[0,1:len(data[0][1])+1],dim=0).numpy()
    #nump_token=torch.max(token_embeddings[0,1:len(data[0][1])+1],dim=0)[0].numpy()
    nump_token=list(nump_token)
    nump_token.insert(0,seq_id)
    writer.writerow(nump_token)
input_afp.close()
output_afp.flush()
output_afp.close()
print('finish work')
end_time = time.time()
run_time = end_time - start_time
print(f"{run_time:.4f}s")


