from Bio import SeqIO
from ViennaRNA import RNA
import numpy as np
import pandas as pd

def get_structural_kmers(seq, structure, k=1):
    structural_symbols = []
    ptable = RNA.ptable(structure)  
    
    for i in range(len(seq)):
        base = seq[i]
        pair_idx = ptable[i+1] 
        
        if pair_idx == 0:
            symbol = f"{base}·" 
        elif pair_idx > i+1:
            symbol = f"{base}(" 
        else:
            symbol = f"{base})"  
        structural_symbols.append(symbol)
    
    return [
        "".join(structural_symbols[i:i+k])
        for i in range(len(structural_symbols) - k + 1)
    ]

input_fasta = "input.fasta"
records = []
with open(input_fasta) as f:
    lines = f.readlines()
    for i in range(0, len(lines), 3):  
        header = lines[i].strip()
        seq = lines[i+1].strip()
        structure = lines[i+2].strip()
        records.append((header, seq, structure))

all_symbols = [f"{b}{s}" for b in "AGCT" for s in ["·", "(", ")"]]
all_kmers = [f"{a1}"
                      for a1 in all_symbols
                      #for a2 in all_symbols
                      #for a3 in all_symbols
                      #for a4 in all_symbols
                      ]

data = []
headers = []

for header, seq, structure in records:    
    structural_kmers = get_structural_kmers(seq, structure, k=1)
    total_kmers = len(structural_kmers)
    kmer_counts = {k: structural_kmers.count(k) for k in all_kmers}
    kmer_freq = {k: count / total_kmers for k, count in kmer_counts.items()}
    row = [kmer_freq.get(k, 0) for k in all_kmers]
    data.append(row)
    headers.append(header)

df = pd.DataFrame(data, index=headers, columns=all_kmers)
df.to_csv("output.csv",encoding="gbk")
