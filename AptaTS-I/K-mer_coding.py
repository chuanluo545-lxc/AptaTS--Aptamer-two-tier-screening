from Bio import SeqIO
import pandas as pd
from collections import defaultdict
import os

def calculate_kmer_frequency(sequence, k=4):
    kmer_count = defaultdict(int)
    total_kmers = len(sequence) - k + 1
    for i in range(total_kmers):
        kmer = sequence[i:i+k]
        kmer_count[kmer] += 1
    kmer_frequency = {kmer: count / total_kmers for kmer, count in kmer_count.items()}
    return kmer_frequency

def process_fasta_to_csv(fasta_file, csv_file, k=4):
    all_kmers = [a + b + c + d for a in 'ACGT' for b in 'ACGT'for c in 'ACGT'for d in 'ACGT']

    data = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        kmer_frequency = calculate_kmer_frequency(sequence, k=4)
        row = {kmer: kmer_frequency.get(kmer, 0) for kmer in all_kmers}
        row['label'] = record.id  
        data.append(row)

    df = pd.DataFrame(data)

    columns = ['label'] + [col for col in df.columns if col != 'label']
    df = df[columns]

    df.to_csv(csv_file, index=False)

fasta_file = "input.csv"

csv_file = "output.csv"

process_fasta_to_csv(fasta_file, csv_file)
