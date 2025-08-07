import pandas as pd
import os
import re
import math


def parse_hla_fasta(fasta_file):
    hla_seq_map = {}
    current_hla = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_hla and current_seq:
                    hla_seq_map[current_hla] = ''.join(current_seq)
                
                match = re.search(r'([A-Z]\*\d+(?::\d+)+(?:[A-Z])?)', line)
                if match:
                    current_hla = match.group(1)
                    current_seq = []
            else:
                if current_hla is not None:
                    current_seq.append(line)
    
    if current_hla and current_seq:
        hla_seq_map[current_hla] = ''.join(current_seq)
    
    return hla_seq_map

def get_hla_sequence(allele, hla_seq_map):
    matching_keys = [key for key in hla_seq_map.keys() if key.startswith(allele)]
    if matching_keys:
        return hla_seq_map[matching_keys[0]]
    return None


def create_dataset(neoantigen_file, output_dir, valid_ratio=0.1, random_state=42, hla_fasta_file="hla_prot.fasta"):
    neoantigen_df = pd.read_csv(neoantigen_file)

    assert neoantigen_df['peptide'].notna().all(), "Peptide column contains NaN values"
    neoantigen_df['peptide'] = neoantigen_df['peptide'].str.upper()

    assert neoantigen_df['HLA'].notna().all(), "HLA column contains NaN values"

    assert ((neoantigen_df['immunogenicity'] == 0) | (neoantigen_df['immunogenicity'] == 1)).all(), "Immunogenicity column must contain only 0 or 1"

    hla_seq_map = parse_hla_fasta(hla_fasta_file)

    neoantigen_df['HLA_sequence'] = neoantigen_df['HLA'].apply(lambda x: get_hla_sequence(x, hla_seq_map))

    assert neoantigen_df['HLA_sequence'].notna().all(), "HLA format is incorrect or not found in the FASTA file"
    
    neoantigen_df = neoantigen_df.rename(columns={
        'peptide': 'sentence1',
        'HLA_sequence': 'sentence2',
        'immunogenicity': 'label'
    })

    neoantigen_df = neoantigen_df[['sentence1', 'sentence2', 'label']]
    
    os.makedirs(output_dir, exist_ok=True)
    
    if valid_ratio > 0:
        neoantigen_df = neoantigen_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        valid_size = math.ceil(len(neoantigen_df) * valid_ratio)
        valid_df = neoantigen_df[:valid_size]
        train_df = neoantigen_df[valid_size:]
        
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        valid_df.to_csv(os.path.join(output_dir, "valid.csv"), index=False)
        print(f"Training set size: {len(train_df)}, Validation set size: {len(valid_df)}")
    else:
        neoantigen_df.to_csv(os.path.join(output_dir, "valid.csv"), index=False)
        print(f"Validation set size: {len(neoantigen_df)}")


if __name__ == "__main__":
    create_dataset(
        neoantigen_file="example/input.csv",
        output_dir="example",
        valid_ratio=0.2,
        random_state=42
    )
