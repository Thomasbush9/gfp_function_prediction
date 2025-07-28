from pathlib import Path
import pandas as pd
import re


def load_dataset(path:str, sep:None)->pd.DataFrame:
    return pd.read_csv(path, sep=sep)

def load_seq_(path:str):
    with open(path, 'r') as f:
        file = f.readlines()
        seq = ''.join([s.strip('\n') for s in file[1:]])
        mapping_db_seq = {str(i):i+1 for i in range(len(seq))}
        return seq, mapping_db_seq


def parse_mutation(mutation_str:str):
    match = re.match(r'^S([A-Za-z])(\d+)(.+)$', mutation_str)
    if match:
        return match.groups()
    else:
        raise ValueError(f"Invalid mutation format: {mutation_str}")

def mutate_seq(src:str, idx:str, dest:str, seq:str, mapping:dict, mapping_db_seq)->str:
    idx = mapping_db_seq[idx]
    new_seq =  seq[:idx] + dest + seq[idx + 1:]
    return new_seq

def mutate_sequence(mutation_string, seq, mapping_db_seq):
    if pd.isna(mutation_string):
        return None
    try:
        mutations = mutation_string.split(':')
        for m in mutations:
            src, idx, dest = parse_mutation(m)
            if idx in mapping_db_seq:
                mapped_idx = mapping_db_seq[idx]
                mutated_seq = seq[:mapped_idx] + dest + seq[mapped_idx + 1:]
                return mutated_seq
    except Exception:
        return None
    return None
