import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from utils import (
    generate_cluster_fasta,
    generate_fasta_data,
    generate_yaml_data,
    load_dataset,
    load_seq_,
    mutate_sequence,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset of mutations
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--msa", type=str, default=None)
    parser.add_argument("--original", type=str, required=True)
    parser.add_argument(
        "file_type", type=str, choices=["cluster", "fasta", "yaml"], default="fasta"
    )
    args = parser.parse_args()

    if not args.msa:
        msa = "empty"
    else:
        msa = args.msa

    original_seq_path = args.original
    dataset_path = args.data
    mode = args.file_type
    seq, mapping = load_seq_(original_seq_path)
    dataset = load_dataset(dataset_path, sep="\t")
    dataset["seq_mutated"] = dataset["aaMutations"].apply(
        lambda muts: mutate_sequence(muts, seq=seq, mapping_db_seq=mapping)
    )
    # 2. Set up directory paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", timestamp)
    training_data_dir = os.path.join(data_dir, "training_data")

    # Create directories
    os.makedirs(training_data_dir, exist_ok=True)

    if mode == "yaml":
        generate_yaml_data(dataset, msa, training_data_dir, data_dir)
    elif mode == "fasta":
        generate_fasta_data(dataset, msa, training_data_dir, data_dir)
    else:
        generate_cluster_fasta(dataset, training_data_dir, data_dir)


print("All the files have been generated correctly:")
