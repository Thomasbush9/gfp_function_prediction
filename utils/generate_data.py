import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from utils import load_dataset, load_seq_, mutate_sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset of mutations
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--msa", type=str, default=None)
    parser.add_argument("--original", type=str, required=True)
    args = parser.parse_args()

    if not args.msa:
        msa = "empty"
    else:
        msa = args.msa

    original_seq_path = args.original
    dataset_path = args.data
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

    # 3. Generate YAML files and record index
    index_records = []
    for idx, row in tqdm(dataset.iterrows(), desc="Generating data"):
        mutated_seq = row["seq_mutated"]

        data_seq = {
            "version": 1,
            "sequences": [
                {"protein": {"id": str(idx), "sequence": mutated_seq, "msa": msa}}
            ],
        }

        filename = f"seq_{idx:05}.yaml"
        filepath = os.path.join(training_data_dir, filename)

        try:
            with open(filepath, "w") as file:
                yaml.dump(data_seq, file, sort_keys=False)
            index_records.append({"idx": idx, "filename": filename})
        except Exception as e:
            print(f"[✗] Failed to write {filename}: {e}")
    # 4. Save index.csv
    index_df = pd.DataFrame(index_records)
    index_df.to_csv(os.path.join(data_dir, "index.csv"), index=False)
    print(f"[✓] Index file written to: {os.path.join(data_dir, 'index.csv')}")
