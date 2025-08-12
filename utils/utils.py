import os
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import except_
from tqdm import tqdm


def load_dataset(path: str, sep: None) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)


# def load_seq_(path: str):
#     with open(path, "r") as f:
#         file = f.readlines()
#         seq = "".join([s.strip("\n") for s in file[1:]])
#         mapping_db_seq = {str(i): i + 1 for i in range(len(seq))}
#         return seq, mapping_db_seq
def load_seq_(path: str, return_meta: bool = False):
    """
    Load a single-entry FASTA/A3M file.

    Parameters
    ----------
    path : str
        Path to the FASTA/A3M file.
    return_meta : bool, optional
        If True, also return a dict with 'type', 'idx', 'path', and 'header'.

    Returns
    -------
    seq : str
        The sequence from the file.
    mapping_db_seq : dict
        Maps sequence position (as str) to 1-based index.
    meta : dict, optional
        Returned only if return_meta=True.
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines or not lines[0].startswith(">"):
        raise ValueError(
            f"File {path} does not look like a FASTA/A3M with a header on the first line."
        )

    header = lines[0]
    seq = "".join(lines[1:])

    mapping_db_seq = {str(i): i + 1 for i in range(len(seq))}

    if return_meta:
        header_body = header[1:]  # remove ">"
        parts = header_body.split("|", maxsplit=2)
        seq_type, idx, msa_path = (parts + [None, None, None])[:3]

        meta = {"type": seq_type, "idx": idx, "path": msa_path, "header": header}
        return seq, mapping_db_seq, meta

    return seq, mapping_db_seq


def parse_mutation(mutation_str: str):
    match = re.match(r"^S([A-Za-z])(\d+)(.+)$", mutation_str)
    if match:
        return match.groups()
    else:
        raise ValueError(f"Invalid mutation format: {mutation_str}")


def mutate_seq(
    src: str, idx: str, dest: str, seq: str, mapping: dict, mapping_db_seq
) -> str:
    idx = mapping_db_seq[idx]
    new_seq = seq[:idx] + dest + seq[idx + 1 :]
    return new_seq


def mutate_sequence(mutation_string, seq, mapping_db_seq):
    if pd.isna(mutation_string):
        return None
    try:
        mutations = mutation_string.split(":")
        for m in mutations:
            src, idx, dest = parse_mutation(m)
            if idx in mapping_db_seq:
                mapped_idx = mapping_db_seq[idx]
                mutated_seq = seq[:mapped_idx] + dest + seq[mapped_idx + 1 :]
                return mutated_seq
    except Exception:
        return None
    return None

    # Calculate the original distribution of num_mut
    def get_numb_mut(mut: str) -> int:
        if type(mut) == str:
            n = len(mut.split(":"))
        else:
            return 0
        return n


def balanced_sampling(dataset: pd.DataFrame, n: int, output_file="balanced_subset.txt"):
    """
    Generate a balanced subset of the dataset that maintains the original distribution
    of num_mut while selecting n samples (where n < N, total dataset size).

    Parameters:
    -----------
    dataset : pd.DataFrame
        The input dataset with a 'num_mut' column
    n : int
        Number of samples to select (must be < len(dataset))
    output_file : str
        Path to the output .txt file

    Returns:
    --------
    pd.DataFrame
        The balanced subset with original indices preserved
    """
    # Validate input
    if n >= len(dataset):
        raise ValueError(f"n ({n}) must be less than dataset size ({len(dataset)})")

    dataset["num_mut"] = dataset["aaMutations"].apply(lambda mut: get_numb_mut(mut))
    original_dist = dataset["num_mut"].value_counts(normalize=True)

    # Calculate how many samples to select from each num_mut category
    target_counts = (original_dist * n).round().astype(int)

    # Ensure we don't exceed the available samples in each category
    available_counts = dataset["num_mut"].value_counts()
    final_counts = {}

    for num_mut in target_counts.index:
        target_count = target_counts[num_mut]
        available_count = available_counts.get(num_mut, 0)
        final_counts[num_mut] = min(target_count, available_count)

    # Adjust if we have fewer samples than requested
    total_selected = sum(final_counts.values())
    if total_selected < n:
        # Distribute remaining samples proportionally
        remaining = n - total_selected
        for num_mut in sorted(
            original_dist.index, key=lambda x: original_dist[x], reverse=True
        ):
            if remaining <= 0:
                break
            available = available_counts.get(num_mut, 0) - final_counts.get(num_mut, 0)
            to_add = min(remaining, available)
            final_counts[num_mut] = final_counts.get(num_mut, 0) + to_add
            remaining -= to_add

    # Sample from each category
    balanced_subset = []

    for num_mut, count in final_counts.items():
        if count > 0:
            # Get indices for this num_mut category
            category_indices = dataset[dataset["num_mut"] == num_mut].index
            # Randomly sample without replacement
            selected_indices = np.random.choice(
                category_indices, size=count, replace=False
            )
            balanced_subset.extend(selected_indices)

    # Create the balanced dataset
    balanced_dataset = dataset.loc[balanced_subset].copy()

    # Save to txt file with idx and n_mut
    output_data = []
    for idx in balanced_subset:
        n_mut = dataset.loc[idx, "num_mut"]
        output_data.append(f"{idx}\t{n_mut}")

    with open(output_file, "w") as f:
        f.write("idx\tn_mut\n")
        for line in output_data:
            f.write(line + "\n")

    print(f"Balanced subset created with {len(balanced_dataset)} samples")
    print(f"Original distribution:")
    print(dataset["num_mut"].value_counts(normalize=True).sort_index())
    print(f"\nBalanced subset distribution:")
    print(balanced_dataset["num_mut"].value_counts(normalize=True).sort_index())
    print(f"\nResults saved to: {output_file}")

    return balanced_dataset


def get_numb_mut(mut: str) -> int:
    """Helper function to count number of mutations from mutation string"""
    if type(mut) == str:
        n = len(mut.split(":"))
    else:
        return 0
    return n


# === Generate Data Functions ===
def generate_yaml_data(dataset: pd.DataFrame, msa, training_data_dir, data_dir):
    """
    Function to generate yaml data for Boltz Predictions.

    Args:
    - dataset: dataset with the mutated sequences, it must have the following
      columns: seq_mutated
    - msa: path to the msa path, if None it will be skipped
    - training_data_dir: where to save the mutated sequences
    - data_dir: where to save the idx

    It saves each mutated sequences in the yaml format in its own separate folder with this structure:
    data_dir
        index.csv
        training_data
            seq_00001
                seq_0001.yaml
    """
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


def generate_fasta_data(dataset: pd.DataFrame, msa, training_data_dir, data_dir):
    """
    Function to generate fasta data for Boltz Predictions.

    Args:
    - dataset: dataset with the mutated sequences, it must have the following
      columns: seq_mutated
    - msa: path to the msa path, if None it will be skipped
    - training_data_dir: where to save the mutated sequences
    - data_dir: where to save the idx

    It saves each mutated sequences in the fasta format in its own separate folder with this structure:
    data_dir
        index.csv
        training_data
            seq_00001
                seq_0001.fasta.txt
    """
    index_records = []

    for idx, row in tqdm(dataset.iterrows(), desc="Generating data"):
        mutated_seq = row["seq_mutated"]
        header = f">A|{idx}|{msa}"
        filename = f"seq_{idx:05}.fasta.txt"
        filepath = os.path.join(training_data_dir, filename)

        try:
            with open(filepath, "w") as f:
                f.write(header + "\n")
                f.write(mutated_seq + "\n")
            index_records.append({"idx": idx, "filename": filename})
        except Exception as e:
            print(f"[x] Failed to write {filename}: {e}")
    index_df = pd.DataFrame(index_records)
    index_df.to_csv(os.path.join(data_dir, "index.csv"), index=False)
    print(f"[✓] Index file written to: {os.path.join(data_dir, 'index.csv')}")


# === easy converter ===
def fasta2yaml(path: str):
    """
    Function to convert .fasta.txt files in Boltx format to .yaml in Boltz format

    Args:
    - path: path to fasta files to convert
    """
    if os.path.isfile(path):
        seq, mapping, meta = load_seq_(path, return_meta=True)
        type_ = meta["type"]
        idx = meta["idx"]
        msa = meta["path"]
        header = meta["header"]

        old_suffix = ".fasta.txt"
        new_suffix = ".yaml"

        new_path = path.removesuffix(old_suffix) + new_suffix
        data_seq = {
            "version": 1,
            "sequences": [{"protein": {"id": idx, "sequence": seq, "msa": msa}}],
        }

        try:
            with open(new_path, "w") as file:
                yaml.dump(data_seq, file, sort_keys=False)
        except Exception as e:
            print(f"[✗] Failed to write {new_path}: {e}")


def yaml2fasta(path: str):
    """
    Load protein info from YAML and save as FASTA-like text file.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML file.
    output_path : str
        Path to save the output FASTA (.txt) file.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    old_suffix = ".yaml"
    new_suffix = ".fasta.txt"
    output_path = path.removesuffix(old_suffix) + new_suffix

    # Expect structure: version, sequences -> list of protein dicts
    seq_entry = data["sequences"][0]["protein"]

    protein_id = seq_entry["id"]  # e.g., "A"
    msa_path = seq_entry["msa"]  # e.g., "raw_data/gfp_msa_b5fdc_0.a3m"
    sequence = seq_entry["sequence"]  # long amino acid string

    # Build header
    header = f">{protein_id}|protein|{msa_path}"

    # Write to output
    with open(output_path, "w") as out_f:
        out_f.write(header + "\n")
        out_f.write(sequence + "\n")

    print(f"[INFO] Saved FASTA file to {output_path}")


def converter(path: str, src: Literal["fasta", "yaml"]):
    """
    Converts a file or a directory from source to the other format

    Args:
    -path: [dir, file]
    - src: the source file that you want to covert, Literal['fasta', 'yaml']
    """
    # handle files:
    function = fasta2yaml if src == "fasta" else yaml2fasta
    suffix = "." + "yaml" if src == "yaml" else ".txt"
    if os.path.isfile(path):
        function(path)
        print(f"File {path} correctly converted")
    else:
        for file in tqdm(Path(path).iterdir()):
            if file.suffix == suffix:
                function(str(file))
            else:
                continue
        print(f"Directory {path} correctly converted")
