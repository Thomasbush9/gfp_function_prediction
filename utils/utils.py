import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(path: str, sep: None) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)


def load_seq_(path: str):
    with open(path, "r") as f:
        file = f.readlines()
        seq = "".join([s.strip("\n") for s in file[1:]])
        mapping_db_seq = {str(i): i + 1 for i in range(len(seq))}
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

    # Calculate the original distribution of num_mut
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
