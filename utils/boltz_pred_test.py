import os
import re
import shutil
from argparse import ArgumentParser
from ast import parse
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.utils import balanced_sampling

if __name__ == "__main__":
    print("Generating subsample")

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument(
        "--yaml_dir", type=str, required=True, help="Directory containing YAML files"
    )

    timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_subsample"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", timestamp)
    os.makedirs(data_dir, exist_ok=True)
    sub_dir = os.path.join(data_dir, "subsample")
    os.makedirs(sub_dir, exist_ok=True)

    args = parser.parse_args()

    data_path = Path(args.dataset)
    n = args.n
    yaml_dir = Path(args.yaml_dir)

    # Load the dataset
    print(f"Loading dataset from: {data_path}")
    dataset = pd.read_csv(data_path, sep="\t")

    # Generate balanced subsample and create txt file
    output_txt_path = os.path.join(data_dir, "balanced_subset.txt")
    balanced_dataset = balanced_sampling(dataset, n, output_txt_path)

    # Copy corresponding YAML files
    print(f"Copying YAML files from: {yaml_dir}")
    copied_count = 0

    # Determine the maximum index to calculate padding length
    max_idx = max(balanced_dataset.index)
    padding_length = len(str(max_idx))

    for idx in balanced_dataset.index:
        # Format index with zero padding
        padded_idx = str(idx).zfill(padding_length)
        yaml_filename = f"seq_{padded_idx}.yaml"
        source_yaml_path = yaml_dir / yaml_filename
        dest_yaml_path = Path(sub_dir) / yaml_filename

        if source_yaml_path.exists():
            shutil.copy2(source_yaml_path, dest_yaml_path)
            copied_count += 1
        else:
            print(f"Warning: YAML file not found: {source_yaml_path}")

    print(f"Successfully copied {copied_count} YAML files to: {data_dir}")
    print(f"Total files in output directory: {len(list(Path(data_dir).glob('*')))}")
