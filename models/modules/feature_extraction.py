import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm

from .protein import Protein

"""
It will expect a dir path structure like this:
    dir/
        seq_0000/
            boltz_results/
                predictions/
                    seq_0000/
                        .cif structure prediction
                        confidence files
it will generate a data loader with the files:

"""


def extract_data_from_dir(dir: Path):
    dir_name = dir.name
    cif_file_path = (
        dir
        / f"boltz_results_{dir.name}"
        / "predictions"
        / dir.name
        / f"{dir.name}_model_0.cif"
    )
    cif_file = Protein(cif_file_path)
    return cif_file


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)

    args = parser.parse_args()

    dir_path = Path(args.dir)

    for i in dir_path.iterdir():
        cif = extract_data_from_dir(i)
        print(cif.neigh_idx)
        break
