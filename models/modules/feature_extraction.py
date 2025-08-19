import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .protein import Protein
from .utils import list2onehot

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
    """
    Args:
    - dir: Path object, path to a single directory of predictions

    Returns:
    - cif file,
    - confidences: Tensor, (num_res, )
    """
    dir_pred = dir / f"boltz_results_{dir.name}" / "predictions" / dir.name
    cif_file_path = dir_pred / f"{dir.name}_model_0.cif"
    confs = np.load(dir_pred / f"plddt_{dir.name}_model_0.npz")
    cif_file = Protein(cif_file_path)
    return cif_file, torch.Tensor(confs["plddt"])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)

    args = parser.parse_args()

    dir_path = Path(args.dir)

    for i in dir_path.iterdir():
        cif, confs = extract_data_from_dir(i)
        adj_mat = list2onehot(cif.neigh_idx, 238)
        coords = torch.Tensor(cif.coord)
        feat = torch.cat((coords, confs.unsqueeze(dim=-1)), dim=1)
