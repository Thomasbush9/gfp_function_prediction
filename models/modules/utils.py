# Script to facilitate .cif feature extraction, inspiration by:
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def rotate_points(P, Q):
    """Rotates a set of of points using the Kabsch algorithm"""
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    D = np.linalg.det(V @ U.T)
    E = np.array([[1, 0, 0], [0, 1, 0], [0, 0, D]])
    R = V @ E @ U.T
    Pnew = (R @ P.T).T
    return Pnew


### Returns a np.ndarray of positions where two sequences differ
def get_mutation_position(s1, s2):
    """Get the position of mutations between two sequences"""
    return np.where(np.array(list(s1)) != np.array(list(s2)))[0]


def get_shared_indices(idx1, idx2):
    """Get the intersection between two sets of indices"""
    i1 = np.where(np.in1d(idx1, idx2))[0]
    i2 = np.where(np.in1d(idx2, idx1))[0]
    return i1, i2


def list2onehot(idxs: List, dim: int) -> torch.Tensor:
    """
    It takes as input a list of lists, each list is a residue containing the
    idx of the neighbour residues.

    Args:
    - idxs: List[List]
    - dim: int, number of residues

    Returns:
    - mat: Tensor, a the adjacencacy matrix (n_res, n_res)
    """
    shape = (dim, dim)
    mat = torch.zeros(shape, dtype=torch.float32)
    lengths = torch.tensor([len(r) for r in idxs])
    row_idx = torch.arange(dim).repeat_interleave(lengths)
    col_idx = torch.tensor([i for r in idxs for i in r])
    mat[row_idx, col_idx] = 1
    return mat


def load_strain(es_dir: Path):
    """
    Args:
    - es_dir: effective strain directory of .csv files

    Returns:
    - strain: torch.Tensor (n_files, n_residues)
    - filenames: list of str, names of the files (without extension)
    """
    tensors = []
    filenames = []

    for file in sorted(es_dir.iterdir()):
        if file.suffix == ".csv":
            df = pd.read_csv(file)
            tensors.append(torch.tensor(df["strain"].values))
            filenames.append(file.stem)  # or file.name if you want the .csv extension

    strain = torch.stack(tensors, dim=0)
    return strain, filenames
