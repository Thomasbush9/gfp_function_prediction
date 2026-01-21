# Script to facilitate .cif feature extraction, inspiration by:
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib


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
def neigh_list_to_edge_index(idxs: List, dim: int = None, undirected: bool = False):
    """
    Convert neighbor-list adjacency into edge_index (COO).
    idxs[i] = list of neighbors of node i.

    Returns:
      edge_index: LongTensor shape (2, E)
    """
    lengths = torch.tensor([len(r) for r in idxs], dtype=torch.long)
    row = torch.arange(len(idxs), dtype=torch.long).repeat_interleave(lengths)
    col = torch.tensor([j for r in idxs for j in r], dtype=torch.long)

    edge_index = torch.stack([row, col], dim=0)  # (2, E)

    if undirected:
        rev = edge_index.flip(0)                 # (2, E) swapped rows/cols
        edge_index = torch.cat([edge_index, rev], dim=1)

    # Optional: remove duplicates / self-loops if needed
    # edge_index = torch.unique(edge_index, dim=1)
    # edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)

    return edge_index

def load_strain(combined_es_path: Path):
    """
    Args:
    - combined_es_path:  effective strain directory of .joblib file 

    Returns:
    - strain: torch.Tensor (n_files, n_residues)
    - filenames: list of str, names of the files (without extension)
    """
    tensors = []
    filenames = []

    data = joblib.load(combined_es_path)
    for seq_idx, df in data.items():
        filenames.append(seq_idx)
        tensors.append(torch.Tensor(df["strain"].values))

    strain = torch.stack(tensors, dim=0)
    return strain, filenames

def load_esm_data(dir_path: Path) -> torch.Tensor:
    """Load ESM embeddings saved as embeddings.npy"""
    embedding_path = dir_path / "embeddings.npy"

    if embedding_path.exists():
        arr = np.load(embedding_path)  # numpy array
        embeddings = torch.from_numpy(arr).float()
        return embeddings
    else:
        print(f"ESM embeddings not found in {dir_path}")
        pass

