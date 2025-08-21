import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from .protein import Protein
from .utils import list2onehot, load_strain

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


def to_pyg_data(edges_ij, coords, feat):
    edge_index = torch.as_tensor(edges_ij, dtype=torch.long).t().contiguous()  # [2, E]
    pos = torch.as_tensor(coords, dtype=torch.float32)  # (n, 3)
    x = torch.as_tensor(feat, dtype=torch.float32)  # (n, 4)
    return Data(x=x, pos=pos, edge_index=edge_index)


# write shards
def write_shards(pyg_iter, out_dir, shard_size=4096):
    out = Path(out_dir).expanduser()  # expand "~"
    out.mkdir(parents=True, exist_ok=True)

    buf, shard_id, total = [], 0, 0
    for d in pyg_iter:
        buf.append(d)
        total += 1
        if len(buf) == shard_size:
            torch.save(buf, out / f"shard_{shard_id:05d}.pt")
            buf.clear()
            shard_id += 1

    if buf:  # flush last partial shard
        torch.save(buf, out / f"shard_{shard_id:05d}.pt")
        shard_id += 1

    return total, shard_id


def edges_from_neighbors(neigh_idx):
    edges = []
    for i, nbrs in enumerate(neigh_idx):
        for j in nbrs:
            edges.append((i, j))
    # For undirected graphs, optionally add the reverse and/or dedup:
    edges = list(set(tuple(sorted(e)) for e in edges))  # undirected unique
    return edges


# From dense adjacency A (n x n, 0/1):
def edges_from_dense_adj(A):
    import torch

    A = torch.as_tensor(A)
    ii, jj = torch.nonzero(A, as_tuple=True)
    return list(zip(ii.tolist(), jj.tolist()))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--es_dir", type=str, required=False, default=None)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--shard_size", type=int, default=4096)

    args = parser.parse_args()

    dir_path = Path(args.dir)
    paths = [i for i in dir_path.iterdir() if i.is_dir()]
    # find better application of strain
    es_dir = Path(args.es_dir) if args.es_dir else None
    if es_dir:
        strain = load_strain(es_dir)
        # strain is (N, 238)-> we can stack it to feat as 238, 5 tot x N

    pbar = tqdm(paths, desc="Starting")

    def sample_generator():
        for i in pbar:
            pbar.set_description_str(f"File: {i.name}")
            pbar.refresh()
            cif, confs = extract_data_from_dir(i)
            adj_mat = list2onehot(cif.neigh_idx, 238)  # -> dense (238,238)
            coords = torch.tensor(cif.coord, dtype=torch.float32)  # (238,3)
            feat = torch.cat((coords, confs.unsqueeze(-1)), dim=1)  # (238,4)
            # pack into a PyG Data
            yield to_pyg_data(adj_mat, coords, feat)

    total, nshards = write_shards(
        sample_generator(), args.out, shard_size=args.shard_size
    )
    print(
        f"Wrote {total} samples across {nshards} shards to {Path(args.out).expanduser()}"
    )
    print(f"wrote {total} samples across {nshards} shards to {args.out}")
