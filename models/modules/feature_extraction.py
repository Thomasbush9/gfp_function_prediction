import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

try:
    from .protein import Protein
    from .utils import list2onehot, load_strain, load_esm_data, neigh_list_to_edge_index
except ImportError:
    from protein import Protein

    from utils import list2onehot, load_strain, load_esm_data


def load_target_values(tsv_path: Path) -> dict:
    """
    Load target values from TSV file.

    Args:
        tsv_path: Path to the TSV file containing target values

    Returns:
        Dictionary mapping sequence indices to target values
    """
    df = pd.read_csv(tsv_path, sep="\t")

    # Create mapping from sequence index to target value
    # The sequence numbers in CIF files directly correspond to TSV row indices
    target_mapping = {}

    for idx, row in df.iterrows():
        # Use the row index as the sequence number
        seq_idx = f"seq_{idx:05d}"
        median_brightness = row["medianBrightness"]
        target_mapping[seq_idx] = median_brightness

    return target_mapping

#TODO: add support for new format
def extract_data_from_dir(dir: Path, target_mapping: dict = None):
    """
    Extract data from a single directory of predictions.

    Args:
        dir: Path object, path to a single directory of predictions
        target_mapping: Dictionary mapping sequence indices to target values

    Returns:
        - cif file
        - confidences: Tensor, (num_res, )
        - target_value: float or None
    """
    # Find the CIF file in the boltz results
    boltz_dir = dir / "boltz"
    cif_file_path = None
    conf_file_path = None
    # Search for CIF and confidence files
    if boltz_dir.exists():
        # Look for CIF and confidence files
        for file in boltz_dir.glob("*.cif"):
            cif_file_path = file
        for file in boltz_dir.glob("plddt_*.npz"):
            conf_file_path = file

    if cif_file_path is None or conf_file_path is None:
        raise FileNotFoundError(f"Could not find CIF or confidence files in {dir}")

    # Load confidence scores
    confs = np.load(conf_file_path)
    conf_tensor = torch.Tensor(confs["plddt"])

    # Load protein structure
    cif_file = Protein(cif_file_path)

    # Get target value if available
    target_value = None
    if target_mapping is not None:
        # dir is like .../seq_35209
        seq_idx = dir.name  # already "seq_35209"
        target_value = target_mapping.get(seq_idx)

    # if target_mapping is not None:
    #     # Extract sequence index from the CIF file path
    #     # The sequence index is in the path like: .../seq_27173.fasta/...
    #     for part in cif_file_path.parts:
    #         if "seq_" in part and ".fasta" in part:
    #             # Extract the sequence number from seq_XXXXX.fasta
    #             seq_part = part.split("_")[1].split(".")[0]  # Get the number part
    #             seq_idx = f"seq_{seq_part.zfill(5)}"
    #             target_value = target_mapping.get(seq_idx)
    #             print(f"Found sequence {seq_idx} in {part}, target: {target_value}")
    #             break
    #
    return cif_file, conf_tensor, target_value


def to_pyg_data(edges_ij, coords, feat, target_value=None):
    """
    Convert to PyTorch Geometric Data object.

    Args:
        edges_ij: Edge indices
        coords: 3D coordinates
        feat: Node features
        target_value: Target value for the graph

    Returns:
        PyG Data object
    """
    edge_index = torch.as_tensor(edges_ij, dtype=torch.long).t().contiguous()  # [2, E]
    pos = torch.as_tensor(coords, dtype=torch.float32)  # (n, 3)
    x = torch.as_tensor(feat, dtype=torch.float32)  # (n, 4)

    # Create Data object with target value
    data = Data(x=x, pos=pos, edge_index=edge_index)

    if target_value is not None:
        data.y = torch.tensor([target_value], dtype=torch.float32)

    return data


def edges_from_neighbors(neigh_idx):
    """Convert neighbor indices to edge list."""
    edges = []
    for i, nbrs in enumerate(neigh_idx):
        for j in nbrs:
            edges.append((i, j))
    # For undirected graphs, optionally add the reverse and/or dedup:
    edges = list(set(tuple(sorted(e)) for e in edges))  # undirected unique
    return edges


def edges_from_dense_adj(A):
    """Convert dense adjacency matrix to edge list."""
    A = torch.as_tensor(A)
    ii, jj = torch.nonzero(A, as_tuple=True)
    return list(zip(ii.tolist(), jj.tolist()))


def write_shards(pyg_iter, out_dir, shard_size=4096):
    """Write PyG data to shards."""
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
def write_tensor_shards(sample_iter, out_dir, shard_size=4096):
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    shard_id, total = 0, 0

    # shard buffers (lists that will be stacked)
    pos_buf = []
    xstruct_buf = []
    xesm_buf = []
    y_buf = []

    edge_cat = []
    edge_ptr = [0]  # cumulative edge count

    def flush():
        nonlocal shard_id, pos_buf, xstruct_buf, xesm_buf, y_buf, edge_cat, edge_ptr

        if not pos_buf:
            return

        shard = {
            "pos": torch.stack(pos_buf, dim=0),         # (S,238,3)
            "x_struct": torch.stack(xstruct_buf, dim=0),# (S,238,F)
            "x_esm": torch.stack(xesm_buf, dim=0),      # (S,238,960)
            "y": torch.stack(y_buf, dim=0),             # (S,)
            "edge_index": torch.cat(edge_cat, dim=1),   # (2,sumE)
            "edge_ptr": torch.tensor(edge_ptr, dtype=torch.long),  # (S+1,)
        }

        shard_path = out / f"shard_{shard_id:05d}.pt"
        torch.save(shard, shard_path)
        torch.save(len(pos_buf), shard_path.with_suffix(".meta.pt"))

        shard_id += 1

        # reset buffers
        pos_buf.clear()
        xstruct_buf.clear()
        xesm_buf.clear()
        y_buf.clear()
        edge_cat.clear()
        edge_ptr = [0]

    for sample in sample_iter:
        # sample is a dict-like payload we yield (see generator below)
        pos_buf.append(sample["pos"])
        xstruct_buf.append(sample["x_struct"])
        xesm_buf.append(sample["x_esm"])
        y_buf.append(sample["y"])

        ei = sample["edge_index"]
        edge_cat.append(ei)
        edge_ptr.append(edge_ptr[-1] + ei.size(1))

        total += 1
        if len(pos_buf) >= shard_size:
            flush()

    flush()
    return total, shard_id


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Extract features from protein structure predictions"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing sequence predictions",
    )
    parser.add_argument(
        "--tsv", type=str, required=True, help="Path to TSV file with target values"
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output directory for processed data"
    )
    parser.add_argument(
        "--shard_size", type=int, default=4096, help="Size of each shard file"
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="medianBrightness",
        help="Column name in TSV file containing target values",
    )

    args = parser.parse_args()

    # Load target values
    print(f"Loading target values from {args.tsv}")
    target_mapping = load_target_values(Path(args.tsv))
    print(f"Loaded {len(target_mapping)} target values")

    # Get all sequence directories
    
    dir_path = Path(args.dir)
    paths = [p for p in dir_path.iterdir() if p.is_dir() and p.name.startswith("seq_")]

    print(f"Found {len(paths)} sequence directories")

    # Load effective strain if provided
    es_dir = dir_path / "es" / "combined.joblib"
    strain_all, strain_ids = (None, None)
    if es_dir:
        strain_all, strain_ids = load_strain(es_dir)
        strain_index = {name: idx for idx, name in enumerate(strain_ids)}
        print(f"Loaded effective strain for {strain_all.shape[0]} sequences")
    else:
        strain_index = {} 
    print(f"Loaded effective strain data from {es_dir}")

    pbar = tqdm(paths, desc="Processing sequences")

    def sample_generator():
        successful = 0
        failed = 0

        for i in pbar:
            pbar.set_description_str(f"Processing: {i.name}")
            seq_name = i.name
            pbar.refresh()

            try:
                # Extract data from directory
                cif, confs, target_value = extract_data_from_dir(i, target_mapping)
# Create adjacency matrix (you said you want to keep this step)
                adj_mat = list2onehot(cif.neigh_idx, 238)  # (238,238) float32

# Convert to edge_index for storage
                ii, jj = torch.nonzero(adj_mat, as_tuple=True)
                edge_index = torch.stack([ii, jj], dim=0).to(torch.long).contiguous()  # (2,E)

# Coordinates
                coords = torch.tensor(cif.coord, dtype=torch.float32)  # (238,3)

# Structural features: coords + confidence (+ strain)
                confs = confs.to(torch.float32).view(-1, 1)            # (238,1)
                x_struct = torch.cat([coords, confs], dim=1)           # (238,4)

                if strain_all is not None and seq_name in strain_index:
                    s = strain_all[strain_index[seq_name]]
                    s = s.to(torch.float32)
                    s = torch.nan_to_num(s, nan=0.0).view(-1, 1)
                else:
                    s = torch.zeros((x_struct.shape[0], 1), dtype=torch.float32)
                if s.shape[0] != x_struct.shape[0]:
                    raise ValueError(f"Strain length {s.shape[0]} not correct")
                x_struct = torch.cat([x_struct, s], dim=1)         # (238,5)

# ESM
                esm_embeddings = load_esm_data(i)
                if esm_embeddings is None:
                    raise ValueError("Missing ESM embeddings")  # or handle with zeros

                esm = esm_embeddings.squeeze(0)   # (240,960)
                esm = esm[1:-1]                   # (238,960)
                if esm.shape[0] != x_struct.shape[0]:
                    raise ValueError(f"ESM length {esm.shape[0]} != residues {x_struct.shape[0]}")

# Cast to fp16 for storage
                pos = coords.to(torch.float16)
                x_struct = x_struct.to(torch.float16)
                x_esm = esm.to(torch.float16)

                y = torch.tensor(target_value, dtype=torch.float32)  # keep y float32

                yield {
                    "pos": pos,
                    "x_struct": x_struct,
                    "x_esm": x_esm,
                    "edge_index": edge_index,
                    "y": y,
                }

            except Exception as e:
                print(f"Failed to process {i.name}: {e}")
                failed += 1
                continue

        print(f"Successfully processed: {successful}")
        print(f"Failed to process: {failed}")

    # Write shards
    print(f"Writing shards to {args.out}")
    total, nshards = write_shards(
        sample_generator(), args.out, shard_size=args.shard_size
    )

    print(
        f"Wrote {total} samples across {nshards} shards to {Path(args.out).expanduser()}"
    )
    print(f"Output directory: {args.out}")
