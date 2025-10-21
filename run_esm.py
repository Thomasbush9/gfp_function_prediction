#!/usr/bin/env python3
from pathlib import Path
import os, csv, hashlib, tempfile
from argparse import ArgumentParser
from typing import Union, Sequence, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from safetensors.torch import save_file, safe_open
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from utils.utils import load_seq_

MODEL = "esmc_600m"

# ---------- utils ----------
def seq_sha1(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()

def shard_dir(root: Path, key: str) -> Path:
    h = hashlib.sha1(key.encode()).hexdigest()
    return root / h[:2] / h[2:4]

def file_id_from_path(p: str) -> str:
    return Path(p).stem  # e.g., seq_11947

def save_one(out_dir: Path, seq_path: str, seq: str,
             logits: torch.Tensor, emb: torch.Tensor,
             dtype: torch.dtype = torch.float16,
             verify_existing: bool = True) -> Path:
    file_id = file_id_from_path(seq_path)
    d = shard_dir(out_dir, file_id); d.mkdir(parents=True, exist_ok=True)
    out_file = d / f"{file_id}.{MODEL}.safetensors"
    sha1 = seq_sha1(seq)

    if out_file.exists():
        if verify_existing:
            with safe_open(str(out_file), framework="pt", device="cpu") as f:
                meta = f.metadata()
            if meta.get("sha1") != sha1:
                raise RuntimeError(
                    f"Hash mismatch for {out_file}: {meta.get('sha1')} != {sha1} (src {seq_path})"
                )
        return out_file

    data = {
        "emb": emb.squeeze(0).to(dtype).contiguous(),      # [L,D]
        "logits": logits.squeeze(0).to(dtype).contiguous() # [L,V]
    }
    meta = {
        "model": MODEL,
        "src_path": seq_path,
        "file_id": file_id,
        "sha1": sha1,
        "seq_len": str(data["emb"].shape[0]),
        "dtype": "fp16" if dtype == torch.float16 else "fp32",
    }

    # atomic write
    with tempfile.NamedTemporaryFile(dir=d, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    save_file(data, str(tmp_path), metadata=meta)
    os.replace(tmp_path, out_file)
    return out_file

# ---------- embedding ----------
def embed_sequence(client: ESMC, sequence: Union[str, ESMProtein]):
    """Return (logits, embeddings) on CPU. Shapes: (1, L, V), (1, L, D)."""
    protein = sequence if isinstance(sequence, ESMProtein) else ESMProtein(sequence)
    enc = client.encode(protein)
    out = client.logits(enc, LogitsConfig(sequence=True, return_embeddings=True))
    return out.logits.sequence.detach().to("cpu"), out.embeddings.detach().to("cpu")

def batch_embed(
    client: ESMC,
    inputs: Sequence[Union[str, ESMProtein]],
    max_workers: int = 1,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    n = len(inputs)
    logits:     List[Any] = [None] * n
    embeddings: List[Any] = [None] * n

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(embed_sequence, client, seq): i for i, seq in enumerate(inputs)}
        for fut in tqdm(as_completed(futures), total=n, desc="Embedding"):
            i = futures[fut]
            try:
                logit, emb = fut.result()
                logits[i] = logit
                embeddings[i] = emb
            except Exception as e:
                logits[i] = e
                embeddings[i] = e

    errors = [(i, x) for i, x in enumerate(logits) if isinstance(x, Exception)]
    if errors:
        msgs = "\n".join([f"  idx {i}: {err}" for i, err in errors])
        raise RuntimeError(f"{len(errors)} sequences failed:\n{msgs}")

    logits = [t for t in logits if isinstance(t, torch.Tensor)]
    embeddings = [t for t in embeddings if isinstance(t, torch.Tensor)]
    return logits, embeddings

# ---------- main ----------
def main():
    ap = ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Text file: one FASTA path per line")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_workers", type=int, default=1, help="Threaded per-seq encode")
    ap.add_argument("--dtype", choices=["fp16","fp32"], default="fp16")
    ap.add_argument("--verify_existing", action="store_true", help="Verify sha1 if file exists")
    args = ap.parse_args()

    input_file = Path(args.inputs)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # offline-friendly envs
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    torch.set_grad_enabled(False)

    # model
    client = ESMC.from_pretrained(MODEL).to("cuda")

    # load paths & sequences
    with input_file.open() as f:
        paths = [line.strip() for line in f if line.strip()]

    proteins = []
    sequences = []
    for p in paths:
        seq = load_seq_(p)[0]
        sequences.append(seq)
        proteins.append(ESMProtein(seq))

    # embed
    logits_list, emb_list = batch_embed(client, proteins, max_workers=args.max_workers)

    # manifest (single-run; Slurm arrays can produce multiple and be merged later)
    manifest = output_dir / "manifest_run.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_id", "sha1", "file", "model", "src_path", "L", "dtype"])
        for seq_path, seq, logits, emb in tqdm(
            list(zip(paths, sequences, logits_list, emb_list)),
            desc="Saving", total=len(logits_list)
        ):
            out_file = save_one(
                output_dir, seq_path, seq, logits, emb,
                dtype=dtype, verify_existing=args.verify_existing
            )
            L = int(emb.squeeze(0).shape[0])
            w.writerow([file_id_from_path(seq_path), seq_sha1(seq), str(out_file),
                        MODEL, seq_path, L, args.dtype])

if __name__ == "__main__":
    main()
