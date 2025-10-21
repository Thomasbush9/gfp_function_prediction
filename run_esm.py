from pathlib import Path
import os
from argparse import ArgumentParser
from tqdm import tqdm
from utils.utils import load_seq_
from typing import Union, Sequence, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import pandas as pd
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


def embed_sequence(client: ESMC, sequence: Union[str, ESMProtein]):
    """Return (logits, embeddings) on CPU. Shapes: (1, L, V), (1, L, D)."""
    protein = sequence if isinstance(sequence, ESMProtein) else ESMProtein(sequence)
    enc = client.encode(protein)
    out = client.logits(enc, LogitsConfig(sequence=True, return_embeddings=True))
    # move to CPU to avoid GPU-memory blowup and ensure cat compatibility
    return out.logits.sequence.detach().to("cpu"), out.embeddings.detach().to("cpu")

# --- batch wrapper (threaded; set max_workers=1 for safest CUDA use) ---
def batch_embed(
    client: ESMC,
    inputs: Sequence[Union[str, ESMProtein]],
    max_workers: int = 1,   # GPU models are often not thread-safe; 1 is safest
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Runs embed_sequence for each input. Preserves order, returns lists of tensors.
    """
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

    # check for errors
    errors = [(i, x) for i, x in enumerate(logits) if isinstance(x, Exception)]
    if errors:
        msgs = "\n".join([f"  idx {i}: {err}" for i, err in errors])
        raise RuntimeError(f"{len(errors)} sequences failed:\n{msgs}")

    # type narrowing
    logits = [t for t in logits if isinstance(t, torch.Tensor)]
    embeddings = [t for t in embeddings if isinstance(t, torch.Tensor)]
    return logits, embeddings

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--inputs", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()
    input_file = args.inputs
    output_dir = Path(args.out_dir)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    #define the model:
    client = ESMC.from_pretrained("esmc_600m").to("cuda")
    
    #load list of sequences:
    with open(input_file) as f:
        paths = [line.strip() for line in f]
    # converts sequences to protein objects:
    proteins = [ESMProtein(load_seq_(path)[0]) for path in paths]
    logits, embeddings = batch_embed(client, proteins)


    

        
