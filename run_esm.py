from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from argparse import ArgumentParser
from pathlib import Path 

from utils.utils import load_seq_



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--fasta_list", type=str, required=True,
                        help="Path to .txt file where each line is a fasta file path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save outputs, one subdir per fasta")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read fasta paths from .txt file
    with open(args.fasta_list, "r") as f:
        fasta_paths = [line.strip() for line in f if line.strip()]

    client = ESMC.from_pretrained("esmc_300m").to("cuda")

    for fasta_path_str in fasta_paths:
        fasta_path = Path(fasta_path_str)
        seq, mapping_db_seq = load_seq_(fasta_path)
        protein = ESMProtein(sequence=seq)
        protein_tensor = client.encode(protein)
        logits_output = client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )

        # Prepare output subdirectory by unique sequence index provided in mapping_db_seq (use the first/only entry)
        seq_index = str(next(iter(mapping_db_seq.values())))
        fasta_output_dir = output_dir / seq_index
        fasta_output_dir.mkdir(parents=True, exist_ok=True)

        # Save logits and embeddings as .npy files
        logits_np = logits_output.logits.cpu().numpy()
        embeddings_np = logits_output.embeddings.cpu().numpy()

        import numpy as np  # local import for safe script-level use

        np.save(fasta_output_dir / "logits.npy", logits_np)
        np.save(fasta_output_dir / "embeddings.npy", embeddings_np)



