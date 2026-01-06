from .utils import generate_mutation_dataset, load_seq_
from pathlib import Path
import yaml
from argparse import ArgumentParser



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seq", type=str)
    parser.add_argument("--n", type=int)


    args = parser.parse_args()
    seq, _ = load_seq_(args.seq, fasta=False)
    print(seq)
    df = generate_mutation_dataset(seq, args.n)
    print(df.head(4))





