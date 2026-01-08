from .utils import generate_mutation_dataset, load_seq_
from argparse import ArgumentParser
from pathlib import Path



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seq", type=str)
    parser.add_argument("--n", type=int)
    parser.add_argument("--out_path", type=str)


    args = parser.parse_args()
    seq, _ = load_seq_(args.seq, fasta=False)
    df = generate_mutation_dataset(seq, args.n)
    df.to_csv(Path(args.out_path)/"GARBRB3.csv", seq="\t")





