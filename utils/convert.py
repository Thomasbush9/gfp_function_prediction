from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

from utils.utils import converter


def delete_sources(path: Path, src: Literal["fasta", "yaml"]):
    """Delete source files after conversion."""
    if path.is_file():
        path.unlink(missing_ok=True)
        return

    # Directory: delete files matching the source type
    if src == "yaml":
        exts = {".yaml", ".yml"}
    else:  # src == "fasta"
        exts = {".fasta", ".fa", ".faa", ".txt"}  # adjust to your real inputs

    for p in path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--src", choices=["fasta", "yaml"], required=True)
    parser.add_argument(
        "--delete", action="store_true", help="Delete source files after conversion"
    )
    args = parser.parse_args()

    path = Path(args.path)
    converter(str(path), args.src)

    if args.delete:
        delete_sources(path, args.src)
