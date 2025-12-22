from argparse import ArgumentParser
from pathlib import Path

from mpi4py import MPI
from tqdm import tqdm

# Script to generate es in parallel using mpi:


def parallel_setup():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank
    size = comm.Get_size
    return comm, rank, size
