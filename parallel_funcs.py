from pathlib import Path
import numpy as np
from mpi4py import MPI

#Inspiration for the code from: https://github.com/TizianoCausin01/temporal_context/blob/main/python_scripts/src/parallel/parallel_funcs.py

#===Review parallel functions ===

def parallel_setup():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank
    size = comm.Get_size
    return comm, rank, size

def
