# torch
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch import optim
# torch geometric for gnn:
import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn
# import utils libraries:
from tqdm import tqdm
from pathlib import Path
import os

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats:Tensor, adj_matrix:Tensor)->Tensor:


