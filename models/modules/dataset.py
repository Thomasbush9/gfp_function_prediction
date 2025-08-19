# dataset_sharded_pt.py
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.loader import (
    DataLoader as PyGDataLoader,  # use this loader for PyG Data
)


class ShardedPTDataset(Dataset):
    """
    Expects shards like shard_00000.pt, shard_00001.pt, ...
    Each shard is a list of torch_geometric.data.Data objects.
    Keeps a small cache of loaded shards in RAM (default: 1).
    """

    def __init__(
        self, shards_dir, pattern="shard_*.pt", map_location="cpu", cache_shards=1
    ):
        self.paths = sorted(Path(shards_dir).expanduser().glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"No shards matching {pattern} in {shards_dir}")
        self.map_location = map_location
        self.cache_shards = cache_shards

        # Build (path, offset) index by scanning shard lengths once
        self.index = []
        for p in self.paths:
            lst = torch.load(
                p, map_location="cpu", weights_only=False
            )  # loads ONE shard to get its length
            self.index.extend([(p, i) for i in range(len(lst))])

        # tiny LRU cache of shards
        self._cache = {}  # path -> list[Data]
        self._cache_order = []  # paths in LRU order

    def __len__(self):
        return len(self.index)

    def _get_shard(self, path):
        if path in self._cache:
            return self._cache[path]
        data_list = torch.load(path, map_location=self.map_location, weights_only=False)
        self._cache[path] = data_list
        self._cache_order.append(path)
        if len(self._cache_order) > self.cache_shards:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return data_list

    def __getitem__(self, idx):
        p, off = self.index[idx]
        shard = self._get_shard(p)
        return shard[off]


# test it:
if __name__ == "__main__":
    dir = "/Users/thomasbush/Downloads/shards"
    ds = ShardedPTDataset(dir)
    loader = DataLoader(ds, batch_size=16, shuffle=True)  # consider other param
    device = torch.device("mps")
    for batch in loader:
        print(batch)
