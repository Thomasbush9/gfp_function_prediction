from torch.utils.data import Dataset
from pathlib import Path
import torch
from torch_geometric.data import Data, DataLoader
class ShardedListOfDictDataset(Dataset):
    """
    Expects shards like shard_00000.pt, shard_00001.pt, ...
    Each shard is a Python list of sample dicts.
    Each sample dict contains: pos, x_struct, x_esm, edge_index, y
    Keeps a small cache of loaded shards in RAM.
    """

    def __init__(self, shards_dir, pattern="shard_*.pt", map_location="cpu", cache_shards=1):
        self.paths = sorted(Path(shards_dir).expanduser().glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"No shards matching {pattern} in {shards_dir}")

        self.map_location = map_location
        self.cache_shards = cache_shards

        # Build (path, offset) index by scanning shard lengths once
        self.index = []
        for p in self.paths:
            shard_list = torch.load(p, map_location="cpu", weights_only=False)
            self.index.extend([(p, i) for i in range(len(shard_list))])

        # tiny LRU cache
        self._cache = {}       # path -> list[dict]
        self._cache_order = [] # LRU order

    def __len__(self):
        return len(self.index)

    def _get_shard(self, path):
        if path in self._cache:
            return self._cache[path]
        shard_list = torch.load(path, map_location=self.map_location, weights_only=False)
        self._cache[path] = shard_list
        self._cache_order.append(path)
        if len(self._cache_order) > self.cache_shards:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return shard_list

    def __getitem__(self, idx):
        p, off = self.index[idx]
        shard_list = self._get_shard(p)
        s = shard_list[off]  # dict

        y = s["y"]
        if torch.is_tensor(y) and y.dim() == 0:
            y = y.view(1)

        return Data(
            pos=s["pos"],
            x_struct=s["x_struct"],
            x_esm=s["x_esm"],
            edge_index=s["edge_index"],
            y=y
        )

if __name__ == "__main__":
    dir = "/Users/thomasbush/Downloads/shards"
    ds = ShardedListOfDictDataset(dir, cache_shards=1)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    device = torch.device("mps")
    for batch in loader:
        batch = batch.to(device)
        print(batch)
        break

