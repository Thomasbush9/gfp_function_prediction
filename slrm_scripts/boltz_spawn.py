import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import runpy
runpy.run_module("boltz", run_name="__main__")
