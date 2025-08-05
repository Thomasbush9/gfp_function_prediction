from collections import defaultdict

import numpy as np
import torch


def to_numpy_fp16(x):
    # Convert list to tensor if needed
    if isinstance(x, list):
        # If it's a list of tensors, stack them
        if all(isinstance(i, torch.Tensor) for i in x):
            x = torch.stack(x)
        else:
            # Otherwise, convert list of floats to tensor
            x = torch.tensor(x)

    return x.cpu().numpy().astype(np.float16)


class PairformerHookCollector:
    """Collector for pairformer hidden representations."""

    def __init__(self):
        self.hidden_reps = defaultdict(list)
        self.hooks = []

    def register_hooks(self, model):
        pairformer = (
            model.pairformer_module._orig_mod
            if getattr(model, "is_pairformer_compiled", False)
            else model.pairformer_module
        )

        def hook_fn(module, inputs, outputs):
            try:
                s, z = outputs
                self.hidden_reps["s"].append(s.detach().cpu())
                self.hidden_reps["z"].append(z.detach().cpu())
                print(f"[HOOK FIRED] s: {s.shape}, z: {z.shape}")
            except Exception as e:
                print(f"[HOOK ERROR] {e}")

        self.hooks.append(pairformer.register_forward_hook(hook_fn))

    def get(self):
        return dict(self.hidden_reps)

    def clear(self):
        self.hidden_reps.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


if __name__ == "__main__":
    pass
    # implementation:

    hidden = collector.get()

    # Save the tensors to file
    # torch.save(hidden["s"], out_dir / "hidden_s.pt")
    # torch.save(hidden["z"], out_dir / "hidden_z.pt")
    # collector.clear()
    np.save(out_dir / "hidden_s.npy", to_numpy_fp16(hidden["s"]))
    np.save(out_dir / "hidden_z.npy", to_numpy_fp16(hidden["z"]))

    collector.clear()
