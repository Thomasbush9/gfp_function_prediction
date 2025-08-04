import torch
import click
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import numpy as np
import traceback
import torch.nn.functional as F

from boltz.model.models.boltz2 import Boltz2
from boltz.data import const


class PairformerHookCollector:
    """Collects pairformer outputs (s, z) using forward hooks."""
    def __init__(self):
        self.hidden_reps = defaultdict(list)
        self.hooks = []

    def register_hooks(self, model):
        """Attach hook to pairformer."""
        if getattr(model, "is_pairformer_compiled", False):
            pairformer_module = model.pairformer_module._orig_mod
        else:
            pairformer_module = model.pairformer_module

        def pairformer_hook(module, input, output):
            try:
                s, z = output
                self.hidden_reps["pairformer_s"].append(s.detach().cpu())
                self.hidden_reps["pairformer_z"].append(z.detach().cpu())
                print(f"[HOOK FIRED] s: {s.shape}, z: {z.shape}")
            except Exception as e:
                print(f"[HOOK ERROR] {e}")

        self.hooks.append(pairformer_module.register_forward_hook(pairformer_hook))

    def clear(self):
        self.hidden_reps.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_representations(self) -> Dict[str, List[torch.Tensor]]:
        return dict(self.hidden_reps)


def load_model_with_hooks(checkpoint_path):
    """Loads Boltz2 model without overriding constructor kwargs (fixes key errors)."""
    try:
        model = Boltz2.load_from_checkpoint(
            checkpoint_path,
            strict=False,
            map_location="cpu",  # or "cuda" if needed
        )
        print("✅ Model loaded from checkpoint")

        # Set prediction parameters manually
        model.predict_args = {
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "max_parallel_samples": 5,
        }
        return model

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


@click.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path())
def main(checkpoint, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_with_hooks(checkpoint)
    model.eval()

    # Register hook
    collector = PairformerHookCollector()
    collector.register_hooks(model)

    # Dummy test batch
    batch_size, seq_len = 1, 50
    token_index = torch.randint(0, 20, (batch_size, seq_len))
    num_res_types = len(const.residue_types)

    test_batch = {
        "token_index": token_index,
        "res_type": F.one_hot(token_index.clamp_max(num_res_types - 1), num_classes=num_res_types).float(),
        "token_pad_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "atom_pad_mask": torch.ones(batch_size, seq_len, 3, dtype=torch.bool),
        "token_bonds": torch.zeros(batch_size, seq_len, seq_len),
        "type_bonds": torch.zeros(batch_size, seq_len, seq_len, dtype=torch.long),
        "mol_type": torch.zeros(batch_size, seq_len),
        "affinity_token_mask": torch.zeros(batch_size, seq_len, dtype=torch.bool),
        "affinity_mw": torch.ones(batch_size),
        "coords": torch.randn(batch_size, 1, seq_len, 3),
        "atom_resolved_mask": torch.ones(batch_size, seq_len, 3, dtype=torch.bool),
        "frames_idx": torch.zeros(batch_size, 1, seq_len),
        "frame_resolved_mask": torch.ones(batch_size, 1, seq_len, dtype=torch.bool),
        "pdb_id": ["dummy_protein"],
        "idx_dataset": torch.zeros(batch_size, dtype=torch.long),
    }

    try:
        collector.clear()

        with torch.no_grad():
            predictions = model(
                feats=test_batch,
                recycling_steps=model.predict_args["recycling_steps"],
                num_sampling_steps=model.predict_args["sampling_steps"],
                diffusion_samples=model.predict_args["diffusion_samples"],
                max_parallel_samples=model.predict_args["max_parallel_samples"],
                run_confidence_sequentially=True,
            )

        hidden_reps = collector.get_representations()
        print(f"\n✅ Hook captured {len(hidden_reps['pairformer_s'])} steps.")

        # Save output
        torch.save(
            {
                "predictions": predictions,
                "hidden_reps": hidden_reps,
                "test_batch_info": {"batch_size": batch_size, "seq_len": seq_len},
            },
            out_dir / "full_output.pt",
        )
        torch.save(hidden_reps, out_dir / "pairformer_reps.pt")

        print(f"\n💾 Saved hidden reps to: {out_dir/'pairformer_reps.pt'}")

        print("\n📦 Prediction keys:")
        for k, v in predictions.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {type(v)}")

    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
        traceback.print_exc()
    finally:
        collector.remove_hooks()


if __name__ == "__main__":
    main()
