#!/usr/bin/env bash
set -euo pipefail

#input paths:
SEQ_FASTA="/n/home06/tbush/gfp_function_prediction/data/trial_10/seq_18277.fasta"
A3M="/n/home06/tbush/gfp_function_prediction/data/outputs/msa/uniref.a3m"
OUT_DIR="/n/home06/tbush/gfp_function_prediction/data/outputs/boltz"

mkdir -p "$OUT_DIR"

# conda activation 
if [-f "/n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/etc/profile.d/conda.sh" ]; then
	source /n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/etc/profile.d/conda.sh
fi
BOLTZ_ENV="/n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz"
mamba activate "$BOLTZ_ENV"

# Small sanity
[ -s "$SEQ_FASTA" ] || { echo "seq fasta missing: $SEQ_FASTA"; exit 20; }
[ -s "$A3M" ] || { echo "a3m missing/empty: $A3M"; exit 21; }
file -L "$A3M" | grep -qi text || { echo "a3m not text: $A3M"; exit 22; }

# Build temporary FASTA with MSA in header
SAMPLE="$(basename "$SEQ_FASTA")"; SAMPLE="${SAMPLE%.txt}"; SAMPLE="${SAMPLE%.fa}"; SAMPLE="${SAMPLE%.fasta}"
TMP_FASTA="$(mktemp --suffix=.fasta)"
SEQ="$(tail -n +2 "$SEQ_FASTA")"
echo ">${SAMPLE}|${A3M}" > "$TMP_FASTA"
echo "$SEQ" >> "$TMP_FASTA"
echo "[BOLTZ] tmp fasta: $TMP_FASTA"

# (Optional) quick versions to catch env mismatches
python - <<'PY'
import sys, torch
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_avail:", torch.cuda.is_available())
try:
    import triton; print("triton:", triton.__version__)
except Exception as e:
    print("triton import FAILED:", e)
PY

# Run boltz
boltz predict "$TMP_FASTA" \
	  --cache /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db \
	    --out_dir "$OUT_DIR" \
	      --devices 1 --accelerator gpu


# Require outputs
if ! find "$OUT_DIR" -maxdepth 1 -type f | grep -q .; then
	  echo "no outputs written to $OUT_DIR"; exit 23
  fi
  echo "[BOLTZ] OK: outputs in $OUT_DIR"
