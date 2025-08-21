#!/usr/bin/env bash
set -euo pipefail

# --- user inputs ---

INPUT_FASTA="/n/home06/tbush/gfp_function_prediction/data/trial_10/seq_18277.fasta"
OUT_DIR="/n/home06/tbush/gfp_function_prediction/data/outputs/msa"
MMSEQ2_DB="/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db"
THREADS="${THREADS:-16}"

# ColabFold CLI in PATH
export PATH="/n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz/localcolabfold/colabfold-conda/bin:$PATH"
export COLABFOLD_DB="/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/colabfold_db"

mkdir -p "$OUT_DIR"

echo "[MSA] Input: $INPUT_FASTA"
echo "[MSA] Out dir: $OUT_DIR"
echo "[MSA] Threads: $THREADS"

# 1) Run colabfold_search (writes mmseqs DBs, then unpacks text .a3m)
colabfold_search "$INPUT_FASTA" "$MMSEQ2_DB" "$OUT_DIR" --thread "$THREADS" --gpu 1

# 2) Find unpacked text .a3m (not the mmseqs DB "a3m" files)
mapfile -t A3M_TXT < <(find "$OUT_DIR" -maxdepth 1 -type f -name "*.a3m" | sort)
if (( ${#A3M_TXT[@]} == 0 )); then
  echo "[MSA] ERROR: no text .a3m found in $OUT_DIR" >&2
  exit 10
fi
A3M="${A3M_TXT[0]}"
echo "[MSA] Found A3M: $A3M"

# 3) Sanitize: remove NULs and blank lines
CLEAN_A3M="${OUT_DIR}/msa.clean.a3m"
tr -d '\000' < "$A3M" | grep -v '^[[:space:]]*$' > "$CLEAN_A3M"

# 4) Validate
file -L "$CLEAN_A3M" | grep -qi 'text' || { echo "[MSA] ERROR: cleaned A3M not text"; exit 11; }
head -n1 "$CLEAN_A3M" | grep -q '^>' || { echo "[MSA] ERROR: cleaned A3M missing '>' header"; exit 12; }

echo "[MSA] Clean A3M: $CLEAN_A3M"
# Print just the path so the next script can capture it
echo "$CLEAN_A3M"
BASH
chmod +x msa_generate.sh

