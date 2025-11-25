#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 DIR N OUT_DIR WORKER_SLRM PYTHON_SCRIPT [-- py-args...]"
  exit 1
fi

DIR="$1"; N="$2"; OUT_DIR="$3"; WORKER="$4"; PY="$5"
shift 5

# Optional Python args after a literal --
PY_ARGS=()
if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then shift; fi
  PY_ARGS=("$@")
fi

# Sanity checks
[[ -d "$DIR" ]] || { echo "ERROR: DIR not found: $DIR"; exit 2; }
[[ -f "$WORKER" ]] || { echo "ERROR: WORKER_SLRM not found: $WORKER"; exit 3; }
[[ -f "$PY" ]] || { echo "ERROR: PYTHON_SCRIPT not found: $PY"; exit 4; }

# HF cache (so weights load on GPU nodes)
export HF_HOME="${HF_HOME:-/n/home06/tbush/hf_cache}"
echo "HF_HOME=${HF_HOME}"

mkdir -p "$OUT_DIR" /n/home06/tbush/job_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
CHUNK_DIR="${OUT_DIR}/chunks_${STAMP}"
MASTER_LIST="${CHUNK_DIR}/master.list"
mkdir -p "$CHUNK_DIR"
: > "$MASTER_LIST"

echo "[1/4] Scanning FASTA files in: $DIR"
find -L "$DIR" -maxdepth 1 -type f -name '*.fasta' -print > "$MASTER_LIST"
NUM_FILES=$(wc -l < "$MASTER_LIST" | tr -d ' ')
echo "  Found $NUM_FILES FASTA files"
if [[ "$NUM_FILES" -eq 0 ]]; then
  echo "No FASTA files found. Exiting."
  exit 0
fi

echo "[2/4] Splitting into up to $N chunks..."
# Split into nearly equal chunks; handles N > NUM_FILES gracefully
split -d -n r/"$N" --additional-suffix=.list "$MASTER_LIST" "${CHUNK_DIR}/chunk_"
# Normalize names to chunk00, chunk01, ...
i=0
for f in "${CHUNK_DIR}"/chunk_*; do
  mv "$f" "$(printf "%s/chunk%02d.list" "$CHUNK_DIR" "$i")"
  ((i++))
done
NUM_CHUNKS=$i
echo "  Made $NUM_CHUNKS chunks at: $CHUNK_DIR"

if [[ "$NUM_CHUNKS" -eq 0 ]]; then
  echo "Unexpected: split produced 0 chunks."
  exit 5
fi

ARRAY_SPEC="0-$((NUM_CHUNKS - 1))"
echo "[3/4] Submitting array: $ARRAY_SPEC"
echo "  Worker: $WORKER"
echo "  Python: $PY"
echo "  OutDir: $OUT_DIR"

# IMPORTANT: show the job id back to you
JID=$(sbatch --parsable --array="$ARRAY_SPEC" "$WORKER" "$CHUNK_DIR" "$OUT_DIR" "$PY" -- "${PY_ARGS[@]}")
echo "[4/4] Submitted batch job: $JID"
echo "$JID" > "${OUT_DIR}/last_submission.txt"

echo
echo "Monitor with:"
echo "  squeue -j $JID"
echo "  tail -f /n/home06/tbush/job_logs/esm_embed.${JID}_*.out  # as tasks start"

