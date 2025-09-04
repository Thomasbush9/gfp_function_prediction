#!/bin/bash
set -euo pipefail

# Usage: ./collect_and_submit.sh ROOT_DIR
# Example: ./collect_and_submit.sh /data/chunks_20250829_121523

ROOT_DIR="${1:-}"
SCRIPT_DIR="${2:-}"
WT_PATH="${3:-}"
ARRAY_MAX_CONCURRENCY="${ARRAY_MAX_CONCURRENCY:-10}"

if [[ -z "$ROOT_DIR" || -z "$SCRIPT_DIR" || -z "$WT_PATH" ]]; then
  echo "Usage: $0 ROOT_DIR SCRIPT_DIR WT_PATH"
  exit 1
fi

ROOT_DIR="$(realpath "$ROOT_DIR")"
SCRIPT_DIR="$(realpath "$SCRIPT_DIR")"
WT_PATH="$(realpath "$WT_PATH")"

# Two manifests: CIF inputs and future output CSVs
CIF_MANIFEST="${ROOT_DIR}/cif_manifest.txt"
OUT_MANIFEST="${ROOT_DIR}/out_manifest.txt"
: > "$CIF_MANIFEST"
: > "$OUT_MANIFEST"

echo "Scanning for CIF files under $ROOT_DIR ..."
for array_dir in "$ROOT_DIR"/*/; do
  [[ -d "$array_dir" ]] || continue
  boltz_dir="$array_dir/boltz"
  [[ -d "$boltz_dir" ]] || { echo "skip: no boltz in $array_dir"; continue; }

  # ensure es/ exists
  es_dir="$array_dir/es"
  mkdir -p "$es_dir"

  # find all CIFs in this boltz
  find "$boltz_dir" -type f -name '*.cif' | sort | while read -r cif; do
    array_id="$(basename "$array_dir")"
    protein_id="$(basename "$cif" .cif)"

    # input manifest
    printf "%s\t%s\n" "$array_id" "$cif" >> "$CIF_MANIFEST"

    # output manifest: one CSV per protein in es/
    out_csv="$es_dir/${protein_id}_output.csv"
    printf "%s\t%s\n" "$array_id" "$out_csv" >> "$OUT_MANIFEST"
  done
done

NUM_TASKS=$(wc -l < "$CIF_MANIFEST")
if (( NUM_TASKS == 0 )); then
  echo "No CIFs found; nothing to submit."
  exit 0
fi

echo "Found $NUM_TASKS CIF files total."
echo "CIF manifest : $CIF_MANIFEST"
echo "OUT manifest : $OUT_MANIFEST"

ARRAY_JOB_ID="$(
  sbatch --parsable \
    --array=1-"$NUM_TASKS"%${ARRAY_MAX_CONCURRENCY} \
    --export=ALL,CIF_MANIFEST="$CIF_MANIFEST",OUT_MANIFEST="$OUT_MANIFEST",SCRIPT_DIR="$SCRIPT_DIR",WT_PATH="$WT_PATH" \
    run_es_array.slrm
)"

echo "Submitted array job ${ARRAY_JOB_ID} with ${NUM_TASKS} tasks."

