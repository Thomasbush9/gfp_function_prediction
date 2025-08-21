#!/bin/bash
set -euo pipefail

# Usage: ./split_and_submit.sh INPUT_DIR N OUTPUT_PARENT_DIR
# Example: ./split_and_submit.sh /data/images 5 /data/jobs
# Optional: set ARRAY_MAX_CONCURRENCY (default 10)

INPUT_DIR="${1:-}"
N="${2:-}"
OUTPUT_PARENT_DIR="${3:-}"
ARRAY_MAX_CONCURRENCY="${ARRAY_MAX_CONCURRENCY:-10}"

if [[ -z "${INPUT_DIR}" || -z "${N}" || -z "${OUTPUT_PARENT_DIR}" ]]; then
  echo "Usage: $0 INPUT_DIR N OUTPUT_PARENT_DIR"
  exit 1
fi
if ! [[ "$N" =~ ^[0-9]+$ && "$N" -ge 1 ]]; then
  echo "N must be a positive integer"
  exit 1
fi

# Normalize to absolute paths
INPUT_DIR="$(realpath -m "$INPUT_DIR")"
OUTPUT_PARENT_DIR="$(realpath -m "$OUTPUT_PARENT_DIR")"

# Make timestamped output directory that will also hold logs + manifest
TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_PARENT_DIR}/chunks_${TS}"
mkdir -p "$OUTPUT_DIR"

echo "Writing chunk files to: $OUTPUT_DIR"

# Collect files (absolute paths), null-safe & sorted
mapfile -d '' -t files < <(find "$INPUT_DIR" -maxdepth 1 -type f -print0 | sort -z)
total=${#files[@]}
if (( total == 0 )); then
  echo "No files found in $INPUT_DIR"
  exit 1
fi

# Compute chunk size (ceil division)
chunk_size=$(( (total + N - 1) / N ))

# Split into N chunks (skip empties if N > total)
for ((i=0; i<N; i++)); do
  start=$(( i * chunk_size ))
  end=$(( start + chunk_size ))
  (( end > total )) && end=$total
  (( start >= end )) && continue  # skip empty chunk

  out="${OUTPUT_DIR}/id_${i}.txt"
  : > "$out"
  for ((j=start; j<end; j++)); do
    # Write absolute paths (they already are if INPUT_DIR was absolute)
    printf '%s\n' "${files[j]}" >> "$out"
  done
  echo "Wrote $(wc -l < "$out") paths -> $out"
done

# -------- Build manifest (stable order) & submit array --------
echo "Building manifest and submitting array..."

MANIFEST="${OUTPUT_DIR}/filelist.manifest"
: > "$MANIFEST"

# Deterministic: sort by name; include only non-empty chunk files
while IFS= read -r -d '' f; do
  [[ -s "$f" ]] && realpath -s "$f" >> "$MANIFEST"
done < <(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'id_*.txt' -print0 | sort -z)

NUM_TASKS=$(wc -l < "$MANIFEST")
if (( NUM_TASKS == 0 )); then
  echo "No non-empty id_*.txt chunk files found; nothing to submit."
  exit 0
fi

echo "Submitting ${NUM_TASKS} array tasks (max concurrent: ${ARRAY_MAX_CONCURRENCY})..."

# --parsable returns just the job ID so we can print it nicely
ARRAY_JOB_ID="$(
  sbatch --parsable \
    --array=1-"$NUM_TASKS"%${ARRAY_MAX_CONCURRENCY} \
    --export=ALL,MANIFEST="$MANIFEST",BASE_OUTPUT_DIR="$OUTPUT_DIR" \
    single_prediction_array.slrm
)"

echo "Submitted array job ${ARRAY_JOB_ID} with ${NUM_TASKS} tasks."
echo "Chunks dir: $OUTPUT_DIR"
echo "Manifest:   $MANIFEST"

