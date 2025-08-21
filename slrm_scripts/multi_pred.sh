#!/bin/bash
set -euo pipefail

# Usage: ./split_and_submit.sh INPUT_DIR N OUTPUT_PARENT_DIR
# Example: ./split_and_submit.sh /data/images 5 /data/jobs

INPUT_DIR="${1:-}"
N="${2:-}"
OUTPUT_PARENT_DIR="${3:-}"

if [[ -z "${INPUT_DIR}" || -z "${N}" || -z "${OUTPUT_PARENT_DIR}" ]]; then
  echo "Usage: $0 INPUT_DIR N OUTPUT_PARENT_DIR"
  exit 1
fi

if ! [[ "$N" =~ ^[0-9]+$ && "$N" -ge 1 ]]; then
  echo "N must be a positive integer"
  exit 1
fi

# Make timestamped output directory
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
    printf '%s\n' "${files[j]}" >> "$out"
  done
  echo "Wrote $(wc -l < "$out") paths -> $out"
done


# Submit sbatch for each chunk file
echo "Submitting Slurm jobs..."
for txt in "$OUTPUT_DIR"/id_*.txt; do
  [[ -s "$txt" ]] || continue  # skip if empty (belt & suspenders)
  # Submit and capture job id
  submit_out="$(sbatch single_prediction.slrm "$txt" "$OUTPUT_DIR")"
  job_id="$(awk '{print $NF}' <<<"$submit_out")"
  echo "Submitted $txt -> Job $job_id"
done
echo "All done."


