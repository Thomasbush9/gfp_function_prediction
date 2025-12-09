#!/bin/bash
set -euo pipefail

# Usage: ./split_and_run_msa.sh INPUT_DIR N OUTPUT_PARENT_DIR [ARRAY_MAX_CONCURRENCY]
# Example: ./split_and_run_msa.sh /data/fasta 5 /data/jobs 100

INPUT_DIR="${1:-}"
N="${2:-}"
OUTPUT_PARENT_DIR="${3:-}"
ARRAY_MAX_CONCURRENCY="${4:-100}"

if [[ -z "${INPUT_DIR}" || -z "${N}" || -z "${OUTPUT_PARENT_DIR}" ]]; then
  echo "Usage: $0 INPUT_DIR N OUTPUT_PARENT_DIR [ARRAY_MAX_CONCURRENCY]"
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

# Collect .fasta and .fa files (absolute paths), null-safe & sorted
mapfile -d '' -t files < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -name "*.fasta" -o -name "*.fa" \) -print0 | sort -z)
total=${#files[@]}
if (( total == 0 )); then
  echo "No .fasta or .fa files found in $INPUT_DIR"
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
    # Write absolute paths
    printf '%s\n' "${files[j]}" >> "$out"
  done
  echo "Wrote $(wc -l < "$out") paths -> $out"
done

# -------- Create total_paths.txt and processed_paths.txt --------
echo "Creating total_paths.txt and processed_paths.txt..."

TOTAL_PATHS_FILE="${OUTPUT_DIR}/total_paths.txt"
PROCESSED_PATHS_FILE="${OUTPUT_DIR}/processed_paths.txt"

# Collect all paths from all chunk files into total_paths.txt, sorted
: > "$TOTAL_PATHS_FILE"
while IFS= read -r -d '' f; do
  [[ -s "$f" ]] && cat "$f" >> "$TOTAL_PATHS_FILE"
done < <(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'id_*.txt' -print0 | sort -z)
sort -u "$TOTAL_PATHS_FILE" -o "$TOTAL_PATHS_FILE"

# Create empty processed_paths.txt
: > "$PROCESSED_PATHS_FILE"

echo "Created $TOTAL_PATHS_FILE with $(wc -l < "$TOTAL_PATHS_FILE") total paths"
echo "Created empty $PROCESSED_PATHS_FILE"

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

# Get script directory to find run_msa_array.slrm
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSA_SCRIPT="${SCRIPT_DIR}/run_msa_array.slrm"

if [[ ! -f "$MSA_SCRIPT" ]]; then
  echo "ERROR: MSA script not found at $MSA_SCRIPT"
  exit 1
fi

# --parsable returns just the job ID so we can print it nicely
# Export ORIGINAL_FASTA_DIR and SCRIPT_DIR so post-processing can access them
ARRAY_JOB_ID="$(
  sbatch --parsable \
    --array=1-"$NUM_TASKS"%${ARRAY_MAX_CONCURRENCY} \
    --export=ALL,MANIFEST="$MANIFEST",BASE_OUTPUT_DIR="$OUTPUT_DIR",ORIGINAL_FASTA_DIR="$INPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
    "$MSA_SCRIPT"
)"

echo "Submitted array job ${ARRAY_JOB_ID} with ${NUM_TASKS} tasks."
echo "Chunks dir: $OUTPUT_DIR"
echo "Manifest:   $MANIFEST"

