#!/bin/bash
set -euo pipefail

# Usage: ./run_msa_then_boltz.sh INPUT_DIR N OUTPUT_PARENT_DIR MAX_FILES_PER_JOB [ARRAY_MAX_CONCURRENCY]
# Example: ./run_msa_then_boltz.sh /data/fasta 5 /data/jobs 10 100

INPUT_DIR="${1:-}"
N="${2:-}"
OUTPUT_PARENT_DIR="${3:-}"
MAX_FILES_PER_JOB="${4:-}"
ARRAY_MAX_CONCURRENCY="${5:-100}"

if [[ -z "${INPUT_DIR}" || -z "${N}" || -z "${OUTPUT_PARENT_DIR}" || -z "${MAX_FILES_PER_JOB}" ]]; then
  echo "Usage: $0 INPUT_DIR N OUTPUT_PARENT_DIR MAX_FILES_PER_JOB [ARRAY_MAX_CONCURRENCY]"
  echo "  INPUT_DIR: Directory containing original .fasta files"
  echo "  N: Number of chunks for MSA generation"
  echo "  OUTPUT_PARENT_DIR: Parent directory for output"
  echo "  MAX_FILES_PER_JOB: Maximum number of .yaml files per boltz array job"
  echo "  ARRAY_MAX_CONCURRENCY: Maximum concurrent array tasks (default: 100)"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSA_SCRIPT="${SCRIPT_DIR}/split_and_run_msa.sh"
POST_PROCESS_SCRIPT="${SCRIPT_DIR}/post_process_msa.slrm"
BOLTZ_SCRIPT="${SCRIPT_DIR}/split_and_run_boltz.sh"

# Normalize paths
INPUT_DIR="$(realpath -m "$INPUT_DIR")"
OUTPUT_PARENT_DIR="$(realpath -m "$OUTPUT_PARENT_DIR")"

echo "==============================================="
echo "Orchestrating MSA generation -> Post-processing -> Boltz prediction"
echo "==============================================="
echo "Input dir: $INPUT_DIR"
echo "Output parent dir: $OUTPUT_PARENT_DIR"
echo ""

# Step 1: Launch MSA array job
echo "Step 1: Launching MSA generation..."
MSA_JOB_OUTPUT=$("$MSA_SCRIPT" "$INPUT_DIR" "$N" "$OUTPUT_PARENT_DIR" "$ARRAY_MAX_CONCURRENCY")

# Extract MSA job ID and output directory from the output
MSA_JOB_ID=$(echo "$MSA_JOB_OUTPUT" | grep -oP 'Submitted array job \K[0-9]+' || echo "")
OUTPUT_DIR=$(echo "$MSA_JOB_OUTPUT" | grep -oP 'Chunks dir: \K[^ ]+' || echo "")

if [[ -z "$MSA_JOB_ID" || -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: Failed to extract MSA job ID or output directory"
  echo "MSA output: $MSA_JOB_OUTPUT"
  exit 1
fi

echo "  MSA job ID: ${MSA_JOB_ID}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Note: Post-processing (FASTA->YAML) will run automatically at the end of MSA job"
echo ""

# Step 2: Launch boltz array job that depends on MSA completion
# Post-processing is now integrated into the MSA job, so boltz depends directly on MSA
echo "Step 2: Launching boltz prediction (depends on MSA job ${MSA_JOB_ID})..."

BOLTZ_WRAPPER="${SCRIPT_DIR}/run_boltz_wrapper.slrm"
if [[ ! -f "$BOLTZ_WRAPPER" ]]; then
  echo "ERROR: Boltz wrapper script not found at $BOLTZ_WRAPPER"
  exit 1
fi

# Submit boltz wrapper job with dependency on MSA completion
# Post-processing is integrated into MSA job, so we depend on MSA job directly
# Export SCRIPT_DIR so wrapper can find split_and_run_boltz.sh
BOLTZ_WRAPPER_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:${MSA_JOB_ID} \
  --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",MAX_FILES_PER_JOB="$MAX_FILES_PER_JOB",ARRAY_MAX_CONCURRENCY="$ARRAY_MAX_CONCURRENCY",SCRIPT_DIR="$SCRIPT_DIR" \
  "$BOLTZ_WRAPPER")

# Note: The actual boltz array job ID will be submitted by the wrapper script
# We track the wrapper job ID here
BOLTZ_JOB_ID="$BOLTZ_WRAPPER_JOB_ID"

if [[ -z "$BOLTZ_JOB_ID" ]]; then
  echo "ERROR: Failed to submit boltz job"
  exit 1
fi

echo "  Boltz job ID: ${BOLTZ_JOB_ID}"
echo ""

echo "==============================================="
echo "All jobs submitted successfully:"
echo "  MSA job: ${MSA_JOB_ID} (includes post-processing: FASTA->YAML)"
echo "  Boltz job: ${BOLTZ_JOB_ID} (depends on MSA job)"
echo "==============================================="
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo "  squeue -j ${MSA_JOB_ID},${BOLTZ_JOB_ID}"

