#!/bin/bash
set -euo pipefail

# Usage: ./organize_boltz_outputs.sh BASE_OUTPUT_DIR BOLTZ_CHUNKS_DIR
# Example: ./organize_boltz_outputs.sh /data/chunks_20240101_120000 /data/chunks_20240101_120000/boltz_chunks_20240101_120000

BASE_OUTPUT_DIR="${1:-}"
BOLTZ_CHUNKS_DIR="${2:-}"

if [[ -z "${BASE_OUTPUT_DIR}" || -z "${BOLTZ_CHUNKS_DIR}" ]]; then
  echo "Usage: $0 BASE_OUTPUT_DIR BOLTZ_CHUNKS_DIR"
  echo "  BASE_OUTPUT_DIR: Base output directory containing sequence directories (seq_idx/)"
  echo "  BOLTZ_CHUNKS_DIR: Directory containing boltz_output/ with predictions"
  exit 1
fi

# Normalize to absolute paths
BASE_OUTPUT_DIR="$(realpath -m "$BASE_OUTPUT_DIR")"
BOLTZ_CHUNKS_DIR="$(realpath -m "$BOLTZ_CHUNKS_DIR")"
BOLTZ_OUTPUT_BASE="${BOLTZ_CHUNKS_DIR}/boltz_output"

if [[ ! -d "$BOLTZ_OUTPUT_BASE" ]]; then
  echo "WARNING: Boltz output directory not found: $BOLTZ_OUTPUT_BASE"
  exit 0
fi

echo "==============================================="
echo "Organizing boltz outputs"
echo "  Base dir: $BASE_OUTPUT_DIR"
echo "  Boltz output: $BOLTZ_OUTPUT_BASE"
echo "==============================================="

# Find all prediction directories
# Pattern: boltz_output/chunk_X/boltz_results_chunk_X/predictions/seq_idx/
mapfile -d '' prediction_dirs < <(find "$BOLTZ_OUTPUT_BASE" -type d -path "*/predictions/*" -print0 | sort -z)

if (( ${#prediction_dirs[@]} == 0 )); then
  echo "No prediction directories found in $BOLTZ_OUTPUT_BASE"
  exit 0
fi

echo "Found ${#prediction_dirs[@]} prediction directories"

PROCESSED_COUNT=0
SKIPPED_COUNT=0

for pred_dir in "${prediction_dirs[@]}"; do
  # Extract sequence ID from directory name (handle both seq_idx and idx formats)
  PRED_DIR_NAME=$(basename "$pred_dir")
  
  # Extract numeric ID (remove 'seq_' prefix if present)
  if [[ "$PRED_DIR_NAME" =~ ^seq_ ]]; then
    SEQ_ID="${PRED_DIR_NAME#seq_}"
  else
    SEQ_ID="$PRED_DIR_NAME"
  fi
  
  # Skip if SEQ_ID is empty or invalid
  if [[ -z "$SEQ_ID" ]]; then
    echo "WARNING: Could not extract sequence ID from: $pred_dir"
    ((SKIPPED_COUNT++)) || true
    continue
  fi
  
  # Always use seq_ format for target directory to match existing structure
  # This ensures boltz outputs go into seq_idx/boltz/ matching the existing seq_idx/msa/ structure
  TARGET_DIR="${BASE_OUTPUT_DIR}/seq_${SEQ_ID}/boltz"
  mkdir -p "$TARGET_DIR"
  
  # Find all model files and determine the highest model number
  # Pattern: *_model_N.* where N is the model number
  HIGHEST_MODEL=-1
  
  while IFS= read -r -d '' file; do
    # Extract model number from filename (e.g., seq_34073_model_9.cif -> 9)
    if [[ "$(basename "$file")" =~ _model_([0-9]+)\. ]]; then
      MODEL_NUM="${BASH_REMATCH[1]}"
      if [[ "$MODEL_NUM" =~ ^[0-9]+$ ]] && (( MODEL_NUM > HIGHEST_MODEL )); then
        HIGHEST_MODEL=$MODEL_NUM
      fi
    fi
  done < <(find "$pred_dir" -type f -print0)
  
  if (( HIGHEST_MODEL < 0 )); then
    echo "WARNING: No model files found in $pred_dir"
    ((SKIPPED_COUNT++)) || true
    continue
  fi
  
  # Copy only files for the highest model number
  FILES_COPIED=0
  while IFS= read -r -d '' file; do
    if [[ "$(basename "$file")" =~ _model_${HIGHEST_MODEL}\. ]]; then
      cp "$file" "$TARGET_DIR/"
      ((FILES_COPIED++)) || true
    fi
  done < <(find "$pred_dir" -type f -print0)
  
  if (( FILES_COPIED > 0 )); then
    echo "Organized $SEQ_ID: copied ${FILES_COPIED} files for model ${HIGHEST_MODEL} -> $TARGET_DIR"
    ((PROCESSED_COUNT++)) || true
  else
    echo "WARNING: No files copied for $SEQ_ID (model ${HIGHEST_MODEL})"
    ((SKIPPED_COUNT++)) || true
  fi
done

echo "==============================================="
echo "Organization complete"
echo "  Processed: $PROCESSED_COUNT sequences"
echo "  Skipped: $SKIPPED_COUNT sequences"
echo "==============================================="

