#!/bin/bash

#enter the output directory
ROOT_DIR=$1

#check if the output directory exists
if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Output directory $ROOT_DIR does not exist."
    exit 1
fi

#check if the output directory is empty
if [ -z "$(ls -A $ROOT_DIR)" ]; then
    echo "Error: Output directory $ROOT_DIR is empty."
    exit 1
fi

# find most recent boltz output directory in the output directory (by modification time)
boltz_output_dir=$(find "$ROOT_DIR" -maxdepth 1 -type d -name "*boltz_chunk*" -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$boltz_output_dir" ] || [ ! -d "$boltz_output_dir" ]; then
    echo "Error: No boltz_chunks directory found in $ROOT_DIR"
    exit 1
fi

BOLTZ_OUTPUT_DIR="$boltz_output_dir"
echo "Using boltz chunks directory: $BOLTZ_OUTPUT_DIR"

# get paths to file not processed
TOT_FILES_BOLTZ=$BOLTZ_OUTPUT_DIR/tot_filesboltz.txt
PROCESSED_PATHS_FILE=$BOLTZ_OUTPUT_DIR/processed_paths.txt

# find unprocessed seq IDs
UNPROCESSED_SEQ_IDS=$(mktemp)
comm -23 <(sort -u <(grep -oE 'seq_[0-9]+' $TOT_FILES_BOLTZ)) <(sort -u <(grep -oE 'seq_[0-9]+' $PROCESSED_PATHS_FILE)) > $UNPROCESSED_SEQ_IDS

# get full paths from tot_filesboltz.txt for unprocessed seq IDs
touch $ROOT_DIR/boltz_unprocessed_paths.txt
while read seq_id; do
    grep "$seq_id" $TOT_FILES_BOLTZ
done < $UNPROCESSED_SEQ_IDS > $ROOT_DIR/boltz_unprocessed_paths.txt
rm $UNPROCESSED_SEQ_IDS

# get number of unprocessed paths
NUM_UNPROCESSED_PATHS=$(wc -l < $ROOT_DIR/boltz_unprocessed_paths.txt)

# print number of unprocessed paths
echo "Number of unprocessed paths: $NUM_UNPROCESSED_PATHS"

# print unprocessed paths
cat $ROOT_DIR/boltz_unprocessed_paths.txt

# if no unprocessed paths, exit successfully
if [ "$NUM_UNPROCESSED_PATHS" -eq 0 ]; then
    echo "All paths processed. Exiting."
    exit 0
fi

# Launch retry workflow for unprocessed paths
echo "==============================================="
echo "Launching boltz retry for unprocessed paths"
echo "==============================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/pipeline_config.yaml"

# Read configuration
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

MAX_FILES_PER_JOB=$(python3 "${SCRIPT_DIR}/parse_config.py" "$CONFIG_FILE" "boltz.max_files_per_job")
ARRAY_MAX_CONCURRENCY=$(python3 "${SCRIPT_DIR}/parse_config.py" "$CONFIG_FILE" "boltz.array_max_concurrency")

if [ -z "$MAX_FILES_PER_JOB" ] || [ -z "$ARRAY_MAX_CONCURRENCY" ]; then
    echo "Error: Failed to read boltz configuration from $CONFIG_FILE"
    exit 1
fi

echo "Max files per job: $MAX_FILES_PER_JOB"
echo "Array max concurrency: $ARRAY_MAX_CONCURRENCY"

# Create new timestamped boltz_chunks directory
TS="$(date +%Y%m%d_%H%M%S)"
NEW_BOLTZ_CHUNKS_DIR="${ROOT_DIR}/boltz_chunks_${TS}"
mkdir -p "$NEW_BOLTZ_CHUNKS_DIR"

echo "Creating new boltz chunks directory: $NEW_BOLTZ_CHUNKS_DIR"

# Read unprocessed paths into array
mapfile -t unprocessed_paths < "$ROOT_DIR/boltz_unprocessed_paths.txt"
total=${#unprocessed_paths[@]}

# Calculate number of chunks needed
NUM_CHUNKS=$(( (total + MAX_FILES_PER_JOB - 1) / MAX_FILES_PER_JOB ))

echo "Creating ${NUM_CHUNKS} chunk directories (max ${MAX_FILES_PER_JOB} files per chunk)..."

# Create chunk directories and copy yaml files
for ((i=0; i<NUM_CHUNKS; i++)); do
    start=$(( i * MAX_FILES_PER_JOB ))
    end=$(( start + MAX_FILES_PER_JOB ))
    (( end > total )) && end=$total
    (( start >= end )) && continue

    CHUNK_DIR="${NEW_BOLTZ_CHUNKS_DIR}/chunk_${i}"
    mkdir -p "$CHUNK_DIR"

    # Copy yaml files to chunk directory
    for ((j=start; j<end; j++)); do
        if [ -f "${unprocessed_paths[j]}" ]; then
            cp "${unprocessed_paths[j]}" "$CHUNK_DIR/"
        fi
    done

    file_count=$(( end - start ))
    echo "Created chunk_${i} with ${file_count} files"
done

# Create boltz_tot_files.txt and processed_paths.txt
echo "Creating boltz_tot_files.txt and processed_paths.txt..."

TOT_FILES_BOLTZ_NEW="${NEW_BOLTZ_CHUNKS_DIR}/boltz_tot_files.txt"
PROCESSED_PATHS_FILE_NEW="${NEW_BOLTZ_CHUNKS_DIR}/processed_paths.txt"

# Write all unprocessed paths to boltz_tot_files.txt
: > "$TOT_FILES_BOLTZ_NEW"
for path in "${unprocessed_paths[@]}"; do
    echo "$path" >> "$TOT_FILES_BOLTZ_NEW"
done
sort -u "$TOT_FILES_BOLTZ_NEW" -o "$TOT_FILES_BOLTZ_NEW"

# Create empty processed_paths.txt
: > "$PROCESSED_PATHS_FILE_NEW"

echo "Created $TOT_FILES_BOLTZ_NEW with $(wc -l < "$TOT_FILES_BOLTZ_NEW") total paths"
echo "Created empty $PROCESSED_PATHS_FILE_NEW"

# Build manifest and submit array job
echo "Building manifest and submitting array job..."

MANIFEST="${NEW_BOLTZ_CHUNKS_DIR}/chunkdirs.manifest"
: > "$MANIFEST"

# Include only non-empty chunk directories
while IFS= read -r -d '' chunk_dir; do
    if [[ -d "$chunk_dir" ]] && [[ -n "$(find "$chunk_dir" -maxdepth 1 -name "*.yaml" -type f)" ]]; then
        realpath -s "$chunk_dir" >> "$MANIFEST"
    fi
done < <(find "$NEW_BOLTZ_CHUNKS_DIR" -maxdepth 1 -type d -name 'chunk_*' -print0 | sort -z)

NUM_TASKS=$(wc -l < "$MANIFEST")
if (( NUM_TASKS == 0 )); then
    echo "No non-empty chunk directories found; nothing to submit."
    exit 0
fi

echo "Submitting ${NUM_TASKS} array tasks (max concurrent: ${ARRAY_MAX_CONCURRENCY})..."

BOLTZ_SCRIPT="${SCRIPT_DIR}/run_boltz_array.slrm"
if [[ ! -f "$BOLTZ_SCRIPT" ]]; then
    echo "ERROR: Boltz array script not found at $BOLTZ_SCRIPT"
    exit 1
fi

# Submit array job
ARRAY_JOB_ID="$(
    sbatch --parsable \
        --array=1-"$NUM_TASKS"%${ARRAY_MAX_CONCURRENCY} \
        --export=ALL,MANIFEST="$MANIFEST",BASE_OUTPUT_DIR="$NEW_BOLTZ_CHUNKS_DIR" \
        "$BOLTZ_SCRIPT"
)"

echo "Submitted array job ${ARRAY_JOB_ID} with ${NUM_TASKS} tasks."
echo "Chunks dir: $NEW_BOLTZ_CHUNKS_DIR"
echo "Manifest:   $MANIFEST"

# Submit post-processing job
ORGANIZE_SCRIPT="${SCRIPT_DIR}/run_boltz_organize.slrm"
if [[ -f "$ORGANIZE_SCRIPT" ]]; then
    echo ""
    echo "Submitting post-processing job to organize boltz outputs..."
    ORGANIZE_JOB_ID=$(sbatch --parsable \
        --dependency=afterok:${ARRAY_JOB_ID} \
        --export=ALL,BASE_OUTPUT_DIR="$ROOT_DIR",BOLTZ_CHUNKS_DIR="$NEW_BOLTZ_CHUNKS_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
        "$ORGANIZE_SCRIPT")
    
    if [[ -n "$ORGANIZE_JOB_ID" ]]; then
        echo "Submitted organize job ${ORGANIZE_JOB_ID} (depends on array job ${ARRAY_JOB_ID})"
    else
        echo "WARNING: Failed to submit organize job"
    fi
else
    echo "WARNING: Organize script not found at $ORGANIZE_SCRIPT, skipping post-processing"
fi

echo "==============================================="
echo "Retry workflow launched successfully"
echo "==============================================="
