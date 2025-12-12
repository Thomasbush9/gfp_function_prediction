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

BOLTZ_CHECK_SKIP=false
if [ -z "$boltz_output_dir" ] || [ ! -d "$boltz_output_dir" ]; then
    echo "Warning: No boltz_chunks directory found in $ROOT_DIR"
    echo "  Skipping boltz checking, will only check ES outputs if manifests exist"
    BOLTZ_CHECK_SKIP=true
else
    BOLTZ_OUTPUT_DIR="$boltz_output_dir"
    echo "Using boltz chunks directory: $BOLTZ_OUTPUT_DIR"
fi

# Check boltz outputs if boltz_chunks directory exists
if [ "$BOLTZ_CHECK_SKIP" = false ]; then
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
    echo "Number of unprocessed boltz paths: $NUM_UNPROCESSED_PATHS"

    # print unprocessed paths
    if [ "$NUM_UNPROCESSED_PATHS" -gt 0 ]; then
        cat $ROOT_DIR/boltz_unprocessed_paths.txt
    fi

    # Launch retry workflow for unprocessed paths if any exist
    if [ "$NUM_UNPROCESSED_PATHS" -gt 0 ]; then
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
        BOLTZ_RECYCLING_STEPS=$(python3 "${SCRIPT_DIR}/parse_config.py" "$CONFIG_FILE" "boltz.recycling_steps")
        BOLTZ_DIFFUSION_SAMPLES=$(python3 "${SCRIPT_DIR}/parse_config.py" "$CONFIG_FILE" "boltz.diffusion_samples")

        if [ -z "$MAX_FILES_PER_JOB" ] || [ -z "$ARRAY_MAX_CONCURRENCY" ] || [ -z "$BOLTZ_RECYCLING_STEPS" ] || [ -z "$BOLTZ_DIFFUSION_SAMPLES" ]; then
            echo "Error: Failed to read boltz configuration from $CONFIG_FILE"
            exit 1
        fi

        echo "Max files per job: $MAX_FILES_PER_JOB"
        echo "Array max concurrency: $ARRAY_MAX_CONCURRENCY"
        echo "Recycling steps: $BOLTZ_RECYCLING_STEPS"
        echo "Diffusion samples: $BOLTZ_DIFFUSION_SAMPLES"

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
        else
            echo "Submitting ${NUM_TASKS} array tasks (max concurrent: ${ARRAY_MAX_CONCURRENCY})..."

            BOLTZ_SCRIPT="${SCRIPT_DIR}/run_boltz_array.slrm"
            if [[ ! -f "$BOLTZ_SCRIPT" ]]; then
                echo "ERROR: Boltz array script not found at $BOLTZ_SCRIPT"
            else
                # Submit array job
                ARRAY_JOB_ID="$(
                    sbatch --parsable \
                        --array=1-"$NUM_TASKS"%${ARRAY_MAX_CONCURRENCY} \
                        --export=ALL,MANIFEST="$MANIFEST",BASE_OUTPUT_DIR="$NEW_BOLTZ_CHUNKS_DIR",BOLTZ_RECYCLING_STEPS="$BOLTZ_RECYCLING_STEPS",BOLTZ_DIFFUSION_SAMPLES="$BOLTZ_DIFFUSION_SAMPLES" \
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
            fi
        fi

        echo "==============================================="
        echo "Boltz retry workflow launched successfully"
        echo "==============================================="
    else
        echo "All boltz paths processed."
    fi
fi

# Get script directory if not already set
if [ -z "${SCRIPT_DIR:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [ -z "${CONFIG_FILE:-}" ]; then
    CONFIG_FILE="${SCRIPT_DIR}/pipeline_config.yaml"
fi

# Check ES outputs if manifests exist
CIF_MANIFEST="${ROOT_DIR}/cif_manifest.txt"
OUT_MANIFEST="${ROOT_DIR}/out_manifest.txt"

if [[ -f "$CIF_MANIFEST" ]] && [[ -f "$OUT_MANIFEST" ]]; then
    echo ""
    echo "==============================================="
    echo "Checking ES outputs"
    echo "==============================================="
    
    # Find unprocessed CIF files (those without corresponding CSV outputs)
    ES_UNPROCESSED_TMP=$(mktemp)
    : > "$ES_UNPROCESSED_TMP"
    
    # Read both manifests into arrays for matching
    mapfile -t cif_lines < "$CIF_MANIFEST"
    mapfile -t out_lines < "$OUT_MANIFEST"
    
    # Create a lookup map for output paths by array_id and protein_id
    declare -A out_path_map
    for out_line in "${out_lines[@]}"; do
        array_id=$(echo "$out_line" | cut -f1)
        out_path=$(echo "$out_line" | cut -f2-)
        protein_id=$(basename "$out_path" _output.csv)
        key="${array_id}:${protein_id}"
        out_path_map["$key"]="$out_path"
    done
    
    # Check each CIF file
    for cif_line in "${cif_lines[@]}"; do
        array_id=$(echo "$cif_line" | cut -f1)
        cif_path=$(echo "$cif_line" | cut -f2-)
        protein_id=$(basename "$cif_path" .cif)
        key="${array_id}:${protein_id}"
        
        if [[ -n "${out_path_map[$key]:-}" ]]; then
            out_path="${out_path_map[$key]}"
            if [[ ! -f "$out_path" ]]; then
                # Output CSV doesn't exist, add to unprocessed
                printf "%s\t%s\n" "$array_id" "$cif_path" >> "$ES_UNPROCESSED_TMP"
            fi
        else
            # No corresponding output path found, add to unprocessed
            printf "%s\t%s\n" "$array_id" "$cif_path" >> "$ES_UNPROCESSED_TMP"
        fi
    done
    
    NUM_UNPROCESSED_CIFS=$(wc -l < "$ES_UNPROCESSED_TMP")
    
    if [[ "$NUM_UNPROCESSED_CIFS" -gt 0 ]]; then
        echo "Number of unprocessed CIF files: $NUM_UNPROCESSED_CIFS"
        echo "Unprocessed CIF files:"
        cat "$ES_UNPROCESSED_TMP"
        
        echo ""
        echo "==============================================="
        echo "Launching ES retry for unprocessed CIF files"
        echo "==============================================="
        
        # Read ES configuration
        ES_SCRIPT_DIR=$(python3 "${SCRIPT_DIR}/parse_config.py" "$CONFIG_FILE" "es.script_dir")
        ES_WT_PATH=$(python3 "${SCRIPT_DIR}/parse_config.py" "$CONFIG_FILE" "es.wt_path")
        ES_ARRAY_MAX_CONCURRENCY=$(python3 "${SCRIPT_DIR}/parse_config.py" "$CONFIG_FILE" "es.array_max_concurrency")
        
        if [[ -z "$ES_SCRIPT_DIR" ]] || [[ -z "$ES_WT_PATH" ]] || [[ -z "$ES_ARRAY_MAX_CONCURRENCY" ]]; then
            echo "WARNING: Missing ES configuration, skipping ES retry"
            echo "  Required: es.script_dir, es.wt_path, es.array_max_concurrency"
        else
            ES_ARRAY_MAX_CONCURRENCY="${ES_ARRAY_MAX_CONCURRENCY:-10}"
            
            echo "ES script dir: $ES_SCRIPT_DIR"
            echo "ES WT path: $ES_WT_PATH"
            echo "ES array max concurrency: $ES_ARRAY_MAX_CONCURRENCY"
            
            # Create new manifests with only unprocessed CIFs
            ES_CIF_MANIFEST_NEW="${ROOT_DIR}/cif_manifest_retry_$(date +%Y%m%d_%H%M%S).txt"
            ES_OUT_MANIFEST_NEW="${ROOT_DIR}/out_manifest_retry_$(date +%Y%m%d_%H%M%S).txt"
            : > "$ES_CIF_MANIFEST_NEW"
            : > "$ES_OUT_MANIFEST_NEW"
            
            # Build new manifests from unprocessed CIFs
            while IFS=$'\t' read -r array_id cif_path; do
                protein_id=$(basename "$cif_path" .cif)
                key="${array_id}:${protein_id}"
                
                if [[ -n "${out_path_map[$key]:-}" ]]; then
                    # Use existing output path from original manifest
                    out_path="${out_path_map[$key]}"
                    printf "%s\t%s\n" "$array_id" "$cif_path" >> "$ES_CIF_MANIFEST_NEW"
                    printf "%s\t%s\n" "$array_id" "$out_path" >> "$ES_OUT_MANIFEST_NEW"
                else
                    # Generate expected output path
                    array_dir="${ROOT_DIR}/${array_id}"
                    es_dir="$array_dir/es"
                    mkdir -p "$es_dir"
                    out_csv="$es_dir/${protein_id}_output.csv"
                    printf "%s\t%s\n" "$array_id" "$cif_path" >> "$ES_CIF_MANIFEST_NEW"
                    printf "%s\t%s\n" "$array_id" "$out_csv" >> "$ES_OUT_MANIFEST_NEW"
                fi
            done < "$ES_UNPROCESSED_TMP"
            
            NUM_TASKS=$(wc -l < "$ES_CIF_MANIFEST_NEW")
            
            if [[ "$NUM_TASKS" -gt 0 ]]; then
                echo "Submitting ${NUM_TASKS} ES array tasks (max concurrent: ${ES_ARRAY_MAX_CONCURRENCY})..."
                
                # run_es_array.slrm is in the slrm_scripts directory (where checker_boltz.sh is)
                # SCRIPT_DIR here refers to slrm_scripts directory
                ES_ARRAY_SCRIPT="${SCRIPT_DIR}/run_es_array.slrm"
                if [[ ! -f "$ES_ARRAY_SCRIPT" ]]; then
                    echo "ERROR: ES array script not found at $ES_ARRAY_SCRIPT"
                else
                    # Note: We pass ES_SCRIPT_DIR (from config, where main.py is) as SCRIPT_DIR env var
                    # to run_es_array.slrm, which uses it to cd into and run main.py
                    ES_ARRAY_JOB_ID="$(
                        sbatch --parsable \
                            --array=1-"$NUM_TASKS"%${ES_ARRAY_MAX_CONCURRENCY} \
                            --export=ALL,CIF_MANIFEST="$ES_CIF_MANIFEST_NEW",OUT_MANIFEST="$ES_OUT_MANIFEST_NEW",SCRIPT_DIR="$ES_SCRIPT_DIR",WT_PATH="$ES_WT_PATH" \
                            "$ES_ARRAY_SCRIPT"
                    )"
                    
                    if [[ -n "$ES_ARRAY_JOB_ID" ]]; then
                        echo "Submitted ES array job ${ES_ARRAY_JOB_ID} with ${NUM_TASKS} tasks."
                        echo "CIF manifest: $ES_CIF_MANIFEST_NEW"
                        echo "OUT manifest: $ES_OUT_MANIFEST_NEW"
                        echo "ES script dir (for main.py): $ES_SCRIPT_DIR"
                    else
                        echo "ERROR: Failed to submit ES array job"
                    fi
                fi
            fi
        fi
    else
        echo "All ES CIF files processed."
    fi
    
    rm -f "$ES_UNPROCESSED_TMP"
else
    echo "ES manifests not found, skipping ES check."
    echo "  Looking for: $CIF_MANIFEST and $OUT_MANIFEST"
fi
