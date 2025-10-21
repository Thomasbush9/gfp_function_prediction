#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 DIR N WORKER_SLRM PYTHON_SCRIPT [-- py-args...]"
  exit 1
fi 

DIR="$1";
N="$2"; 
OUT_DIR="$3";
shift 3
# WORKER="$4";
# PYTHON_SCRIPT="$5";
# shift 3
#
#we define the args for the python scritp 
# PY_ARGS=()
# if [[$# -gt 0 ]]; then
#   if [["$1"=="--"]]; then shift; fi 
#   PY_ARGS=($@)
# fi 
#
# make a working folder to hold the chunk list for each array
STAMP="$(date +%Y%m%d_%H%M%S)"
CHUNK_DIR="${OUT_DIR}/chunks_${STAMP}"
mkdir -p "$CHUNK_DIR"

# build a newline separeted list of files: 
MASTER_LIST="${CHUNK_DIR}/master.list"
: > "$MASTER_LIST"

find -L "$DIR" -maxdepth 1 -type f -name "*.fasta" -print > "$MASTER_LIST"
#check that there is at least one file 
NUM_FILES=$(wc -l < "$MASTER_LIST" | tr -d " ")
if [[ "$NUM_FILES" -eq 0 ]]; then 
  echo "No fasta files found in: $DIR"
  exit 1
fi 

# split the files into equal chunks: 

split -d -n r/$N --additional-suffix=.list "$MASTER_LIST" "${CHUNK_DIR}/chunk_"
i=0
for f in "${CHUNK_DIR}"/chunk_*; do 
  mv "$f" "$(printf "%s/chunk%02d.list" "$CHUNK_DIR" "$i")"
  ((i++))
done 

# here submit the job
echo "Division completed: $NUM_FILES submited in $CHUNK_DIR"
# sbatch --array=0-$((N-1)) "$WORKER" "$CHUNK_DIR" "$PYTHON_SCRIPT" 

