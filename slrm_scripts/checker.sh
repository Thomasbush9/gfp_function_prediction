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

# find boltz output directory in the output directory
boltz_output_dir=$(find $ROOT_DIR -maxdepth 1 -name "*boltz_chunk*" -type d )
# get complete path combinging root and boltz
BOLTZ_OUTPUT_DIR=$boltz_output_dir

# get paths to file not processed
TOT_FILES_BOLTZ=$BOLTZ_OUTPUT_DIR/tot_filesboltz.txt
PROCESSED_PATHS_FILE=$BOLTZ_OUTPUT_DIR/processed_paths.txt

# find unprocessed seq IDs
UNPROCESSED_SEQ_IDS=$(mktemp)
comm -23 <(sort -u <(grep -oE 'seq_[0-9]+' $TOT_FILES_BOLTZ)) <(sort -u <(grep -oE 'seq_[0-9]+' $PROCESSED_PATHS_FILE)) > $UNPROCESSED_SEQ_IDS

# get full paths from tot_filesboltz.txt for unprocessed seq IDs
touch $BOLTZ_OUTPUT_DIR/unprocessed_paths.txt
while read seq_id; do
    grep "$seq_id" $TOT_FILES_BOLTZ
done < $UNPROCESSED_SEQ_IDS > $BOLTZ_OUTPUT_DIR/unprocessed_paths.txt
rm $UNPROCESSED_SEQ_IDS

# get number of unprocessed paths
NUM_UNPROCESSED_PATHS=$(wc -l < $BOLTZ_OUTPUT_DIR/unprocessed_paths.txt)

# print number of unprocessed paths
echo "Number of unprocessed paths: $NUM_UNPROCESSED_PATHS"

# print unprocessed paths
cat $BOLTZ_OUTPUT_DIR/unprocessed_paths.txt

# exit with error if there are unprocessed paths