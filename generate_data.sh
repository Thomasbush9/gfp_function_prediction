#!/bin/bash

# Step 1: Define paths
MAIN_DIR=~/projects/gfp_function_prediction  # change if needed
DATA_DIR=$MAIN_DIR/data/raw_data
SCRIPT_DIR=$MAIN_DIR/scripts

# Step 2: Create data directory and download
mkdir -p $DATA_DIR
cd $DATA_DIR

echo "⬇️ Downloading dataset..."
wget -q https://github.com/Thomasbush9/gfp_function_prediction/releases/download/v1.0/raw_data.zip
unzip -q raw_data.zip
echo "✅ Dataset extracted to $DATA_DIR"

# Step 3: Define input files
DATA_FILE="$DATA_DIR/amino_acid_genotypes_to_brightness.tsv"
ORIGINAL_FILE="$DATA_DIR/P42212.fasta.txt"

# Step 4: Run Python script with arguments
cd $SCRIPT_DIR
echo "🚀 Running generate_data.py..."
python generate_data.py --data "$DATA_FILE" --original "$ORIGINAL_FILE"

echo "✅ All done."

