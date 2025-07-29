#!/bin/bash

# Step 1: Define paths
MAIN_DIR=~/projects/gfp_function_prediction
DATA_DIR=$MAIN_DIR/data/raw_data
SCRIPT_DIR=$MAIN_DIR/scripts

# Step 2: Create data directory and download files
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "⬇️ Downloading dataset from Google Drive..."

# Replace with your actual file IDs
DATA_FILE_ID="YOUR_TSV_FILE_ID"       # amino_acid_genotypes_to_brightness.tsv
ORIGINAL_FILE_ID="YOUR_FASTA_FILE_ID" # P42212.fasta.txt

# Download using gdown
pip show gdown >/dev/null 2>&1 || pip install gdown
gdown "https://drive.google.com/uc?id=1ysB_unEXbgcGwQIHCCH0E2TpPlDa3eAm" -O raw_data.zip
unzip raw_data.zip
echo "✅ Files downloaded to $DATA_DIR"

# Step 3: Define input files
DATA_FILE="$DATA_DIR/amino_acid_genotypes_to_brightness.tsv"
ORIGINAL_FILE="$DATA_DIR/P42212.fasta.txt"

# Step 4: Run Python script with arguments
cd "$SCRIPT_DIR"
echo "🚀 Running generate_data.py..."
python generate_data.py --data "$DATA_FILE" --original "$ORIGINAL_FILE"

echo "✅ All done."

