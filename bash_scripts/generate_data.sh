#!/bin/bash

# Step 1: Automatically infer paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MAIN_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$MAIN_DIR/data/raw_data"
SCRIPT_PY="$MAIN_DIR/utils"

# Step 2: Create data directory and download files
mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || { echo "❌ Cannot cd into $DATA_DIR"; exit 1; }

echo "⬇️  Downloading dataset from Google Drive..."

# Ensure gdown is installed
pip show gdown >/dev/null 2>&1 || pip install gdown

# Replace with your actual ZIP file ID (single archive containing both files)
# FILE_ID="1ysB_unEXbgcGwQIHCCH0E2TpPlDa3eAm"
FILE_ID="1e9cRYt4ctQ1vPb64UUhR3IubauKqRzen"
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O raw_data.zip

# Unzip the archive
unzip -q raw_data.zip
echo "✅ Files downloaded to $DATA_DIR"

# Step 3: Define input files
DATA_FILE="$DATA_DIR/amino_acid_genotypes_to_brightness.tsv"
ORIGINAL_FILE="$DATA_DIR/P42212.fasta.txt"
MSA_FILE="$DATA_DIR/gfp_msa_b5fdc_0.a3m"

# Step 4: Run Python script with arguments
cd "$SCRIPT_PY" || { echo "❌ Cannot cd into $SCRIPT_PY"; exit 1; }
echo "🚀 Running generate_data.py..."
python generate_data.py --data "$DATA_FILE" --original "$ORIGINAL_FILE" --msa "$MSA_FILE" --file_type cluster

echo "✅ All done."

