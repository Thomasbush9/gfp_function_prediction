#!/bin/bash

# Get the directory where the script resides
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MAIN_DIR="$(dirname "$SCRIPT_DIR")"

echo "[+] Cloning Protein Deformation Analysis Library repository into $MAIN_DIR"
cd "$MAIN_DIR" || { echo "❌ Failed to cd into $MAIN_DIR"; exit 1; }

git clone https://github.com/mirabdi/PDAnalysis

echo "[+] Installing Protein Deformation Analysis Library in editable mode with CUDA support"
cd PDAnalysis || { echo "❌ Failed to cd into PDA"; exit 1; }

python setup.py install

