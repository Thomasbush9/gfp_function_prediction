#!/bin/bash

# Get the directory where the script resides
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MAIN_DIR="$(dirname "$SCRIPT_DIR")"

echo "[+] Cloning Boltz repository into $MAIN_DIR"
cd "$MAIN_DIR" || { echo "❌ Failed to cd into $MAIN_DIR"; exit 1; }

git clone https://github.com/jwohlwend/boltz.git

echo "[+] Installing Boltz in editable mode with CUDA support"
cd boltz || { echo "❌ Failed to cd into boltz"; exit 1; }
pip install -e .[cuda]

