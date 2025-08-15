#!/bin/bash
# Run this once per setup

# Load Python module from cluster environment
module purge
module load python/3.12.8-fasrc01  # change to your cluster's available python module

echo "[+] Creating conda environment for Boltz"
mamba create --name boltz-infer python=3.10 -y

# Ensure mamba/conda activation works in non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate environment
mamba activate boltz-infer

echo "[+] Installing requirements"
# pip install -r requirements.txt  # optional if you use this


