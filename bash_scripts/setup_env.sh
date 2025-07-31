#!/bin/bash
# Run this once per setup

#!/bin/bash
# Run this once per setup

echo "[+] Creating conda environment for Boltz"
mamba create -y -n boltz-infer python=3.10

# Ensure conda can be activated in non-interactive shell
source $(conda info --base)/etc/profile.d/conda.sh
mamba activate boltz-infer

echo "[+] Installing requirements"
pip install -r requirements.txt  # optional if you use this

