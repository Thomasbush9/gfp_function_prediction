#!/bin/bash
# Run this once per setup

echo "[+] Creating conda environment for Boltz"
conda create -y -n boltz-infer python=3.10
conda activate boltz-infer

echo "[+] Installing requirements"
pip install -r requirements.txt  # optional if you use this

