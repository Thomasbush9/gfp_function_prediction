#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./setup_es_analysis_env.sh <REPO_URL>
#
# Example:
#   ./setup_es_analysis_env.sh https://github.com/your-org/your-repo.git

# Hardcoded repo URL
REPO_URL="https://github.com/mirabdi/PDAnalysis.git"

# Load cluster modules
module load python/3.12.8-fasrc01 gcc/14.2.0-fasrc01

# Where to create env + clone
ENV_PREFIX="$HOME/envs/es-analysis"
TARGET_SRC_DIR="$HOME/es-analysis-src"

echo "[+] Using environment prefix: $ENV_PREFIX"
echo "[+] Will clone repo into   : $TARGET_SRC_DIR"

# Ensure conda/mamba are available
if ! command -v conda >/dev/null; then
  echo "[!] conda not found. Load it with a module if needed." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! command -v mamba >/dev/null; then
  echo "[!] mamba not found. Install with: conda install -n base -c conda-forge mamba" >&2
  exit 1
fi

# Create env if missing
if [[ ! -d "$ENV_PREFIX" ]]; then
  echo "[+] Creating conda environment 'es-analysis'"
  mamba create -y -p "$ENV_PREFIX" python=3.10
else
  echo "[i] Environment already exists at $ENV_PREFIX"
fi

echo "[+] Activating environment"
conda activate "$ENV_PREFIX"

# Clone or update repo
if [[ -d "$TARGET_SRC_DIR/.git" ]]; then
  echo "[i] Repo already present — pulling latest."
  git -C "$TARGET_SRC_DIR" pull --ff-only
elif [[ -e "$TARGET_SRC_DIR" ]]; then
  echo "[!] $TARGET_SRC_DIR exists but is not a git repo. Remove it or choose another path." >&2
  exit 1
else
  echo "[+] Cloning $REPO_URL into $TARGET_SRC_DIR"
  git clone "$REPO_URL" "$TARGET_SRC_DIR"
fi

# Install requirements if available
if [[ -f "$TARGET_SRC_DIR/requirements.txt" ]]; then
  echo "[+] Installing requirements.txt"
  pip install -U pip
  pip install -r "$TARGET_SRC_DIR/requirements.txt"
elif [[ -f "$TARGET_SRC_DIR/setup.py" || -f "$TARGET_SRC_DIR/pyproject.toml" ]]; then
  echo "[+] Installing package in editable mode"
  pip install -U pip
  pip install -e "$TARGET_SRC_DIR"
else
  echo "[i] No requirements.txt or setup.py found — skipping dependency install."
fi

echo "[✓] Environment ready: es-analysis"
echo "[✓] Source cloned at: $TARGET_SRC_DIR"
echo
echo "Next time, activate with:"
echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\""
echo "  conda activate \"$ENV_PREFIX\""

