#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=256GB
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomasbush52@gmail.com
#SBATCH --output=/n/home06/tbush/job_logs/%x.%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# --- Load modules ---
module load python/3.12.8-fasrc01 gcc/14.2.0-fasrc01
source "$(conda info --base)/etc/profile.d/conda.sh"
# prefer prefix activation
ES_ENV_PREFIX="${ES_ENV_PREFIX:-/n/home06/tbush/envs/es-analysis}"
conda activate "$ES_ENV_PREFIX"


ROOT_DIR="${1:-}"
SCRIPT_DIR="${2:-}"
WT_PATH="${3:-}"

ROOT_DIR="$(realpath "$ROOT_DIR")"
SCRIPT_DIR="$(realpath "$SCRIPT_DIR")"
WT_PATH="$(realpath "$WT_PATH")"

# Get directory containing this script (slrm_scripts directory)
ES_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ES_ARRAY_SCRIPT="${ES_SCRIPT_DIR}/run_es_array.slrm"
find "$ROOT_DIR" -type f -name "*.cif" -printf "%p\n" \
| awk -F'[_/]' '
{
  seq = $NF;               # filename
  match(seq, /seq_[0-9]+/);
  seq_id = substr(seq, RSTART, RLENGTH);

  match(seq, /model_[0-9]+/);
  model = substr(seq, RSTART+6, RLENGTH-6);

  if (!(seq_id in max) || model > max[seq_id]) {
    max[seq_id] = model;
    file[seq_id] = $0;
  }
}
END {
  for (s in file)
    print file[s];
}' \
| xargs -r realpath > "$ROOT_DIR/paths.txt"

cd "$SCRIPT_DIR"
srun python main.py --parallel True --protA "$WT_PATH" --path_list "$ROOT_DIR/paths.txt" --out_dir "$ROOT_DIR" --method all --min_plddt 70 --lddt_cutoffs 0.125 0.25 0.5 1

