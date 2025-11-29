#!/bin/bash

INPUT_FASTA=$1
TEMP_FASTA="prot_pipeline.fasta"
BOLTZ_OUTPUT_DIR=$2

BOLTZ_CACHE="/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db"
MSA_PATH="/n/home06/tbush/gfp_function_prediction/data/raw_data/gfp_msa_b5fdc_0.a3m"

THREADS=$SLURM_CPUS_PER_TASK 
# Set GPU specifications for mmseq2 and boltz calculations
export CUDA_VISIBLE_DEVICES=0
export NUM_GPU_DEVICES=1


echo "Loading modules..."
module load python/3.12.8-fasrc01 gcc/14.2.0-fasrc01 cuda/12.9.1-fasrc01 cudnn/9.10.2.21_cuda12-fasrc01

# Create the FASTA file for Boltz with MSA path in header
echo "Creating $TEMP_FASTA..."

# Read original header and sequence
ORIGINAL_HEADER=$(head -n1 "$INPUT_FASTA")
NEW_HEADER="${ORIGINAL_HEADER}${MSA_PATH}"
SEQUENCE=$(tail -n+2 "$INPUT_FASTA")
echo "$NEW_HEADER"
echo "$NEW_HEADER" > "$TEMP_FASTA"
echo "$SEQUENCE" >> "$TEMP_FASTA"

echo "Created Boltz FASTA file:"
echo "Sequence length: $(echo "$SEQUENCE" | wc -c)"

# ==========================================
# STEP 4: Run Boltz Prediction
# ==========================================
echo ""
echo "STEP 4: Running Boltz prediction..."
echo "----------------------------------------"

# Activate Boltz environment
mamba activate /n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz

# Create Boltz output directory
mkdir -p "$BOLTZ_OUTPUT_DIR"


echo "Command: boltz predict $TEMP_FASTA --cache $BOLTZ_CACHE --out_dir $BOLTZ_OUTPUT_DIR --devices $NUM_GPU_DEVICES --accelerator gpu"

# Start Boltz timing
boltz_start=$(date +%s)

# Run Boltz prediction
boltz predict "$TEMP_FASTA" --cache "$BOLTZ_CACHE" --out_dir "$BOLTZ_OUTPUT_DIR" --devices $NUM_GPU_DEVICES --accelerator gpu --recycling_steps 4  --diffusion_samples 10
python -c "import triton; print(triton.__version__)"
python -c "import boltz; print(boltz.__version__)"

# Check if Boltz succeeded
if [ $? -ne 0 ]; then
	    echo "ERROR: Boltz prediction failed!"
	        exit 1
	fi

	# Calculate Boltz runtime
	boltz_end=$(date +%s)
	boltz_runtime=$((boltz_end - boltz_start))
	boltz_hours=$(echo "scale=2; $boltz_runtime/3600" | bc)

	echo "Boltz prediction completed successfully!"
	echo "Boltz runtime: $boltz_runtime seconds (${boltz_hours} hours)"
