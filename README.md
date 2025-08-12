# GFP Function Prediction from Mutation

This project aims to predict GFP functionality (fluorescence) from mutate sequences of proteins. We are going to compare different models such as [Boltz](https://github.com/jwohlwend/boltz/tree/main) and [OpenFold](https://github.com/aqlaboratory/openfold/tree/main).

We are going to compare several metrics and test a custom model to predict GOF or LOF.

## Project Structure

gfp_function_prediction/
    - bash_scripts/ # bash scripts for setup
    - scripts/ #python script for custom analysis
    - data/ # test data for boltz

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Thomasbush9/gfp_function_prediction
cd gfp_function_prediction
```

### 2. Create Environment

```bash
source bash_scripts/setup_env.sh
```

This will create a basic conda env named "boltz-enfer" and install Python 3.10

### 3. Install Boltz 2

```bash
source bash_scripts/dwn_boltz.sh
```

This will clone the Boltz 2 repository and install it (it will be necessary to apply hooks later)

### 4. Download and Generate the GFP Data

```bash
source bash_scripts/generate_data.sh
```
This will:

1. Create data, output directory
2. Download the .tsv with the mutated sequence and the original sequence in FASTA format
3. Generate YAML files required for boltz using 'generate_data.py' script, files are saved inside date_time directory inside data/

### Generate and Run predictions

```python
python run_deformation_analysis --Path to predictions --o output path
```
## TODO:

-[V ] Decide what to do with the MSA: either generate one for each sequence or change the first line

-[ ] Plan prediction run calibrating the memory constraints (currently with hidden representations: 0.055gb per predictions, full gfp: 3.025tb)

-[ ] Cluster script to cover the full dataset

-[ ] Build custom model for the function prediction

- [ ] Apply SE(3) Graph Attention Neural Networks
