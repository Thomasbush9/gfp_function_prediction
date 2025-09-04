# GFP Function Prediction from Mutation

This project aims to predict GFP functionality (fluorescence) from mutate sequences of proteins. We are going to compare different models such as [Boltz](https://github.com/jwohlwend/boltz/tree/main) and [OpenFold](https://github.com/aqlaboratory/openfold/tree/main).

We are going to compare several metrics and test a custom model to predict GOF or LOF.

## Project Structure

gfp_function_prediction/
    - bash_scripts/ # bash scripts for setup
    - scripts/ #python script for custom analysis
    - data/ # test data for boltz
    - models/ # contains the ML model for function prediction
    - slrm_scripts/ # scripts for kempner
    - notebooks/
    - utils/ # utility functions scripts


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

## Kempner Cluster Instructions:

### Set-up Env and Dataset Generation:

1. Start an interactive session, like:

```{bash}
salloc --partition=kempner_requeue --account=kempner_dev --ntasks=1 --cpus-per-task=16 --mem=16G --gres=gpu:1 --time=00-03:00:00
```

2. Load python and conda modules, and run from the project dir:

```{bash}
source bash_scripts/setup_env.sh
# download the raw data and generate .fasta dir with 50k sequences:
source bash_scripts/generate_data.sh type
```
Where,
-type: is the type of data that you generates, options are: fasta, yaml and cluster

3. Optional, create a sub-sample of the data:

```{bash}
python utils/boltz_pred_test.py --dataset --n --data_dir --seed
```

Where,

- dataset: is the path to the .tsv original dataset
- n: number of sequence in the sample
- data_dir: path to the directory containing the full data

You can set the number of array concurrency inside the script or by doing:
```{bash}
export ARRAY_MAX_CONCURRENCY=
```
It generates a balanced sample (same num_mutation distribution) in a new directory called: data_dir_data_timestamp.

4. Converting a dataset to another file format, you can:

```{bash}
python utils/converted.py --path --src
# it converts the files in the path dir to the src file extension
```

## Generating Boltz Predictions on the Kempner Cluster

To generate the Boltz predictions on the Kempner cluster you run from the login node the following command:

```{bash}
source slrm_scripts/split_and_pred.sh INPUT_DIR N OUT_DIR
```

The script will:

- Divide the input dir files into n sets, generate .txt containing the path to each .fasta (one per set)
- create an out_dir/chunks_timestamp/ directory where the predictions will be stored

- start N jobs launching the script: slrm_scripts/single_prediction.slrm n times (you can modify the resource of each job by modifying this script)

- Predictions are saved as:

out_dir/chunks_timestamp/
    job_id/
        boltz/ # prediction boltz
        msa/ # msa generated

## Generate Effective Strain Data from Boltz-2's Predictions:

You can generate Effective Strain data from Boltz's predictions by:

1. Set up the effective strain environment by running this:

```{bash}
source bash_scripts/setup_es_analysis.sh
```

This will clone the Effective Strain repository in your home directory, create a mamba env with all the necessary packages to run the analysis

2. Launch the Job Arrays:

```{bash}
source slrm_scripts/run_es.sh ROOT_DIR SCRIPT_DIR WT_PATH ARRAY_MAX_CONCURRENCY
```

Where:
- ROOT_DIR= directory with the prediction of BOLTZ, in our case is the main directory with the job arrays
- SCRIPT_DIR= path to the effective strain directory
- WT_PATH= path to the structure file (.cif) of the wildtype

By the default it will launch n jobs as many proteins have been predicted. The effective strain is saved as: ROOT_DIR/es/output.csv

## Generate Training Data for Model:

You can generate tensor data for training your model by:

```{bash}
python models/modules/feature_extraction.py --dir --out --shard_size
```

Currently, the script generates n chunks depending on the shrd size. It expects a single with:
It will expect a dir path structure like this:
    dir/
        seq_0000/
            boltz_results/
                predictions/
                    seq_0000/
                        .cif structure prediction
                        confidence files
The data will be:

- coordinates (N, residues, 3)
- confidence (N, residues, 1)
- Effective strain (N, residues, 1)
- adj_matrix (N, residues, residues)


## TODO:

- change generate_data.sh ->.fasta support
