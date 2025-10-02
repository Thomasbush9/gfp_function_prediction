# SE(3) GAT Protein Function Prediction Pipeline

This pipeline processes protein structure predictions and trains an SE(3) equivariant Graph Attention Network for protein function prediction.

## Quick Start

### 1. Run the Complete Pipeline

```bash
python run_pipeline.py \
    --input_dir /Users/thomasbush/Downloads/chunks_20250829_121523 \
    --tsv_path /Users/thomasbush/Documents/ML/gfp_tryout/data/amino_acid_genotypes_to_brightness.tsv \
    --processed_data_dir outputs/processed_data \
    --model_output_dir outputs/se3gat_model \
    --batch_size 8 \
    --epochs 100 \
    --test_equivariance
```

### 2. Run Example (with your data)

```bash
python example_usage.py
```

## Pipeline Steps

### Step 1: Feature Extraction

The pipeline first extracts features from protein structure predictions:

```bash
python -m models.modules.feature_extraction_updated \
    --dir /Users/thomasbush/Downloads/chunks_20250829_121523 \
    --tsv /Users/thomasbush/Documents/ML/gfp_tryout/data/amino_acid_genotypes_to_brightness.tsv \
    --out outputs/processed_data \
    --shard_size 4096
```

**Input Requirements:**
- Directory with structure: `seq_XXXXX/boltz/.../predictions/.../`
- TSV file with target values (column: `medianBrightness`)
- Optional: Effective strain data

**Output:**
- Sharded PyTorch Geometric data files
- Each shard contains multiple protein structures with features and targets

### Step 2: Model Training

Train the SE(3) GAT model on the processed data:

```bash
python train_se3gat.py \
    --data_dir outputs/processed_data \
    --output_dir outputs/se3gat_model \
    --batch_size 8 \
    --epochs 100 \
    --learning_rate 1e-3 \
    --hidden_dim 128 \
    --num_layers 3 \
    --num_heads 8 \
    --test_equivariance
```

**Output:**
- Trained model weights (`best_model.pt`)
- Training curves (`training_curves.png`)
- Training configuration (`training_config.json`)
- Training history (`training_history.json`)

## Data Format

### Input Directory Structure
```
chunks_20250829_121523/
├── 32483690_1/
│   ├── boltz/
│   │   └── seq_00403.fasta/
│   │       └── boltz_results_seq_00403.fasta_prot_pipeline/
│   │           └── predictions/
│   │               └── seq_00403.fasta_prot_pipeline/
│   │                   ├── seq_00403.fasta_prot_pipeline_model_0.cif
│   │                   └── plddt_seq_00403.fasta_prot_pipeline_model_0.npz
│   ├── es/
│   └── msa/
├── 32483690_2/
└── ...
```

### TSV File Format
```tsv
aaMutations     uniqueBarcodes  medianBrightness        std
        3645    3.7192121319    0.106991789588
SA108D  1       1.30103000388
SA108D:SN144D:SI186V:SM231T:SL234P      1       1.30103123972
```

## Model Architecture

The SE(3) GAT model processes protein structures as graphs:

- **Nodes**: Amino acid residues (238 per protein)
- **Features**: 3D coordinates + confidence scores
- **Edges**: Spatial proximity between residues
- **Target**: Scalar function prediction (e.g., brightness)

### Key Features:
- ✅ **SE(3) Equivariant**: Predictions are consistent under 3D rotations/translations
- ✅ **Geometric Awareness**: Understands 3D protein structure
- ✅ **Attention Mechanism**: Focuses on important residues
- ✅ **Scalable**: Handles variable protein sizes

## Training Parameters

### Model Parameters
- `scalar_dim`: Number of scalar features per residue (default: 4)
- `vector_dim`: Dimension of vector features (default: 1)
- `hidden_dim`: Hidden layer dimension (default: 128)
- `num_layers`: Number of GAT layers (default: 3)
- `num_heads`: Number of attention heads (default: 8)
- `dropout`: Dropout rate (default: 0.1)

### Training Parameters
- `batch_size`: Batch size for training (default: 8)
- `epochs`: Number of training epochs (default: 100)
- `learning_rate`: Learning rate (default: 1e-3)
- `device`: Device to use (auto, cpu, cuda, mps)

## Output Files

### Processed Data
```
outputs/processed_data/
├── shard_00000.pt
├── shard_00001.pt
└── ...
```

### Model Output
```
outputs/se3gat_model/
├── best_model.pt              # Best model weights
├── training_curves.png        # Training visualization
├── training_config.json       # Training configuration
├── training_history.json      # Training metrics
└── checkpoint_epoch_*.pt      # Model checkpoints
```

## Usage Examples

### 1. Basic Training
```bash
python run_pipeline.py \
    --input_dir /path/to/structures \
    --tsv_path /path/to/targets.tsv \
    --processed_data_dir outputs/data \
    --model_output_dir outputs/model
```

### 2. Custom Training Parameters
```bash
python run_pipeline.py \
    --input_dir /path/to/structures \
    --tsv_path /path/to/targets.tsv \
    --processed_data_dir outputs/data \
    --model_output_dir outputs/model \
    --batch_size 16 \
    --epochs 200 \
    --learning_rate 5e-4 \
    --hidden_dim 256 \
    --num_layers 4 \
    --num_heads 16
```

### 3. Skip Feature Extraction
```bash
python run_pipeline.py \
    --input_dir /path/to/structures \
    --tsv_path /path/to/targets.tsv \
    --processed_data_dir outputs/data \
    --model_output_dir outputs/model \
    --skip_extraction
```

### 4. Skip Training
```bash
python run_pipeline.py \
    --input_dir /path/to/structures \
    --tsv_path /path/to/targets.tsv \
    --processed_data_dir outputs/data \
    --model_output_dir outputs/model \
    --skip_training
```

## Monitoring Training

### Training Output
```
Epoch 1/100 - Train Loss: 1.5068, Train MAE: 1.0015 - Val Loss: 0.3149, Val MAE: 0.4652
Epoch 2/100 - Train Loss: 1.4295, Train MAE: 0.9024 - Val Loss: 0.1796, Val MAE: 0.2937
...
Training completed! Best validation loss: 0.1796
```

### SE(3) Equivariance Test
```
Original output: -0.0801
Rotated output: -0.0801
Equivariance error: 0.000000
✓ SE(3) equivariance test passed!
```

## Troubleshooting

### Common Issues

#### 1. Data Not Found
```
Error: Could not find CIF or confidence files in /path/to/dir
```
**Solution**: Check that your directory structure matches the expected format.

#### 2. Device Mismatch
```
Error: torch.cat(): all input tensors must be on the same device
```
**Solution**: The pipeline automatically handles device placement. Ensure PyTorch is properly installed.

#### 3. Memory Issues
```
Error: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU:
```bash
--batch_size 4 --device cpu
```

#### 4. TSV File Format
```
Error: Could not find target values
```
**Solution**: Ensure your TSV file has the correct column names and format.

### Performance Tips

1. **Use appropriate batch size**: Start with 8, adjust based on memory
2. **Enable mixed precision**: For large models on CUDA
3. **Use MPS on Apple Silicon**: Automatic device detection
4. **Monitor training curves**: Check for overfitting
5. **Test SE(3) equivariance**: Ensure geometric consistency

## Advanced Usage

### Custom Feature Extraction
Modify `models/modules/feature_extraction_updated.py` to:
- Add new features
- Change preprocessing steps
- Handle different file formats

### Custom Model Architecture
Modify `models/SEnGAT.py` to:
- Change network architecture
- Add new layers
- Implement different attention mechanisms

### Custom Training Loop
Modify `train_se3gat.py` to:
- Add new loss functions
- Implement custom metrics
- Add data augmentation

## Dependencies

```bash
# Core dependencies
torch>=1.9.0
torch-geometric>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
tqdm>=4.60.0

# Optional for visualization
seaborn>=0.11.0
plotly>=5.0.0
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{se3gat_pipeline2024,
  title={SE(3) Equivariant Graph Attention Network Pipeline for Protein Function Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/se3gat-pipeline}
}
```

## License

This project is licensed under the MIT License.

## Support

For questions and support, please open an issue on the GitHub repository.
