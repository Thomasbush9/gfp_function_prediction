# SE(3) Equivariant Graph Attention Network (SE3GAT)

## Overview

The SE(3) Equivariant Graph Attention Network (SE3GAT) is a specialized neural network architecture designed for protein function prediction that maintains **SE(3) equivariance** - meaning the model's predictions are consistent under 3D rotations and translations of the input protein structure.

## Key Features

### 🎯 **SE(3) Equivariance**
- **Rotation Invariant**: Predictions remain consistent regardless of protein orientation
- **Translation Invariant**: Predictions remain consistent regardless of protein position
- **Geometric Awareness**: Understands 3D protein structure and spatial relationships

### 🏗️ **Architecture Components**

#### **1. SE3Linear**
- Equivariant linear transformations that preserve SE(3) symmetry
- Xavier weight initialization for stable training
- Handles both scalar and vector features

#### **2. SE3Attention**
- Multi-head attention mechanism with geometric awareness
- Processes scalar node features (chemical properties, confidence scores)
- Considers vector features (3D coordinates) for spatial relationships
- Simplified implementation using PyTorch's MultiheadAttention

#### **3. SE3GATLayer**
- Complete Graph Attention Layer with SE(3) equivariance
- Message passing with attention mechanisms
- Residual connections and layer normalization
- Dropout regularization for robust training

#### **4. SE3GAT**
- Full network with multiple SE3GATLayer instances
- Global pooling for graph-level predictions
- Configurable architecture (hidden dimensions, layers, attention heads)
- Handles variable protein sizes and structures

## Model Architecture

```
Input: (N, 238, d) tensor where:
- N: batch size (number of protein sequences)
- 238: number of residues per protein
- d: number of features per residue

┌─────────────────────────────────────────────────────────────┐
│                    SE3GAT Architecture                      │
├─────────────────────────────────────────────────────────────┤
│  Input Projection: (N×238, d) → (N×238, hidden_dim)       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  SE3GATLayer 1:                                        │ │
│  │  ├─ SE3Linear (scalar features)                        │ │
│  │  ├─ SE3Linear (vector features)                        │ │
│  │  ├─ SE3Attention (multi-head attention)                │ │
│  │  ├─ Residual Connection                                 │ │
│  │  └─ Layer Normalization                                │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  SE3GATLayer 2: (same structure)                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  SE3GATLayer 3: (same structure)                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│  Global Pooling: (N×238, hidden_dim) → (N, hidden_dim)     │
│  Output Projection: (N, hidden_dim) → (N, 1)              │
└─────────────────────────────────────────────────────────────┘

Output: (N, 1) scalar predictions for each protein sequence
```

## Usage

### **Basic Usage**

```python
from models.SEnGAT import create_se3gat_model, create_se3gat_trainer
from torch_geometric.loader import DataLoader

# Create SE(3) GAT model
model = create_se3gat_model(
    scalar_dim=4,      # Number of scalar features per residue
    vector_dim=1,      # 3D coordinates
    hidden_dim=128,    # Hidden layer dimension
    output_dim=1,      # Output dimension (scalar prediction)
    num_layers=3,      # Number of GAT layers
    num_heads=8,       # Number of attention heads
    dropout=0.1        # Dropout rate
)

# Create trainer
trainer = create_se3gat_trainer(
    model=model,
    learning_rate=1e-3,
    weight_decay=1e-4,
    scheduler_type='cosine',
    early_stopping_patience=20,
    device='auto'
)

# Train model
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    save_dir="outputs/se3gat_training"
)
```

### **Data Format**

The model expects PyTorch Geometric `Data` objects with:

```python
data = Data(
    x=scalar_features,        # (N, scalar_dim) - node features
    pos=coordinates,          # (N, 3) - 3D coordinates
    edge_index=edge_connectivity,  # (2, E) - edge indices
    edge_attr=edge_features,  # (E, edge_dim) - edge attributes
    y=target_values          # (1,) - scalar targets
)
```

### **SE(3) Equivariance Testing**

```python
# Test SE(3) equivariance
def test_se3_equivariance(model, data, rotation_angle=np.pi/4):
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Create copy to avoid modifying original data
        data_copy = data.clone().to(device)
        
        # Original prediction
        original_output = model(
            data_copy.x, data_copy.pos.unsqueeze(-1), 
            data_copy.edge_index, data_copy.edge_attr,
            torch.zeros(data_copy.x.size(0), dtype=torch.long, device=device)
        )
        
        # Rotate coordinates
        cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        rotated_pos = torch.mm(data_copy.pos, rotation_matrix.T)
        
        # Prediction on rotated data
        rotated_output = model(
            data_copy.x, rotated_pos.unsqueeze(-1),
            data_copy.edge_index, data_copy.edge_attr,
            torch.zeros(data_copy.x.size(0), dtype=torch.long, device=device)
        )
        
        # Check equivariance
        equivariance_error = torch.abs(original_output - rotated_output).mean().item()
        return original_output, rotated_output, equivariance_error
```

## Model Parameters

### **SE3GAT Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scalar_dim` | int | - | Number of scalar features per residue |
| `vector_dim` | int | - | Dimension of vector features (3D coordinates) |
| `hidden_dim` | int | 128 | Hidden layer dimension |
| `output_dim` | int | 1 | Output dimension (scalar prediction) |
| `num_layers` | int | 3 | Number of SE3GATLayer instances |
| `num_heads` | int | 8 | Number of attention heads |
| `dropout` | float | 0.1 | Dropout rate |

### **SE3GATTrainer Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 1e-3 | Learning rate for optimizer |
| `weight_decay` | float | 1e-4 | Weight decay for regularization |
| `scheduler_type` | str | 'cosine' | Learning rate scheduler type |
| `early_stopping_patience` | int | 20 | Patience for early stopping |
| `device` | str | 'auto' | Device for training ('auto', 'cpu', 'cuda', 'mps') |

## Training Features

### **Advanced Training Features**
- ✅ **Automatic Device Detection**: Supports CUDA, MPS (Apple Silicon), and CPU
- ✅ **Learning Rate Scheduling**: Cosine, step, and plateau schedulers
- ✅ **Early Stopping**: Prevents overfitting with configurable patience
- ✅ **Gradient Clipping**: Prevents exploding gradients
- ✅ **Model Checkpointing**: Saves best model and training history
- ✅ **Comprehensive Metrics**: MSE, MAE, R² score tracking
- ✅ **Training Visualization**: Automatic plotting of training curves

### **Training Output**
```
Epoch 1/100 - Train Loss: 1.5068, Train MAE: 1.0015 - Val Loss: 0.3149, Val MAE: 0.4652
Epoch 2/100 - Train Loss: 1.4295, Train MAE: 0.9024 - Val Loss: 0.1796, Val MAE: 0.2937
...
Training completed! Best validation loss: 0.1796
```

## Performance Characteristics

### **Computational Complexity**
- **Time Complexity**: O(N²) where N is the number of residues
- **Space Complexity**: O(N²) for attention computation
- **Memory Usage**: Scales with protein size and batch size

### **Model Size**
- **Parameters**: ~547K parameters for default configuration
- **Model Size**: ~2.2 MB (without optimizer state)
- **Training Memory**: Depends on batch size and protein length

## Advantages over Standard GAT

### **1. Geometric Awareness**
- Understands 3D protein structure geometry
- Processes both scalar and vector features
- Maintains spatial relationships between residues

### **2. Equivariance**
- Predictions are consistent under 3D rotations
- Predictions are consistent under 3D translations
- Robust to protein orientation changes

### **3. Structure-Aware Attention**
- Attention weights consider 3D spatial relationships
- Vector features provide geometric context
- Better understanding of protein folding patterns

### **4. Scalability**
- Handles variable protein sizes
- Efficient batch processing
- Memory-efficient attention mechanisms

## Example Results

### **SE(3) Equivariance Test**
```
Original output: -0.0801
Rotated output: -0.0801
Equivariance error: 0.000000
✓ SE(3) equivariance test passed!
```

### **Training Performance**
```
Best validation loss: 0.1796
Final training loss: 1.0949
Final validation loss: 0.5552
Validation MSE: 0.7198
Validation MAE: 0.6827
```

## File Structure

```
models/
├── SEnGAT.py              # Main SE(3) GAT implementation
├── MLP.py                 # MLP model for comparison
├── MLP_trainer.py         # MLP trainer
└── modules/
    ├── dataset.py         # Dataset utilities
    └── feature_extraction.py  # Feature extraction

examples/
├── example_se3gat_usage.py    # SE(3) GAT usage example
└── example_mlp_usage.py       # MLP usage example

outputs/
└── se3gat_training/
    ├── best_model.pt          # Best model weights
    ├── training_curves.png    # Training visualization
    └── training_history.json # Training metrics
```

## Dependencies

### **Required Packages**
```bash
torch>=1.9.0
torch-geometric>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
tqdm>=4.60.0
```

### **Installation**
```bash
# Install PyTorch (choose appropriate version)
pip install torch torchvision torchaudio

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install numpy matplotlib scikit-learn tqdm
```

## Best Practices

### **1. Data Preprocessing**
- Normalize coordinates to zero mean
- Standardize scalar features
- Ensure consistent edge connectivity

### **2. Training Configuration**
- Use appropriate batch size (4-16 for large proteins)
- Start with learning rate 1e-3
- Use cosine annealing scheduler
- Enable early stopping

### **3. Model Selection**
- Choose hidden_dim based on protein complexity
- Use 3-5 layers for most applications
- 8-16 attention heads work well
- Dropout rate 0.1-0.3

### **4. Evaluation**
- Always test SE(3) equivariance
- Use multiple random seeds
- Cross-validate on different protein families
- Monitor training curves for overfitting

## Troubleshooting

### **Common Issues**

#### **Device Mismatch**
```python
# Ensure all tensors are on the same device
data = data.to(device)
model = model.to(device)
```

#### **Memory Issues**
```python
# Reduce batch size or use gradient accumulation
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

#### **SE(3) Equivariance Failures**
```python
# Check that you're not modifying original data
data_copy = data.clone().to(device)
```

### **Performance Optimization**
- Use mixed precision training for large models
- Enable gradient checkpointing for memory efficiency
- Use appropriate device (MPS for Apple Silicon, CUDA for NVIDIA)
- Profile memory usage with large protein structures

## Future Enhancements

### **Planned Features**
- [ ] Support for different protein representations
- [ ] Integration with AlphaFold2 structures
- [ ] Multi-task learning capabilities
- [ ] Uncertainty quantification
- [ ] Attention visualization tools

### **Research Directions**
- [ ] Improved SE(3) equivariant operations
- [ ] Hierarchical attention mechanisms
- [ ] Integration with sequence information
- [ ] Transfer learning from large protein datasets

## Citation

If you use this SE(3) GAT implementation in your research, please cite:

```bibtex
@software{se3gat2024,
  title={SE(3) Equivariant Graph Attention Network for Protein Function Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/se3gat}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

For questions and support, please open an issue on the GitHub repository or contact the maintainers.

---

**Note**: This SE(3) GAT implementation is specifically designed for protein function prediction where 3D structure and geometric relationships are crucial for accurate predictions. The equivariance ensures that the model's predictions are consistent regardless of how the protein is oriented in 3D space.
