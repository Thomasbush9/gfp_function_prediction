#!/usr/bin/env python3
"""
Example usage of SE(3) GAT model for protein function prediction.

This script demonstrates how to:
1. Create an SE(3) GAT model
2. Set up a trainer
3. Train the model
4. Test SE(3) equivariance
"""

import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt

# Import our custom modules
from models.SEnGAT import create_se3gat_model, create_se3gat_trainer


def create_dummy_protein_data(num_proteins: int = 10, seq_len: int = 238, 
                             num_features: int = 4, num_edges: int = 500):
    """
    Create dummy protein data for demonstration.
    
    Args:
        num_proteins: Number of protein structures
        seq_len: Length of each protein sequence
        num_features: Number of features per residue
        num_edges: Number of edges in the graph
        
    Returns:
        List of Data objects
    """
    data_list = []
    
    for i in range(num_proteins):
        # Create random node features (scalar features)
        scalar_x = torch.randn(seq_len, num_features)
        
        # Create random 3D coordinates (vector features)
        vector_x = torch.randn(seq_len, 3)
        
        # Create random edge connectivity
        edge_index = torch.randint(0, seq_len, (2, num_edges))
        edge_index = to_undirected(edge_index)  # Make undirected
        
        # Create random edge attributes
        edge_attr = torch.randn(edge_index.size(1), 3)  # 3D edge attributes
        
        # Create random target (scalar function prediction)
        y = torch.randn(1)
        
        # Create PyG Data object
        data = Data(
            x=scalar_x,
            pos=vector_x,  # 3D coordinates
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )
        
        data_list.append(data)
    
    return data_list


def test_se3_equivariance(model, data, rotation_angle: float = np.pi/4):
    """
    Test SE(3) equivariance by rotating the input and checking if the output
    transforms consistently.
    
    Args:
        model: SE(3) GAT model
        data: Input protein data
        rotation_angle: Angle for rotation test
        
    Returns:
        Tuple of (original_output, rotated_output, equivariance_error)
    """
    model.eval()
    
    # Get the device from the model
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Create a copy of data and move to the same device as the model
        data_copy = data.clone()
        data_copy = data_copy.to(device)
        
        # Original prediction
        original_output = model(
            data_copy.x, data_copy.pos.unsqueeze(-1), data_copy.edge_index, 
            data_copy.edge_attr, torch.zeros(data_copy.x.size(0), dtype=torch.long, device=device)
        )
        
        # Create rotation matrix around z-axis
        cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        # Rotate coordinates
        rotated_pos = torch.mm(data_copy.pos, rotation_matrix.T)
        
        # Prediction on rotated data
        rotated_output = model(
            data_copy.x, rotated_pos.unsqueeze(-1), data_copy.edge_index,
            data_copy.edge_attr, torch.zeros(data_copy.x.size(0), dtype=torch.long, device=device)
        )
        
        # Check equivariance (output should be the same for rotation-invariant tasks)
        equivariance_error = torch.abs(original_output - rotated_output).mean().item()
        
        return original_output, rotated_output, equivariance_error


def visualize_training_history(history, save_path: str = None):
    """Visualize training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MAE curves
    axes[0, 1].plot(history['train_mae'], label='Train MAE')
    axes[0, 1].plot(history['val_mae'], label='Val MAE')
    axes[0, 1].set_title('Training and Validation MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning curves
    axes[1, 0].plot(history['train_loss'], label='Train')
    axes[1, 0].plot(history['val_loss'], label='Val')
    axes[1, 0].set_title('Loss Curves')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # MAE comparison
    axes[1, 1].plot(history['train_mae'], label='Train')
    axes[1, 1].plot(history['val_mae'], label='Val')
    axes[1, 1].set_title('MAE Curves')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def main():
    """Main function demonstrating SE(3) GAT usage."""
    print("=== SE(3) GAT Protein Function Prediction Example ===\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dummy data
    print("Creating dummy protein data...")
    data_list = create_dummy_protein_data(
        num_proteins=20, 
        seq_len=238, 
        num_features=4, 
        num_edges=500
    )
    
    print(f"Created {len(data_list)} protein structures")
    print(f"Each protein has {data_list[0].x.size(0)} residues")
    print(f"Feature dimension: {data_list[0].x.size(1)}")
    print(f"Number of edges: {data_list[0].edge_index.size(1)}")
    
    # Split data
    train_data = data_list[:15]
    val_data = data_list[15:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create SE(3) GAT model
    print("\nCreating SE(3) GAT model...")
    model = create_se3gat_model(
        scalar_dim=4,  # Number of scalar features per residue
        vector_dim=1,  # 3D coordinates
        hidden_dim=128,
        output_dim=1,
        num_layers=3,
        num_heads=8,
        dropout=0.1
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Model info: {model.get_model_info()}")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = create_se3gat_trainer(
        model=model,
        learning_rate=1e-3,
        weight_decay=1e-4,
        scheduler_type='cosine',
        early_stopping_patience=10,
        device='auto'
    )
    
    print(f"Trainer created on device: {trainer.device}")
    
    # Test SE(3) equivariance before training
    print("\nTesting SE(3) equivariance...")
    test_data = train_data[0]
    original_output, rotated_output, equivariance_error = test_se3_equivariance(
        model, test_data
    )
    
    print(f"Original output: {original_output.item():.4f}")
    print(f"Rotated output: {rotated_output.item():.4f}")
    print(f"Equivariance error: {equivariance_error:.6f}")
    
    if equivariance_error < 1e-3:
        print("✓ SE(3) equivariance test passed!")
    else:
        print("⚠ SE(3) equivariance test failed - this is expected before training")
    
    # Create data loaders
    from torch_geometric.loader import DataLoader
    
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Train the model
    print("\nStarting training...")
    save_dir = Path("outputs/se3gat_training")
    
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=20,
            save_dir=save_dir
        )
        
        print("\nTraining completed!")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Final training loss: {history['train_loss'][-1]:.4f}")
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        
        # Test SE(3) equivariance after training
        print("\nTesting SE(3) equivariance after training...")
        original_output, rotated_output, equivariance_error = test_se3_equivariance(
            model, test_data
        )
        
        print(f"Original output: {original_output.item():.4f}")
        print(f"Rotated output: {rotated_output.item():.4f}")
        print(f"Equivariance error: {equivariance_error:.6f}")
        
        if equivariance_error < 1e-3:
            print("✓ SE(3) equivariance test passed!")
        else:
            print("⚠ SE(3) equivariance test failed - model may need more training")
        
        # Visualize training history
        print("\nGenerating training curves...")
        visualize_training_history(history, save_dir / "training_curves.png")
        
        # Make predictions
        print("\nMaking predictions...")
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            predictions = []
            targets = []
            
            for batch in val_loader:
                batch = batch.to(device)
                
                pred = model(
                    batch.x, batch.pos.unsqueeze(-1), batch.edge_index,
                    batch.edge_attr, batch.batch
                )
                
                # Extract individual predictions and targets
                for i in range(pred.size(0)):
                    predictions.append(pred[i].item())
                    targets.append(batch.y[i].item())
            
            predictions = np.array(predictions)
            targets = np.array(targets)
            
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            
            print(f"Validation MSE: {mse:.4f}")
            print(f"Validation MAE: {mae:.4f}")
            print(f"Sample predictions: {predictions[:5]}")
            print(f"Sample targets: {targets[:5]}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        print("This might be due to missing dependencies or device issues.")
        print("Make sure PyTorch and PyTorch Geometric are properly installed.")
    
    print("\n=== Example completed ===")


if __name__ == "__main__":
    main()
