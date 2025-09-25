#!/usr/bin/env python3
"""
Example usage of MLP model and trainer for protein function prediction.

This script demonstrates how to:
1. Create an MLP model
2. Set up a trainer
3. Train the model
4. Make predictions
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Import our custom modules
from models.MLP import create_mlp_model
from models.MLP_trainer import create_trainer


def create_dummy_data(num_samples: int = 100, seq_len: int = 238, num_features: int = 4):
    """
    Create dummy data for demonstration.
    
    Args:
        num_samples: Number of protein sequences
        seq_len: Length of each sequence (238 for GFP)
        num_features: Number of features per residue
        
    Returns:
        Tuple of (features, targets)
    """
    # Create random features: (num_samples, seq_len, num_features)
    features = torch.randn(num_samples, seq_len, num_features)
    
    # Create dummy targets (scalar values for each sequence)
    # In real usage, these would be actual function predictions
    targets = torch.randn(num_samples)
    
    return features, targets


def main():
    """Main function demonstrating MLP usage."""
    print("=== MLP Protein Function Prediction Example ===\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dummy data
    print("Creating dummy data...")
    train_features, train_targets = create_dummy_data(num_samples=80)
    val_features, val_targets = create_dummy_data(num_samples=20)
    
    print(f"Training data shape: {train_features.shape}")
    print(f"Validation data shape: {val_features.shape}")
    print(f"Number of features per residue: {train_features.shape[-1]}")
    
    # Create data loaders
    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create MLP model
    print("\nCreating MLP model...")
    model = create_mlp_model(
        input_dim=4,  # Number of features per residue
        hidden_dims=[512, 256, 128, 64],  # Hidden layer dimensions
        dropout_rate=0.3,  # Dropout rate
        activation='relu',  # Activation function
        use_batch_norm=True,  # Use batch normalization
        residual_connections=False  # No residual connections for this example
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Model info: {model.get_model_info()}")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = create_trainer(
        model=model,
        learning_rate=1e-3,
        weight_decay=1e-4,
        scheduler_type='cosine',
        early_stopping_patience=10,
        use_mixed_precision=False,  # Set to True if using CUDA
        device='auto'
    )
    
    print(f"Trainer created on device: {trainer.device}")
    
    # Train the model
    print("\nStarting training...")
    save_dir = Path("outputs/mlp_training")
    
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=20,
            save_dir=save_dir,
            plot_training=True
        )
        
        print("\nTraining completed!")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Final training loss: {history['train_loss'][-1]:.4f}")
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        
        # Make predictions on validation set
        print("\nMaking predictions...")
        predictions = trainer.predict(val_loader)
        print(f"Prediction shape: {predictions.shape}")
        print(f"Sample predictions: {predictions[:5].flatten()}")
        print(f"Sample targets: {val_targets[:5].numpy()}")
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics = trainer.evaluate(val_loader)
        print(f"Validation metrics: {metrics}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("This might be due to missing dependencies or device issues.")
        print("Make sure PyTorch is properly installed.")
    
    print("\n=== Example completed ===")


if __name__ == "__main__":
    main()
