#!/usr/bin/env python3
"""
Simple training loop to test SE(3) GAT on shard data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import our models
from models.SEnGAT import create_se3gat_model


def load_shard_data(shard_path: str, batch_size: int = 8):
    """Load data from a single shard."""
    shard = torch.load(shard_path)
    print(f"Loaded shard with {len(shard)} samples")
    
    # Check if samples have target values
    has_targets = hasattr(shard[0], 'y') and shard[0].y is not None
    print(f"Samples have target values: {has_targets}")
    
    if has_targets:
        print(f"Sample target value: {shard[0].y.item()}")
    
    # Create data loader
    loader = DataLoader(shard, batch_size=batch_size, shuffle=True)
    return loader


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        # Extract features
        scalar_x = batch.x.to(device)
        vector_x = batch.pos.unsqueeze(-1).to(device)  # (N, 3, 1)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device) if hasattr(batch, 'edge_attr') and batch.edge_attr is not None else None
        y = batch.y.to(device) if hasattr(batch, 'y') and batch.y is not None else None
        batch_idx = batch.batch.to(device) if hasattr(batch, 'batch') else None
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(scalar_x, vector_x, edge_index, edge_attr, batch_idx)
        
        # Compute loss
        if y is not None:
            loss = criterion(predictions.squeeze(), y.float())
        else:
            # If no targets, use dummy loss
            loss = torch.mean(predictions ** 2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def main():
    """Main training function."""
    print("=== SE(3) GAT Test Training ===")
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    shard_path = "/Users/thomasbush/Downloads/training/shard_00000.pt"
    if not Path(shard_path).exists():
        print(f"Error: Shard file {shard_path} not found!")
        return
    
    loader = load_shard_data(shard_path, batch_size=16)
    
    # Create model
    print("\nCreating SE(3) GAT model...")
    model = create_se3gat_model(
        scalar_dim=4,  # 3 coordinates + 1 confidence
        vector_dim=1,  # 3D coordinates
        hidden_dim=64,  # Smaller for testing
        output_dim=1,
        num_layers=2,   # Fewer layers for testing
        num_heads=4,    # Fewer heads for testing
        dropout=0.1
    )
    
    model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"\nStarting training for 100 epochs...")
    print(f"Batch size: 16")
    print(f"Learning rate: 1e-3")
    
    # Training loop
    for epoch in range(100):
        train_loss = train_epoch(model, loader, optimizer, criterion, device)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {train_loss:.6f}")
    
    print(f"\nTraining completed!")
    print(f"Final loss: {train_loss:.6f}")
    
    # Test prediction on a sample
    print("\nTesting prediction on a sample...")
    model.eval()
    with torch.no_grad():
        sample = next(iter(loader))
        sample = sample.to(device)
        
        scalar_x = sample.x.to(device)
        vector_x = sample.pos.unsqueeze(-1).to(device)
        edge_index = sample.edge_index.to(device)
        edge_attr = sample.edge_attr.to(device) if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else None
        batch_idx = sample.batch.to(device) if hasattr(sample, 'batch') else None
        
        prediction = model(scalar_x, vector_x, edge_index, edge_attr, batch_idx)
        
        print(f"Prediction shape: {prediction.shape}")
        print(f"Sample predictions: {prediction.squeeze()[:5].cpu().numpy()}")
        
        if hasattr(sample, 'y') and sample.y is not None:
            print(f"Target values: {sample.y.squeeze()[:5].cpu().numpy()}")


if __name__ == "__main__":
    main()
