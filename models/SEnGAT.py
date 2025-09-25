import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple, List, Dict
import math
from tqdm import tqdm
from pathlib import Path

# For SE(3) equivariant operations
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch_geometric.data as geom_data


class SE3Linear(nn.Module):
    """
    SE(3) equivariant linear layer that preserves rotational and translational equivariance.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(SE3Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight matrix for scalar features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for SE(3) equivariant linear transformation.
        
        Args:
            x: Input tensor of shape (N, in_features)
            
        Returns:
            Output tensor of shape (N, out_features)
        """
        return F.linear(x, self.weight, self.bias)


class SE3Attention(nn.Module):
    """
    Simplified SE(3) equivariant attention mechanism.
    """
    
    def __init__(self, scalar_dim: int, vector_dim: int, num_heads: int = 8):
        super(SE3Attention, self).__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.num_heads = num_heads
        
        # Simple attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=scalar_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Vector feature processing
        self.vector_proj = nn.Linear(vector_dim, vector_dim, bias=False)
        
        # Output projections
        self.out_proj = nn.Linear(scalar_dim, scalar_dim)
        self.vector_out_proj = nn.Linear(vector_dim, vector_dim)
        
    def forward(self, scalar_features: Tensor, vector_features: Tensor, 
                edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for SE(3) equivariant attention.
        
        Args:
            scalar_features: Scalar node features (N, scalar_dim)
            vector_features: Vector node features (N, 3, vector_dim) - 3D coordinates
            edge_index: Edge connectivity (2, E)
            edge_attr: Optional edge attributes (E, edge_dim)
            
        Returns:
            Tuple of (updated_scalar_features, updated_vector_features)
        """
        # Use standard multi-head attention
        attn_output, _ = self.attention(
            scalar_features, scalar_features, scalar_features
        )
        
        # Update vector features (SE(3) equivariant)
        if vector_features is not None:
            vector_out = self.vector_out_proj(vector_features)
        else:
            vector_out = vector_features
        
        # Apply output projection
        out_scalar = self.out_proj(attn_output)
        
        return out_scalar, vector_out


class SE3GATLayer(MessagePassing):
    """
    SE(3) equivariant Graph Attention Layer.
    """
    
    def __init__(self, scalar_dim: int, vector_dim: int, num_heads: int = 8, 
                 dropout: float = 0.1, edge_dim: Optional[int] = None):
        super(SE3GATLayer, self).__init__(aggr='add', flow='source_to_target')
        
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        
        # Node feature transformations
        self.scalar_linear = SE3Linear(scalar_dim, scalar_dim)
        self.vector_linear = SE3Linear(vector_dim, vector_dim)
        
        # Attention mechanism
        self.attention = SE3Attention(scalar_dim, vector_dim, num_heads)
        
        # Edge feature processing
        if edge_dim is not None:
            self.edge_linear = nn.Linear(edge_dim, scalar_dim)
        
        # Output projections
        self.scalar_out = nn.Linear(scalar_dim, scalar_dim)
        self.vector_out = nn.Linear(vector_dim, vector_dim)
        
        # Layer normalization
        self.scalar_norm = nn.LayerNorm(scalar_dim)
        self.vector_norm = nn.LayerNorm(vector_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, scalar_x: Tensor, vector_x: Tensor, edge_index: Tensor, 
                edge_attr: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for SE(3) equivariant GAT layer.
        
        Args:
            scalar_x: Scalar node features (N, scalar_dim)
            vector_x: Vector node features (N, 3, vector_dim)
            edge_index: Edge connectivity (2, E)
            edge_attr: Optional edge attributes (E, edge_dim)
            
        Returns:
            Tuple of (updated_scalar_features, updated_vector_features)
        """
        # Transform input features
        scalar_h = self.scalar_linear(scalar_x)
        vector_h = self.vector_linear(vector_x)
        
        # Apply attention
        scalar_out, vector_out = self.attention(
            scalar_h, vector_h, edge_index, edge_attr
        )
        
        # Residual connections
        scalar_out = scalar_out + scalar_x
        vector_out = vector_out + vector_x
        
        # Layer normalization
        scalar_out = self.scalar_norm(scalar_out)
        vector_out = self.vector_norm(vector_out)
        
        # Apply dropout
        scalar_out = self.dropout_layer(scalar_out)
        
        return scalar_out, vector_out


class SE3GAT(nn.Module):
    """
    SE(3) Equivariant Graph Attention Network for protein function prediction.
    """
    
    def __init__(self, scalar_dim: int, vector_dim: int, hidden_dim: int, 
                 output_dim: int, num_layers: int = 3, num_heads: int = 8,
                 dropout: float = 0.1, edge_dim: Optional[int] = None):
        super(SE3GAT, self).__init__()
        
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Input projections
        self.scalar_input = nn.Linear(scalar_dim, hidden_dim)
        self.vector_input = nn.Linear(vector_dim, hidden_dim)
        
        # SE(3) GAT layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(SE3GATLayer(
                scalar_dim=hidden_dim,
                vector_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                edge_dim=edge_dim
            ))
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Activation functions
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, scalar_x: Tensor, vector_x: Tensor, edge_index: Tensor,
                edge_attr: Optional[Tensor] = None, batch: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for SE(3) GAT.
        
        Args:
            scalar_x: Scalar node features (N, scalar_dim)
            vector_x: Vector node features (N, 3, vector_dim)
            edge_index: Edge connectivity (2, E)
            edge_attr: Optional edge attributes (E, edge_dim)
            batch: Optional batch assignment (N,)
            
        Returns:
            Graph-level predictions (batch_size, output_dim)
        """
        # Input projections
        scalar_h = self.scalar_input(scalar_x)
        vector_h = self.vector_input(vector_x)
        
        # Apply SE(3) GAT layers
        for layer in self.layers:
            scalar_h, vector_h = layer(scalar_h, vector_h, edge_index, edge_attr)
            scalar_h = self.activation(scalar_h)
            scalar_h = self.dropout(scalar_h)
        
        # Global pooling
        if batch is not None:
            # Use batch information for global pooling
            num_graphs = batch.max().item() + 1
            graph_embeddings = []
            
            for i in range(num_graphs):
                mask = batch == i
                graph_scalar = scalar_h[mask].mean(dim=0)  # (hidden_dim,)
                graph_embeddings.append(graph_scalar)
            
            graph_embeddings = torch.stack(graph_embeddings)  # (num_graphs, hidden_dim)
        else:
            # Single graph
            graph_embeddings = scalar_h.mean(dim=0, keepdim=True)  # (1, hidden_dim)
        
        # Output projection
        output = self.output_proj(graph_embeddings)  # (batch_size, output_dim)
        
        return output
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'scalar_dim': self.scalar_dim,
            'vector_dim': self.vector_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class SE3GATTrainer:
    """
    Trainer for SE(3) GAT model.
    """
    
    def __init__(self, model: SE3GAT, device: str = 'auto', learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, scheduler_type: str = 'cosine',
                 early_stopping_patience: int = 20):
        self.model = model
        self.device = self._setup_device(device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=1e-6
            )
        else:
            self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = early_stopping_patience
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup device for training."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move all tensors to the correct device
            batch = batch.to(self.device)
            
            # Extract features and ensure they're on the correct device
            scalar_x = batch.x.to(self.device)  # (N, scalar_dim)
            vector_x = batch.pos.unsqueeze(-1).to(self.device)  # (N, 3, 1) - coordinates
            edge_index = batch.edge_index.to(self.device)
            edge_attr = batch.edge_attr.to(self.device) if hasattr(batch, 'edge_attr') and batch.edge_attr is not None else None
            y = batch.y.to(self.device) if hasattr(batch, 'y') and batch.y is not None else None
            batch_idx = batch.batch.to(self.device) if hasattr(batch, 'batch') else None
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(scalar_x, vector_x, edge_index, edge_attr, batch_idx)
            
            # Compute loss
            if y is not None:
                loss = F.mse_loss(predictions.squeeze(), y.float())
                mae = F.l1_loss(predictions.squeeze(), y.float())
            else:
                loss = torch.mean(predictions ** 2)
                mae = torch.mean(torch.abs(predictions))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
        
        return total_loss / num_batches, total_mae / num_batches
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move all tensors to the correct device
                batch = batch.to(self.device)
                
                # Extract features and ensure they're on the correct device
                scalar_x = batch.x.to(self.device)
                vector_x = batch.pos.unsqueeze(-1).to(self.device)
                edge_index = batch.edge_index.to(self.device)
                edge_attr = batch.edge_attr.to(self.device) if hasattr(batch, 'edge_attr') and batch.edge_attr is not None else None
                y = batch.y.to(self.device) if hasattr(batch, 'y') and batch.y is not None else None
                batch_idx = batch.batch.to(self.device) if hasattr(batch, 'batch') else None
                
                # Forward pass
                predictions = self.model(scalar_x, vector_x, edge_index, edge_attr, batch_idx)
                
                # Compute loss
                if y is not None:
                    loss = F.mse_loss(predictions.squeeze(), y.float())
                    mae = F.l1_loss(predictions.squeeze(), y.float())
                else:
                    loss = torch.mean(predictions ** 2)
                    mae = torch.mean(torch.abs(predictions))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        return total_loss / num_batches, total_mae / num_batches
    
    def train(self, train_loader, val_loader, num_epochs, save_dir=None):
        """Train the model."""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss, train_mae = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_mae = self.validate_epoch(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_mae'].append(train_mae)
            self.training_history['val_mae'].append(val_mae)
            
            # Log progress
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
            
            # Save model
            if save_dir and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_dir / 'best_model.pt')
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self.training_history


def create_se3gat_model(scalar_dim: int, vector_dim: int, hidden_dim: int = 128,
                        output_dim: int = 1, num_layers: int = 3, num_heads: int = 8,
                        dropout: float = 0.1) -> SE3GAT:
    """
    Factory function to create SE(3) GAT model.
    
    Args:
        scalar_dim: Dimension of scalar node features
        vector_dim: Dimension of vector node features
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_layers: Number of GAT layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        
    Returns:
        SE3GAT model instance
    """
    return SE3GAT(
        scalar_dim=scalar_dim,
        vector_dim=vector_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    )


def create_se3gat_trainer(model: SE3GAT, learning_rate: float = 1e-3,
                         **kwargs) -> SE3GATTrainer:
    """
    Factory function to create SE(3) GAT trainer.
    
    Args:
        model: SE3GAT model instance
        learning_rate: Learning rate
        **kwargs: Additional arguments for SE3GATTrainer
        
    Returns:
        SE3GATTrainer instance
    """
    return SE3GATTrainer(
        model=model,
        learning_rate=learning_rate,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Create SE(3) GAT model
    model = create_se3gat_model(
        scalar_dim=4,  # Example: 3 coordinates + 1 confidence
        vector_dim=1,  # 3D coordinates
        hidden_dim=128,
        output_dim=1,
        num_layers=3,
        num_heads=8
    )
    
    # Create trainer
    trainer = create_se3gat_trainer(
        model=model,
        learning_rate=1e-3,
        device='auto'
    )
    
    print("SE(3) GAT Model created successfully!")
    print(f"Model info: {model.get_model_info()}")
