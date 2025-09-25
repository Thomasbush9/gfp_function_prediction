import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for protein function prediction.
    
    Takes input of shape (batch_size, 238, d) where:
    - batch_size: number of protein sequences
    - 238: number of residues per protein
    - d: number of features per residue
    
    Outputs a scalar prediction for each protein sequence.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.3,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        residual_connections: bool = False
    ):
        """
        Initialize MLP model.
        
        Args:
            input_dim: Number of features per residue (d)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'gelu', 'swish')
            use_batch_norm: Whether to use batch normalization
            residual_connections: Whether to use residual connections
        """
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.residual_connections = residual_connections
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # SiLU is the same as Swish
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        self._build_network()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_network(self):
        """Build the network architecture."""
        # Input projection: (batch_size, 238, d) -> (batch_size, 238, hidden_dims[0])
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dims[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(len(self.hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])
            )
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(self.hidden_dims[i + 1]))
            self.dropouts.append(nn.Dropout(self.dropout_rate))
        
        # Global pooling layer (mean pooling over sequence length)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final prediction layer
        self.output_layer = nn.Linear(self.hidden_dims[-1], 1)
        
        # Additional dropout for output
        self.output_dropout = nn.Dropout(self.dropout_rate)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 238, d)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size, seq_len, features = x.shape
        
        # Reshape for processing: (batch_size * seq_len, features)
        x = x.view(-1, features)
        
        # Input projection
        x = self.input_projection(x)
        x = self.activation(x)
        
        # Hidden layers
        for i, (layer, dropout) in enumerate(zip(self.hidden_layers, self.dropouts)):
            residual = x if self.residual_connections and i > 0 else None
            
            x = layer(x)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = dropout(x)
            
            # Residual connection
            if residual is not None and x.shape == residual.shape:
                x = x + residual
        
        # Reshape back to (batch_size, seq_len, hidden_dim)
        x = x.view(batch_size, seq_len, -1)
        
        # Global pooling: (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim)
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        x = self.global_pool(x)  # (batch_size, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch_size, hidden_dim)
        
        # Final prediction
        x = self.output_dropout(x)
        x = self.output_layer(x)
        
        return x
    
    def get_model_info(self) -> Dict:
        """Get model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'residual_connections': self.residual_connections,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class MLPTrainer:
    """
    Trainer class for MLP model with comprehensive training loop.
    """
    
    def __init__(
        self,
        model: MLP,
        device: str = 'auto',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_type: str = 'cosine',
        warmup_epochs: int = 5,
        early_stopping_patience: int = 20,
        save_best_only: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: MLP model instance
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            scheduler_type: Learning rate scheduler type
            warmup_epochs: Number of warmup epochs
            early_stopping_patience: Patience for early stopping
            save_best_only: Whether to save only the best model
        """
        self.model = model
        self.device = self._setup_device(device)
        self.model.to(self.device)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.early_stopping_patience = early_stopping_patience
        self.save_best_only = save_best_only
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        logger.info(f"MLP Trainer initialized on device: {self.device}")
        logger.info(f"Model info: {self.model.get_model_info()}")
    
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
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=1e-6
            )
        elif self.scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif self.scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            # Move batch to device
            if hasattr(batch, 'x'):
                # PyTorch Geometric Data format
                x = batch.x.to(self.device)
                y = batch.y.to(self.device) if hasattr(batch, 'y') else None
            else:
                # Standard tensor format
                x = batch[0].to(self.device)
                y = batch[1].to(self.device) if len(batch) > 1 else None
            
            # Reshape if needed: (batch_size, seq_len, features)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # Add batch dimension if single sample
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(x)
            
            # Compute loss
            if y is not None:
                loss = F.mse_loss(predictions.squeeze(), y.float())
                mae = F.l1_loss(predictions.squeeze(), y.float())
            else:
                # If no labels, use dummy loss (for testing)
                loss = torch.mean(predictions ** 2)
                mae = torch.mean(torch.abs(predictions))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{mae.item():.4f}'
            })
        
        return total_loss / num_batches, total_mae / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                if hasattr(batch, 'x'):
                    x = batch.x.to(self.device)
                    y = batch.y.to(self.device) if hasattr(batch, 'y') else None
                else:
                    x = batch[0].to(self.device)
                    y = batch[1].to(self.device) if len(batch) > 1 else None
                
                # Reshape if needed
                if len(x.shape) == 2:
                    x = x.unsqueeze(0)
                
                # Forward pass
                predictions = self.model(x)
                
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
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
            
        Returns:
            Training history dictionary
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Save directory: {save_dir}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss, train_mae = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_mae = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if self.scheduler_type == 'plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_mae'].append(train_mae)
            self.training_history['val_mae'].append(val_mae)
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}"
            )
            
            # Save model
            if save_dir:
                self._save_checkpoint(save_dir, epoch, val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if save_dir and self.save_best_only:
                    self._save_best_model(save_dir)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
        return self.training_history
    
    def _save_checkpoint(self, save_dir: Path, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pt')
    
    def _save_best_model(self, save_dir: Path):
        """Save the best model."""
        torch.save(self.model.state_dict(), save_dir / 'best_model.pt')
        
        # Save model info
        model_info = self.model.get_model_info()
        with open(save_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """Make predictions on a dataset."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                if hasattr(batch, 'x'):
                    x = batch.x.to(self.device)
                else:
                    x = batch[0].to(self.device)
                
                if len(x.shape) == 2:
                    x = x.unsqueeze(0)
                
                pred = self.model(x)
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)


def create_mlp_model(
    input_dim: int,
    hidden_dims: List[int] = [512, 256, 128],
    dropout_rate: float = 0.3,
    **kwargs
) -> MLP:
    """
    Factory function to create MLP model.
    
    Args:
        input_dim: Number of input features per residue
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout rate
        **kwargs: Additional arguments for MLP
        
    Returns:
        MLP model instance
    """
    return MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        **kwargs
    )


def create_trainer(
    model: MLP,
    learning_rate: float = 1e-3,
    **kwargs
) -> MLPTrainer:
    """
    Factory function to create MLP trainer.
    
    Args:
        model: MLP model instance
        learning_rate: Learning rate
        **kwargs: Additional arguments for MLPTrainer
        
    Returns:
        MLPTrainer instance
    """
    return MLPTrainer(
        model=model,
        learning_rate=learning_rate,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_mlp_model(
        input_dim=4,  # Example: 3 coordinates + 1 confidence
        hidden_dims=[512, 256, 128, 64],
        dropout_rate=0.3
    )
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        learning_rate=1e-3,
        device='auto'
    )
    
    print("MLP Model created successfully!")
    print(f"Model info: {model.get_model_info()}")
