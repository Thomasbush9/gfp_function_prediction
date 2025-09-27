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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPTrainer:
    """
    Comprehensive trainer class for MLP model with advanced training features.
    
    Features:
    - Automatic device detection (CUDA, MPS, CPU)
    - Multiple learning rate schedulers
    - Early stopping with patience
    - Model checkpointing and saving
    - Training visualization
    - Comprehensive metrics tracking
    - Gradient clipping
    - Mixed precision training support
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_type: str = 'cosine',
        warmup_epochs: int = 5,
        early_stopping_patience: int = 20,
        save_best_only: bool = True,
        use_mixed_precision: bool = False,
        gradient_clip_norm: float = 1.0
    ):
        """
        Initialize MLP trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            scheduler_type: Learning rate scheduler type ('cosine', 'step', 'plateau', 'warmup_cosine')
            warmup_epochs: Number of warmup epochs
            early_stopping_patience: Patience for early stopping
            save_best_only: Whether to save only the best model
            use_mixed_precision: Whether to use mixed precision training
            gradient_clip_norm: Gradient clipping norm
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
        self.use_mixed_precision = use_mixed_precision
        self.gradient_clip_norm = gradient_clip_norm
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Initialize scheduler
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_r2': [],
            'val_r2': [],
            'learning_rate': []
        }
        
        logger.info(f"MLP Trainer initialized on device: {self.device}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
        logger.info(f"Gradient clipping: {self.gradient_clip_norm}")
    
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
                self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
        elif self.scheduler_type == 'warmup_cosine':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        else:
            return None
    
    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with comprehensive metrics."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        all_predictions = []
        all_targets = []
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
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions = self.model(x)
                    
                    if y is not None:
                        loss = F.mse_loss(predictions.squeeze(), y.float())
                        mae = F.l1_loss(predictions.squeeze(), y.float())
                    else:
                        loss = torch.mean(predictions ** 2)
                        mae = torch.mean(torch.abs(predictions))
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(x)
                
                if y is not None:
                    loss = F.mse_loss(predictions.squeeze(), y.float())
                    mae = F.l1_loss(predictions.squeeze(), y.float())
                else:
                    loss = torch.mean(predictions ** 2)
                    mae = torch.mean(torch.abs(predictions))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
            
            # Store predictions and targets for R² calculation
            if y is not None:
                all_predictions.extend(predictions.squeeze().cpu().numpy())
                all_targets.extend(y.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{mae.item():.4f}',
                'lr': f'{self._get_current_lr():.2e}'
            })
        
        # Calculate R² score
        if all_predictions and all_targets:
            r2 = r2_score(all_targets, all_predictions)
        else:
            r2 = 0.0
        
        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches,
            'r2': r2
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch with comprehensive metrics."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        all_predictions = []
        all_targets = []
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
                
                # Store predictions and targets for R² calculation
                if y is not None:
                    all_predictions.extend(predictions.squeeze().cpu().numpy())
                    all_targets.extend(y.cpu().numpy())
        
        # Calculate R² score
        if all_predictions and all_targets:
            r2 = r2_score(all_targets, all_predictions)
        else:
            r2 = 0.0
        
        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches,
            'r2': r2
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: Optional[Union[str, Path]] = None,
        plot_training: bool = True
    ) -> Dict:
        """
        Train the model with comprehensive monitoring.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
            plot_training: Whether to plot training curves
            
        Returns:
            Training history dictionary
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Save directory: {save_dir}")
        logger.info(f"Initial learning rate: {self._get_current_lr():.2e}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if self.scheduler_type == 'plateau':
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_mae'].append(train_metrics['mae'])
            self.training_history['val_mae'].append(val_metrics['mae'])
            self.training_history['train_r2'].append(train_metrics['r2'])
            self.training_history['val_r2'].append(val_metrics['r2'])
            self.training_history['learning_rate'].append(self._get_current_lr())
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train MAE: {train_metrics['mae']:.4f}, Train R²: {train_metrics['r2']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f}, Val MAE: {val_metrics['mae']:.4f}, Val R²: {val_metrics['r2']:.4f} - "
                f"LR: {self._get_current_lr():.2e}"
            )
            
            # Save model
            if save_dir:
                self._save_checkpoint(save_dir, epoch, val_metrics['loss'])
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                if save_dir and self.save_best_only:
                    self._save_best_model(save_dir)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Plot training curves
        if plot_training and save_dir:
            self._plot_training_curves(save_dir)
        
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
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pt')
    
    def _save_best_model(self, save_dir: Path):
        """Save the best model."""
        torch.save(self.model.state_dict(), save_dir / 'best_model.pt')
        
        # Save training history
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save trainer configuration
        config = {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'scheduler_type': self.scheduler_type,
            'early_stopping_patience': self.early_stopping_patience,
            'use_mixed_precision': self.use_mixed_precision,
            'gradient_clip_norm': self.gradient_clip_norm
        }
        with open(save_dir / 'trainer_config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def _plot_training_curves(self, save_dir: Path):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE curves
        axes[0, 1].plot(self.training_history['train_mae'], label='Train MAE')
        axes[0, 1].plot(self.training_history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Training and Validation MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # R² curves
        axes[1, 0].plot(self.training_history['train_r2'], label='Train R²')
        axes[1, 0].plot(self.training_history['val_r2'], label='Val R²')
        axes[1, 0].set_title('Training and Validation R²')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate curve
        axes[1, 1].plot(self.training_history['learning_rate'])
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint['training_history']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
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
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                if hasattr(batch, 'x'):
                    x = batch.x.to(self.device)
                    y = batch.y.to(self.device) if hasattr(batch, 'y') else None
                else:
                    x = batch[0].to(self.device)
                    y = batch[1].to(self.device) if len(batch) > 1 else None
                
                if len(x.shape) == 2:
                    x = x.unsqueeze(0)
                
                pred = self.model(x)
                all_predictions.extend(pred.squeeze().cpu().numpy())
                
                if y is not None:
                    all_targets.extend(y.cpu().numpy())
        
        if all_targets:
            mse = mean_squared_error(all_targets, all_predictions)
            mae = mean_absolute_error(all_targets, all_predictions)
            r2 = r2_score(all_targets, all_predictions)
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
        else:
            return {
                'predictions': all_predictions
            }


def create_trainer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    **kwargs
) -> MLPTrainer:
    """
    Factory function to create MLP trainer.
    
    Args:
        model: PyTorch model to train
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
    from MLP import create_mlp_model
    
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
        device='auto',
        scheduler_type='cosine',
        early_stopping_patience=20
    )
    
    print("MLP Trainer created successfully!")
    print(f"Device: {trainer.device}")
    print(f"Mixed precision: {trainer.use_mixed_precision}")
