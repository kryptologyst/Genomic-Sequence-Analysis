"""Training utilities for genomic sequence models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from tqdm import tqdm
import os
from pathlib import Path
import json
import time

from ..metrics import ModelEvaluator
from ..utils.core import Config

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            restore_best_weights: Whether to restore best weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss.
            model: Model to potentially restore weights.
            
        Returns:
            bool: True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class ModelTrainer:
    """Trainer class for genomic sequence models."""
    
    def __init__(self, config: Config, model: nn.Module, 
                 device: torch.device):
        """Initialize trainer.
        
        Args:
            config: Training configuration.
            model: Model to train.
            device: Device to train on.
        """
        self.config = config
        self.model = model
        self.device = device
        self.evaluator = ModelEvaluator()
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(patience=5)
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "learning_rate": []
        }
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (sequences, labels) in enumerate(progress_bar):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            test_loader: Optional test data loader.
            
        Returns:
            Dict containing training results and history.
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} parameters")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_accuracy"].append(val_acc)
            self.history["learning_rate"].append(self.optimizer.param_groups[0]['lr'])
            
            # Log progress
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Final evaluation
        results = {
            "history": self.history,
            "training_time": training_time,
            "best_val_loss": best_val_loss,
            "final_epoch": epoch + 1
        }
        
        if test_loader:
            logger.info("Evaluating on test set...")
            test_results = self.evaluator.evaluate_model(
                self.model, test_loader, self.device
            )
            results["test_results"] = test_results
        
        return results
    
    def save_checkpoint(self, epoch: int, val_loss: float, 
                       is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch.
            val_loss: Current validation loss.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.to_dict()
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']


class LearningRateScheduler:
    """Learning rate scheduler with various strategies."""
    
    def __init__(self, optimizer: optim.Optimizer, strategy: str = "cosine",
                 **kwargs):
        """Initialize scheduler.
        
        Args:
            optimizer: PyTorch optimizer.
            strategy: Scheduler strategy ("cosine", "step", "plateau").
            **kwargs: Additional scheduler arguments.
        """
        self.optimizer = optimizer
        self.strategy = strategy
        
        if strategy == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=kwargs.get('T_max', 100)
            )
        elif strategy == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif strategy == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10)
            )
        else:
            raise ValueError(f"Unknown scheduler strategy: {strategy}")
    
    def step(self, metric: Optional[float] = None) -> None:
        """Step the scheduler.
        
        Args:
            metric: Optional metric for plateau scheduler.
        """
        if self.strategy == "plateau" and metric is not None:
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate.
        
        Returns:
            float: Current learning rate.
        """
        return self.optimizer.param_groups[0]['lr']
