"""Core utilities for genomic sequence analysis."""

import random
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import logging
import os
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device with fallback.
    
    Returns:
        torch.device: CUDA, MPS (Apple Silicon), or CPU device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional log file path.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def create_directories(base_path: Union[str, Path], subdirs: List[str]) -> None:
    """Create directory structure.
    
    Args:
        base_path: Base directory path.
        subdirs: List of subdirectories to create.
    """
    base_path = Path(base_path)
    for subdir in subdirs:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)


class Config:
    """Configuration class for genomic sequence analysis."""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """Initialize configuration.
        
        Args:
            config_dict: Optional dictionary to override defaults.
        """
        # Default configuration
        self.seed = 42
        self.device = get_device()
        
        # Data configuration
        self.seq_len = 50
        self.num_samples = 1000
        self.batch_size = 16
        self.train_split = 0.7
        self.val_split = 0.15
        self.test_split = 0.15
        
        # Model configuration
        self.model_type = "cnn"  # "cnn" or "transformer"
        self.hidden_dim = 128
        self.num_layers = 2
        self.dropout = 0.1
        
        # Training configuration
        self.learning_rate = 0.001
        self.num_epochs = 10
        self.weight_decay = 1e-4
        
        # Evaluation configuration
        self.eval_metrics = ["accuracy", "auroc", "auprc", "f1"]
        
        # Paths
        self.data_dir = "data"
        self.checkpoint_dir = "checkpoints"
        self.assets_dir = "assets"
        self.logs_dir = "logs"
        
        # Override with provided config
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary.
        
        Returns:
            Dict: Configuration as dictionary.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
