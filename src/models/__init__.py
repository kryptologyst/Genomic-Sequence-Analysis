"""Neural network models for genomic sequence analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model).
            
        Returns:
            torch.Tensor: Positionally encoded tensor.
        """
        return x + self.pe[:x.size(0), :]


class GenomicCNN(nn.Module):
    """1D CNN for genomic sequence classification."""
    
    def __init__(
        self,
        input_channels: int = 4,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
        seq_len: int = 50
    ):
        """Initialize CNN model.
        
        Args:
            input_channels: Number of input channels (4 for one-hot DNA).
            hidden_dim: Hidden dimension size.
            num_classes: Number of output classes.
            dropout: Dropout rate.
            seq_len: Input sequence length.
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate flattened size after convolutions
        # 50 -> 25 -> 12 -> 6 (after 3 pooling operations)
        self.fc1 = nn.Linear(hidden_dim * 6, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, seq_len).
            
        Returns:
            torch.Tensor: Output logits.
        """
        # First conv block
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Second conv block
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Third conv block
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


class GenomicTransformer(nn.Module):
    """Transformer model for genomic sequence classification."""
    
    def __init__(
        self,
        vocab_size: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1,
        max_len: int = 500
    ):
        """Initialize transformer model.
        
        Args:
            vocab_size: Vocabulary size (4 for DNA bases).
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer layers.
            num_classes: Number of output classes.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
        """
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len).
            
        Returns:
            torch.Tensor: Output logits.
        """
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence classification."""
    
    def __init__(self, d_model: int):
        """Initialize attention pooling.
        
        Args:
            d_model: Model dimension.
        """
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention pooling.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            torch.Tensor: Pooled tensor of shape (batch_size, d_model).
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (batch_size, seq_len, 1)
        
        # Apply attention weights
        pooled = torch.sum(x * attn_weights, dim=1)  # (batch_size, d_model)
        
        return pooled


class GenomicHybridModel(nn.Module):
    """Hybrid CNN-Transformer model for genomic sequences."""
    
    def __init__(
        self,
        input_channels: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        seq_len: int = 50
    ):
        """Initialize hybrid model.
        
        Args:
            input_channels: Number of input channels.
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer layers.
            num_classes: Number of output classes.
            dropout: Dropout rate.
            seq_len: Input sequence length.
        """
        super().__init__()
        
        # CNN feature extractor
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # Project CNN features to transformer dimension
        self.projection = nn.Linear(64, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Attention pooling and classification
        self.attention_pooling = AttentionPooling(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, seq_len).
            
        Returns:
            torch.Tensor: Output logits.
        """
        # CNN feature extraction
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Transpose for transformer (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # Project to transformer dimension
        x = self.projection(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Attention pooling
        x = self.attention_pooling(x)
        
        # Classification
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


def create_model(
    model_type: str,
    config: Dict[str, Any]
) -> nn.Module:
    """Create model based on configuration.
    
    Args:
        model_type: Type of model ("cnn", "transformer", "hybrid").
        config: Model configuration.
        
    Returns:
        nn.Module: Initialized model.
    """
    if model_type == "cnn":
        return GenomicCNN(
            input_channels=config.get("input_channels", 4),
            hidden_dim=config.get("hidden_dim", 128),
            num_classes=config.get("num_classes", 2),
            dropout=config.get("dropout", 0.1),
            seq_len=config.get("seq_len", 50)
        )
    elif model_type == "transformer":
        return GenomicTransformer(
            vocab_size=config.get("vocab_size", 4),
            d_model=config.get("d_model", 128),
            nhead=config.get("nhead", 8),
            num_layers=config.get("num_layers", 4),
            num_classes=config.get("num_classes", 2),
            dropout=config.get("dropout", 0.1),
            max_len=config.get("max_len", 500)
        )
    elif model_type == "hybrid":
        return GenomicHybridModel(
            input_channels=config.get("input_channels", 4),
            d_model=config.get("d_model", 128),
            nhead=config.get("nhead", 8),
            num_layers=config.get("num_layers", 2),
            num_classes=config.get("num_classes", 2),
            dropout=config.get("dropout", 0.1),
            seq_len=config.get("seq_len", 50)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
