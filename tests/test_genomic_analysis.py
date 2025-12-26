"""Unit tests for genomic sequence analysis."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from src.utils.core import Config, set_seed, get_device
from src.data import GenomicSequenceEncoder, GenomicDataset, generate_synthetic_data
from src.models import GenomicCNN, GenomicTransformer, create_model
from src.metrics import MetricsCalculator, ModelEvaluator


class TestConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        assert config.seed == 42
        assert config.seq_len == 50
        assert config.num_samples == 1000
        assert config.model_type == "cnn"
    
    def test_config_override(self):
        """Test configuration override."""
        config_dict = {"seq_len": 100, "num_samples": 500}
        config = Config(config_dict)
        assert config.seq_len == 100
        assert config.num_samples == 500
        assert config.seed == 42  # Should keep default


class TestGenomicSequenceEncoder:
    """Test genomic sequence encoder."""
    
    def test_one_hot_encoding(self):
        """Test one-hot encoding."""
        encoder = GenomicSequenceEncoder(seq_len=10, encoding_type="one_hot")
        sequence = "ATCGATCGAT"
        encoded = encoder.encode_sequence(sequence)
        
        assert encoded.shape == (4, 10)
        assert encoded.dtype == np.float32
        assert np.sum(encoded) == 10  # Should have 10 ones
    
    def test_integer_encoding(self):
        """Test integer encoding."""
        encoder = GenomicSequenceEncoder(seq_len=10, encoding_type="integer")
        sequence = "ATCGATCGAT"
        encoded = encoder.encode_sequence(sequence)
        
        assert encoded.shape == (10,)
        assert encoded.dtype == np.int64
        assert len(np.unique(encoded)) <= 4  # Should have at most 4 unique values
    
    def test_sequence_padding(self):
        """Test sequence padding for shorter sequences."""
        encoder = GenomicSequenceEncoder(seq_len=10, encoding_type="one_hot")
        sequence = "ATC"  # Shorter than seq_len
        encoded = encoder.encode_sequence(sequence)
        
        assert encoded.shape == (4, 10)
        assert np.sum(encoded) == 3  # Only 3 bases encoded


class TestGenomicDataset:
    """Test genomic dataset."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        sequences = ["ATCG", "GCTA", "TTAA"]
        labels = [0, 1, 0]
        encoder = GenomicSequenceEncoder(seq_len=4, encoding_type="one_hot")
        
        dataset = GenomicDataset(sequences, labels, encoder)
        
        assert len(dataset) == 3
        assert dataset.sequences == sequences
        assert dataset.labels == labels
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        sequences = ["ATCG", "GCTA"]
        labels = [0, 1]
        encoder = GenomicSequenceEncoder(seq_len=4, encoding_type="one_hot")
        
        dataset = GenomicDataset(sequences, labels, encoder)
        
        item = dataset[0]
        assert isinstance(item[0], torch.Tensor)
        assert isinstance(item[1], int)
        assert item[0].shape == (4, 4)  # One-hot encoded


class TestModels:
    """Test neural network models."""
    
    def test_cnn_model(self):
        """Test CNN model creation and forward pass."""
        model = GenomicCNN(input_channels=4, seq_len=50, num_classes=2)
        
        # Test forward pass
        x = torch.randn(2, 4, 50)  # Batch of 2 sequences
        output = model(x)
        
        assert output.shape == (2, 2)  # Batch size, num classes
    
    def test_transformer_model(self):
        """Test Transformer model creation and forward pass."""
        model = GenomicTransformer(vocab_size=4, seq_len=50, num_classes=2)
        
        # Test forward pass
        x = torch.randint(0, 4, (2, 50))  # Batch of 2 sequences
        output = model(x)
        
        assert output.shape == (2, 2)  # Batch size, num classes
    
    def test_model_creation_function(self):
        """Test model creation function."""
        config = {"input_channels": 4, "num_classes": 2, "seq_len": 50}
        
        cnn_model = create_model("cnn", config)
        transformer_model = create_model("transformer", config)
        
        assert isinstance(cnn_model, GenomicCNN)
        assert isinstance(transformer_model, GenomicTransformer)
    
    def test_model_parameter_count(self):
        """Test model parameter counting."""
        from src.models import count_parameters
        
        model = GenomicCNN(input_channels=4, seq_len=50, num_classes=2)
        param_count = count_parameters(model)
        
        assert param_count > 0
        assert isinstance(param_count, int)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_metrics_calculator(self):
        """Test metrics calculator."""
        calculator = MetricsCalculator()
        
        # Add some test data
        predictions = torch.tensor([0, 1, 0, 1])
        labels = torch.tensor([0, 1, 1, 1])
        probabilities = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]])
        
        calculator.update(predictions, labels, probabilities)
        metrics = calculator.compute_metrics()
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auroc" in metrics
        assert "auprc" in metrics
    
    def test_calibration_analyzer(self):
        """Test calibration analyzer."""
        from src.metrics import CalibrationAnalyzer
        
        analyzer = CalibrationAnalyzer(n_bins=5)
        
        # Test with perfect calibration
        probabilities = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1, 1])
        
        results = analyzer.analyze_calibration(probabilities, labels)
        
        assert "ece" in results
        assert "mce" in results
        assert "brier_score" in results
        assert len(results["bin_accuracies"]) == 5


class TestDataGeneration:
    """Test synthetic data generation."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        sequences, labels = generate_synthetic_data(num_samples=100, seq_len=20)
        
        assert len(sequences) == 100
        assert len(labels) == 100
        assert all(len(seq) == 20 for seq in sequences)
        assert all(label in [0, 1] for label in labels)
        assert all(base in "ACGT" for seq in sequences for base in seq)
    
    def test_data_distribution(self):
        """Test data distribution."""
        sequences, labels = generate_synthetic_data(num_samples=1000, seq_len=20, promoter_prob=0.3)
        
        promoter_count = sum(labels)
        non_promoter_count = len(labels) - promoter_count
        
        # Should be roughly 30% promoters (within reasonable tolerance)
        assert 0.2 <= promoter_count / len(labels) <= 0.4


class TestUtilities:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        first_run = np.random.random()
        
        set_seed(42)
        second_run = np.random.random()
        
        assert first_run == second_run
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]


if __name__ == "__main__":
    pytest.main([__file__])
