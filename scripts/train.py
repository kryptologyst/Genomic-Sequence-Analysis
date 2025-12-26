"""Main training script for genomic sequence analysis."""

import argparse
import logging
import yaml
from pathlib import Path
import torch
import numpy as np

from src.utils.core import Config, set_seed, setup_logging, get_device
from src.data import GenomicSequenceEncoder, generate_synthetic_data, create_data_loaders
from src.models import create_model
from src.train import ModelTrainer
from src.metrics import ModelEvaluator

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Config: Configuration object.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train genomic sequence classification model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory to save/load data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config.data_dir = args.data_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.seed = args.seed
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device()
    config.device = device
    logger.info(f"Using device: {device}")
    
    # Create directories
    Path(config.data_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.assets_dir).mkdir(parents=True, exist_ok=True)
    Path(config.logs_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    logger.info("Generating synthetic genomic data...")
    sequences, labels = generate_synthetic_data(
        num_samples=config.num_samples,
        seq_len=config.seq_len
    )
    
    # Create encoder
    encoder = GenomicSequenceEncoder(
        seq_len=config.seq_len,
        encoding_type="one_hot" if config.model_type == "cnn" else "integer"
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        sequences=sequences,
        labels=labels,
        encoder=encoder,
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
        batch_size=config.batch_size
    )
    
    # Create model
    model_config = {
        "input_channels": 4,
        "hidden_dim": config.hidden_dim,
        "num_classes": 2,
        "dropout": config.dropout,
        "seq_len": config.seq_len,
        "d_model": config.hidden_dim,
        "nhead": 8,
        "num_layers": config.num_layers
    }
    
    model = create_model(config.model_type, model_config)
    model = model.to(device)
    
    logger.info(f"Created {config.model_type} model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    
    # Create trainer
    trainer = ModelTrainer(config, model, device)
    
    # Train model
    logger.info("Starting training...")
    results = trainer.train(train_loader, val_loader, test_loader)
    
    # Save results
    results_path = Path(config.assets_dir) / "training_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    else:
                        json_results[key][k] = v
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value
        
        import json
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Training completed. Results saved to {results_path}")
    
    # Print final metrics
    if "test_results" in results:
        test_metrics = results["test_results"]["metrics"]
        logger.info("Final Test Metrics:")
        logger.info(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}")
        logger.info(f"  F1-Score: {test_metrics.get('f1', 0):.4f}")
        logger.info(f"  AUROC: {test_metrics.get('auroc', 0):.4f}")
        logger.info(f"  AUPRC: {test_metrics.get('auprc', 0):.4f}")


if __name__ == "__main__":
    main()
