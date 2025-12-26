"""Model comparison script for genomic sequence analysis."""

import argparse
import logging
import yaml
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

from src.utils.core import Config, set_seed, setup_logging, get_device
from src.data import GenomicSequenceEncoder, generate_synthetic_data, create_data_loaders
from src.models import create_model
from src.metrics import ModelEvaluator
from src.train import ModelTrainer

logger = logging.getLogger(__name__)


def train_and_evaluate_model(
    model_type: str,
    config: Config,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device
) -> Dict[str, Any]:
    """Train and evaluate a single model.
    
    Args:
        model_type: Type of model to train.
        config: Training configuration.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        test_loader: Test data loader.
        device: Device to train on.
        
    Returns:
        Dictionary containing training and evaluation results.
    """
    logger.info(f"Training {model_type} model...")
    
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
    
    model = create_model(model_type, model_config)
    model = model.to(device)
    
    # Create trainer
    trainer = ModelTrainer(config, model, device)
    
    # Train model
    training_results = trainer.train(train_loader, val_loader, test_loader)
    
    # Evaluate on test set
    evaluator = ModelEvaluator()
    test_results = evaluator.evaluate_model(model, test_loader, device)
    
    return {
        "model_type": model_type,
        "training_results": training_results,
        "test_results": test_results,
        "model": model
    }


def create_comparison_plots(results_list: List[Dict[str, Any]], output_dir: Path):
    """Create comparison plots for multiple models.
    
    Args:
        results_list: List of model results.
        output_dir: Directory to save plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics for comparison
    model_names = []
    accuracies = []
    f1_scores = []
    aurocs = []
    auprcs = []
    training_times = []
    
    for result in results_list:
        model_names.append(result["model_type"].upper())
        metrics = result["test_results"]["metrics"]
        accuracies.append(metrics["accuracy"])
        f1_scores.append(metrics["f1"])
        aurocs.append(metrics["auroc"])
        auprcs.append(metrics["auprc"])
        training_times.append(result["training_results"]["training_time"])
    
    # Create metrics comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy comparison
    axes[0, 0].bar(model_names, accuracies, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # F1-Score comparison
    axes[0, 1].bar(model_names, f1_scores, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('F1-Score Comparison')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_ylim(0, 1)
    for i, v in enumerate(f1_scores):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # AUROC comparison
    axes[1, 0].bar(model_names, aurocs, color='orange', alpha=0.7)
    axes[1, 0].set_title('AUROC Comparison')
    axes[1, 0].set_ylabel('AUROC')
    axes[1, 0].set_ylim(0, 1)
    for i, v in enumerate(aurocs):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Training time comparison
    axes[1, 1].bar(model_names, training_times, color='pink', alpha=0.7)
    axes[1, 1].set_title('Training Time Comparison')
    axes[1, 1].set_ylabel('Training Time (seconds)')
    for i, v in enumerate(training_times):
        axes[1, 1].text(i, v + max(training_times) * 0.01, f'{v:.1f}s', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed metrics table
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'F1-Score': f1_scores,
        'AUROC': aurocs,
        'AUPRC': auprcs,
        'Training Time (s)': training_times
    })
    
    # Save metrics table
    metrics_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    # Create training curves comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, result in enumerate(results_list):
        history = result["training_results"]["history"]
        model_name = result["model_type"].upper()
        
        # Training loss
        axes[0].plot(history["train_loss"], 
                    label=f'{model_name} Train', 
                    color=colors[i % len(colors)], 
                    linestyle='-')
        axes[0].plot(history["val_loss"], 
                    label=f'{model_name} Val', 
                    color=colors[i % len(colors)], 
                    linestyle='--')
        
        # Training accuracy
        axes[1].plot(history["train_accuracy"], 
                     label=f'{model_name} Train', 
                     color=colors[i % len(colors)], 
                     linestyle='-')
        axes[1].plot(history["val_accuracy"], 
                     label=f'{model_name} Val', 
                     color=colors[i % len(colors)], 
                     linestyle='--')
    
    axes[0].set_title('Training Loss Comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Training Accuracy Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare genomic sequence models")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to base configuration file")
    parser.add_argument("--models", nargs='+', default=["cnn", "transformer", "hybrid"],
                       help="Models to compare")
    parser.add_argument("--output_dir", type=str, default="assets/comparison",
                       help="Directory to save comparison results")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to generate")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load base configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Set random seed
    set_seed(config_dict["seed"])
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data once for all models
    logger.info("Generating synthetic data...")
    sequences, labels = generate_synthetic_data(
        num_samples=args.num_samples,
        seq_len=config_dict["seq_len"]
    )
    
    # Create data loaders
    encoder = GenomicSequenceEncoder(
        seq_len=config_dict["seq_len"],
        encoding_type="one_hot"  # Use one-hot for all models for fair comparison
    )
    
    train_loader, val_loader, test_loader = create_data_loaders(
        sequences=sequences,
        labels=labels,
        encoder=encoder,
        train_split=config_dict["train_split"],
        val_split=config_dict["val_split"],
        test_split=config_dict["test_split"],
        batch_size=config_dict["batch_size"]
    )
    
    # Train and evaluate each model
    results_list = []
    
    for model_type in args.models:
        logger.info(f"Starting comparison for {model_type} model...")
        
        # Create model-specific config
        model_config = config_dict.copy()
        model_config["model_type"] = model_type
        model_config["num_epochs"] = args.num_epochs
        
        config = Config(model_config)
        
        # Train and evaluate
        result = train_and_evaluate_model(
            model_type, config, train_loader, val_loader, test_loader, device
        )
        results_list.append(result)
        
        logger.info(f"Completed {model_type} model training")
    
    # Create comparison plots
    logger.info("Creating comparison visualizations...")
    create_comparison_plots(results_list, output_dir)
    
    # Print summary
    logger.info("Model Comparison Summary:")
    logger.info("=" * 50)
    
    for result in results_list:
        model_type = result["model_type"].upper()
        metrics = result["test_results"]["metrics"]
        
        logger.info(f"{model_type} Model:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1']:.4f}")
        logger.info(f"  AUROC: {metrics['auroc']:.4f}")
        logger.info(f"  AUPRC: {metrics['auprc']:.4f}")
        logger.info(f"  Training Time: {result['training_results']['training_time']:.2f}s")
        logger.info("")
    
    # Find best model
    best_model = max(results_list, key=lambda x: x["test_results"]["metrics"]["auroc"])
    logger.info(f"Best performing model: {best_model['model_type'].upper()} "
                f"(AUROC: {best_model['test_results']['metrics']['auroc']:.4f})")
    
    logger.info(f"Comparison results saved to {output_dir}")


if __name__ == "__main__":
    main()
