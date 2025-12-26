"""Evaluation script for genomic sequence models."""

import argparse
import logging
import yaml
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.core import Config, set_seed, setup_logging, get_device
from src.data import GenomicSequenceEncoder, generate_synthetic_data, create_data_loaders
from src.models import create_model
from src.metrics import ModelEvaluator
from src.train import ModelTrainer

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, config: Config, device: torch.device):
    """Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        config: Model configuration.
        device: Device to load model on.
        
    Returns:
        Loaded model.
    """
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def create_visualizations(results: dict, output_dir: Path):
    """Create evaluation visualizations.
    
    Args:
        results: Evaluation results dictionary.
        output_dir: Directory to save visualizations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion Matrix
    if "confusion_matrix" in results["metrics"]:
        cm = results["metrics"]["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Promoter', 'Promoter'],
                   yticklabels=['Non-Promoter', 'Promoter'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # ROC Curve
    if "predictions" in results and "probabilities" in results:
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        labels = results["labels"]
        probabilities = results["probabilities"]
        
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            pos_probs = probabilities[:, 1]
        else:
            pos_probs = probabilities
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(labels, pos_probs)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {results["metrics"]["auroc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(labels, pos_probs)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {results["metrics"]["auprc"]:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Calibration Plot
    if "calibration" in results:
        calib = results["calibration"]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(calib["bin_confidences"], calib["bin_accuracies"], 'o-', 
               linewidth=2, markersize=8, label='Model')
        ax.plot([0, 1], [0, 1], '--', linewidth=2, label='Perfect Calibration')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Calibration Plot (ECE = {calib["ece"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'calibration_plot.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate genomic sequence model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="assets/evaluation",
                       help="Directory to save evaluation results")
    parser.add_argument("--num_test_samples", type=int, default=500,
                       help="Number of test samples to generate")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(config_dict)
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Generate test data
    logger.info("Generating test data...")
    sequences, labels = generate_synthetic_data(
        num_samples=args.num_test_samples,
        seq_len=config.seq_len
    )
    
    # Create encoder and data loader
    encoder = GenomicSequenceEncoder(
        seq_len=config.seq_len,
        encoding_type="one_hot" if config.model_type == "cnn" else "integer"
    )
    
    _, _, test_loader = create_data_loaders(
        sequences=sequences,
        labels=labels,
        encoder=encoder,
        train_split=0.0,
        val_split=0.0,
        test_split=1.0,
        batch_size=config.batch_size
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(
        model, test_loader, device, return_predictions=True
    )
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info(results["classification_report"])
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualizations(results, output_dir)
    
    # Save detailed results
    results_path = output_dir / "evaluation_results.json"
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
    
    logger.info(f"Evaluation completed. Results saved to {output_dir}")
    logger.info(f"Key metrics:")
    logger.info(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
    logger.info(f"  F1-Score: {results['metrics']['f1']:.4f}")
    logger.info(f"  AUROC: {results['metrics']['auroc']:.4f}")
    logger.info(f"  AUPRC: {results['metrics']['auprc']:.4f}")


if __name__ == "__main__":
    main()
