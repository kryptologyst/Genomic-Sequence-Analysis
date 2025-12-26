"""Evaluation metrics for genomic sequence analysis."""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate comprehensive metrics for genomic sequence classification."""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """Initialize metrics calculator.
        
        Args:
            class_names: Optional class names for display.
        """
        self.class_names = class_names or ["Non-Promoter", "Promoter"]
        self.reset()
    
    def reset(self) -> None:
        """Reset all stored metrics."""
        self.predictions = []
        self.labels = []
        self.probabilities = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor, 
               probabilities: Optional[torch.Tensor] = None) -> None:
        """Update metrics with new batch.
        
        Args:
            predictions: Predicted class labels.
            labels: True class labels.
            probabilities: Predicted class probabilities.
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy())
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute comprehensive metrics.
        
        Returns:
            Dict containing all computed metrics.
        """
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(labels, predictions)
        metrics["precision"] = precision_score(labels, predictions, average="weighted")
        metrics["recall"] = recall_score(labels, predictions, average="weighted")
        metrics["f1"] = f1_score(labels, predictions, average="weighted")
        
        # Per-class metrics
        precision_per_class = precision_score(labels, predictions, average=None)
        recall_per_class = recall_score(labels, predictions, average=None)
        f1_per_class = f1_score(labels, predictions, average=None)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f"precision_{class_name.lower().replace('-', '_')}"] = precision_per_class[i]
            metrics[f"recall_{class_name.lower().replace('-', '_')}"] = recall_per_class[i]
            metrics[f"f1_{class_name.lower().replace('-', '_')}"] = f1_per_class[i]
        
        # ROC and PR metrics (if probabilities available)
        if self.probabilities:
            probabilities = np.array(self.probabilities)
            
            # For binary classification, use positive class probabilities
            if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                pos_probs = probabilities[:, 1]
            else:
                pos_probs = probabilities
            
            try:
                metrics["auroc"] = roc_auc_score(labels, pos_probs)
                metrics["auprc"] = average_precision_score(labels, pos_probs)
            except ValueError as e:
                logger.warning(f"Could not compute ROC/PR metrics: {e}")
                metrics["auroc"] = 0.0
                metrics["auprc"] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics["confusion_matrix"] = cm
        
        return metrics
    
    def get_classification_report(self) -> str:
        """Get detailed classification report.
        
        Returns:
            str: Formatted classification report.
        """
        metrics = self.compute_metrics()
        
        report = "Classification Report\n"
        report += "=" * 50 + "\n"
        report += f"Accuracy: {metrics.get('accuracy', 0):.4f}\n"
        report += f"Precision (Weighted): {metrics.get('precision', 0):.4f}\n"
        report += f"Recall (Weighted): {metrics.get('recall', 0):.4f}\n"
        report += f"F1-Score (Weighted): {metrics.get('f1', 0):.4f}\n"
        
        if "auroc" in metrics:
            report += f"AUROC: {metrics['auroc']:.4f}\n"
        if "auprc" in metrics:
            report += f"AUPRC: {metrics['auprc']:.4f}\n"
        
        report += "\nPer-Class Metrics:\n"
        report += "-" * 30 + "\n"
        
        for class_name in self.class_names:
            class_key = class_name.lower().replace('-', '_')
            precision = metrics.get(f"precision_{class_key}", 0)
            recall = metrics.get(f"recall_{class_key}", 0)
            f1 = metrics.get(f"f1_{class_key}", 0)
            
            report += f"{class_name}:\n"
            report += f"  Precision: {precision:.4f}\n"
            report += f"  Recall: {recall:.4f}\n"
            report += f"  F1-Score: {f1:.4f}\n"
        
        return report


class CalibrationAnalyzer:
    """Analyze model calibration for genomic sequence classification."""
    
    def __init__(self, n_bins: int = 10):
        """Initialize calibration analyzer.
        
        Args:
            n_bins: Number of bins for calibration analysis.
        """
        self.n_bins = n_bins
    
    def analyze_calibration(self, probabilities: np.ndarray, 
                          labels: np.ndarray) -> Dict[str, Any]:
        """Analyze model calibration.
        
        Args:
            probabilities: Predicted probabilities for positive class.
            labels: True binary labels.
            
        Returns:
            Dict containing calibration metrics and bin information.
        """
        # Create bins
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # Calculate calibration metrics
        ece = 0  # Expected Calibration Error
        mce = 0  # Maximum Calibration Error
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        # Brier Score
        brier_score = np.mean((probabilities - labels) ** 2)
        
        return {
            "ece": ece,
            "mce": mce,
            "brier_score": brier_score,
            "bin_accuracies": np.array(bin_accuracies),
            "bin_confidences": np.array(bin_confidences),
            "bin_counts": np.array(bin_counts),
            "bin_centers": (bin_lowers + bin_uppers) / 2
        }


class ModelEvaluator:
    """Comprehensive model evaluation for genomic sequences."""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """Initialize model evaluator.
        
        Args:
            class_names: Optional class names for display.
        """
        self.class_names = class_names or ["Non-Promoter", "Promoter"]
        self.metrics_calculator = MetricsCalculator(class_names)
        self.calibration_analyzer = CalibrationAnalyzer()
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """Evaluate model on dataset.
        
        Args:
            model: PyTorch model to evaluate.
            data_loader: Data loader for evaluation.
            device: Device to run evaluation on.
            return_predictions: Whether to return predictions and probabilities.
            
        Returns:
            Dict containing evaluation results.
        """
        model.eval()
        self.metrics_calculator.reset()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in data_loader:
                sequences, labels = batch
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                # Forward pass
                logits = model(sequences)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update metrics
                self.metrics_calculator.update(predictions, labels, probabilities)
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_metrics()
        
        # Analyze calibration
        all_probabilities = np.array(all_probabilities)
        if all_probabilities.ndim == 2 and all_probabilities.shape[1] == 2:
            pos_probs = all_probabilities[:, 1]
        else:
            pos_probs = all_probabilities
        
        calibration_results = self.calibration_analyzer.analyze_calibration(
            pos_probs, np.array(all_labels)
        )
        
        # Combine results
        results = {
            "metrics": metrics,
            "calibration": calibration_results,
            "classification_report": self.metrics_calculator.get_classification_report()
        }
        
        if return_predictions:
            results["predictions"] = np.array(all_predictions)
            results["labels"] = np.array(all_labels)
            results["probabilities"] = all_probabilities
        
        return results
    
    def create_leaderboard(self, results_dict: Dict[str, Dict[str, Any]]) -> str:
        """Create a leaderboard from multiple model results.
        
        Args:
            results_dict: Dictionary mapping model names to evaluation results.
            
        Returns:
            str: Formatted leaderboard.
        """
        leaderboard = "Model Performance Leaderboard\n"
        leaderboard += "=" * 60 + "\n"
        leaderboard += f"{'Model':<20} {'Accuracy':<10} {'F1':<10} {'AUROC':<10} {'AUPRC':<10}\n"
        leaderboard += "-" * 60 + "\n"
        
        for model_name, results in results_dict.items():
            metrics = results["metrics"]
            accuracy = metrics.get("accuracy", 0)
            f1 = metrics.get("f1", 0)
            auroc = metrics.get("auroc", 0)
            auprc = metrics.get("auprc", 0)
            
            leaderboard += f"{model_name:<20} {accuracy:<10.4f} {f1:<10.4f} "
            leaderboard += f"{auroc:<10.4f} {auprc:<10.4f}\n"
        
        return leaderboard
