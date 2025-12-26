#!/usr/bin/env python3
"""Quick start script for genomic sequence analysis."""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False


def main():
    """Main quick start function."""
    logger.info("🚀 Genomic Sequence Analysis - Quick Start")
    logger.info("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        logger.error("Please run this script from the project root directory")
        sys.exit(1)
    
    # Create necessary directories
    directories = ["data", "checkpoints", "assets", "logs"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        logger.info(f"✓ Created directory: {dir_name}")
    
    # Install dependencies
    logger.info("\n📦 Installing dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        logger.warning("Failed to install requirements. Please install manually.")
    
    # Run tests
    logger.info("\n🧪 Running tests...")
    if not run_command("python -m pytest tests/ -v", "Running unit tests"):
        logger.warning("Some tests failed. Check the output above.")
    
    # Train a quick model
    logger.info("\n🏋️ Training a quick CNN model...")
    if not run_command(
        "python scripts/train.py --config configs/default.yaml --num_epochs 5 --num_samples 500",
        "Training CNN model"
    ):
        logger.warning("Training failed. Check the output above.")
    
    # Train a transformer model
    logger.info("\n🤖 Training a Transformer model...")
    if not run_command(
        "python scripts/train.py --config configs/transformer.yaml --num_epochs 5 --num_samples 500",
        "Training Transformer model"
    ):
        logger.warning("Transformer training failed. Check the output above.")
    
    # Run model comparison
    logger.info("\n📊 Running model comparison...")
    if not run_command(
        "python scripts/compare_models.py --num_epochs 5 --num_samples 500",
        "Model comparison"
    ):
        logger.warning("Model comparison failed. Check the output above.")
    
    # Check if Streamlit is available
    logger.info("\n🌐 Checking Streamlit demo...")
    try:
        import streamlit
        logger.info("✓ Streamlit is available")
        logger.info("To run the demo: streamlit run demo/app.py")
    except ImportError:
        logger.warning("Streamlit not available. Install with: pip install streamlit")
    
    # Summary
    logger.info("\n🎉 Quick start completed!")
    logger.info("\nNext steps:")
    logger.info("1. Explore the trained models in checkpoints/")
    logger.info("2. View comparison results in assets/comparison/")
    logger.info("3. Run the interactive demo: streamlit run demo/app.py")
    logger.info("4. Check out the Jupyter notebook: notebooks/demo.ipynb")
    logger.info("\n⚠️  Remember: This is a research demonstration tool only. Not for clinical use.")


if __name__ == "__main__":
    main()
