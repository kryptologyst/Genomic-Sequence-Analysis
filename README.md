# Genomic Sequence Analysis

A research-ready toolkit for genomic sequence classification using deep learning. This project demonstrates state-of-the-art approaches for distinguishing between promoter and non-promoter DNA sequences.

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS A RESEARCH DEMONSTRATION TOOL ONLY**

- **NOT INTENDED FOR CLINICAL DIAGNOSIS OR MEDICAL ADVICE**
- **FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**
- **ALWAYS CONSULT QUALIFIED HEALTHCARE PROFESSIONALS FOR MEDICAL DECISIONS**
- **MODEL PERFORMANCE MAY NOT REFLECT REAL-WORLD CLINICAL SCENARIOS**

## Features

- **Multiple Model Architectures**: CNN, Transformer, and Hybrid models
- **Comprehensive Evaluation**: AUROC, AUPRC, calibration analysis, and more
- **Interactive Demo**: Streamlit-based web interface
- **Reproducible Research**: Deterministic seeding and proper configuration management
- **Modern Stack**: PyTorch 2.x, Python 3.10+, with proper type hints
- **Educational Focus**: Clear documentation and examples

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Genomic-Sequence-Analysis.git
cd Genomic-Sequence-Analysis

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Training a Model

```bash
# Train with default CNN model
python scripts/train.py --config configs/default.yaml

# Train with Transformer model
python scripts/train.py --config configs/transformer.yaml

# Custom configuration
python scripts/train.py --config configs/default.yaml --num_epochs 20 --learning_rate 0.0005
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Project Structure

```
genomic-sequence-analysis/
├── src/                    # Source code
│   ├── models/            # Neural network models
│   ├── data/              # Data processing utilities
│   ├── metrics/           # Evaluation metrics
│   ├── train/             # Training utilities
│   └── utils/             # Core utilities
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── demo/                  # Streamlit demo application
├── tests/                 # Unit tests
├── assets/                # Generated outputs and visualizations
├── data/                  # Data directory
├── checkpoints/           # Model checkpoints
└── logs/                  # Training logs
```

## Model Architectures

### 1. CNN Model (`GenomicCNN`)
- 1D convolutional layers with batch normalization
- Max pooling for feature extraction
- Fully connected layers for classification
- Optimized for one-hot encoded DNA sequences

### 2. Transformer Model (`GenomicTransformer`)
- Multi-head self-attention mechanism
- Positional encoding for sequence understanding
- Global average pooling for classification
- Handles variable-length sequences

### 3. Hybrid Model (`GenomicHybridModel`)
- Combines CNN feature extraction with Transformer processing
- CNN layers extract local patterns
- Transformer layers capture long-range dependencies
- Attention-based pooling for final classification

## Data Format

The toolkit works with DNA sequences in standard format:

- **Input**: DNA sequences as strings (A, C, G, T)
- **Encoding**: One-hot encoding for CNN, integer encoding for Transformer
- **Labels**: Binary classification (0: non-promoter, 1: promoter)
- **Length**: Configurable sequence length (default: 50 bases)

### Synthetic Data Generation

For demonstration purposes, the toolkit includes synthetic data generation:

- **Non-promoter sequences**: Random DNA sequences
- **Promoter sequences**: Sequences containing known promoter motifs (TATA box, CAAT box, GC box)
- **Balanced dataset**: Configurable class distribution

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted metrics
- **AUROC**: Area Under ROC Curve
- **AUPRC**: Area Under Precision-Recall Curve

### Calibration Metrics
- **Expected Calibration Error (ECE)**: Measures calibration quality
- **Maximum Calibration Error (MCE)**: Worst-case calibration error
- **Brier Score**: Probabilistic accuracy measure

### Visualization
- Confusion matrices
- ROC and PR curves
- Calibration plots
- Training progress curves

## Configuration

Models and training are configured via YAML files:

```yaml
# Model configuration
model_type: "cnn"  # "cnn", "transformer", "hybrid"
hidden_dim: 128
num_layers: 2
dropout: 0.1

# Training configuration
learning_rate: 0.001
num_epochs: 10
batch_size: 16
weight_decay: 1e-4

# Data configuration
seq_len: 50
num_samples: 1000
train_split: 0.7
val_split: 0.15
test_split: 0.15
```

## Usage Examples

### Basic Training

```python
from src.utils.core import Config
from src.data import GenomicSequenceEncoder, generate_synthetic_data
from src.models import create_model
from src.train import ModelTrainer

# Load configuration
config = Config({"model_type": "cnn", "seq_len": 50})

# Generate data
sequences, labels = generate_synthetic_data(1000, 50)

# Create model and train
encoder = GenomicSequenceEncoder(seq_len=50)
model = create_model("cnn", {"input_channels": 4, "num_classes": 2})
trainer = ModelTrainer(config, model, device)
results = trainer.train(train_loader, val_loader)
```

### Model Evaluation

```python
from src.metrics import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_model(model, test_loader, device)
print(results["classification_report"])
```

### Custom Sequence Analysis

```python
# Analyze a custom sequence
sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
encoded = encoder.encode_sequence(sequence)
prediction = model(torch.tensor(encoded).unsqueeze(0))
```

## Development

### Code Quality

The project uses modern Python development practices:

- **Type hints**: Full type annotation coverage
- **Code formatting**: Black and Ruff for consistent style
- **Testing**: Pytest for unit tests
- **Documentation**: Google-style docstrings

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

### Code Formatting

```bash
# Format code
black src/ scripts/ demo/

# Lint code
ruff check src/ scripts/ demo/
```

## Limitations and Future Work

### Current Limitations
- Synthetic data only (no real genomic datasets)
- Binary classification only
- Fixed sequence length
- No sequence alignment or variant calling

### Future Enhancements
- Integration with real genomic datasets
- Multi-class classification (different promoter types)
- Variable-length sequence handling
- Integration with bioinformatics tools (BLAST, BWA, etc.)
- Transfer learning from pre-trained genomic models

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes all linting checks
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{genomic_sequence_analysis,
  title={Genomic Sequence Analysis: A Deep Learning Toolkit},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Genomic-Sequence-Analysis}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Streamlit team for the interactive demo framework
- Bioinformatics community for inspiration and datasets
- Contributors and users who provide feedback

---

**Remember: This is a research demonstration tool. Not for clinical use.**
# Genomic-Sequence-Analysis
