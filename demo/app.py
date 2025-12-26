"""Streamlit demo for genomic sequence analysis."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json
from typing import Dict, Any, List, Tuple

from src.utils.core import Config, set_seed, get_device
from src.data import GenomicSequenceEncoder, generate_synthetic_data
from src.models import create_model
from src.metrics import ModelEvaluator, CalibrationAnalyzer

# Page configuration
st.set_page_config(
    page_title="Genomic Sequence Analysis",
    page_layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer banner
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Research Demo Disclaimer</h4>
    <p><strong>This is a research demonstration tool only.</strong></p>
    <ul>
        <li>Not intended for clinical diagnosis or medical advice</li>
        <li>Results are for educational and research purposes only</li>
        <li>Always consult qualified healthcare professionals for medical decisions</li>
        <li>Model performance may not reflect real-world clinical scenarios</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🧬 Genomic Sequence Analysis Demo</h1>', 
            unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["cnn", "transformer", "hybrid"],
    help="Choose the neural network architecture"
)

# Sequence parameters
seq_len = st.sidebar.slider(
    "Sequence Length",
    min_value=20,
    max_value=100,
    value=50,
    help="Length of DNA sequences to analyze"
)

num_samples = st.sidebar.slider(
    "Number of Samples",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100,
    help="Number of sequences to generate for training"
)

# Training parameters
learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.0001,
    max_value=0.01,
    value=0.001,
    format="%.4f",
    help="Learning rate for model training"
)

num_epochs = st.sidebar.slider(
    "Number of Epochs",
    min_value=5,
    max_value=50,
    value=10,
    help="Number of training epochs"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoder' not in st.session_state:
    st.session_state.encoder = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'test_results' not in st.session_state:
    st.session_state.test_results = None

@st.cache_data
def generate_data(num_samples: int, seq_len: int) -> Tuple[List[str], List[int]]:
    """Generate synthetic genomic data."""
    return generate_synthetic_data(num_samples, seq_len)

@st.cache_resource
def create_model_and_encoder(model_type: str, seq_len: int, hidden_dim: int = 128):
    """Create model and encoder."""
    device = get_device()
    
    # Create encoder
    encoding_type = "one_hot" if model_type == "cnn" else "integer"
    encoder = GenomicSequenceEncoder(seq_len=seq_len, encoding_type=encoding_type)
    
    # Create model
    model_config = {
        "input_channels": 4,
        "hidden_dim": hidden_dim,
        "num_classes": 2,
        "dropout": 0.1,
        "seq_len": seq_len,
        "d_model": hidden_dim,
        "nhead": 8,
        "num_layers": 2
    }
    
    model = create_model(model_type, model_config)
    model = model.to(device)
    
    return model, encoder, device

def train_model(model, encoder, device, sequences, labels, learning_rate, num_epochs):
    """Train the model."""
    from torch.utils.data import DataLoader, TensorDataset
    from torch import optim
    import torch.nn as nn
    
    # Prepare data
    encoded_sequences = []
    for seq in sequences:
        encoded = encoder.encode_sequence(seq)
        encoded_sequences.append(encoded)
    
    # Convert to tensors
    X = torch.stack([torch.tensor(seq, dtype=torch.float32) for seq in encoded_sequences])
    y = torch.tensor(labels, dtype=torch.long)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    train_losses = []
    train_accuracies = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        progress_bar.progress((epoch + 1) / num_epochs)
        status_text.text(f'Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}')
    
    progress_bar.empty()
    status_text.empty()
    
    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies
    }

def evaluate_model(model, encoder, device, sequences, labels):
    """Evaluate the model."""
    evaluator = ModelEvaluator()
    
    # Prepare test data
    encoded_sequences = []
    for seq in sequences:
        encoded = encoder.encode_sequence(seq)
        encoded_sequences.append(encoded)
    
    X = torch.stack([torch.tensor(seq, dtype=torch.float32) for seq in encoded_sequences])
    y = torch.tensor(labels, dtype=torch.long)
    
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Evaluate
    results = evaluator.evaluate_model(model, dataloader, device, return_predictions=True)
    
    return results

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🚀 Training", "📊 Results", "🔬 Analysis"])

with tab1:
    st.header("Welcome to Genomic Sequence Analysis")
    
    st.markdown("""
    This demo showcases deep learning models for genomic sequence classification, specifically 
    distinguishing between promoter and non-promoter DNA sequences.
    
    ### Features:
    - **Multiple Model Architectures**: CNN, Transformer, and Hybrid models
    - **Real-time Training**: Train models interactively with different parameters
    - **Comprehensive Evaluation**: Detailed metrics and visualizations
    - **Sequence Analysis**: Upload and analyze your own DNA sequences
    
    ### How to Use:
    1. Configure model parameters in the sidebar
    2. Go to the Training tab to train your model
    3. View results and metrics in the Results tab
    4. Analyze specific sequences in the Analysis tab
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Information")
        st.info(f"""
        **Selected Model**: {model_type.upper()}
        **Sequence Length**: {seq_len} bases
        **Training Samples**: {num_samples:,}
        **Learning Rate**: {learning_rate:.4f}
        **Epochs**: {num_epochs}
        """)
    
    with col2:
        st.subheader("Quick Start")
        if st.button("🚀 Start Training", type="primary"):
            st.session_state.start_training = True
            st.rerun()

with tab2:
    st.header("Model Training")
    
    if st.button("🔄 Generate New Data", type="secondary"):
        st.session_state.training_results = None
        st.session_state.test_results = None
    
    # Generate data
    with st.spinner("Generating synthetic genomic data..."):
        sequences, labels = generate_data(num_samples, seq_len)
    
    st.success(f"Generated {len(sequences)} sequences")
    
    # Show data distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Label Distribution")
        label_counts = np.bincount(labels)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Non-Promoter', 'Promoter'], label_counts, color=['#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Count')
        ax.set_title('Sequence Label Distribution')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Sample Sequences")
        sample_df = {
            'Sequence': sequences[:5],
            'Label': ['Promoter' if l == 1 else 'Non-Promoter' for l in labels[:5]]
        }
        st.dataframe(sample_df, use_container_width=True)
    
    # Training section
    st.subheader("Model Training")
    
    if st.button("🏋️ Train Model", type="primary"):
        with st.spinner("Creating model and encoder..."):
            model, encoder, device = create_model_and_encoder(model_type, seq_len)
            st.session_state.model = model
            st.session_state.encoder = encoder
        
        with st.spinner("Training model..."):
            training_results = train_model(
                model, encoder, device, sequences, labels, 
                learning_rate, num_epochs
            )
            st.session_state.training_results = training_results
        
        with st.spinner("Evaluating model..."):
            test_results = evaluate_model(model, encoder, device, sequences, labels)
            st.session_state.test_results = test_results
        
        st.success("Training completed!")
        st.rerun()
    
    # Show training progress
    if st.session_state.training_results:
        st.subheader("Training Progress")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(st.session_state.training_results["train_losses"], label='Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(st.session_state.training_results["train_accuracies"], label='Training Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Training Accuracy')
            ax.legend()
            st.pyplot(fig)

with tab3:
    st.header("Model Results")
    
    if st.session_state.test_results is None:
        st.info("Please train a model first in the Training tab.")
    else:
        results = st.session_state.test_results
        metrics = results["metrics"]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("F1-Score", f"{metrics['f1']:.4f}")
        with col3:
            st.metric("AUROC", f"{metrics['auroc']:.4f}")
        with col4:
            st.metric("AUPRC", f"{metrics['auprc']:.4f}")
        
        # Detailed metrics
        st.subheader("Detailed Classification Report")
        st.text(results["classification_report"])
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = metrics["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Promoter', 'Promoter'],
                   yticklabels=['Non-Promoter', 'Promoter'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        # Calibration analysis
        if "calibration" in results:
            st.subheader("Model Calibration")
            calib = results["calibration"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Expected Calibration Error", f"{calib['ece']:.4f}")
            with col2:
                st.metric("Brier Score", f"{calib['brier_score']:.4f}")
            
            # Calibration plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(calib["bin_confidences"], calib["bin_accuracies"], 'o-', label='Model')
            ax.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Accuracy')
            ax.set_title('Calibration Plot')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

with tab4:
    st.header("Sequence Analysis")
    
    if st.session_state.model is None:
        st.info("Please train a model first in the Training tab.")
    else:
        st.subheader("Analyze Custom Sequences")
        
        # Text input for sequences
        sequence_input = st.text_area(
            "Enter DNA sequences (one per line):",
            placeholder="ATCGATCGATCG...\nGCTAGCTAGCTA...",
            height=150
        )
        
        if sequence_input:
            sequences = [seq.strip().upper() for seq in sequence_input.split('\n') if seq.strip()]
            
            if sequences:
                st.subheader("Analysis Results")
                
                # Analyze each sequence
                results = []
                for i, seq in enumerate(sequences):
                    if len(seq) != seq_len:
                        st.warning(f"Sequence {i+1} length ({len(seq)}) doesn't match model input length ({seq_len})")
                        continue
                    
                    # Encode and predict
                    encoded = st.session_state.encoder.encode_sequence(seq)
                    X = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(st.session_state.model.device if hasattr(st.session_state.model, 'device') else 'cpu')
                    
                    with torch.no_grad():
                        logits = st.session_state.model(X)
                        probabilities = torch.softmax(logits, dim=1)
                        prediction = torch.argmax(logits, dim=1)
                    
                    results.append({
                        'Sequence': seq,
                        'Prediction': 'Promoter' if prediction.item() == 1 else 'Non-Promoter',
                        'Confidence': probabilities[0][prediction.item()].item(),
                        'Non-Promoter Prob': probabilities[0][0].item(),
                        'Promoter Prob': probabilities[0][1].item()
                    })
                
                if results:
                    st.dataframe(results, use_container_width=True)
                    
                    # Visualization
                    if len(results) > 1:
                        st.subheader("Prediction Distribution")
                        pred_counts = {}
                        for r in results:
                            pred = r['Prediction']
                            pred_counts[pred] = pred_counts.get(pred, 0) + 1
                        
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.pie(pred_counts.values(), labels=pred_counts.keys(), autopct='%1.1f%%')
                        ax.set_title('Prediction Distribution')
                        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Genomic Sequence Analysis Demo | Research and Educational Use Only</p>
    <p>Not for clinical diagnosis or medical advice</p>
</div>
""", unsafe_allow_html=True)
