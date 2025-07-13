# HiViFAN: Hierarchical Visual-Financial Attention Networks for Multi-Modal NFT Market Analysis

## Overview

HiViFAN is a comprehensive visual analytics framework that integrates computer vision and financial analysis through multi-modal deep learning for NFT market analysis. The system combines multi-scale visual feature extraction with temporal market dynamics modeling through attention mechanisms to predict NFT prices and analyze market behavior.

### Key Features

- **Multi-Scale Visual Processing**: Four-level feature pyramid capturing NFT attributes from pixel-level to compositional patterns
- **Dynamic Cross-Modal Attention**: Volatility-aware fusion mechanism adapting to market conditions
- **Temporal Coherence Modeling**: Dilated convolutions with graph attention for market dynamics
- **Interactive Visual Analytics**: Real-time dashboards with GPU-accelerated visualizations
- **Theoretical Foundations**: Information-theoretic bounds guiding architectural design decisions

## System Requirements

### Hardware Requirements
- **Minimum**: NVIDIA GPU with 8GB VRAM (RTX 3070 or equivalent)
- **Recommended**: NVIDIA RTX 3090 or A100 with 24GB+ VRAM
- **RAM**: 32GB system memory minimum, 64GB recommended
- **Storage**: 100GB free space for datasets and model checkpoints

### Software Dependencies
- Python 3.9 or higher
- CUDA 11.8 or higher
- PyTorch 2.0 or higher with CUDA support

## Installation

### Environment Setup


# Clone the repository
git clone https://anonymous.4open.science/r/HiViFAN:-464A/
cd HiViFAN

# Create conda environment
conda create -n hivifan python=3.9
conda activate hivifan

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional dependencies
pip install -r requirements.txt


### Docker Installation (Alternative)


# Build Docker image
docker build -t hivifan:latest .

# Run container with GPU support
docker run --gpus all -v $(pwd):/workspace -p 8050:8050 hivifan:latest


## Dataset Preparation

### Required Datasets

The experiments require NFT transaction data and corresponding visual assets. We provide preprocessing scripts for the following collections:

- **CryptoPunks**: 167,492 transactions (primary dataset)
- **Bored Ape Yacht Club**: 42,891 transactions
- **Azuki**: 38,234 transactions  
- **Art Blocks Curated**: 67,123 transactions

### Data Structure


data/
├── cryptopunks/
│   ├── images/           # NFT image files
│   ├── transactions.csv  # Transaction history
│   ├── metadata.json     # Collection metadata
│   └── attributes.json   # Trait information
├── bored_apes/
├── azuki/
└── art_blocks/


### Dataset Download and Preprocessing


# Download datasets (requires authentication)
python scripts/download_datasets.py --collections cryptopunks bored_apes azuki art_blocks

# Preprocess data for training
python scripts/preprocess_data.py --input_dir data/raw --output_dir data/processed

# Extract visual features
python scripts/extract_visual_features.py --data_dir data/processed


## Model Training

### Configuration

Model configuration is managed through YAML files in the `configs/` directory. The default configuration provides optimal settings for CryptoPunks analysis.


# configs/hivifan_default.yaml
model:
  visual_embed_dim: 768
  market_embed_dim: 512
  fusion_dim: 1024
  pyramid_levels: [4, 8, 16, 32]
  attention_heads: 16

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  gradient_clip_norm: 1.0

### Training Execution


# Single GPU training
python train.py --config configs/hivifan_default.yaml --data_dir data/processed/cryptopunks

# Multi-GPU distributed training
python -m torch.distributed.launch --nproc_per_node=4 train.py \
    --config configs/hivifan_default.yaml \
    --data_dir data/processed/cryptopunks \
    --distributed

# Resume from checkpoint
python train.py --config configs/hivifan_default.yaml \
    --resume checkpoints/hivifan_cryptopunks/best_model.pth

### Monitoring Training Progress


# Launch TensorBoard
tensorboard --logdir logs/

# View training metrics in browser
# Navigate to http://localhost:6006


## Model Evaluation

### Reproducing Paper Results


# Evaluate on test set
python evaluate.py --model_path checkpoints/hivifan_cryptopunks/best_model.pth \
    --data_dir data/processed/cryptopunks \
    --output_dir results/cryptopunks

# Cross-collection evaluation
python evaluate_cross_collection.py --model_path checkpoints/hivifan_cryptopunks/best_model.pth \
    --collections bored_apes azuki art_blocks \
    --output_dir results/cross_collection

# Statistical significance testing
python scripts/statistical_analysis.py --results_dir results/ \
    --output_file results/statistical_tests.json

### Baseline Comparisons


# Train and evaluate all baseline methods
python experiments/run_baselines.py --data_dir data/processed/cryptopunks \
    --methods clip_based vit_temporal cnn_lstm vanilla_vae \
    --output_dir results/baselines

# Generate comparison tables
python scripts/generate_comparison_tables.py --results_dir results/ \
    --output_file results/comparison_table.tex

### Ablation Studies


# Run comprehensive ablation experiments
python experiments/ablation_study.py --config configs/ablation_configs/ \
    --data_dir data/processed/cryptopunks \
    --output_dir results/ablation

# Analyze component contributions
python scripts/analyze_ablation_results.py --results_dir results/ablation \
    --output_file results/ablation_analysis.json


## Interactive Visual Analytics

### Dashboard Launch


# Start visual analytics dashboard
python dashboard/app.py --config configs/dashboard_config.yaml \
    --model_path checkpoints/hivifan_cryptopunks/best_model.pth \
    --data_dir data/processed/cryptopunks \
    --port 8050

# Access dashboard at http://localhost:8050


### Dashboard Features

The interactive dashboard provides several visualization components:

- **Real-time Latent Space Visualization**: GPU-accelerated t-SNE with price color mapping
- **Attention Flow Diagrams**: Dynamic visualization of model attention patterns
- **3D Market Sentiment Surfaces**: Interactive exploration of visual-financial relationships
- **Trait Evolution Trees**: Hierarchical visualization of visual trait propagation

### Generating Static Visualizations


# Create publication-ready figures
python scripts/generate_figures.py --model_path checkpoints/hivifan_cryptopunks/best_model.pth \
    --data_dir data/processed/cryptopunks \
    --output_dir figures/ \
    --formats pdf png svg

# Generate attention visualization examples
python scripts/attention_examples.py --model_path checkpoints/hivifan_cryptopunks/best_model.pth \
    --input_images examples/sample_nfts/ \
    --output_dir figures/attention_examples/

## Performance Optimization

### GPU Memory Optimization

# Enable gradient checkpointing for reduced memory usage
python train.py --config configs/hivifan_default.yaml \
    --gradient_checkpointing \
    --mixed_precision

# Use model sharding for large-scale inference
python inference.py --model_path checkpoints/hivifan_cryptopunks/best_model.pth \
    --data_dir data/processed/cryptopunks \
    --shard_model \
    --batch_size 128


### Distributed Inference


# Multi-GPU batch inference
python -m torch.distributed.launch --nproc_per_node=4 inference_distributed.py \
    --model_path checkpoints/hivifan_cryptopunks/best_model.pth \
    --data_dir data/processed/ \
    --collections cryptopunks bored_apes azuki art_blocks


## Customization and Extension

### Adding New NFT Collections


# Create new collection configuration
python scripts/create_collection_config.py --collection_name new_collection \
    --image_dir path/to/images \
    --metadata_file path/to/metadata.json

# Preprocess new collection data
python scripts/preprocess_data.py --input_dir data/new_collection \
    --output_dir data/processed/new_collection \
    --config configs/new_collection_config.yaml


### Model Architecture Modifications

The codebase supports modular architecture modifications through configuration files:


# Custom architecture configuration
model:
  visual_encoder:
    type: "custom_pyramid"
    levels: [2, 4, 8, 16, 32]
    channels: [128, 256, 512, 1024, 2048]
  
  attention_mechanism:
    type: "enhanced_cross_modal"
    heads: 32
    volatility_adaptation: true
    temperature_scaling: true


## Testing and Validation

### Unit Tests


# Run comprehensive test suite
python -m pytest tests/ -v --cov=hivifan

# Test specific components
python -m pytest tests/test_visual_pyramid.py -v
python -m pytest tests/test_attention_mechanism.py -v
python -m pytest tests/test_data_loading.py -v


### Integration Tests


# End-to-end pipeline testing
python tests/integration/test_full_pipeline.py --data_dir data/test_samples

# Performance benchmarking
python tests/benchmarks/benchmark_inference.py --model_path checkpoints/hivifan_cryptopunks/best_model.pth

## Troubleshooting

### Common Issues

**Out of Memory Errors**: Reduce batch size in configuration files or enable gradient checkpointing. For inference, use model sharding or distributed processing.

**CUDA Compatibility**: Ensure PyTorch CUDA version matches your system CUDA installation. Reinstall PyTorch with correct CUDA version if necessary.

**Data Loading Errors**: Verify dataset directory structure matches expected format. Run data validation scripts to check file integrity.

**Dashboard Connection Issues**: Check firewall settings and ensure port 8050 is available. Use `--host 0.0.0.0` flag for remote access.

### Performance Optimization

For optimal performance on different hardware configurations:

- **RTX 3070/3080**: Use batch_size=16, enable mixed precision training
- **RTX 3090**: Use batch_size=32, standard configuration
- **A100**: Use batch_size=64, enable large model variants
