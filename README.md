# HANCross: Hierarchical Attention Networks for Cross-Modal Pattern Recognition

## Overview

HANCross is a theoretically-grounded framework for cross-modal learning between hierarchical visual features and temporal sequences. The system implements information-theoretic principles to optimally integrate multi-scale visual processing with adaptive temporal modeling through dynamic attention mechanisms. While validated on visual-temporal prediction in digital asset markets, the framework generalizes to diverse pattern recognition tasks requiring integration of visual and temporal modalities.

### Key Contributions

- **Information-Theoretic Architecture Design**: Optimal latent dimensionality and multi-scale decomposition derived through variational information bottleneck principles
- **Multi-Scale Visual Feature Pyramid**: Theoretically-justified four-level hierarchy based on frequency analysis and mutual information maximization
- **Dynamic Cross-Modal Attention**: Volatility-aware gating mechanism that adaptively balances visual and temporal features according to data characteristics
- **Temporal Coherence Modeling**: Dilated convolutions with exponentially increasing receptive fields capturing patterns from hourly to weekly scales
- **Generalization Capability**: Robust cross-domain performance without architectural modifications or extensive fine-tuning

## System Requirements

### Hardware Requirements
- **Minimum**: NVIDIA GPU with 8GB VRAM (RTX 3070 or equivalent)
- **Recommended**: NVIDIA RTX 3090 or A100 with 24GB+ VRAM for large-scale experiments
- **RAM**: 32GB system memory minimum, 64GB recommended for full dataset processing
- **Storage**: 150GB free space for datasets, features, and model checkpoints

### Software Dependencies
- Python 3.9 or higher
- CUDA 11.8 or higher with cuDNN 8.6+
- PyTorch 2.0 or higher with CUDA support
- NumPy 1.23+ with MKL acceleration

## Installation

### Environment Setup

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/HANCross-BB62/
cd HANCross

# Create conda environment with dependencies
conda create -n hancross python=3.9
conda activate hancross

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Docker Installation (Recommended for Reproducibility)

```bash
# Build Docker image with all dependencies
docker build -t hancross:latest .

# Run container with GPU support and volume mounting
docker run --gpus all -v $(pwd):/workspace -p 8888:8888 hancross:latest

# For interactive development
docker run --gpus all -it -v $(pwd):/workspace hancross:latest bash
```

## Dataset Preparation

### Supported Dataset Formats

HANCross processes visual-temporal paired data with the following structure:

```
data/
├── dataset_name/
│   ├── visual/              # Visual inputs (images/videos)
│   ├── temporal/            # Temporal sequences (CSV/NPY)
│   ├── metadata.json        # Dataset configuration
│   └── splits/              # Train/val/test indices
│       ├── train.json
│       ├── val.json
│       └── test.json
```

### Visual-Temporal Dataset Examples

The framework has been validated on multiple datasets demonstrating different visual styles and temporal characteristics:

- **Digital Asset Markets**: 167,492 NFT transactions with pixel art to generative compositions
- **Weather Prediction**: Satellite imagery with meteorological time series
- **Traffic Flow**: Street camera feeds with vehicle count sequences
- **Medical Monitoring**: X-ray sequences with patient vital signs

### Data Preprocessing Pipeline

```bash
# Generic preprocessing for visual-temporal datasets
python scripts/preprocess_visual_temporal.py \
    --visual_dir data/raw/visual \
    --temporal_dir data/raw/temporal \
    --output_dir data/processed \
    --visual_size 224 \
    --temporal_window 168 \
    --temporal_stride 1

# Compute dataset statistics for normalization
python scripts/compute_statistics.py \
    --data_dir data/processed \
    --output_file data/processed/statistics.json

# Validate data integrity and format
python scripts/validate_dataset.py \
    --data_dir data/processed \
    --check_visual --check_temporal --check_alignment
```

## Model Architecture Configuration

### Information-Theoretic Parameter Selection

The framework derives optimal parameters through theoretical analysis:

```yaml
# configs/hancross_optimal.yaml
model:
  # Latent dimension from information bottleneck analysis
  latent_dim: 256  # Derived from I(V,T;Y) ≈ 4.3 nats
  
  # Multi-scale pyramid from frequency analysis
  pyramid_levels: 4
  pyramid_resolutions: [224, 112, 56, 28]
  pyramid_channels: [64, 128, 256, 512]
  
  # Cross-modal attention configuration
  attention:
    num_heads: 16  # From pattern clustering analysis
    head_dim: 64
    volatility_adaptation: true
    adaptation_strength: 0.2
  
  # Temporal modeling parameters
  temporal:
    kernel_size: 7
    dilation_rates: [1, 2, 4, 8, 16]
    hidden_dim: 512

training:
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 1e-5
  gradient_clip_norm: 1.0
  mixed_precision: true
```

## Training Procedures

### Standard Training

```bash
# Train with default configuration
python train.py \
    --config configs/hancross_optimal.yaml \
    --data_dir data/processed/dataset_name \
    --experiment_name hancross_baseline

# Distributed training across multiple GPUs
torchrun --nproc_per_node=4 train_distributed.py \
    --config configs/hancross_optimal.yaml \
    --data_dir data/processed/dataset_name \
    --experiment_name hancross_distributed
```

### Advanced Training Options

```bash
# Information-theoretic regularization
python train.py \
    --config configs/hancross_optimal.yaml \
    --beta 0.1 \
    --mutual_information_estimation mine \
    --regularization_weight 0.01

# Progressive multi-scale training
python train_progressive.py \
    --config configs/hancross_optimal.yaml \
    --start_level 1 \
    --end_level 4 \
    --epochs_per_level 25

# Curriculum learning with volatility scheduling
python train_curriculum.py \
    --config configs/hancross_optimal.yaml \
    --volatility_schedule linear \
    --initial_volatility 0.1 \
    --final_volatility 0.5
```

### Training Monitoring

```bash
# Launch TensorBoard with custom metrics
tensorboard --logdir experiments/ --port 6006

# Monitor information-theoretic metrics
python scripts/monitor_information_metrics.py \
    --log_dir experiments/hancross_baseline \
    --metrics mutual_information compression_ratio attention_entropy
```

## Evaluation and Analysis

### Comprehensive Evaluation Suite

```bash
# Standard evaluation metrics
python evaluate.py \
    --model_path experiments/hancross_baseline/best_model.pth \
    --data_dir data/processed/dataset_name \
    --metrics r2 mae rmse correlation \
    --output_dir results/standard_evaluation

# Cross-modal alignment analysis
python analyze_cross_modal.py \
    --model_path experiments/hancross_baseline/best_model.pth \
    --data_dir data/processed/dataset_name \
    --analysis_type cca attention_consistency information_flow

# Hierarchical feature analysis
python analyze_pyramid_features.py \
    --model_path experiments/hancross_baseline/best_model.pth \
    --data_dir data/processed/dataset_name \
    --visualize_features --compute_mutual_information
```

### Cross-Domain Generalization

```bash
# Zero-shot evaluation on new domains
python evaluate_cross_domain.py \
    --model_path experiments/hancross_baseline/best_model.pth \
    --source_domain dataset_name \
    --target_domains weather traffic medical \
    --output_dir results/cross_domain

# Few-shot adaptation analysis
python few_shot_adaptation.py \
    --model_path experiments/hancross_baseline/best_model.pth \
    --target_domain new_dataset \
    --num_shots 10 50 100 500 1000 \
    --adaptation_layers fusion_only all
```

### Ablation Studies

```bash
# Systematic component ablation
python ablation_study.py \
    --base_config configs/hancross_optimal.yaml \
    --ablate pyramid attention temporal gating \
    --data_dir data/processed/dataset_name \
    --output_dir results/ablation

# Information-theoretic analysis
python information_analysis.py \
    --model_path experiments/hancross_baseline/best_model.pth \
    --data_dir data/processed/dataset_name \
    --compute_bounds --validate_theory
```

## Inference and Deployment

### Optimized Inference

```bash
# Single sample inference
python inference.py \
    --model_path experiments/hancross_baseline/best_model.pth \
    --visual_input samples/image.jpg \
    --temporal_input samples/sequence.csv \
    --output_format json

# Batch inference with optimization
python inference_batch.py \
    --model_path experiments/hancross_baseline/best_model.pth \
    --data_dir data/test \
    --batch_size 128 \
    --use_tensorrt --mixed_precision
```

### Model Optimization for Deployment

```bash
# Quantization for edge deployment
python optimize_model.py \
    --model_path experiments/hancross_baseline/best_model.pth \
    --optimization_type int8_quantization \
    --calibration_data data/processed/calibration \
    --output_path models/hancross_int8.pth

# ONNX export for cross-platform deployment
python export_onnx.py \
    --model_path experiments/hancross_baseline/best_model.pth \
    --output_path models/hancross.onnx \
    --opset_version 16 \
    --dynamic_axes batch_size
```

## Theoretical Validation

### Information-Theoretic Analysis Tools

```bash
# Validate theoretical predictions
python validate_theory.py \
    --model_path experiments/hancross_baseline/best_model.pth \
    --data_dir data/processed/dataset_name \
    --theory_checks latent_capacity scale_information attention_optimality

# Mutual information estimation
python estimate_mutual_information.py \
    --data_dir data/processed/dataset_name \
    --estimation_method mine kde variational \
    --num_samples 10000 \
    --output_file results/mutual_information_analysis.json

# Attention mechanism analysis
python analyze_attention_theory.py \
    --model_path experiments/hancross_baseline/best_model.pth \
    --data_dir data/processed/dataset_name \
    --compute_entropy --compute_specialization --validate_gating
```

## Extending the Framework

### Adding New Visual Encoders

```python
# custom_encoders/efficient_pyramid.py
from hancross.models.visual import BaseVisualEncoder

class EfficientPyramid(BaseVisualEncoder):
    def __init__(self, config):
        super().__init__(config)
        # Implementation following information-theoretic principles
        
    def compute_scale_features(self, x, level):
        # Scale-specific processing with theoretical justification
        pass
```

### Implementing Custom Fusion Mechanisms

```python
# custom_fusion/adaptive_fusion.py
from hancross.models.fusion import BaseFusionModule

class AdaptiveFusion(BaseFusionModule):
    def __init__(self, config):
        super().__init__(config)
        # Adaptive fusion with theoretical guarantees
        
    def compute_modality_weights(self, visual_features, temporal_features, context):
        # Information-theoretic weight computation
        pass
```

## Performance Benchmarks

### Computational Efficiency Analysis

Performance metrics on standard hardware configurations:

| Hardware | Batch Size | Inference Time | Memory Usage | Throughput |
|----------|------------|----------------|--------------|------------|
| RTX 3070 | 16 | 52.3ms | 6.8GB | 306 samples/s |
| RTX 3090 | 32 | 46.8ms | 12.4GB | 683 samples/s |
| A100 40GB | 64 | 42.1ms | 18.7GB | 1520 samples/s |
| A100 80GB | 128 | 38.9ms | 31.2GB | 3290 samples/s |

### Scaling Analysis

```bash
# Benchmark scaling behavior
python benchmark_scaling.py \
    --model_path experiments/hancross_baseline/best_model.pth \
    --data_sizes 1000 10000 100000 1000000 \
    --hardware_configs configs/hardware/ \
    --output_dir results/scaling_analysis
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{hancross2024,
  title={Hierarchical Attention Networks for Cross-Modal Pattern Recognition in Visual-Financial Data},
  author={Victor, Kombou and Xia, Qi and Gao, Jianbin and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  note={Under Review}
}
```

## License

This project is released under the MIT License. See LICENSE file for details.

## Acknowledgments

This work was supported in part by the National Natural Science Foundation of China (No. U22B2029) and Key Laboratory of Intelligent Space TTC&O (Space Engineering University), Ministry of Education (No. CYK2024-02-02).
