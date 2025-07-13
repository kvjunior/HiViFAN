"""
Configuration and Shared Utilities for HiViFAN
IEEE TVCG Submission - Configuration Management and Utility Functions

This module provides comprehensive configuration management, data processing pipelines,
evaluation metrics, and shared utilities ensuring reproducibility and efficiency
across all components of the HiViFAN system. The implementation emphasizes numerical
stability, efficient data handling, and rigorous experimental control.

The utilities support multi-threaded preprocessing, deterministic operations,
and seamless integration with popular experiment tracking frameworks while
maintaining the highest standards of scientific computing.

Authors: [Anonymized for Review]
Version: 1.0.0
License: MIT
"""

import os
import json
import yaml
import pickle
import hashlib
import random
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
import torchvision.models as models

from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_fscore_support, roc_auc_score,
    average_precision_score, confusion_matrix
)
import wandb
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class GlobalConfig:
    """
    Comprehensive global configuration for the HiViFAN system.
    Provides centralized configuration management with validation and versioning.
    """
    
    # Project settings
    project_name: str = "HiViFAN_TVCG"
    experiment_name: str = "baseline"
    version: str = "1.0.0"
    
    # Data configuration
    data_root: str = "./data"
    dataset_name: str = "cryptopunks"
    image_size: int = 224
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Model configuration
    model_name: str = "hivifan"
    pretrained_weights: Optional[str] = None
    checkpoint_path: Optional[str] = None
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_epochs: int = 5
    warmup_factor: float = 0.1
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Optimization configuration
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7
    
    # Mixed precision training
    use_amp: bool = True
    amp_opt_level: str = "O1"
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    master_port: str = "12355"
    
    # Evaluation configuration
    eval_batch_size: int = 64
    eval_frequency: int = 5
    save_frequency: int = 10
    early_stopping_patience: int = 20
    early_stopping_delta: float = 1e-4
    
    # Augmentation configuration
    augmentation_strength: float = 0.5
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    
    # Loss configuration
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'price': 1.0,
        'efficiency': 0.5,
        'reconstruction': 0.1,
        'kl_divergence': 0.01
    })
    
    # Logging configuration
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    tensorboard_dir: str = "./tensorboard"
    wandb_project: str = "hivifan-tvcg"
    wandb_entity: Optional[str] = None
    use_wandb: bool = True
    log_frequency: int = 100
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    
    # Hardware configuration
    device: str = "cuda"
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    
    # Validation configuration
    validation_metrics: List[str] = field(default_factory=lambda: [
        'mae', 'mse', 'rmse', 'r2', 'mape', 'correlation',
        'price_accuracy', 'directional_accuracy'
    ])
    
    # Debug configuration
    debug: bool = False
    debug_samples: int = 100
    profile: bool = False
    
    def __post_init__(self):
        """Validate configuration and create necessary directories."""
        self._validate_config()
        self._create_directories()
        self._setup_device()
        
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate batch sizes
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.eval_batch_size > 0, "Evaluation batch size must be positive"
        
        # Validate learning rate
        assert 0 < self.learning_rate < 1, "Learning rate must be in (0, 1)"
        
        # Validate epochs
        assert self.num_epochs > 0, "Number of epochs must be positive"
        assert 0 <= self.warmup_epochs <= self.num_epochs, "Invalid warmup epochs"
        
        # Validate paths
        assert self.data_root, "Data root path must be specified"
        
        # Validate distributed settings
        if self.distributed:
            assert self.world_size > 1, "World size must be > 1 for distributed training"
            assert 0 <= self.rank < self.world_size, "Invalid rank"
            
    def _create_directories(self):
        """Create necessary directories for outputs."""
        directories = [
            self.log_dir,
            self.checkpoint_dir,
            self.results_dir,
            self.tensorboard_dir,
            os.path.join(self.checkpoint_dir, self.experiment_name),
            os.path.join(self.results_dir, self.experiment_name),
            os.path.join(self.tensorboard_dir, self.experiment_name)
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def _setup_device(self):
        """Setup computing device configuration."""
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.gpu_ids = []
            
        if self.device == "cuda" and self.gpu_ids:
            # Validate GPU IDs
            available_gpus = torch.cuda.device_count()
            self.gpu_ids = [gpu_id for gpu_id in self.gpu_ids if gpu_id < available_gpus]
            
            if not self.gpu_ids:
                logger.warning("No valid GPU IDs, using GPU 0")
                self.gpu_ids = [0]
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to file."""
        config_dict = self.to_dict()
        config_dict['timestamp'] = datetime.now().isoformat()
        config_dict['version'] = self.version
        
        if path.endswith('.yaml'):
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration format: {path}")
            
    @classmethod
    def load(cls, path: str) -> 'GlobalConfig':
        """Load configuration from file."""
        if path.endswith('.yaml'):
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {path}")
            
        # Remove metadata fields
        config_dict.pop('timestamp', None)
        config_dict.pop('version', None)
        
        return cls(**config_dict)
    
    def get_experiment_id(self) -> str:
        """Generate unique experiment identifier."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_hash = hashlib.md5(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"{self.experiment_name}_{timestamp}_{config_hash}"


class NFTDataset(Dataset):
    """
    Comprehensive dataset class for NFT visual and market data.
    Handles multi-modal data loading, preprocessing, and augmentation.
    """
    
    def __init__(self, 
                 config: GlobalConfig,
                 data_file: str,
                 mode: str = 'train',
                 transform: Optional[Callable] = None):
        """
        Initialize NFT dataset.
        
        Args:
            config: Global configuration object
            data_file: Path to data manifest file
            mode: Dataset mode ('train', 'val', 'test')
            transform: Optional transform pipeline
        """
        self.config = config
        self.mode = mode
        self.transform = transform or self._get_default_transform()
        
        # Load data manifest
        self.data = self._load_data_manifest(data_file)
        
        # Setup data indices
        self._setup_indices()
        
        # Initialize data cache
        self.cache = {} if config.pin_memory else None
        
        # Setup data statistics
        self.stats = self._compute_statistics()
        
    def _load_data_manifest(self, data_file: str) -> pd.DataFrame:
        """Load and validate data manifest."""
        if data_file.endswith('.csv'):
            data = pd.read_csv(data_file)
        elif data_file.endswith('.json'):
            data = pd.read_json(data_file)
        elif data_file.endswith('.parquet'):
            data = pd.read_parquet(data_file)
        else:
            raise ValueError(f"Unsupported data format: {data_file}")
            
        # Validate required columns
        required_columns = ['image_path', 'price', 'timestamp', 'attributes']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Parse attributes if stored as string
        if isinstance(data['attributes'].iloc[0], str):
            data['attributes'] = data['attributes'].apply(json.loads)
            
        return data
    
    def _setup_indices(self):
        """Setup train/val/test indices."""
        n_samples = len(self.data)
        indices = np.arange(n_samples)
        
        if self.mode == 'train':
            self.indices = indices[:int(0.8 * n_samples)]
        elif self.mode == 'val':
            self.indices = indices[int(0.8 * n_samples):int(0.9 * n_samples)]
        else:  # test
            self.indices = indices[int(0.9 * n_samples):]
            
        logger.info(f"Dataset {self.mode}: {len(self.indices)} samples")
        
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics for normalization."""
        stats = {
            'price_mean': self.data['price'].mean(),
            'price_std': self.data['price'].std(),
            'price_min': self.data['price'].min(),
            'price_max': self.data['price'].max(),
            'n_unique_attributes': len(set(
                attr for attrs in self.data['attributes'] 
                for attr in attrs
            ))
        }
        
        # Compute image statistics if needed
        if self.config.compute_image_stats:
            stats.update(self._compute_image_statistics())
            
        return stats
    
    def _compute_image_statistics(self) -> Dict[str, Any]:
        """Compute image statistics for normalization."""
        # Sample subset of images for efficiency
        sample_size = min(1000, len(self.indices))
        sample_indices = np.random.choice(self.indices, sample_size, replace=False)
        
        pixel_values = []
        for idx in sample_indices:
            image_path = os.path.join(
                self.config.data_root, 
                self.data.iloc[idx]['image_path']
            )
            image = Image.open(image_path).convert('RGB')
            image = np.array(image) / 255.0
            pixel_values.append(image.reshape(-1, 3))
            
        pixel_values = np.concatenate(pixel_values, axis=0)
        
        return {
            'image_mean': pixel_values.mean(axis=0).tolist(),
            'image_std': pixel_values.std(axis=0).tolist()
        }
    
    def _get_default_transform(self) -> Callable:
        """Get default transform pipeline based on mode."""
        if self.mode == 'train':
            return A.Compose([
                A.Resize(self.config.image_size, self.config.image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.config.image_size, self.config.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Returns:
            Dictionary containing:
                - visual: Image tensor
                - market: Market features tensor
                - targets: Target values dictionary
        """
        # Get actual data index
        data_idx = self.indices[idx]
        
        # Check cache
        if self.cache is not None and data_idx in self.cache:
            return self.cache[data_idx]
        
        # Load data
        row = self.data.iloc[data_idx]
        
        # Load and transform image
        image_path = os.path.join(self.config.data_root, row['image_path'])
        image = self._load_image(image_path)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Extract market features
        market_features = self._extract_market_features(row)
        
        # Prepare targets
        targets = {
            'prices': torch.tensor(row['price'], dtype=torch.float32),
            'timestamp': torch.tensor(row['timestamp'].timestamp(), dtype=torch.float32),
            'efficiency_targets': self._extract_efficiency_targets(row)
        }
        
        # Create sample dictionary
        sample = {
            'visual': image,
            'market': market_features,
            'targets': targets,
            'metadata': {
                'image_path': row['image_path'],
                'attributes': row['attributes'],
                'index': data_idx
            }
        }
        
        # Cache if enabled
        if self.cache is not None:
            self.cache[data_idx] = sample
            
        return sample
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image with error handling."""
        try:
            image = Image.open(image_path).convert('RGB')
            return np.array(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return placeholder image
            return np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
    
    def _extract_market_features(self, row: pd.Series) -> torch.Tensor:
        """Extract market features from data row."""
        features = []
        
        # Price features
        features.extend([
            row['price'] / self.stats['price_mean'],  # Normalized price
            np.log1p(row['price']),  # Log price
            (row['price'] - self.stats['price_min']) / 
            (self.stats['price_max'] - self.stats['price_min'])  # Min-max normalized
        ])
        
        # Time features
        timestamp = row['timestamp']
        features.extend([
            timestamp.year - 2020,  # Years since 2020
            timestamp.month / 12.0,  # Normalized month
            timestamp.day / 31.0,  # Normalized day
            timestamp.hour / 24.0,  # Normalized hour
            timestamp.weekday() / 7.0  # Normalized weekday
        ])
        
        # Attribute features (simplified encoding)
        attribute_vector = np.zeros(100)  # Fixed size attribute vector
        for i, attr in enumerate(row['attributes'][:100]):
            attribute_vector[i] = hash(attr) % 2  # Binary encoding
        features.extend(attribute_vector.tolist())
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_efficiency_targets(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        """Extract market efficiency targets."""
        # Placeholder implementation - would use actual market data
        return {
            'liquidity': torch.rand(1),
            'price_stability': torch.rand(1),
            'market_depth': torch.rand(1) * 100,
            'trading_volume': torch.rand(1) * 1000
        }
    
    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Get weighted sampler for balanced training."""
        # Compute sample weights based on price distribution
        prices = self.data.iloc[self.indices]['price'].values
        
        # Use inverse frequency weighting
        price_bins = np.histogram(prices, bins=10)[0]
        bin_indices = np.digitize(prices, np.histogram(prices, bins=10)[1])
        
        weights = 1.0 / price_bins[np.clip(bin_indices - 1, 0, 9)]
        weights = weights / weights.sum() * len(weights)
        
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )


class DataProcessor:
    """
    Comprehensive data processing pipeline with multi-threaded preprocessing
    and efficient batch generation.
    """
    
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        
    def create_data_loaders(self, 
                          train_data: str,
                          val_data: str,
                          test_data: Optional[str] = None) -> Dict[str, DataLoader]:
        """
        Create data loaders for training, validation, and testing.
        
        Args:
            train_data: Path to training data
            val_data: Path to validation data
            test_data: Optional path to test data
            
        Returns:
            Dictionary of data loaders
        """
        # Create datasets
        train_dataset = NFTDataset(self.config, train_data, mode='train')
        val_dataset = NFTDataset(self.config, val_data, mode='val')
        
        loaders = {}
        
        # Training loader with weighted sampling
        train_sampler = train_dataset.get_weighted_sampler() if self.config.use_weighted_sampling else None
        
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers,
            collate_fn=self.collate_fn
        )
        
        # Validation loader
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self.collate_fn
        )
        
        # Test loader if provided
        if test_data:
            test_dataset = NFTDataset(self.config, test_data, mode='test')
            loaders['test'] = DataLoader(
                test_dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                collate_fn=self.collate_fn
            )
        
        return loaders
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for batching multi-modal data.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Batched dictionary
        """
        # Stack visual features
        visual = torch.stack([sample['visual'] for sample in batch])
        
        # Stack market features
        market = torch.stack([sample['market'] for sample in batch])
        
        # Batch targets
        targets = {}
        for key in batch[0]['targets']:
            if isinstance(batch[0]['targets'][key], dict):
                targets[key] = {}
                for subkey in batch[0]['targets'][key]:
                    targets[key][subkey] = torch.stack([
                        sample['targets'][key][subkey] for sample in batch
                    ])
            else:
                targets[key] = torch.stack([
                    sample['targets'][key] for sample in batch
                ])
        
        # Collect metadata
        metadata = {
            'image_paths': [sample['metadata']['image_path'] for sample in batch],
            'attributes': [sample['metadata']['attributes'] for sample in batch],
            'indices': [sample['metadata']['index'] for sample in batch]
        }
        
        return {
            'visual': visual,
            'market': market,
            'targets': targets,
            'metadata': metadata
        }
    
    def preprocess_market_data(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess market data with feature engineering and scaling.
        
        Args:
            market_data: Raw market data
            
        Returns:
            Preprocessed feature array
        """
        # Feature engineering
        engineered_features = self._engineer_market_features(market_data)
        
        # Scale features
        if 'market' not in self.scalers:
            self.scalers['market'] = StandardScaler()
            scaled_features = self.scalers['market'].fit_transform(engineered_features)
        else:
            scaled_features = self.scalers['market'].transform(engineered_features)
            
        return scaled_features
    
    def _engineer_market_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Engineer features from market data."""
        features = []
        
        # Price features
        features.append(market_data['price'].values)
        features.append(np.log1p(market_data['price'].values))
        
        # Rolling statistics
        for window in [7, 14, 30]:
            features.append(market_data['price'].rolling(window).mean().fillna(0).values)
            features.append(market_data['price'].rolling(window).std().fillna(0).values)
            
        # Price momentum
        for lag in [1, 7, 30]:
            features.append(market_data['price'].pct_change(lag).fillna(0).values)
            
        # Volume features if available
        if 'volume' in market_data.columns:
            features.append(market_data['volume'].values)
            features.append(np.log1p(market_data['volume'].values))
            
        # Stack features
        return np.column_stack(features)
    
    def augment_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply mixup or cutmix augmentation to batch.
        
        Args:
            batch: Input batch
            
        Returns:
            Augmented batch
        """
        if self.config.use_mixup and np.random.random() < 0.5:
            batch = self._apply_mixup(batch)
        elif self.config.use_cutmix and np.random.random() < 0.5:
            batch = self._apply_cutmix(batch)
            
        return batch
    
    def _apply_mixup(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply mixup augmentation."""
        batch_size = batch['visual'].size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        
        # Random permutation
        index = torch.randperm(batch_size).to(batch['visual'].device)
        
        # Mix visual features
        batch['visual'] = lam * batch['visual'] + (1 - lam) * batch['visual'][index]
        
        # Mix targets
        for key in batch['targets']:
            if isinstance(batch['targets'][key], dict):
                for subkey in batch['targets'][key]:
                    batch['targets'][key][subkey] = (
                        lam * batch['targets'][key][subkey] + 
                        (1 - lam) * batch['targets'][key][subkey][index]
                    )
            else:
                batch['targets'][key] = (
                    lam * batch['targets'][key] + 
                    (1 - lam) * batch['targets'][key][index]
                )
                
        # Store mixing coefficient
        batch['lam'] = lam
        batch['index'] = index
        
        return batch
    
    def _apply_cutmix(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply CutMix augmentation."""
        batch_size = batch['visual'].size(0)
        
        # Sample lambda
        lam = np.random.beta(self.config.cutmix_alpha, self.config.cutmix_alpha)
        
        # Random permutation
        index = torch.randperm(batch_size).to(batch['visual'].device)
        
        # Generate random box
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(
            batch['visual'].size(),
            lam
        )
        
        # Apply CutMix to visual features
        batch['visual'][:, :, bbx1:bbx2, bby1:bby2] = \
            batch['visual'][index, :, bbx1:bbx2, bby1:bby2]
            
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / 
                  (batch['visual'].size()[-2] * batch['visual'].size()[-1]))
        
        # Mix targets
        for key in batch['targets']:
            if isinstance(batch['targets'][key], dict):
                for subkey in batch['targets'][key]:
                    batch['targets'][key][subkey] = (
                        lam * batch['targets'][key][subkey] + 
                        (1 - lam) * batch['targets'][key][subkey][index]
                    )
            else:
                batch['targets'][key] = (
                    lam * batch['targets'][key] + 
                    (1 - lam) * batch['targets'][key][index]
                )
                
        return batch
    
    def _rand_bbox(self, size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
        """Generate random bounding box for CutMix."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


class MetricsCalculator:
    """
    Comprehensive metrics calculator with numerical stability and
    efficient computation for all evaluation metrics.
    """
    
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.metrics_history = defaultdict(list)
        
    def calculate_metrics(self,
                         predictions: torch.Tensor,
                         targets: torch.Tensor,
                         prefix: str = '') -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            prefix: Metric name prefix
            
        Returns:
            Dictionary of computed metrics
        """
        # Convert to numpy for sklearn metrics
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        metrics = {}
        
        # Regression metrics
        metrics[f'{prefix}mae'] = self._safe_metric(
            mean_absolute_error, targets, predictions
        )
        metrics[f'{prefix}mse'] = self._safe_metric(
            mean_squared_error, targets, predictions
        )
        metrics[f'{prefix}rmse'] = np.sqrt(metrics[f'{prefix}mse'])
        metrics[f'{prefix}r2'] = self._safe_metric(
            r2_score, targets, predictions
        )
        
        # Percentage error metrics
        metrics[f'{prefix}mape'] = self._calculate_mape(targets, predictions)
        metrics[f'{prefix}smape'] = self._calculate_smape(targets, predictions)
        
        # Correlation metrics
        metrics[f'{prefix}pearson_r'] = self._calculate_correlation(
            targets, predictions, method='pearson'
        )
        metrics[f'{prefix}spearman_r'] = self._calculate_correlation(
            targets, predictions, method='spearman'
        )
        
        # Market-specific metrics
        metrics[f'{prefix}directional_accuracy'] = self._calculate_directional_accuracy(
            targets, predictions
        )
        metrics[f'{prefix}price_accuracy'] = self._calculate_price_accuracy(
            targets, predictions
        )
        
        # Percentile metrics
        for percentile in [25, 50, 75, 90, 95]:
            metrics[f'{prefix}mae_p{percentile}'] = self._calculate_percentile_mae(
                targets, predictions, percentile
            )
            
        # Update history
        for name, value in metrics.items():
            self.metrics_history[name].append(value)
            
        return metrics
    
    def _safe_metric(self, metric_fn: Callable, y_true: np.ndarray,
                    y_pred: np.ndarray, **kwargs) -> float:
        """Safely compute metric with error handling."""
        try:
            return float(metric_fn(y_true, y_pred, **kwargs))
        except Exception as e:
            logger.warning(f"Error computing metric {metric_fn.__name__}: {e}")
            return float('nan')
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error with stability."""
        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return float('nan')
            
        return float(np.mean(
            np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        ) * 100)
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        
        if not mask.any():
            return float('nan')
            
        return float(np.mean(
            np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
        ) * 100)
    
    def _calculate_correlation(self, y_true: np.ndarray, y_pred: np.ndarray,
                             method: str = 'pearson') -> float:
        """Calculate correlation coefficient."""
        try:
            if method == 'pearson':
                return float(np.corrcoef(y_true, y_pred)[0, 1])
            elif method == 'spearman':
                from scipy.stats import spearmanr
                return float(spearmanr(y_true, y_pred)[0])
            else:
                raise ValueError(f"Unknown correlation method: {method}")
        except Exception:
            return float('nan')
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray,
                                      y_pred: np.ndarray) -> float:
        """Calculate directional accuracy for time series."""
        if len(y_true) < 2:
            return float('nan')
            
        # Calculate directional changes
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # Calculate accuracy
        correct = (true_direction == pred_direction).sum()
        total = len(true_direction)
        
        return float(correct / total) if total > 0 else float('nan')
    
    def _calculate_price_accuracy(self, y_true: np.ndarray,
                                y_pred: np.ndarray,
                                tolerance: float = 0.1) -> float:
        """Calculate price prediction accuracy within tolerance."""
        relative_error = np.abs((y_pred - y_true) / y_true)
        within_tolerance = relative_error <= tolerance
        
        return float(np.mean(within_tolerance))
    
    def _calculate_percentile_mae(self, y_true: np.ndarray, y_pred: np.ndarray,
                                percentile: float) -> float:
        """Calculate MAE for specific percentile of data."""
        threshold = np.percentile(y_true, percentile)
        mask = y_true >= threshold
        
        if not mask.any():
            return float('nan')
            
        return float(mean_absolute_error(y_true[mask], y_pred[mask]))
    
    def calculate_efficiency_metrics(self,
                                   efficiency_predictions: Dict[str, torch.Tensor],
                                   efficiency_targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate market efficiency metrics."""
        metrics = {}
        
        for metric_name in ['liquidity', 'price_stability', 'market_depth', 'trading_volume']:
            if metric_name in efficiency_predictions and metric_name in efficiency_targets:
                pred = efficiency_predictions[metric_name]
                target = efficiency_targets[metric_name]
                
                # Binary metrics (liquidity, price_stability)
                if metric_name in ['liquidity', 'price_stability']:
                    pred_binary = (pred > 0.5).float()
                    target_binary = (target > 0.5).float()
                    
                    # Calculate binary classification metrics
                    metrics[f'{metric_name}_accuracy'] = float(
                        (pred_binary == target_binary).float().mean()
                    )
                    
                    if target_binary.sum() > 0:
                        metrics[f'{metric_name}_auc'] = self._safe_metric(
                            roc_auc_score,
                            target_binary.cpu().numpy(),
                            pred.cpu().numpy()
                        )
                    
                # Continuous metrics
                else:
                    metrics[f'{metric_name}_mae'] = self._safe_metric(
                        mean_absolute_error,
                        target.cpu().numpy(),
                        pred.cpu().numpy()
                    )
                    metrics[f'{metric_name}_r2'] = self._safe_metric(
                        r2_score,
                        target.cpu().numpy(),
                        pred.cpu().numpy()
                    )
                    
        return metrics
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics of all tracked metrics."""
        summary_data = []
        
        for metric_name, values in self.metrics_history.items():
            if values:
                summary_data.append({
                    'metric': metric_name,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'last': values[-1]
                })
                
        return pd.DataFrame(summary_data)


class ExperimentLogger:
    """
    Comprehensive experiment logging with TensorBoard and Weights & Biases
    integration for tracking experiments and visualizations.
    """
    
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.experiment_id = config.get_experiment_id()
        
        # Setup TensorBoard
        self.tb_writer = SummaryWriter(
            log_dir=os.path.join(config.tensorboard_dir, self.experiment_id)
        )
        
        # Setup Weights & Biases
        if config.use_wandb:
            self._setup_wandb()
        else:
            self.wandb_run = None
            
        # Setup file logging
        self._setup_file_logging()
        
        # Track global step
        self.global_step = 0
        
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.experiment_id,
            config=self.config.to_dict(),
            tags=[self.config.dataset_name, self.config.model_name],
            resume='allow'
        )
        self.wandb_run = wandb.run
        
    def _setup_file_logging(self):
        """Setup file-based logging."""
        log_file = os.path.join(
            self.config.log_dir,
            self.experiment_id,
            'experiment.log'
        )
        
        # Create directory
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None,
                   phase: str = 'train'):
        """Log metrics to all tracking systems."""
        if step is None:
            step = self.global_step
            
        # Log to TensorBoard
        for name, value in metrics.items():
            self.tb_writer.add_scalar(f'{phase}/{name}', value, step)
            
        # Log to Weights & Biases
        if self.wandb_run:
            wandb_metrics = {f'{phase}/{name}': value for name, value in metrics.items()}
            wandb.log(wandb_metrics, step=step)
            
        # Log to console
        metric_str = ', '.join([f'{name}: {value:.4f}' for name, value in metrics.items()])
        logger.info(f'[Step {step}] {phase} - {metric_str}')
        
    def log_images(self, images: Dict[str, torch.Tensor], step: Optional[int] = None):
        """Log images to tracking systems."""
        if step is None:
            step = self.global_step
            
        for name, image in images.items():
            # Ensure correct format
            if image.dim() == 4:  # Batch of images
                image = image[0]  # Take first image
            if image.dim() == 3 and image.size(0) in [1, 3]:
                pass  # Correct format
            else:
                logger.warning(f"Skipping image {name} with shape {image.shape}")
                continue
                
            # Log to TensorBoard
            self.tb_writer.add_image(name, image, step)
            
            # Log to Weights & Biases
            if self.wandb_run:
                wandb.log({name: wandb.Image(image)}, step=step)
                
    def log_model_graph(self, model: nn.Module, input_sample: Dict[str, torch.Tensor]):
        """Log model architecture graph."""
        try:
            self.tb_writer.add_graph(
                model,
                [input_sample['visual'], input_sample['market']]
            )
        except Exception as e:
            logger.warning(f"Could not log model graph: {e}")
            
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters and their corresponding metrics."""
        # TensorBoard hyperparameter logging
        self.tb_writer.add_hparams(hparams, metrics)
        
        # Weights & Biases
        if self.wandb_run:
            wandb.config.update(hparams)
            
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: Optional[int] = None):
        """Log current learning rate."""
        if step is None:
            step = self.global_step
            
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.tb_writer.add_scalar(f'train/lr_group_{i}', lr, step)
            
            if self.wandb_run:
                wandb.log({f'train/lr_group_{i}': lr}, step=step)
                
    def log_gradient_statistics(self, model: nn.Module, step: Optional[int] = None):
        """Log gradient statistics for debugging."""
        if step is None:
            step = self.global_step
            
        gradient_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_data = param.grad.data
                
                gradient_stats[f'gradients/{name}_mean'] = grad_data.mean().item()
                gradient_stats[f'gradients/{name}_std'] = grad_data.std().item()
                gradient_stats[f'gradients/{name}_max'] = grad_data.max().item()
                gradient_stats[f'gradients/{name}_min'] = grad_data.min().item()
                
        # Log to TensorBoard
        for name, value in gradient_stats.items():
            self.tb_writer.add_scalar(name, value, step)
            
    def save_checkpoint(self, checkpoint: Dict[str, Any], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(
            self.config.checkpoint_dir,
            self.experiment_id
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'checkpoint_step_{self.global_step}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
            
        # Log to Weights & Biases
        if self.wandb_run:
            wandb.save(checkpoint_path)
            if is_best:
                wandb.save(best_path)
                
    def close(self):
        """Close all logging resources."""
        self.tb_writer.close()
        
        if self.wandb_run:
            wandb.finish()
            
        logger.info(f"Experiment {self.experiment_id} logging closed")


class ReproducibilityManager:
    """
    Ensures reproducibility through seed management and deterministic operations.
    """
    
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.set_seeds()
        self.configure_deterministic_mode()
        
    def set_seeds(self):
        """Set random seeds for all libraries."""
        seed = self.config.seed
        
        # Python random
        random.seed(seed)
        
        # Numpy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Environment variables
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        logger.info(f"Set random seed to {seed}")
        
    def configure_deterministic_mode(self):
        """Configure PyTorch for deterministic operations."""
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            
            # Set environment variable for CUDA operations
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            
            logger.info("Enabled deterministic mode")
        else:
            # Enable benchmarking for performance
            torch.backends.cudnn.benchmark = self.config.benchmark
            logger.info(f"Deterministic mode disabled, benchmark={self.config.benchmark}")
            
    def get_random_states(self) -> Dict[str, Any]:
        """Get current random states for checkpointing."""
        return {
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'torch_cuda_random_state': [
                torch.cuda.get_rng_state(i) 
                for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else None
        }
        
    def set_random_states(self, states: Dict[str, Any]):
        """Restore random states from checkpoint."""
        random.setstate(states['python_random_state'])
        np.random.set_state(states['numpy_random_state'])
        torch.set_rng_state(states['torch_random_state'])
        
        if torch.cuda.is_available() and states['torch_cuda_random_state']:
            for i, state in enumerate(states['torch_cuda_random_state']):
                torch.cuda.set_rng_state(state, i)
                
        logger.info("Restored random states from checkpoint")


def setup_distributed_training(config: GlobalConfig):
    """Setup distributed training environment."""
    if config.distributed:
        # Initialize process group
        torch.distributed.init_process_group(
            backend=config.backend,
            init_method=f'tcp://localhost:{config.master_port}',
            world_size=config.world_size,
            rank=config.rank
        )
        
        # Set device
        torch.cuda.set_device(config.local_rank)
        
        logger.info(
            f"Initialized distributed training: "
            f"rank={config.rank}, world_size={config.world_size}"
        )
        
        # Synchronize
        torch.distributed.barrier()
        
        
def cleanup_distributed_training():
    """Cleanup distributed training resources."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Cleaned up distributed training")


def create_optimizer(model: nn.Module, config: GlobalConfig) -> torch.optim.Optimizer:
    """Create optimizer based on configuration."""
    # Separate parameters for different learning rates
    param_groups = [
        {'params': model.parameters(), 'lr': config.learning_rate}
    ]
    
    # Create optimizer
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
        
    logger.info(f"Created {config.optimizer} optimizer")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, 
                    config: GlobalConfig) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler based on configuration."""
    if config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=config.scheduler_min_lr
        )
    elif config.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=config.scheduler_factor
        )
    elif config.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
        
    logger.info(f"Created {config.scheduler} learning rate scheduler")
    
    return scheduler


def get_model_size(model: nn.Module) -> Dict[str, Any]:
    """Calculate model size and parameter statistics."""
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    # Calculate size in MB
    param_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': param_size_mb
    }


def profile_model(model: nn.Module, input_sample: Dict[str, torch.Tensor], 
                 config: GlobalConfig) -> Dict[str, Any]:
    """Profile model performance and memory usage."""
    from torch.profiler import profile, record_function, ProfilerActivity
    
    model.eval()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_sample['visual'], input_sample['market'])
            
    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("model_forward"):
            with torch.no_grad():
                _ = model(input_sample['visual'], input_sample['market'])
                
    # Get profiling results
    key_averages = prof.key_averages()
    
    # Extract metrics
    total_time = sum([item.cpu_time_total for item in key_averages])
    total_memory = sum([item.cpu_memory_usage for item in key_averages])
    
    profile_results = {
        'total_time_ms': total_time / 1000,
        'total_memory_mb': total_memory / (1024 * 1024),
        'top_operations': []
    }
    
    # Get top operations by time
    for item in sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:10]:
        profile_results['top_operations'].append({
            'name': item.key,
            'cpu_time_ms': item.cpu_time_total / 1000,
            'cuda_time_ms': item.cuda_time_total / 1000,
            'memory_mb': item.cpu_memory_usage / (1024 * 1024)
        })
        
    return profile_results


# Utility functions for data handling

def load_json_lines(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON lines file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_json_lines(data: List[Dict[str, Any]], file_path: str):
    """Save data to JSON lines file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def compute_dataset_statistics(dataset: Dataset) -> Dict[str, Any]:
    """Compute comprehensive dataset statistics."""
    stats = defaultdict(list)
    
    # Sample subset for efficiency
    sample_size = min(1000, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        
        # Visual statistics
        if 'visual' in sample:
            visual = sample['visual']
            stats['visual_mean'].append(visual.mean().item())
            stats['visual_std'].append(visual.std().item())
            
        # Price statistics
        if 'targets' in sample and 'prices' in sample['targets']:
            price = sample['targets']['prices'].item()
            stats['prices'].append(price)
            
    # Aggregate statistics
    aggregated_stats = {
        'visual_mean': np.mean(stats['visual_mean']),
        'visual_std': np.mean(stats['visual_std']),
        'price_mean': np.mean(stats['prices']),
        'price_std': np.std(stats['prices']),
        'price_min': np.min(stats['prices']),
        'price_max': np.max(stats['prices']),
        'price_median': np.median(stats['prices'])
    }
    
    return aggregated_stats


def create_model_summary(model: nn.Module, config: GlobalConfig) -> str:
    """Create detailed model summary."""
    from torchinfo import summary
    
    # Create dummy input
    batch_size = 1
    visual_input = torch.randn(batch_size, 3, config.image_size, config.image_size)
    market_input = torch.randn(batch_size, 128, 512)  # Adjust dimensions as needed
    
    # Generate summary
    model_summary = summary(
        model,
        input_data=[visual_input, market_input],
        verbose=0,
        col_names=['input_size', 'output_size', 'num_params', 'mult_adds'],
        col_width=20,
        row_settings=['var_names']
    )
    
    return str(model_summary)


# Main configuration loading function

def load_and_validate_config(config_path: str) -> GlobalConfig:
    """Load and validate configuration from file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    config = GlobalConfig.load(config_path)
    
    logger.info(f"Loaded configuration from {config_path}")
    logger.info(f"Experiment ID: {config.get_experiment_id()}")
    
    return config


if __name__ == "__main__":
    """Demonstration of configuration and utilities usage."""
    
    # Create default configuration
    config = GlobalConfig()
    
    # Save configuration
    config_path = os.path.join(config.results_dir, 'config.yaml')
    config.save(config_path)
    logger.info(f"Saved configuration to {config_path}")
    
    # Setup reproducibility
    reproducibility = ReproducibilityManager(config)
    
    # Create data processor
    processor = DataProcessor(config)
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator(config)
    
    # Initialize experiment logger
    exp_logger = ExperimentLogger(config)
    
    # Demonstrate metrics calculation
    dummy_predictions = torch.randn(100)
    dummy_targets = torch.randn(100)
    
    metrics = metrics_calc.calculate_metrics(dummy_predictions, dummy_targets, prefix='test_')
    logger.info(f"Calculated metrics: {metrics}")
    
    # Log metrics
    exp_logger.log_metrics(metrics, step=0, phase='test')
    
    # Get model size (dummy model)
    dummy_model = nn.Linear(100, 10)
    model_info = get_model_size(dummy_model)
    logger.info(f"Model info: {model_info}")
    
    # Close logging
    exp_logger.close()
    
    logger.info("Configuration and utilities demonstration completed")
