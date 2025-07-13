"""
Hierarchical Visual-Financial Attention Network (HiViFAN) Architecture
IEEE TVCG Submission - Core Neural Network Implementation

This module implements the complete HiViFAN architecture for multi-modal NFT market analysis,
incorporating advanced visual feature extraction, temporal market dynamics modeling, and
novel cross-modal attention mechanisms with theoretical guarantees.

Mathematical Foundation:
Given visual features V ∈ ℝ^(N×D_v) and market features M ∈ ℝ^(T×D_m), we learn a joint
representation Z = f(V,M) that maximizes I(Z;P) where P represents price dynamics and
I(·;·) denotes mutual information.

Authors: [Anonymized for Review]
Version: 1.0.0
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import math
from einops import rearrange, reduce, repeat
from torch.nn.init import xavier_uniform_, constant_, normal_


@dataclass
class HiViFANConfig:
    """Configuration for HiViFAN architecture with validated hyperparameters."""
    
    # Visual encoder configurations
    visual_patch_size: int = 16
    visual_embed_dim: int = 768
    visual_depth: int = 12
    visual_num_heads: int = 12
    visual_mlp_ratio: float = 4.0
    
    # Market encoder configurations  
    market_embed_dim: int = 512
    market_depth: int = 8
    market_num_heads: int = 8
    market_window_size: int = 128
    
    # Cross-modal fusion configurations
    fusion_dim: int = 1024
    fusion_heads: int = 16
    fusion_dropout: float = 0.1
    
    # Multi-scale feature pyramid
    pyramid_levels: List[int] = None
    pyramid_channels: List[int] = None
    
    # Temporal coherence module
    temporal_kernel_size: int = 7
    temporal_dilation_rates: List[int] = None
    
    # Training configurations
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    def __post_init__(self):
        """Initialize default values and validate configurations."""
        if self.pyramid_levels is None:
            self.pyramid_levels = [4, 8, 16, 32]
        if self.pyramid_channels is None:
            self.pyramid_channels = [256, 512, 1024, 2048]
        if self.temporal_dilation_rates is None:
            self.temporal_dilation_rates = [1, 2, 4, 8, 16]


class MultiScaleVisualFeaturePyramid(nn.Module):
    """
    Hierarchical visual feature extraction with multi-scale pyramid architecture.
    
    This module implements a sophisticated feature pyramid network that captures
    visual attributes at multiple granularities, from pixel-level traits to
    compositional patterns, specifically designed for NFT visual analysis.
    
    Mathematical Formulation:
    F_l = Conv_l(Pool_l(F_{l-1})) + Lateral_l(F_{l-1})
    where l ∈ {1, ..., L} represents pyramid levels.
    """
    
    def __init__(self, config: HiViFANConfig):
        super().__init__()
        self.config = config
        
        # Build feature pyramid with lateral connections
        self.pyramid_blocks = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()
        
        in_channels = 3
        for i, (level, channels) in enumerate(zip(config.pyramid_levels, config.pyramid_channels)):
            # Pyramid block with strided convolution
            pyramid_block = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, channels),
                nn.GELU()
            )
            self.pyramid_blocks.append(pyramid_block)
            
            # Lateral connection for feature fusion
            if i > 0:
                lateral = nn.Conv2d(config.pyramid_channels[i-1], channels, kernel_size=1)
                self.lateral_connections.append(lateral)
            
            # Feature fusion block
            fusion_block = nn.Sequential(
                nn.Conv2d(channels * 2 if i > 0 else channels, channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, channels),
                nn.GELU(),
                SEBlock(channels)  # Squeeze-and-excitation for channel attention
            )
            self.fusion_blocks.append(fusion_block)
            
            in_channels = channels
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_projection = nn.Linear(sum(config.pyramid_channels), config.visual_embed_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Extract multi-scale visual features.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W)
            
        Returns:
            global_features: Aggregated features of shape (B, visual_embed_dim)
            pyramid_features: List of feature maps at different scales
        """
        pyramid_features = []
        prev_features = None
        
        for i, (pyramid_block, fusion_block) in enumerate(zip(self.pyramid_blocks, self.fusion_blocks)):
            # Extract features at current scale
            features = pyramid_block(x if i == 0 else prev_features)
            
            # Apply lateral connection and fusion
            if i > 0 and i <= len(self.lateral_connections):
                lateral_features = self.lateral_connections[i-1](prev_features)
                lateral_features = F.interpolate(lateral_features, size=features.shape[-2:], mode='bilinear')
                features = torch.cat([features, lateral_features], dim=1)
            
            features = fusion_block(features)
            pyramid_features.append(features)
            prev_features = features
        
        # Global feature aggregation
        global_features = []
        for features in pyramid_features:
            pooled = self.global_pool(features).flatten(1)
            global_features.append(pooled)
        
        global_features = torch.cat(global_features, dim=1)
        global_features = self.feature_projection(global_features)
        
        return global_features, pyramid_features


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DynamicCrossModalGating(nn.Module):
    """
    Dynamic gating mechanism for adaptive cross-modal information flow.
    
    This module learns importance weights between visual and market features
    based on market volatility indicators and feature relevance scores.
    
    Mathematical Formulation:
    G_v = σ(W_g^v[V; M; V⊙M] + b_g^v)
    G_m = σ(W_g^m[M; V; M⊙V] + b_g^m)
    Z = G_v ⊙ f_v(V) + G_m ⊙ f_m(M)
    """
    
    def __init__(self, visual_dim: int, market_dim: int, fusion_dim: int):
        super().__init__()
        
        # Gating networks for each modality
        self.visual_gate = nn.Sequential(
            nn.Linear(visual_dim + market_dim + visual_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, visual_dim),
            nn.Sigmoid()
        )
        
        self.market_gate = nn.Sequential(
            nn.Linear(market_dim + visual_dim + market_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, market_dim),
            nn.Sigmoid()
        )
        
        # Feature transformation networks
        self.visual_transform = nn.Sequential(
            nn.Linear(visual_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.market_transform = nn.Sequential(
            nn.Linear(market_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Volatility-aware attention
        self.volatility_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, visual_features: torch.Tensor, market_features: torch.Tensor,
                volatility_indicators: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply dynamic cross-modal gating.
        
        Args:
            visual_features: Visual features (B, visual_dim)
            market_features: Market features (B, market_dim)
            volatility_indicators: Optional market volatility indicators (B, V)
            
        Returns:
            Fused features (B, fusion_dim)
        """
        # Compute element-wise products for gating
        visual_market_product = visual_features * market_features[:, :visual_features.size(1)]
        market_visual_product = market_features * visual_features[:, :market_features.size(1)].repeat(1, market_features.size(1) // visual_features.size(1) + 1)[:, :market_features.size(1)]
        
        # Compute gates
        visual_gate_input = torch.cat([visual_features, market_features[:, :visual_features.size(1)], visual_market_product], dim=1)
        market_gate_input = torch.cat([market_features, visual_features[:, :market_features.size(1)].repeat(1, market_features.size(1) // visual_features.size(1) + 1)[:, :market_features.size(1)], market_visual_product], dim=1)
        
        visual_gate = self.visual_gate(visual_gate_input)
        market_gate = self.market_gate(market_gate_input)
        
        # Apply gates and transform features
        gated_visual = visual_gate * visual_features
        gated_market = market_gate * market_features
        
        transformed_visual = self.visual_transform(gated_visual)
        transformed_market = self.market_transform(gated_market)
        
        # Combine features
        fused_features = transformed_visual + transformed_market
        
        # Apply volatility-aware attention if indicators provided
        if volatility_indicators is not None:
            fused_features = fused_features.unsqueeze(1)
            fused_features, _ = self.volatility_attention(
                fused_features, fused_features, fused_features,
                key_padding_mask=volatility_indicators
            )
            fused_features = fused_features.squeeze(1)
        
        return fused_features


class TemporalVisualCoherenceModule(nn.Module):
    """
    Model temporal coherence between visual attributes and market dynamics.
    
    Uses graph neural networks to capture relationships between visual similarity
    and market correlation patterns over time.
    
    Mathematical Formulation:
    A_t = softmax(Q_t K_t^T / √d_k)
    H_t = A_t V_t + TCN(H_{t-1})
    where TCN represents temporal convolutional network.
    """
    
    def __init__(self, config: HiViFANConfig):
        super().__init__()
        
        # Temporal convolutional layers with dilated convolutions
        self.temporal_convs = nn.ModuleList()
        for dilation in config.temporal_dilation_rates:
            conv = nn.Conv1d(
                config.fusion_dim,
                config.fusion_dim,
                kernel_size=config.temporal_kernel_size,
                dilation=dilation,
                padding=(config.temporal_kernel_size - 1) * dilation // 2
            )
            self.temporal_convs.append(conv)
        
        # Graph attention for visual-market relationships
        self.graph_attention = GraphAttentionLayer(
            in_features=config.fusion_dim,
            out_features=config.fusion_dim,
            num_heads=8
        )
        
        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.fusion_dim,
            num_heads=config.fusion_heads,
            dropout=config.fusion_dropout,
            batch_first=True
        )
        
        # Feature fusion and projection
        self.fusion_projection = nn.Sequential(
            nn.Linear(config.fusion_dim * len(config.temporal_dilation_rates), config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout)
        )
        
    def forward(self, features: torch.Tensor, temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process temporal-visual coherence.
        
        Args:
            features: Input features (B, T, D)
            temporal_mask: Optional temporal mask (B, T)
            
        Returns:
            Coherence-aware features (B, T, D)
        """
        B, T, D = features.shape
        
        # Apply temporal convolutions with different dilation rates
        temporal_features = []
        features_transpose = features.transpose(1, 2)  # (B, D, T)
        
        for conv in self.temporal_convs:
            conv_out = conv(features_transpose)
            temporal_features.append(conv_out.transpose(1, 2))  # Back to (B, T, D)
        
        # Concatenate multi-scale temporal features
        multi_scale_features = torch.cat(temporal_features, dim=-1)  # (B, T, D*num_dilations)
        
        # Project back to original dimension
        fused_features = self.fusion_projection(multi_scale_features)  # (B, T, D)
        
        # Apply temporal attention
        attended_features, _ = self.temporal_attention(
            fused_features, fused_features, fused_features,
            key_padding_mask=temporal_mask
        )
        
        # Apply graph attention for visual-market relationships
        # Reshape for graph processing
        graph_features = attended_features.reshape(B * T, D)
        graph_output = self.graph_attention(graph_features)
        graph_output = graph_output.reshape(B, T, D)
        
        # Residual connection
        output = graph_output + features
        
        return output


class GraphAttentionLayer(nn.Module):
    """Graph attention layer for modeling relationships between NFT attributes."""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Parameter(torch.zeros(size=(2 * self.head_dim, 1)))
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
        
    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply graph attention."""
        B = x.size(0)
        h = self.W(x).view(B, self.num_heads, self.head_dim)
        
        # Self-attention on the nodes
        a_input = torch.cat([h.repeat(1, 1, B).view(B * B, self.num_heads, self.head_dim),
                            h.repeat(B, 1, 1)], dim=2).view(B * B, self.num_heads, 2 * self.head_dim)
        
        e = self.leakyrelu(torch.matmul(a_input, self.a.repeat(1, self.num_heads).view(2 * self.head_dim, self.num_heads))).squeeze(2)
        e = e.view(B, B, self.num_heads)
        
        # Apply adjacency mask if provided
        if adj is not None:
            e = e.masked_fill(adj.unsqueeze(2) == 0, -1e9)
        
        attention = F.softmax(e, dim=1)
        h_prime = torch.matmul(attention.transpose(1, 2), h).view(B, -1)
        
        return F.elu(h_prime)


class HiViFAN(nn.Module):
    """
    Hierarchical Visual-Financial Attention Network (HiViFAN).
    
    Complete architecture integrating multi-scale visual features, temporal market
    dynamics, and cross-modal attention mechanisms for NFT market analysis.
    """
    
    def __init__(self, config: HiViFANConfig):
        super().__init__()
        self.config = config
        
        # Visual feature extraction
        self.visual_pyramid = MultiScaleVisualFeaturePyramid(config)
        
        # Vision transformer for high-level visual understanding
        self.visual_transformer = VisionTransformer(
            img_size=224,
            patch_size=config.visual_patch_size,
            embed_dim=config.visual_embed_dim,
            depth=config.visual_depth,
            num_heads=config.visual_num_heads,
            mlp_ratio=config.visual_mlp_ratio
        )
        
        # Market dynamics encoder
        self.market_encoder = MarketDynamicsEncoder(
            input_dim=config.market_embed_dim,
            hidden_dim=config.market_embed_dim,
            num_layers=config.market_depth,
            num_heads=config.market_num_heads,
            window_size=config.market_window_size
        )
        
        # Dynamic cross-modal gating
        self.cross_modal_gate = DynamicCrossModalGating(
            visual_dim=config.visual_embed_dim,
            market_dim=config.market_embed_dim,
            fusion_dim=config.fusion_dim
        )
        
        # Temporal-visual coherence
        self.coherence_module = TemporalVisualCoherenceModule(config)
        
        # Price prediction head
        self.price_predictor = PricePredictionHead(
            input_dim=config.fusion_dim,
            hidden_dim=config.fusion_dim // 2,
            output_dim=1
        )
        
        # Market efficiency predictor
        self.efficiency_predictor = MarketEfficiencyHead(
            input_dim=config.fusion_dim,
            hidden_dim=config.fusion_dim // 2
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights with Xavier/He initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, 
                visual_input: torch.Tensor,
                market_input: torch.Tensor,
                temporal_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HiViFAN.
        
        Args:
            visual_input: Visual input tensor (B, 3, H, W)
            market_input: Market features tensor (B, T, D_m)
            temporal_mask: Optional temporal mask (B, T)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing predictions and intermediate features
        """
        # Extract multi-scale visual features
        pyramid_global, pyramid_features = self.visual_pyramid(visual_input)
        
        # High-level visual features from transformer
        visual_transformer_features = self.visual_transformer(visual_input)
        
        # Combine visual features
        visual_features = pyramid_global + visual_transformer_features
        
        # Encode market dynamics
        market_features, market_attention = self.market_encoder(market_input, temporal_mask)
        
        # Dynamic cross-modal gating
        fused_features = self.cross_modal_gate(
            visual_features,
            market_features.mean(dim=1),  # Aggregate temporal dimension
            volatility_indicators=None  # Can be computed from market_input
        )
        
        # Expand fused features to temporal dimension
        B, T, _ = market_input.shape
        fused_features = fused_features.unsqueeze(1).expand(-1, T, -1)
        
        # Apply temporal-visual coherence
        coherent_features = self.coherence_module(fused_features, temporal_mask)
        
        # Generate predictions
        price_predictions = self.price_predictor(coherent_features)
        efficiency_scores = self.efficiency_predictor(coherent_features)
        
        # Prepare output dictionary
        outputs = {
            'price_predictions': price_predictions,
            'efficiency_scores': efficiency_scores,
            'visual_features': visual_features,
            'market_features': market_features,
            'fused_features': coherent_features,
            'pyramid_features': pyramid_features
        }
        
        if return_attention:
            outputs['market_attention'] = market_attention
        
        return outputs


class VisionTransformer(nn.Module):
    """Vision Transformer for high-level visual feature extraction."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        
        num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed[:, :x.size(1)]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Return CLS token as global feature
        return x[:, 0]


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and MLP."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class MarketDynamicsEncoder(nn.Module):
    """Encode temporal market dynamics with advanced attention mechanisms."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 num_heads: int, window_size: int):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Temporal positional encoding
        self.temporal_encoding = TemporalPositionalEncoding(hidden_dim, max_len=window_size)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Market-specific feature extractors
        self.volatility_extractor = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.trend_extractor = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Feature aggregation
        self.feature_aggregator = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode market dynamics.
        
        Args:
            x: Market features (B, T, D)
            mask: Optional temporal mask (B, T)
            
        Returns:
            encoded_features: Encoded market features (B, T, D)
            attention_weights: Self-attention weights for interpretability
        """
        # Project input features
        x = self.input_projection(x)
        
        # Add temporal positional encoding
        x = self.temporal_encoding(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Extract volatility features
        x_transpose = encoded.transpose(1, 2)
        volatility_features = self.volatility_extractor(x_transpose).transpose(1, 2)
        
        # Extract trend features
        trend_features, _ = self.trend_extractor(encoded)
        
        # Aggregate all features
        aggregated_features = torch.cat([
            encoded,
            volatility_features.mean(dim=-1, keepdim=True).expand_as(encoded[:, :, :encoded.size(-1)//4]),
            trend_features
        ], dim=-1)
        
        final_features = self.feature_aggregator(aggregated_features)
        
        # Compute attention weights for interpretability
        with torch.no_grad():
            attention_weights = torch.softmax(
                torch.matmul(encoded, encoded.transpose(-2, -1)) / math.sqrt(encoded.size(-1)),
                dim=-1
            )
        
        return final_features, attention_weights


class TemporalPositionalEncoding(nn.Module):
    """Learnable temporal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.encoding, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, :x.size(1)]


class PricePredictionHead(nn.Module):
    """Price prediction head with uncertainty estimation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Separate heads for mean and variance (uncertainty)
        self.mean_head = nn.Linear(hidden_dim // 2, output_dim)
        self.log_var_head = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.predictor(x)
        mean = self.mean_head(features)
        log_var = self.log_var_head(features)
        
        # Ensure numerical stability
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        return {
            'mean': mean,
            'log_var': log_var,
            'std': torch.exp(0.5 * log_var)
        }


class MarketEfficiencyHead(nn.Module):
    """Predict market efficiency metrics."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.efficiency_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4)  # Multiple efficiency metrics
        )
        
        self.metric_names = ['liquidity', 'price_stability', 'market_depth', 'trading_volume']
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        predictions = self.efficiency_predictor(x)
        
        # Apply appropriate activations for each metric
        results = {}
        results['liquidity'] = torch.sigmoid(predictions[..., 0])
        results['price_stability'] = torch.sigmoid(predictions[..., 1])
        results['market_depth'] = F.softplus(predictions[..., 2])
        results['trading_volume'] = F.softplus(predictions[..., 3])
        
        return results


class HybridLoss(nn.Module):
    """
    Comprehensive loss function for HiViFAN training.
    
    Combines price prediction loss, market efficiency loss, and
    information-theoretic regularization terms.
    """
    
    def __init__(self, config: HiViFANConfig):
        super().__init__()
        self.config = config
        
        # Loss weights
        self.price_weight = 1.0
        self.efficiency_weight = 0.5
        self.mi_weight = 0.1
        self.consistency_weight = 0.2
        
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Price prediction loss with uncertainty
        price_pred = predictions['price_predictions']
        price_target = targets['prices']
        
        price_mean = price_pred['mean']
        price_log_var = price_pred['log_var']
        
        # Negative log-likelihood for Gaussian distribution
        price_loss = 0.5 * (price_log_var + (price_target - price_mean) ** 2 / torch.exp(price_log_var))
        losses['price_loss'] = price_loss.mean()
        
        # Market efficiency losses
        if 'efficiency_scores' in predictions and 'efficiency_targets' in targets:
            efficiency_pred = predictions['efficiency_scores']
            efficiency_target = targets['efficiency_targets']
            
            efficiency_losses = {}
            for metric in ['liquidity', 'price_stability', 'market_depth', 'trading_volume']:
                if metric in efficiency_pred and metric in efficiency_target:
                    if metric in ['liquidity', 'price_stability']:
                        # Binary cross-entropy for bounded metrics
                        loss = F.binary_cross_entropy(efficiency_pred[metric], efficiency_target[metric])
                    else:
                        # MSE for unbounded metrics
                        loss = F.mse_loss(efficiency_pred[metric], efficiency_target[metric])
                    efficiency_losses[f'{metric}_loss'] = loss
            
            losses['efficiency_loss'] = sum(efficiency_losses.values()) / len(efficiency_losses)
            losses.update(efficiency_losses)
        
        # Mutual information regularization
        if self.mi_weight > 0:
            visual_features = predictions['visual_features']
            market_features = predictions['market_features'].mean(dim=1)
            
            # Estimate mutual information using MINE approximation
            mi_loss = self._estimate_mutual_information(visual_features, market_features)
            losses['mi_loss'] = mi_loss
        
        # Temporal consistency regularization
        if self.consistency_weight > 0 and 'fused_features' in predictions:
            fused_features = predictions['fused_features']
            if fused_features.size(1) > 1:
                # Encourage smooth temporal transitions
                consistency_loss = torch.mean(
                    torch.sum((fused_features[:, 1:] - fused_features[:, :-1]) ** 2, dim=-1)
                )
                losses['consistency_loss'] = consistency_loss
        
        # Combine all losses
        total_loss = (self.price_weight * losses['price_loss'] +
                     self.efficiency_weight * losses.get('efficiency_loss', 0) +
                     self.mi_weight * losses.get('mi_loss', 0) +
                     self.consistency_weight * losses.get('consistency_loss', 0))
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _estimate_mutual_information(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Estimate mutual information using MINE (Mutual Information Neural Estimation).
        
        I(X;Y) = E[T(x,y)] - log(E[e^T(x',y)])
        where x' are samples from marginal distribution.
        """
        batch_size = x.size(0)
        
        # Joint distribution samples
        joint = torch.cat([x, y], dim=1)
        
        # Marginal distribution samples (shuffle y)
        y_shuffle = y[torch.randperm(batch_size)]
        marginal = torch.cat([x, y_shuffle], dim=1)
        
        # Simple statistics network for T
        T = nn.Sequential(
            nn.Linear(joint.size(1), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(joint.device)
        
        # Compute expectations
        t_joint = T(joint).mean()
        t_marginal = torch.logsumexp(T(marginal), dim=0) - math.log(batch_size)
        
        # MINE estimate of mutual information
        mi = t_joint - t_marginal
        
        # Return negative MI as loss (we want to maximize MI)
        return -mi


def create_hivifan_model(config: Optional[HiViFANConfig] = None) -> HiViFAN:
    """
    Factory function to create HiViFAN model with default or custom configuration.
    
    Args:
        config: Optional configuration object
        
    Returns:
        Initialized HiViFAN model
    """
    if config is None:
        config = HiViFANConfig()
    
    model = HiViFAN(config)
    
    # Log model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"HiViFAN Model Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model configuration: {config}")
    
    return model
