#!/usr/bin/env python3
"""
Sensory Fusion Network - Multi-modal neural fusion for MORPHEUS.

This module implements advanced neural networks for fusing multiple
sensory modalities (tactile, audio, visual) into unified representations.

Features:
- Transformer-based attention mechanisms
- Cross-modal learning and adaptation
- Temporal sequence processing
- Uncertainty quantification
- Hierarchical feature extraction
- Real-time processing optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import math

logger = logging.getLogger(__name__)

@dataclass
class ModalityConfig:
    """Configuration for a single sensory modality."""
    name: str
    input_dim: int
    feature_dim: int
    enabled: bool = True
    weight: float = 1.0
    temporal_window: int = 10
    preprocessing: Optional[str] = None

@dataclass
class FusionResult:
    """Result from sensory fusion processing."""
    fused_embedding: torch.Tensor
    modality_attentions: Dict[str, float]
    uncertainty: torch.Tensor
    temporal_features: Optional[torch.Tensor]
    processing_time: float
    confidence: float
    quality_metrics: Dict[str, float]

class ModalityEncoder(nn.Module):
    """Individual modality encoder with attention mechanism."""
    
    def __init__(self, config: ModalityConfig, hidden_dim: int = 256):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, config.feature_dim)
        )
        
        # Self-attention for temporal modeling
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.feature_dim)
        self.norm2 = nn.LayerNorm(config.feature_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, config.feature_dim)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.feature_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through modality encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of (features, uncertainty)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Self-attention with residual connection
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Compute uncertainty
        uncertainty = self.uncertainty_head(x)
        
        return x, uncertainty

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for fusion."""
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cross-modal attention forward pass."""
        attn_out, attn_weights = self.attention(query, key, value)
        output = self.norm(query + attn_out)
        
        return output, attn_weights

class TemporalFusionBlock(nn.Module):
    """Temporal fusion with recurrent connections."""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Attention over time steps
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Temporal fusion forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, feature_dim)
            
        Returns:
            Temporally fused features
        """
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Temporal attention
        attention_scores = self.temporal_attention(lstm_out)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum over time
        weighted_features = (lstm_out * attention_weights).sum(dim=1)
        
        # Output projection
        output = self.output_proj(weighted_features)
        
        return output

class AdaptiveFusionGate(nn.Module):
    """Adaptive gating mechanism for modality fusion."""
    
    def __init__(self, num_modalities: int, feature_dim: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim
        
        # Gate computation network
        self.gate_net = nn.Sequential(
            nn.Linear(num_modalities * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        # Quality estimation for each modality
        self.quality_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, 1),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
        
    def forward(self, modality_features: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Adaptive fusion gate forward pass.
        
        Args:
            modality_features: List of feature tensors from different modalities
            
        Returns:
            Tuple of (gated_features, attention_weights)
        """
        batch_size = modality_features[0].size(0)
        
        # Concatenate all modality features
        concat_features = torch.cat(modality_features, dim=-1)
        
        # Compute gate weights
        gate_weights = self.gate_net(concat_features)
        
        # Compute quality scores
        quality_scores = []
        for i, (features, quality_net) in enumerate(zip(modality_features, self.quality_nets)):
            quality = quality_net(features).squeeze(-1)
            quality_scores.append(quality)
            
        quality_scores = torch.stack(quality_scores, dim=-1)
        
        # Combine gate weights with quality scores
        final_weights = gate_weights * quality_scores
        final_weights = F.softmax(final_weights, dim=-1)
        
        # Apply weighted fusion
        fused_features = torch.zeros_like(modality_features[0])
        for i, features in enumerate(modality_features):
            fused_features += final_weights[:, i:i+1] * features
            
        # Convert attention weights to dictionary
        attention_dict = {}
        for i in range(len(modality_features)):
            attention_dict[f'modality_{i}'] = final_weights[:, i].mean().item()
            
        return fused_features, attention_dict

class SensoryFusionNetwork(nn.Module):
    """
    Complete sensory fusion network with multi-modal processing.
    
    Combines tactile, audio, and visual inputs using attention mechanisms
    and produces unified sensory representations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Modality configurations
        self.modalities = {}
        modality_configs = config.get('modalities', {})
        
        for name, mod_config in modality_configs.items():
            self.modalities[name] = ModalityConfig(
                name=name,
                input_dim=mod_config['input_dim'],
                feature_dim=mod_config.get('feature_dim', 128),
                enabled=mod_config.get('enabled', True),
                weight=mod_config.get('weight', 1.0),
                temporal_window=mod_config.get('temporal_window', 10)
            )
            
        # Default modalities if not specified
        if not self.modalities:
            self.modalities = {
                'tactile': ModalityConfig('tactile', 64, 128),
                'audio': ModalityConfig('audio', 32, 128),
                'visual': ModalityConfig('visual', 128, 128)
            }
            
        # Network parameters
        self.feature_dim = config.get('feature_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.output_dim = config.get('output_dim', 128)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        
        # Modality encoders
        self.encoders = nn.ModuleDict()
        for name, mod_config in self.modalities.items():
            if mod_config.enabled:
                self.encoders[name] = ModalityEncoder(mod_config, self.hidden_dim)
                
        # Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList()
        num_modalities = len([m for m in self.modalities.values() if m.enabled])
        
        for _ in range(config.get('num_fusion_layers', 3)):
            self.cross_modal_layers.append(
                CrossModalAttention(self.feature_dim, self.num_heads)
            )
            
        # Adaptive fusion gate
        self.fusion_gate = AdaptiveFusionGate(num_modalities, self.feature_dim)
        
        # Temporal fusion
        self.temporal_fusion = TemporalFusionBlock(self.feature_dim, self.hidden_dim)
        
        # Final output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
        
        # Uncertainty estimation
        self.uncertainty_net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation
        self.confidence_net = nn.Sequential(
            nn.Linear(self.output_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"SensoryFusionNetwork initialized with {num_modalities} modalities")
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            
    def forward(self, inputs: Dict[str, torch.Tensor]) -> FusionResult:
        """
        Forward pass through sensory fusion network.
        
        Args:
            inputs: Dictionary mapping modality names to input tensors
            
        Returns:
            FusionResult with fused embeddings and metadata
        """
        start_time = time.time()
        batch_size = next(iter(inputs.values())).size(0)
        
        # Process each modality
        modality_features = []
        modality_uncertainties = []
        modality_names = []
        
        for name, encoder in self.encoders.items():
            if name in inputs and inputs[name] is not None:
                # Ensure input has sequence dimension
                input_tensor = inputs[name]
                if len(input_tensor.shape) == 2:
                    input_tensor = input_tensor.unsqueeze(1)
                    
                features, uncertainty = encoder(input_tensor)
                
                # Take last timestep if sequence
                if features.size(1) > 1:
                    features = features[:, -1, :]
                    uncertainty = uncertainty[:, -1, :]
                else:
                    features = features.squeeze(1)
                    uncertainty = uncertainty.squeeze(1)
                    
                modality_features.append(features)
                modality_uncertainties.append(uncertainty)
                modality_names.append(name)
                
        if not modality_features:
            # Return zero-filled result if no modalities available
            return self._create_empty_result(batch_size, start_time)
            
        # Cross-modal attention processing
        fused_features = modality_features.copy()
        
        for cross_attention in self.cross_modal_layers:
            new_features = []
            for i, query_features in enumerate(fused_features):
                # Use other modalities as key and value
                other_features = [f for j, f in enumerate(fused_features) if j != i]
                if other_features:
                    key_value = torch.stack(other_features, dim=1)  # (batch, num_others, feature_dim)
                    query = query_features.unsqueeze(1)  # (batch, 1, feature_dim)
                    
                    attended, _ = cross_attention(query, key_value, key_value)
                    new_features.append(attended.squeeze(1))
                else:
                    new_features.append(query_features)
                    
            fused_features = new_features
            
        # Adaptive fusion gate
        final_features, attention_weights = self.fusion_gate(fused_features)
        
        # Temporal fusion (if enabled)
        temporal_features = None
        if hasattr(self, 'temporal_fusion') and len(modality_features) > 0:
            # Stack features for temporal processing
            stacked_features = final_features.unsqueeze(1)
            temporal_features = self.temporal_fusion(stacked_features)
            final_features = temporal_features
            
        # Final output projection
        fused_embedding = self.output_layers(final_features)
        
        # Uncertainty estimation
        combined_uncertainty = torch.stack(modality_uncertainties, dim=-1).mean(dim=-1)
        output_uncertainty = self.uncertainty_net(final_features)
        final_uncertainty = (combined_uncertainty + output_uncertainty.squeeze(-1)) / 2
        
        # Confidence estimation
        confidence_score = self.confidence_net(fused_embedding).squeeze(-1)
        avg_confidence = confidence_score.mean().item()
        
        # Quality metrics
        quality_metrics = self._compute_quality_metrics(
            modality_features, 
            fused_embedding, 
            final_uncertainty
        )
        
        processing_time = time.time() - start_time
        
        return FusionResult(
            fused_embedding=fused_embedding,
            modality_attentions=attention_weights,
            uncertainty=final_uncertainty,
            temporal_features=temporal_features,
            processing_time=processing_time,
            confidence=avg_confidence,
            quality_metrics=quality_metrics
        )
        
    def _create_empty_result(self, batch_size: int, start_time: float) -> FusionResult:
        """Create empty result when no modalities are available."""
        return FusionResult(
            fused_embedding=torch.zeros(batch_size, self.output_dim),
            modality_attentions={},
            uncertainty=torch.ones(batch_size),
            temporal_features=None,
            processing_time=time.time() - start_time,
            confidence=0.0,
            quality_metrics={}
        )
        
    def _compute_quality_metrics(self, 
                               modality_features: List[torch.Tensor],
                               fused_embedding: torch.Tensor,
                               uncertainty: torch.Tensor) -> Dict[str, float]:
        """Compute quality metrics for fusion result."""
        
        metrics = {}
        
        if len(modality_features) > 0:
            # Feature diversity (how different are the modalities)
            if len(modality_features) > 1:
                feature_matrix = torch.stack(modality_features, dim=1)
                pairwise_distances = torch.pdist(feature_matrix.view(-1, feature_matrix.size(-1)))
                metrics['feature_diversity'] = pairwise_distances.mean().item()
            else:
                metrics['feature_diversity'] = 0.0
                
            # Fusion coherence (how well do features align)
            coherence_scores = []
            for features in modality_features:
                similarity = F.cosine_similarity(features, fused_embedding, dim=-1)
                coherence_scores.append(similarity.mean().item())
            metrics['fusion_coherence'] = np.mean(coherence_scores)
            
            # Uncertainty confidence (lower uncertainty = higher confidence)
            metrics['uncertainty_confidence'] = (1.0 - uncertainty.mean().item())
            
            # Signal-to-noise ratio estimate
            signal_power = fused_embedding.pow(2).mean().item()
            noise_power = uncertainty.mean().item() + 1e-8
            metrics['signal_to_noise'] = 10 * np.log10(signal_power / noise_power)
            
        return metrics
        
    def get_attention_patterns(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Get attention patterns for interpretability."""
        
        attention_patterns = {}
        
        with torch.no_grad():
            # Set to evaluation mode temporarily
            was_training = self.training
            self.eval()
            
            try:
                result = self.forward(inputs)
                attention_patterns['modality_attention'] = result.modality_attentions
                
                # Additional attention pattern analysis could go here
                
            finally:
                # Restore training mode
                if was_training:
                    self.train()
                    
        return attention_patterns
        
    def adapt_to_modality_quality(self, quality_scores: Dict[str, float]):
        """Adapt fusion weights based on modality quality scores."""
        
        # Update modality weights based on quality
        for name, quality in quality_scores.items():
            if name in self.modalities:
                # Higher quality = higher weight
                self.modalities[name].weight = quality
                
        # Normalize weights
        total_weight = sum(mod.weight for mod in self.modalities.values() if mod.enabled)
        if total_weight > 0:
            for mod in self.modalities.values():
                mod.weight /= total_weight
                
        logger.info(f"Updated modality weights based on quality scores: {quality_scores}")
        
    def freeze_modality(self, modality_name: str):
        """Freeze parameters for a specific modality encoder."""
        
        if modality_name in self.encoders:
            for param in self.encoders[modality_name].parameters():
                param.requires_grad = False
            logger.info(f"Frozen modality encoder: {modality_name}")
            
    def unfreeze_modality(self, modality_name: str):
        """Unfreeze parameters for a specific modality encoder."""
        
        if modality_name in self.encoders:
            for param in self.encoders[modality_name].parameters():
                param.requires_grad = True
            logger.info(f"Unfrozen modality encoder: {modality_name}")
            
    def get_modality_importance(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute importance scores for each modality."""
        
        importance_scores = {}
        
        with torch.no_grad():
            # Get baseline performance with all modalities
            baseline_result = self.forward(inputs)
            baseline_confidence = baseline_result.confidence
            
            # Test performance with each modality removed
            for modality_name in inputs.keys():
                if modality_name in self.encoders:
                    # Create inputs without this modality
                    modified_inputs = {k: v for k, v in inputs.items() if k != modality_name}
                    
                    if modified_inputs:  # Only if other modalities remain
                        modified_result = self.forward(modified_inputs)
                        confidence_drop = baseline_confidence - modified_result.confidence
                        importance_scores[modality_name] = max(0, confidence_drop)
                    else:
                        importance_scores[modality_name] = 1.0  # Only modality available
                        
        return importance_scores
        
    def save_fusion_state(self, filepath: str):
        """Save fusion network state."""
        
        state = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'modalities': {name: {
                'name': mod.name,
                'input_dim': mod.input_dim,
                'feature_dim': mod.feature_dim,
                'enabled': mod.enabled,
                'weight': mod.weight,
                'temporal_window': mod.temporal_window
            } for name, mod in self.modalities.items()}
        }
        
        torch.save(state, filepath)
        logger.info(f"Fusion network state saved to {filepath}")
        
    def load_fusion_state(self, filepath: str):
        """Load fusion network state."""
        
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            self.load_state_dict(checkpoint['model_state_dict'])
            
            if 'modalities' in checkpoint:
                for name, mod_data in checkpoint['modalities'].items():
                    if name in self.modalities:
                        self.modalities[name].weight = mod_data.get('weight', 1.0)
                        self.modalities[name].enabled = mod_data.get('enabled', True)
                        
            logger.info(f"Fusion network state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load fusion state: {e}")

class FusionNetworkManager:
    """Manager class for sensory fusion network with additional utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.network = SensoryFusionNetwork(config)
        self.optimizer = None
        self.scheduler = None
        
        # Performance tracking
        self.performance_history = []
        self.processing_times = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize training components if specified
        if config.get('training', {}).get('enabled', False):
            self._init_training()
            
    def _init_training(self):
        """Initialize training components."""
        
        training_config = self.config.get('training', {})
        
        # Optimizer
        lr = training_config.get('learning_rate', 1e-3)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=lr,
            weight_decay=training_config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=training_config.get('max_epochs', 100)
        )
        
        logger.info("Training components initialized")
        
    def process_batch(self, batch: Dict[str, torch.Tensor]) -> List[FusionResult]:
        """Process a batch of inputs."""
        
        results = []
        
        with torch.no_grad():
            self.network.eval()
            
            batch_size = next(iter(batch.values())).size(0)
            
            for i in range(batch_size):
                # Extract individual sample
                sample = {}
                for modality, data in batch.items():
                    sample[modality] = data[i:i+1]  # Keep batch dimension
                    
                # Process sample
                result = self.network(sample)
                results.append(result)
                
        return results
        
    def update_performance_metrics(self, results: List[FusionResult]):
        """Update performance tracking metrics."""
        
        with self._lock:
            for result in results:
                self.processing_times.append(result.processing_time)
                
                # Keep only recent measurements
                if len(self.processing_times) > 1000:
                    self.processing_times = self.processing_times[-1000:]
                    
                # Update performance history
                performance_entry = {
                    'timestamp': time.time(),
                    'confidence': result.confidence,
                    'uncertainty': result.uncertainty.mean().item(),
                    'processing_time': result.processing_time,
                    'quality_metrics': result.quality_metrics
                }
                
                self.performance_history.append(performance_entry)
                
                # Keep only recent history
                if len(self.performance_history) > 10000:
                    self.performance_history = self.performance_history[-10000:]
                    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        
        with self._lock:
            if not self.performance_history:
                return {}
                
            recent_history = self.performance_history[-100:]  # Last 100 samples
            
            summary = {
                'samples_processed': len(self.performance_history),
                'average_confidence': np.mean([h['confidence'] for h in recent_history]),
                'average_uncertainty': np.mean([h['uncertainty'] for h in recent_history]),
                'average_processing_time': np.mean([h['processing_time'] for h in recent_history]),
                'processing_fps': 1.0 / np.mean(self.processing_times) if self.processing_times else 0,
                'quality_trends': self._compute_quality_trends(recent_history)
            }
            
            return summary
            
    def _compute_quality_trends(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute quality trends from history."""
        
        trends = {}
        
        if len(history) < 10:
            return trends
            
        # Compute trends for key metrics
        metrics = ['confidence', 'uncertainty', 'processing_time']
        
        for metric in metrics:
            values = [h[metric] for h in history]
            
            # Simple linear trend (positive = improving)
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            
            # Invert for metrics where lower is better
            if metric in ['uncertainty', 'processing_time']:
                slope = -slope
                
            trends[f'{metric}_trend'] = slope
            
        return trends
        
    def adapt_network(self, performance_feedback: Dict[str, float]):
        """Adapt network based on performance feedback."""
        
        # Update modality weights based on performance
        modality_quality = {}
        
        for modality in self.network.modalities.keys():
            # Use performance feedback to estimate modality quality
            base_quality = performance_feedback.get(f'{modality}_quality', 0.5)
            modality_quality[modality] = base_quality
            
        self.network.adapt_to_modality_quality(modality_quality)
        
        logger.info(f"Network adapted based on feedback: {performance_feedback}")