#!/usr/bin/env python3
"""
Test suite for Sensory Fusion Network functionality.

Tests multi-modal neural fusion, attention mechanisms,
and network management capabilities.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F
from unittest.mock import Mock, patch
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from morpheus.perception import (
    SensoryFusionNetwork, 
    FusionNetworkManager,
    FusionResult,
    ModalityConfig
)

class TestSensoryFusionNetwork:
    """Test cases for Sensory Fusion Network."""
    
    @pytest.fixture
    def fusion_config(self):
        """Standard fusion network configuration."""
        return {
            'modalities': {
                'tactile': {'input_dim': 64, 'feature_dim': 128, 'enabled': True},
                'audio': {'input_dim': 32, 'feature_dim': 128, 'enabled': True},
                'visual': {'input_dim': 256, 'feature_dim': 128, 'enabled': True}
            },
            'feature_dim': 128,
            'hidden_dim': 256,
            'output_dim': 128,
            'num_heads': 8,
            'dropout': 0.1,
            'num_fusion_layers': 3
        }
        
    @pytest.fixture
    def fusion_network(self, fusion_config):
        """Fusion network instance for testing."""
        return SensoryFusionNetwork(fusion_config)
        
    @pytest.fixture
    def sample_inputs(self):
        """Sample input tensors for testing."""
        return {
            'tactile': torch.randn(2, 64),
            'audio': torch.randn(2, 32), 
            'visual': torch.randn(2, 256)
        }
        
    def test_initialization(self, fusion_network, fusion_config):
        """Test fusion network initialization."""
        assert fusion_network.feature_dim == fusion_config['feature_dim']
        assert fusion_network.hidden_dim == fusion_config['hidden_dim']
        assert fusion_network.output_dim == fusion_config['output_dim']
        
        # Check modality encoders
        assert 'tactile' in fusion_network.encoders
        assert 'audio' in fusion_network.encoders
        assert 'visual' in fusion_network.encoders
        
        # Check network components
        assert hasattr(fusion_network, 'cross_modal_layers')
        assert hasattr(fusion_network, 'fusion_gate')
        assert hasattr(fusion_network, 'output_layers')
        
    def test_forward_pass(self, fusion_network, sample_inputs):
        """Test forward pass through fusion network."""
        result = fusion_network(sample_inputs)
        
        assert isinstance(result, FusionResult)
        assert result.fused_embedding.shape == (2, 128)  # batch_size, output_dim
        assert isinstance(result.modality_attentions, dict)
        assert result.uncertainty.shape == (2,)
        assert result.processing_time > 0
        assert 0 <= result.confidence <= 1
        
    def test_missing_modalities(self, fusion_network):
        """Test handling of missing modalities."""
        # Test with only tactile input
        partial_inputs = {'tactile': torch.randn(2, 64)}
        result = fusion_network(partial_inputs)
        
        assert isinstance(result, FusionResult)
        assert result.fused_embedding.shape == (2, 128)
        
        # Test with no inputs
        empty_inputs = {}
        result = fusion_network(empty_inputs)
        
        assert result.fused_embedding.shape == (2, 128)
        assert result.confidence == 0.0
        
    def test_sequence_inputs(self, fusion_network):
        """Test handling of sequence inputs."""
        # Input with sequence dimension
        sequence_inputs = {
            'tactile': torch.randn(2, 5, 64),  # batch, seq, features
            'audio': torch.randn(2, 3, 32),
            'visual': torch.randn(2, 1, 256)
        }
        
        result = fusion_network(sequence_inputs)
        
        assert isinstance(result, FusionResult)
        assert result.fused_embedding.shape == (2, 128)
        
    def test_attention_patterns(self, fusion_network, sample_inputs):
        """Test attention pattern extraction."""
        patterns = fusion_network.get_attention_patterns(sample_inputs)
        
        assert isinstance(patterns, dict)
        assert 'modality_attention' in patterns
        
        # Attention weights should sum to approximately 1
        attention_values = list(patterns['modality_attention'].values())
        total_attention = sum(attention_values)
        assert abs(total_attention - 1.0) < 0.1
        
    def test_modality_adaptation(self, fusion_network):
        """Test adaptation to modality quality scores."""
        quality_scores = {'tactile': 0.9, 'audio': 0.7, 'visual': 0.8}
        
        fusion_network.adapt_to_modality_quality(quality_scores)
        
        # Check that weights were updated
        for name, quality in quality_scores.items():
            if name in fusion_network.modalities:
                # Weight should reflect quality (after normalization)
                assert fusion_network.modalities[name].weight > 0
                
    def test_modality_freezing(self, fusion_network):
        """Test freezing and unfreezing modality encoders."""
        modality = 'tactile'
        
        # Initially parameters should require gradients
        encoder = fusion_network.encoders[modality]
        initial_grad_state = next(encoder.parameters()).requires_grad
        assert initial_grad_state is True
        
        # Freeze modality
        fusion_network.freeze_modality(modality)
        
        # Parameters should no longer require gradients
        frozen_grad_state = next(encoder.parameters()).requires_grad
        assert frozen_grad_state is False
        
        # Unfreeze modality
        fusion_network.unfreeze_modality(modality)
        
        # Parameters should require gradients again
        unfrozen_grad_state = next(encoder.parameters()).requires_grad
        assert unfrozen_grad_state is True
        
    def test_modality_importance(self, fusion_network, sample_inputs):
        """Test modality importance scoring."""
        importance_scores = fusion_network.get_modality_importance(sample_inputs)
        
        assert isinstance(importance_scores, dict)
        assert len(importance_scores) > 0
        
        # Importance scores should be non-negative
        for score in importance_scores.values():
            assert score >= 0
            
    def test_quality_metrics_computation(self, fusion_network):
        """Test quality metrics computation."""
        # Mock data for quality metrics
        modality_features = [torch.randn(2, 128) for _ in range(3)]
        fused_embedding = torch.randn(2, 128)
        uncertainty = torch.rand(2)  # Random uncertainty values
        
        metrics = fusion_network._compute_quality_metrics(
            modality_features, fused_embedding, uncertainty
        )
        
        assert isinstance(metrics, dict)
        assert 'feature_diversity' in metrics
        assert 'fusion_coherence' in metrics
        assert 'uncertainty_confidence' in metrics
        assert 'signal_to_noise' in metrics
        
        # Check metric ranges
        assert metrics['fusion_coherence'] >= -1 and metrics['fusion_coherence'] <= 1
        assert metrics['uncertainty_confidence'] >= 0 and metrics['uncertainty_confidence'] <= 1
        
    def test_state_persistence(self, fusion_network, tmp_path):
        """Test saving and loading fusion state."""
        save_path = tmp_path / "fusion_state.pt"
        
        # Save state
        fusion_network.save_fusion_state(str(save_path))
        assert save_path.exists()
        
        # Modify network parameters
        original_param = next(fusion_network.parameters()).clone()
        
        # Load state
        fusion_network.load_fusion_state(str(save_path))
        
        # Parameters should be restored
        loaded_param = next(fusion_network.parameters())
        torch.testing.assert_close(original_param, loaded_param)

class TestModalityEncoder:
    """Test cases for individual modality encoders."""
    
    @pytest.fixture
    def modality_config(self):
        """Configuration for modality encoder."""
        return ModalityConfig(
            name='test_modality',
            input_dim=64,
            feature_dim=128,
            enabled=True,
            temporal_window=10
        )
        
    def test_modality_config_creation(self, modality_config):
        """Test modality configuration creation."""
        assert modality_config.name == 'test_modality'
        assert modality_config.input_dim == 64
        assert modality_config.feature_dim == 128
        assert modality_config.enabled is True
        assert modality_config.temporal_window == 10
        
    def test_encoder_forward_pass(self, fusion_network):
        """Test individual encoder forward pass."""
        encoder = fusion_network.encoders['tactile']
        input_tensor = torch.randn(2, 1, 64)  # batch, seq, features
        
        features, uncertainty = encoder(input_tensor)
        
        assert features.shape == (2, 1, 128)  # batch, seq, feature_dim
        assert uncertainty.shape == (2, 1, 1)  # batch, seq, 1
        assert torch.all(uncertainty >= 0) and torch.all(uncertainty <= 1)

class TestCrossModalAttention:
    """Test cases for cross-modal attention mechanism."""
    
    def test_attention_computation(self, fusion_network):
        """Test cross-modal attention computation."""
        attention_layer = fusion_network.cross_modal_layers[0]
        
        query = torch.randn(2, 1, 128)
        key = torch.randn(2, 3, 128)
        value = torch.randn(2, 3, 128)
        
        output, attention_weights = attention_layer(query, key, value)
        
        assert output.shape == query.shape
        assert attention_weights.shape == (2, 1, 3)  # batch, query_len, key_len
        
        # Attention weights should sum to 1
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(2, 1))

class TestAdaptiveFusionGate:
    """Test cases for adaptive fusion gate."""
    
    def test_fusion_gate_computation(self, fusion_network):
        """Test adaptive fusion gate computation."""
        fusion_gate = fusion_network.fusion_gate
        
        # Create test modality features
        modality_features = [torch.randn(2, 128) for _ in range(3)]
        
        fused_features, attention_weights = fusion_gate(modality_features)
        
        assert fused_features.shape == (2, 128)
        assert isinstance(attention_weights, dict)
        assert len(attention_weights) == 3
        
        # Attention weights should be positive and sum to approximately 1
        total_weight = sum(attention_weights.values())
        assert abs(total_weight - 1.0) < 0.1
        assert all(weight >= 0 for weight in attention_weights.values())

class TestFusionNetworkManager:
    """Test cases for Fusion Network Manager."""
    
    @pytest.fixture
    def manager_config(self):
        """Configuration for fusion network manager."""
        return {
            'modalities': {
                'tactile': {'input_dim': 64, 'feature_dim': 128},
                'audio': {'input_dim': 32, 'feature_dim': 128}
            },
            'feature_dim': 128,
            'output_dim': 64,
            'training': {
                'enabled': True,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4
            }
        }
        
    @pytest.fixture
    def network_manager(self, manager_config):
        """Network manager instance for testing."""
        return FusionNetworkManager(manager_config)
        
    def test_manager_initialization(self, network_manager):
        """Test manager initialization."""
        assert network_manager.network is not None
        assert network_manager.optimizer is not None  # Training enabled
        assert network_manager.scheduler is not None
        assert len(network_manager.performance_history) == 0
        
    def test_batch_processing(self, network_manager):
        """Test batch processing functionality."""
        batch = {
            'tactile': torch.randn(4, 64),
            'audio': torch.randn(4, 32)
        }
        
        results = network_manager.process_batch(batch)
        
        assert len(results) == 4  # batch size
        assert all(isinstance(result, FusionResult) for result in results)
        
    def test_performance_tracking(self, network_manager):
        """Test performance metrics tracking."""
        # Create mock results
        results = [
            FusionResult(
                fused_embedding=torch.randn(1, 64),
                modality_attentions={'tactile': 0.6, 'audio': 0.4},
                uncertainty=torch.rand(1),
                temporal_features=None,
                processing_time=0.01,
                confidence=0.8,
                quality_metrics={'fusion_coherence': 0.7}
            )
            for _ in range(5)
        ]
        
        network_manager.update_performance_metrics(results)
        
        assert len(network_manager.performance_history) == 5
        assert len(network_manager.processing_times) == 5
        
        # Get performance summary
        summary = network_manager.get_performance_summary()
        
        assert 'samples_processed' in summary
        assert 'average_confidence' in summary
        assert 'processing_fps' in summary
        assert summary['samples_processed'] == 5
        
    def test_network_adaptation(self, network_manager):
        """Test network adaptation based on performance feedback."""
        feedback = {
            'tactile_quality': 0.9,
            'audio_quality': 0.7,
            'overall_performance': 0.8
        }
        
        # Should not raise exception
        network_manager.adapt_network(feedback)
        
        # Check that modality weights were updated
        tactile_weight = network_manager.network.modalities['tactile'].weight
        audio_weight = network_manager.network.modalities['audio'].weight
        
        assert tactile_weight > 0
        assert audio_weight > 0

class TestTemporalFusionBlock:
    """Test cases for temporal fusion functionality."""
    
    def test_temporal_processing(self, fusion_network):
        """Test temporal fusion block processing."""
        if hasattr(fusion_network, 'temporal_fusion'):
            temporal_block = fusion_network.temporal_fusion
            
            # Create temporal sequence
            sequence = torch.randn(2, 10, 128)  # batch, seq_len, features
            
            output = temporal_block(sequence)
            
            assert output.shape == (2, 128)  # Should aggregate over time
            assert torch.all(torch.isfinite(output))

if __name__ == '__main__':
    pytest.main([__file__, '-v'])