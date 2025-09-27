#!/usr/bin/env python3
"""
Test suite for Dream Orchestrator functionality.

Tests the complete dream system including experience replay,
strategy optimization, and neural learning.
"""

import pytest
import numpy as np
import torch
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from morpheus.dream_sim import (
    DreamOrchestrator,
    DreamConfig, 
    DreamSession,
    NeuralStrategyOptimizer
)
from morpheus.integration import MaterialBridge

class TestDreamOrchestrator:
    """Test cases for Dream Orchestrator."""
    
    @pytest.fixture
    def mock_database(self):
        """Mock database for testing."""
        db = Mock()
        db.get_recent_experiences.return_value = [
            {
                'id': 1,
                'session_id': 'test-session',
                'timestamp': time.time(),
                'primary_material': 'steel',
                'action_type': 'grip',
                'success': True,
                'reward': 1.0,
                'forces': [5.0, 0.0, 0.0],
                'action_params': {'pressure': 5.0}
            },
            {
                'id': 2,
                'session_id': 'test-session', 
                'timestamp': time.time(),
                'primary_material': 'rubber',
                'action_type': 'push',
                'success': False,
                'reward': -0.5,
                'forces': [2.0, 1.0, 0.0],
                'action_params': {'velocity': 0.5}
            }
        ]
        db.store_learned_strategy.return_value = None
        db.store_dream_session.return_value = 'dream-session-123'
        return db
        
    @pytest.fixture
    def mock_material_bridge(self):
        """Mock material bridge for testing."""
        bridge = Mock()
        bridge.materials = {
            'steel': Mock(friction=0.8, restitution=0.2, young_modulus=200e9, density=7800),
            'rubber': Mock(friction=0.6, restitution=0.8, young_modulus=1e6, density=1200)
        }
        bridge.get_material.side_effect = lambda name: bridge.materials.get(name)
        return bridge
        
    @pytest.fixture
    def dream_config(self):
        """Standard dream configuration."""
        return DreamConfig(
            replay_speed=10.0,
            variation_factor=0.2,
            min_improvement=0.1,
            max_iterations=100,
            parallel_dreams=2
        )
        
    @pytest.fixture
    def dream_orchestrator(self, mock_database, mock_material_bridge, dream_config):
        """Dream orchestrator instance for testing."""
        return DreamOrchestrator(mock_database, mock_material_bridge, dream_config)
        
    def test_initialization(self, dream_orchestrator, dream_config):
        """Test dream orchestrator initialization."""
        assert dream_orchestrator.config == dream_config
        assert dream_orchestrator.neural_optimizer is not None
        assert isinstance(dream_orchestrator.neural_optimizer, NeuralStrategyOptimizer)
        assert len(dream_orchestrator.discovered_strategies) == 0
        
    def test_dream_session_basic(self, dream_orchestrator):
        """Test basic dream session execution."""
        result = dream_orchestrator.dream(duration=5.0)
        
        assert isinstance(result, DreamSession)
        assert result.experiences_processed > 0
        assert result.session_id is not None
        assert result.start_time > 0
        assert result.end_time > result.start_time
        
    def test_experience_variation_generation(self, dream_orchestrator):
        """Test experience variation generation."""
        experience = {
            'primary_material': 'steel',
            'action_type': 'grip',
            'forces': [5.0, 0.0, 0.0],
            'action_params': {'pressure': 5.0}
        }
        
        variations = dream_orchestrator._generate_experience_variations(experience)
        
        assert len(variations) > 0
        assert all(isinstance(var, dict) for var in variations)
        assert any(var.get('variation_type') == 'material' for var in variations)
        assert any(var.get('variation_type') == 'force' for var in variations)
        
    def test_strategy_extraction(self, dream_orchestrator):
        """Test strategy extraction from successful variations."""
        original = {
            'primary_material': 'steel',
            'action_type': 'grip',
            'reward': 0.5
        }
        
        variation = {
            'primary_material': 'rubber',
            'action_type': 'grip',
            'variation_type': 'material',
            'reward': 1.0
        }
        
        strategy = dream_orchestrator._extract_strategy(original, variation, 0.5)
        
        assert strategy['improvement_ratio'] == 0.5
        assert strategy['category'] == 'material'
        assert 'steel' in strategy['applicable_materials']
        assert strategy['confidence'] > 0
        
    def test_neural_network_update(self, dream_orchestrator):
        """Test neural network learning from experiences."""
        experience = {
            'primary_material': 'steel',
            'action_type': 'grip',
            'fused_embedding': np.random.randn(128),
            'action_params': {'pressure': 5.0}
        }
        
        variation = experience.copy()
        variation['primary_material'] = 'rubber'
        
        # Test that neural update doesn't crash
        updates = dream_orchestrator._update_neural_network(experience, variation, 0.3)
        assert isinstance(updates, int)
        assert updates >= 0
        
    def test_strategy_consolidation(self, dream_orchestrator):
        """Test strategy consolidation and merging."""
        strategies = [
            {
                'name': 'strategy1',
                'category': 'material',
                'improvement_ratio': 0.5,
                'confidence': 0.8,
                'applicable_materials': ['steel']
            },
            {
                'name': 'strategy2', 
                'category': 'material',
                'improvement_ratio': 0.52,
                'confidence': 0.7,
                'applicable_materials': ['aluminum']
            },
            {
                'name': 'strategy3',
                'category': 'force',
                'improvement_ratio': 0.3,
                'confidence': 0.6,
                'applicable_materials': ['rubber']
            }
        ]
        
        # Create mock replays
        replays = []
        for strategy in strategies:
            replay = Mock()
            replay.strategies_found = [strategy]
            replays.append(replay)
            
        consolidated = dream_orchestrator._consolidate_strategies_neural(replays)
        
        assert len(consolidated) <= len(strategies)
        assert all(s['improvement_ratio'] >= 0 for s in consolidated)
        
    def test_parallel_processing(self, dream_orchestrator):
        """Test parallel experience processing."""
        experiences = [
            {'id': i, 'primary_material': 'steel', 'action_type': 'grip', 'success': True}
            for i in range(10)
        ]
        
        replays = dream_orchestrator._process_experiences_parallel(experiences, max_duration=2.0)
        
        assert len(replays) <= len(experiences)  # Some might be filtered out
        assert all(hasattr(replay, 'original_experience') for replay in replays if replay)
        
    def test_performance_metrics(self, dream_orchestrator):
        """Test performance metrics collection."""
        # Run a short dream session
        result = dream_orchestrator.dream(duration=2.0)
        
        metrics = dream_orchestrator.get_session_metrics()
        
        assert 'total_dreams' in metrics
        assert metrics['total_dreams'] >= 1
        assert 'total_experiences_processed' in metrics
        assert 'average_session_time' in metrics
        
    def test_neural_convergence_computation(self, dream_orchestrator):
        """Test neural network convergence metric."""
        convergence = dream_orchestrator._compute_neural_convergence()
        
        assert isinstance(convergence, float)
        assert 0.0 <= convergence <= 1.0
        
    def test_state_save_load(self, dream_orchestrator, tmp_path):
        """Test saving and loading neural state."""
        save_path = tmp_path / "neural_state.pt"
        
        # Save state
        dream_orchestrator.save_neural_state(str(save_path))
        assert save_path.exists()
        
        # Modify network
        original_params = list(dream_orchestrator.neural_optimizer.parameters())[0].clone()
        
        # Load state
        dream_orchestrator.load_neural_state(str(save_path))
        
        # Verify parameters were restored
        loaded_params = list(dream_orchestrator.neural_optimizer.parameters())[0]
        torch.testing.assert_close(original_params, loaded_params)

class TestNeuralStrategyOptimizer:
    """Test cases for Neural Strategy Optimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Neural strategy optimizer instance."""
        return NeuralStrategyOptimizer(state_dim=64, action_dim=8, hidden_dim=128)
        
    def test_initialization(self, optimizer):
        """Test neural optimizer initialization."""
        assert optimizer.state_dim == 64
        assert optimizer.action_dim == 8
        assert hasattr(optimizer, 'strategy_encoder')
        assert hasattr(optimizer, 'improvement_predictor')
        assert hasattr(optimizer, 'similarity_net')
        
    def test_forward_pass(self, optimizer):
        """Test forward pass through optimizer."""
        batch_size = 4
        state = torch.randn(batch_size, 64)
        action = torch.randn(batch_size, 8)
        
        encoding, improvement = optimizer(state, action)
        
        assert encoding.shape == (batch_size, 64)
        assert improvement.shape == (batch_size, 1)
        assert torch.all(improvement >= 0) and torch.all(improvement <= 1)  # Sigmoid output
        
    def test_similarity_computation(self, optimizer):
        """Test strategy similarity computation."""
        encoding1 = torch.randn(4, 64)
        encoding2 = torch.randn(4, 64)
        
        similarity = optimizer.compute_similarity(encoding1, encoding2)
        
        assert similarity.shape == (4, 1)
        assert torch.all(similarity >= 0) and torch.all(similarity <= 1)
        
    def test_gradient_flow(self, optimizer):
        """Test gradient flow through network."""
        state = torch.randn(2, 64, requires_grad=True)
        action = torch.randn(2, 8, requires_grad=True)
        
        encoding, improvement = optimizer(state, action)
        loss = improvement.mean()
        
        loss.backward()
        
        assert state.grad is not None
        assert action.grad is not None

class TestDreamConfig:
    """Test cases for Dream Configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DreamConfig()
        
        assert config.replay_speed == 10.0
        assert config.variation_factor == 0.2
        assert config.exploration_rate == 0.3
        assert config.min_improvement == 0.1
        assert config.max_iterations == 1000
        assert config.parallel_dreams == 4
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DreamConfig(
            replay_speed=20.0,
            parallel_dreams=8,
            min_improvement=0.05
        )
        
        assert config.replay_speed == 20.0
        assert config.parallel_dreams == 8
        assert config.min_improvement == 0.05
        # Other values should remain default
        assert config.variation_factor == 0.2

if __name__ == '__main__':
    pytest.main([__file__, '-v'])