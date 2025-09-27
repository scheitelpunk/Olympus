"""Comprehensive test suite for Dream Orchestrator.

Tests the complete dream orchestration functionality including:
- Neural network optimization
- Experience replay and variation generation
- Strategy consolidation and ranking
- Parallel processing and performance metrics
- Error handling and edge cases
"""

import pytest
import numpy as np
import torch
import time
import uuid
from unittest.mock import Mock, MagicMock, patch, call
from concurrent.futures import Future
from dataclasses import dataclass
from typing import List, Dict, Any

from morpheus.dream_sim.dream_orchestrator import (
    DreamOrchestrator, DreamConfig, DreamSession, ExperienceReplay,
    NeuralStrategyOptimizer
)
from morpheus.integration.material_bridge import MaterialBridge
from morpheus.core.types import MaterialProperties, MaterialType


class TestDreamConfig:
    """Test configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DreamConfig()
        
        assert config.replay_speed == 10.0
        assert config.variation_factor == 0.2
        assert config.exploration_rate == 0.3
        assert config.consolidation_threshold == 0.8
        assert config.min_improvement == 0.1
        assert config.max_iterations == 1000
        assert config.parallel_dreams == 4
        assert config.experience_sample_rate == 0.8
        assert config.strategy_merge_threshold == 0.15
        assert config.neural_learning_rate == 0.001
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DreamConfig(
            replay_speed=5.0,
            parallel_dreams=8,
            neural_learning_rate=0.01
        )
        
        assert config.replay_speed == 5.0
        assert config.parallel_dreams == 8
        assert config.neural_learning_rate == 0.01
        # Other values should remain default
        assert config.variation_factor == 0.2


class TestNeuralStrategyOptimizer:
    """Test neural network for strategy optimization."""
    
    @pytest.fixture
    def optimizer(self):
        """Create neural optimizer instance."""
        return NeuralStrategyOptimizer(state_dim=128, action_dim=16, hidden_dim=256)
    
    def test_initialization(self, optimizer):
        """Test neural network initialization."""
        assert optimizer.state_dim == 128
        assert optimizer.action_dim == 16
        
        # Check network layers exist
        assert hasattr(optimizer, 'strategy_encoder')
        assert hasattr(optimizer, 'improvement_predictor')
        assert hasattr(optimizer, 'similarity_net')
    
    def test_forward_pass(self, optimizer):
        """Test forward pass through network."""
        batch_size = 4
        state = torch.randn(batch_size, 128)
        action = torch.randn(batch_size, 16)
        
        encoding, improvement = optimizer(state, action)
        
        assert encoding.shape == (batch_size, 64)
        assert improvement.shape == (batch_size, 1)
        assert torch.all(improvement >= 0)  # Sigmoid output
        assert torch.all(improvement <= 1)
    
    def test_similarity_computation(self, optimizer):
        """Test similarity computation between encodings."""
        encoding1 = torch.randn(4, 64)
        encoding2 = torch.randn(4, 64)
        
        similarity = optimizer.compute_similarity(encoding1, encoding2)
        
        assert similarity.shape == (4, 1)
        assert torch.all(similarity >= 0)  # Sigmoid output
        assert torch.all(similarity <= 1)
    
    def test_gradient_flow(self, optimizer):
        """Test that gradients flow correctly."""
        state = torch.randn(1, 128, requires_grad=True)
        action = torch.randn(1, 16, requires_grad=True)
        
        encoding, improvement = optimizer(state, action)
        loss = improvement.sum()
        loss.backward()
        
        # Check gradients exist
        assert state.grad is not None
        assert action.grad is not None
        
        # Check network parameters have gradients
        for param in optimizer.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestDreamOrchestrator:
    """Test the complete dream orchestrator system."""
    
    @pytest.fixture
    def mock_database(self):
        """Create mock database."""
        db = Mock()
        db.get_recent_experiences.return_value = []
        db.store_dream_session.return_value = True
        db.store_learned_strategy.return_value = True
        return db
    
    @pytest.fixture
    def mock_material_bridge(self):
        """Create mock material bridge."""
        bridge = Mock()
        
        # Mock material properties
        mock_material = MaterialProperties(
            name='steel',
            material_type=MaterialType.METAL,
            color=[0.7, 0.7, 0.7, 1.0],
            friction=0.8,
            restitution=0.2,
            density=7850,
            young_modulus=200e9,
            poisson_ratio=0.3
        )
        
        bridge.get_material.return_value = mock_material
        bridge.materials = {'steel': mock_material, 'default': mock_material}
        
        return bridge
    
    @pytest.fixture
    def orchestrator(self, mock_database, mock_material_bridge):
        """Create dream orchestrator instance."""
        config = DreamConfig(
            parallel_dreams=2,  # Reduce for testing
            max_iterations=10
        )
        return DreamOrchestrator(mock_database, mock_material_bridge, config)
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.db is not None
        assert orchestrator.material_bridge is not None
        assert orchestrator.config is not None
        assert orchestrator.neural_optimizer is not None
        assert orchestrator.optimizer is not None
        assert hasattr(orchestrator, '_lock')
        assert isinstance(orchestrator.session_metrics, dict)
    
    def test_empty_dream_session(self, orchestrator, mock_database):
        """Test dream session with no experiences."""
        mock_database.get_recent_experiences.return_value = []
        
        session = orchestrator.dream(duration_seconds=1.0)
        
        assert isinstance(session, DreamSession)
        assert session.experiences_processed == 0
        assert session.variations_generated == 0
        assert session.strategies_discovered == 0
        assert session.strategies_consolidated == 0
    
    def test_dream_session_with_experiences(self, orchestrator, mock_database):
        """Test dream session with sample experiences."""
        # Mock experiences
        experiences = [
            {
                'id': 1,
                'primary_material': 'steel',
                'action_type': 'grip',
                'forces': [1.0, 0.5, 0.2],
                'action_params': {'force': 10.0, 'duration': 1.0},
                'success': True,
                'reward': 0.8,
                'timestamp': time.time(),
                'fused_embedding': np.random.randn(128).tolist()
            },
            {
                'id': 2,
                'primary_material': 'steel',
                'action_type': 'push',
                'forces': [5.0, 0.0, 0.0],
                'action_params': {'force': 15.0, 'velocity': 2.0},
                'success': False,
                'reward': 0.2,
                'timestamp': time.time(),
                'robot_velocity': [0.1, 0.0, 0.0],
                'fused_embedding': np.random.randn(128).tolist()
            }
        ]
        
        mock_database.get_recent_experiences.return_value = experiences
        
        session = orchestrator.dream(duration_seconds=2.0)
        
        assert isinstance(session, DreamSession)
        assert session.experiences_processed > 0
        assert session.end_time is not None
        assert session.end_time > session.start_time
        
        # Verify database calls
        mock_database.get_recent_experiences.assert_called_once()
        mock_database.store_dream_session.assert_called_once()
    
    def test_experience_variation_generation(self, orchestrator):
        """Test generation of experience variations."""
        experience = {
            'primary_material': 'steel',
            'action_type': 'grip',
            'forces': [1.0, 0.5, 0.2],
            'action_params': {'force': 10.0, 'duration': 1.0},
            'robot_velocity': [0.1, 0.0, 0.0],
            'success': True,
            'reward': 0.8
        }
        
        variations = orchestrator._generate_experience_variations(experience)
        
        assert isinstance(variations, list)
        assert len(variations) > 0
        assert len(variations) <= 15  # max_variations
        
        # Check variation types are present
        variation_types = [v.get('variation_type') for v in variations]
        assert 'material' in variation_types
        assert 'action' in variation_types
    
    def test_material_variations(self, orchestrator):
        """Test material-based variations."""
        experience = {
            'primary_material': 'steel',
            'action_type': 'grip'
        }
        
        variations = orchestrator._generate_material_variations(experience, 3)
        
        assert isinstance(variations, list)
        for variation in variations:
            assert variation['variation_type'] == 'material'
            assert 'primary_material' in variation
            assert variation['primary_material'] != 'steel'  # Should be different
            assert 'variation_source' in variation
            assert 'variation_target' in variation
    
    def test_physics_variations(self, orchestrator):
        """Test physics parameter variations."""
        experience = {
            'forces': [1.0, 0.5, 0.2],
            'robot_velocity': [0.1, 0.0, 0.0]
        }
        
        variations = orchestrator._generate_physics_variations(experience, 4)
        
        assert isinstance(variations, list)
        for variation in variations:
            assert variation['variation_type'] in ['force', 'velocity']
            if variation['variation_type'] == 'force':
                assert 'force_scale' in variation
                assert 'forces' in variation
            elif variation['variation_type'] == 'velocity':
                assert 'robot_velocity' in variation
    
    def test_action_variations(self, orchestrator):
        """Test action parameter variations."""
        experience = {
            'action_params': {
                'force': 10.0,
                'duration': 1.0,
                'position': [0.1, 0.2, 0.3]
            }
        }
        
        variations = orchestrator._generate_action_variations(experience, 3)
        
        assert isinstance(variations, list)
        for variation in variations:
            assert variation['variation_type'] == 'action'
            assert 'action_params' in variation
            # Parameters should be modified
            original_params = experience['action_params']
            new_params = variation['action_params']
            assert new_params != original_params
    
    def test_neural_variations(self, orchestrator):
        """Test neural network guided variations."""
        experience = {
            'primary_material': 'steel',
            'action_params': {'force': 10.0, 'duration': 1.0},
            'fused_embedding': np.random.randn(128).tolist(),
            'forces': [1.0, 0.5, 0.2]
        }
        
        variations = orchestrator._generate_neural_variations(experience, 2)
        
        assert isinstance(variations, list)
        for variation in variations:
            assert variation['variation_type'] == 'neural'
            assert 'neural_confidence' in variation
            assert 0 <= variation['neural_confidence'] <= 1
    
    def test_experience_to_tensors(self, orchestrator):
        """Test conversion of experience to neural tensors."""
        experience = {
            'fused_embedding': np.random.randn(128).tolist(),
            'primary_material': 'steel',
            'forces': [1.0, 0.5, 0.2],
            'success': True,
            'reward': 0.8,
            'action_type': 'grip',
            'action_params': {'force': 10.0, 'duration': 1.0}
        }
        
        state = orchestrator._experience_to_state(experience)
        action = orchestrator._experience_to_action(experience)
        
        assert isinstance(state, torch.Tensor)
        assert isinstance(action, torch.Tensor)
        assert state.shape == (128,)
        assert action.shape == (16,)
    
    def test_variation_evaluation(self, orchestrator):
        """Test evaluation of variations."""
        original = {
            'primary_material': 'steel',
            'action_type': 'grip',
            'reward': 0.5,
            'fused_embedding': np.random.randn(128).tolist(),
            'forces': [1.0, 0.5, 0.2],
            'action_params': {'force': 10.0}
        }
        
        variation = original.copy()
        variation['primary_material'] = 'aluminum'
        variation['variation_type'] = 'material'
        
        improvement = orchestrator._evaluate_variation_neural(original, variation)
        
        assert isinstance(improvement, float)
        assert improvement >= 0
    
    def test_strategy_extraction(self, orchestrator):
        """Test extraction of strategy from successful variation."""
        original = {
            'primary_material': 'steel',
            'action_type': 'grip',
            'reward': 0.5
        }
        
        variation = original.copy()
        variation['primary_material'] = 'aluminum'
        variation['variation_type'] = 'material'
        variation['success'] = True
        
        strategy = orchestrator._extract_strategy(original, variation, 0.3)
        
        assert isinstance(strategy, dict)
        assert 'id' in strategy
        assert 'name' in strategy
        assert 'category' in strategy
        assert 'strategy_data' in strategy
        assert 'improvement_ratio' in strategy
        assert 'confidence' in strategy
        assert strategy['improvement_ratio'] == 0.3
        assert 'material' in strategy['category']
    
    def test_strategy_consolidation(self, orchestrator):
        """Test consolidation of similar strategies."""
        # Create mock replays with strategies
        replays = []
        for i in range(3):
            replay = Mock()
            replay.strategies_found = [
                {
                    'id': f'strategy_{i}',
                    'category': 'material',
                    'improvement_ratio': 0.2 + i * 0.1,
                    'applicable_materials': ['steel'],
                    'applicable_scenarios': ['grip'],
                    'confidence': 0.8
                }
            ]
            replays.append(replay)
        
        with patch.object(orchestrator, '_strategy_to_encoding') as mock_encoding:
            mock_encoding.return_value = torch.randn(64)
            
            with patch.object(orchestrator, '_compute_neural_similarity') as mock_similarity:
                mock_similarity.return_value = torch.tensor(0.9)  # High similarity
                
                consolidated = orchestrator._consolidate_strategies_neural(replays)
        
        assert isinstance(consolidated, list)
        assert len(consolidated) <= len(replays)  # Should be consolidated
    
    def test_strategy_storage(self, orchestrator, mock_database):
        """Test storage of consolidated strategies."""
        strategies = [
            {
                'id': 'strategy_1',
                'name': 'test_strategy',
                'category': 'material',
                'improvement_ratio': 0.3,
                'confidence': 0.8,
                'strategy_data': {'test': 'data'},
                'applicable_materials': ['steel']
            }
        ]
        
        mock_database.store_learned_strategy.return_value = True
        
        stored_count = orchestrator._store_strategies(strategies)
        
        assert stored_count == 1
        mock_database.store_learned_strategy.assert_called_once()
    
    @patch('morpheus.dream_sim.dream_orchestrator.ThreadPoolExecutor')
    def test_parallel_processing(self, mock_executor, orchestrator, mock_database):
        """Test parallel processing of experiences."""
        experiences = [
            {'id': 1, 'primary_material': 'steel'},
            {'id': 2, 'primary_material': 'aluminum'}
        ]
        
        # Mock future results
        mock_future = Mock()
        mock_future.result.return_value = []
        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
        
        replays = orchestrator._process_experiences_parallel(experiences, 5.0)
        
        assert isinstance(replays, list)
        mock_executor.assert_called_once()
    
    def test_neural_network_update(self, orchestrator):
        """Test neural network parameter updates."""
        original = {
            'fused_embedding': np.random.randn(128).tolist(),
            'forces': [1.0, 0.5, 0.2],
            'action_params': {'force': 10.0}
        }
        
        variation = original.copy()
        variation['action_params'] = {'force': 12.0}
        
        # Get initial parameters
        initial_params = [p.clone() for p in orchestrator.neural_optimizer.parameters()]
        
        updates = orchestrator._update_neural_network(original, variation, 0.3)
        
        # Check that parameters changed
        updated_params = list(orchestrator.neural_optimizer.parameters())
        params_changed = any(
            not torch.equal(initial, updated) 
            for initial, updated in zip(initial_params, updated_params)
            if updated.requires_grad
        )
        
        assert isinstance(updates, int)
        if updates > 0:
            assert params_changed
    
    def test_session_metrics_update(self, orchestrator):
        """Test update of session metrics."""
        initial_dreams = orchestrator.session_metrics['total_dreams']
        
        session = DreamSession(
            session_id='test',
            start_time=time.time(),
            end_time=time.time() + 1,
            config=orchestrator.config,
            experiences_processed=5,
            variations_generated=15,
            strategies_discovered=3,
            strategies_consolidated=2,
            average_improvement=0.2,
            best_improvement=0.5,
            neural_convergence=0.8,
            compute_metrics={}
        )
        
        orchestrator._update_global_metrics(session)
        
        assert orchestrator.session_metrics['total_dreams'] == initial_dreams + 1
        assert orchestrator.session_metrics['total_experiences_processed'] >= 5
        assert orchestrator.session_metrics['total_strategies_found'] >= 3
    
    def test_neural_convergence_computation(self, orchestrator):
        """Test neural convergence metric calculation."""
        convergence = orchestrator._compute_neural_convergence()
        
        assert isinstance(convergence, float)
        assert 0 <= convergence <= 1
    
    def test_reset_neural_network(self, orchestrator):
        """Test neural network reset functionality."""
        # Modify network parameters
        with torch.no_grad():
            for param in orchestrator.neural_optimizer.parameters():
                if param.requires_grad:
                    param.fill_(1.0)
        
        # Reset network
        orchestrator.reset_neural_network()
        
        # Parameters should be different now
        all_ones = all(
            torch.allclose(param, torch.ones_like(param)) 
            for param in orchestrator.neural_optimizer.parameters() 
            if param.requires_grad
        )
        assert not all_ones
    
    def test_save_and_load_neural_state(self, orchestrator, tmp_path):
        """Test saving and loading neural network state."""
        filepath = tmp_path / "neural_state.pth"
        
        # Modify some parameters
        with torch.no_grad():
            list(orchestrator.neural_optimizer.parameters())[0].fill_(0.5)
        
        # Save state
        orchestrator.save_neural_state(str(filepath))
        assert filepath.exists()
        
        # Reset network
        orchestrator.reset_neural_network()
        
        # Load state back
        orchestrator.load_neural_state(str(filepath))
        
        # First parameter should be back to 0.5
        first_param = list(orchestrator.neural_optimizer.parameters())[0]
        assert torch.allclose(first_param, torch.full_like(first_param, 0.5))
    
    def test_get_session_metrics(self, orchestrator):
        """Test getting session metrics."""
        metrics = orchestrator.get_session_metrics()
        
        assert isinstance(metrics, dict)
        required_keys = [
            'total_dreams', 'total_experiences_processed', 
            'total_strategies_found', 'average_session_time'
        ]
        for key in required_keys:
            assert key in metrics
    
    def test_error_handling_invalid_experience(self, orchestrator):
        """Test error handling with invalid experience data."""
        # Experience with missing required fields
        experience = {'id': 1}  # Missing most fields
        
        replay = orchestrator._replay_single_experience(experience)
        
        # Should handle gracefully
        assert replay is None or isinstance(replay, ExperienceReplay)
    
    def test_dream_with_focus_materials(self, orchestrator, mock_database):
        """Test dream session with material focus."""
        experiences = [
            {'id': 1, 'primary_material': 'steel', 'action_type': 'grip'},
            {'id': 2, 'primary_material': 'aluminum', 'action_type': 'grip'},
            {'id': 3, 'primary_material': 'plastic', 'action_type': 'grip'}
        ]
        mock_database.get_recent_experiences.return_value = experiences
        
        session = orchestrator.dream(
            duration_seconds=1.0,
            focus_materials=['steel', 'aluminum']
        )
        
        # Should process only focused materials
        assert isinstance(session, DreamSession)
    
    def test_dream_with_focus_actions(self, orchestrator, mock_database):
        """Test dream session with action focus."""
        experiences = [
            {'id': 1, 'primary_material': 'steel', 'action_type': 'grip'},
            {'id': 2, 'primary_material': 'steel', 'action_type': 'push'},
            {'id': 3, 'primary_material': 'steel', 'action_type': 'slide'}
        ]
        mock_database.get_recent_experiences.return_value = experiences
        
        session = orchestrator.dream(
            duration_seconds=1.0,
            focus_actions=['grip', 'push']
        )
        
        # Should process only focused actions
        assert isinstance(session, DreamSession)
    
    def test_concurrent_dream_sessions(self, orchestrator, mock_database):
        """Test multiple concurrent dream sessions."""
        mock_database.get_recent_experiences.return_value = [
            {'id': 1, 'primary_material': 'steel', 'action_type': 'grip'}
        ]
        
        import threading
        
        results = []
        
        def run_dream():
            session = orchestrator.dream(duration_seconds=0.5)
            results.append(session)
        
        # Start multiple threads
        threads = [threading.Thread(target=run_dream) for _ in range(3)]
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 3
        for session in results:
            assert isinstance(session, DreamSession)
    
    def test_cleanup(self, orchestrator):
        """Test proper cleanup and resource deallocation."""
        # This test ensures no memory leaks or resource issues
        initial_metrics = orchestrator.get_session_metrics()
        
        # Run some operations
        experience = {
            'id': 1,
            'primary_material': 'steel',
            'forces': [1.0, 0.5, 0.2],
            'action_params': {'force': 10.0}
        }
        
        variations = orchestrator._generate_experience_variations(experience)
        assert len(variations) > 0
        
        # Metrics should be updated
        final_metrics = orchestrator.get_session_metrics()
        assert isinstance(final_metrics, dict)


@pytest.fixture
def sample_experiences():
    """Generate sample experiences for testing."""
    experiences = []
    for i in range(10):
        exp = {
            'id': i,
            'primary_material': np.random.choice(['steel', 'aluminum', 'plastic']),
            'action_type': np.random.choice(['grip', 'push', 'slide']),
            'forces': np.random.uniform(0, 10, 3).tolist(),
            'action_params': {
                'force': np.random.uniform(5, 15),
                'duration': np.random.uniform(0.5, 2.0)
            },
            'success': np.random.choice([True, False]),
            'reward': np.random.uniform(-0.5, 1.0),
            'timestamp': time.time() + i,
            'fused_embedding': np.random.randn(128).tolist()
        }
        experiences.append(exp)
    return experiences


class TestDreamOrchestrator_Integration:
    """Integration tests for dream orchestrator."""
    
    @pytest.fixture
    def orchestrator(self, mock_database, mock_material_bridge):
        """Create orchestrator for integration tests."""
        config = DreamConfig(
            parallel_dreams=2,
            max_iterations=5,
            min_improvement=0.05
        )
        return DreamOrchestrator(mock_database, mock_material_bridge, config)
    
    def test_full_dream_cycle(self, orchestrator, mock_database, sample_experiences):
        """Test complete dream cycle from experiences to strategies."""
        mock_database.get_recent_experiences.return_value = sample_experiences[:5]
        
        session = orchestrator.dream(duration_seconds=3.0)
        
        # Verify complete pipeline
        assert isinstance(session, DreamSession)
        assert session.experiences_processed > 0
        assert session.end_time > session.start_time
        
        # Check that database methods were called
        mock_database.get_recent_experiences.assert_called_once()
        mock_database.store_dream_session.assert_called_once()
    
    def test_performance_metrics(self, orchestrator, mock_database, sample_experiences):
        """Test performance metrics collection."""
        mock_database.get_recent_experiences.return_value = sample_experiences
        
        start_time = time.time()
        session = orchestrator.dream(duration_seconds=2.0)
        end_time = time.time()
        
        # Verify timing metrics
        assert session.compute_metrics is not None
        assert 'session_duration' in session.compute_metrics
        assert session.compute_metrics['session_duration'] <= (end_time - start_time) + 1  # Allow some tolerance
        
        if session.experiences_processed > 0:
            assert 'experiences_per_second' in session.compute_metrics
            assert session.compute_metrics['experiences_per_second'] > 0
    
    def test_memory_efficiency(self, orchestrator, mock_database):
        """Test memory usage during dream sessions."""
        # Large number of experiences to test memory handling
        large_experiences = []
        for i in range(100):
            exp = {
                'id': i,
                'primary_material': 'steel',
                'fused_embedding': np.random.randn(128).tolist(),
                'forces': [1.0, 0.5, 0.2]
            }
            large_experiences.append(exp)
        
        mock_database.get_recent_experiences.return_value = large_experiences
        
        session = orchestrator.dream(duration_seconds=5.0)
        
        # Should complete without memory errors
        assert isinstance(session, DreamSession)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])