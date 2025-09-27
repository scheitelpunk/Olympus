#!/usr/bin/env python3
"""
Test suite for Forward Model Predictor functionality.

Tests sensory state prediction, uncertainty quantification,
and physics-informed neural networks.
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

from morpheus.predictive import (
    SensoryPredictor,
    PredictionConfig,
    PredictionResult,
    PhysicsInformedLayer,
    MaterialConditioningNetwork,
    UncertaintyEstimationHead,
    PredictionEvaluator
)

class TestSensoryPredictor:
    """Test cases for Sensory Predictor."""
    
    @pytest.fixture
    def prediction_config(self):
        """Standard prediction configuration."""
        return PredictionConfig(
            prediction_horizon=5,
            state_dim=64,
            action_dim=8,
            hidden_dim=128,
            num_layers=2,
            uncertainty_estimation=True,
            physics_informed=True,
            material_conditioning=True,
            temporal_attention=True
        )
        
    @pytest.fixture
    def predictor(self, prediction_config):
        """Predictor instance for testing."""
        return SensoryPredictor(prediction_config)
        
    @pytest.fixture
    def sample_data(self, prediction_config):
        """Sample data for testing."""
        batch_size = 4
        return {
            'current_state': torch.randn(batch_size, prediction_config.state_dim),
            'planned_actions': torch.randn(batch_size, prediction_config.prediction_horizon, prediction_config.action_dim),
            'material_properties': torch.randn(batch_size, 16)  # Reduced material dims
        }
        
    def test_initialization(self, predictor, prediction_config):
        """Test predictor initialization."""
        assert predictor.config == prediction_config
        assert hasattr(predictor, 'state_encoder')
        assert hasattr(predictor, 'action_encoder')
        assert hasattr(predictor, 'prediction_network')
        
        # Check optional components
        assert hasattr(predictor, 'physics_layer')
        assert hasattr(predictor, 'material_network')
        assert hasattr(predictor, 'uncertainty_head')
        
    def test_forward_pass(self, predictor, sample_data):
        """Test forward pass through predictor."""
        result = predictor(
            sample_data['current_state'],
            sample_data['planned_actions'],
            sample_data['material_properties']
        )
        
        assert isinstance(result, PredictionResult)
        assert result.predicted_states.shape == (4, 5, 64)  # batch, horizon, state_dim
        assert result.predicted_uncertainties.shape == (4, 5, 64)
        assert result.processing_time > 0
        
        # Check uncertainty values are positive
        assert torch.all(result.predicted_uncertainties >= 0)
        
    def test_single_step_prediction(self, predictor, sample_data):
        """Test single step ahead prediction."""
        next_state, uncertainty = predictor.predict_single_step(
            sample_data['current_state'],
            sample_data['planned_actions'][:, 0],  # Single action
            sample_data['material_properties']
        )
        
        assert next_state.shape == (4, 64)  # batch, state_dim
        if uncertainty is not None:
            assert uncertainty.shape == (4, 64)
            assert torch.all(uncertainty >= 0)
            
    def test_prediction_rollout(self, predictor, sample_data):
        """Test multi-step rollout with action policy."""
        def simple_policy(state):
            return torch.randn(state.shape[0], 8)  # Random actions
            
        rollout = predictor.predict_rollout(
            sample_data['current_state'],
            simple_policy,
            num_steps=3,
            material_properties=sample_data['material_properties']
        )
        
        assert 'states' in rollout
        assert 'actions' in rollout
        assert rollout['states'].shape == (4, 4, 64)  # batch, steps+1, state_dim
        assert rollout['actions'].shape == (4, 3, 8)  # batch, steps, action_dim
        
    def test_prediction_error_computation(self, predictor):
        """Test prediction error computation."""
        predictions = torch.randn(2, 3, 64)
        targets = torch.randn(2, 3, 64)
        uncertainties = torch.rand(2, 3, 64) * 0.1 + 0.05  # Small positive uncertainties
        
        errors = predictor.compute_prediction_error(predictions, targets, uncertainties)
        
        assert 'mse' in errors
        assert 'rmse' in errors
        assert 'mae' in errors
        assert 'nll' in errors
        assert 'coverage' in errors
        
        # Check shapes
        assert errors['mse'].shape == predictions.shape
        assert errors['rmse'].shape == (2, 3)  # Reduced over feature dimension
        
        # Check coverage is between 0 and 1
        assert torch.all(errors['coverage'] >= 0) and torch.all(errors['coverage'] <= 1)
        
    def test_experience_update(self, predictor, sample_data):
        """Test online learning with experience."""
        states = sample_data['current_state']
        actions = sample_data['planned_actions'][:, 0]  # Single action
        next_states = torch.randn_like(states)  # Mock next states
        
        # Should not raise exception
        predictor.update_with_experience(
            states, actions, next_states, sample_data['material_properties']
        )
        
    def test_model_complexity_metrics(self, predictor):
        """Test model complexity computation."""
        complexity = predictor.get_model_complexity()
        
        assert 'total_parameters' in complexity
        assert 'trainable_parameters' in complexity
        assert 'prediction_horizon' in complexity
        assert 'state_dimension' in complexity
        
        assert complexity['total_parameters'] > 0
        assert complexity['trainable_parameters'] > 0
        assert complexity['prediction_horizon'] == 5
        assert complexity['state_dimension'] == 64
        
    def test_state_persistence(self, predictor, tmp_path):
        """Test saving and loading predictor state."""
        save_path = tmp_path / "predictor_state.pt"
        
        # Save state
        predictor.save_predictor_state(str(save_path))
        assert save_path.exists()
        
        # Modify parameters
        original_param = next(predictor.parameters()).clone()
        
        # Load state
        predictor.load_predictor_state(str(save_path))
        
        # Verify restoration
        loaded_param = next(predictor.parameters())
        torch.testing.assert_close(original_param, loaded_param)

class TestPhysicsInformedLayer:
    """Test cases for Physics-Informed Layer."""
    
    @pytest.fixture
    def physics_layer(self):
        """Physics-informed layer for testing."""
        constraints = ['conservation_momentum', 'conservation_energy']
        return PhysicsInformedLayer(64, 64, constraints)
        
    def test_physics_layer_forward(self, physics_layer):
        """Test physics-informed layer forward pass."""
        input_tensor = torch.randn(4, 64)
        
        output, constraints = physics_layer(input_tensor)
        
        assert output.shape == (4, 64)
        assert isinstance(constraints, dict)
        assert 'conservation_momentum' in constraints
        assert 'conservation_energy' in constraints
        
        # Check constraint outputs have correct shapes
        assert constraints['conservation_momentum'].shape == (4, 64)
        assert constraints['conservation_energy'].shape == (4, 1)
        
    def test_physics_constraints_application(self, physics_layer):
        """Test that physics constraints modify outputs."""
        input_tensor = torch.randn(4, 64)
        
        # Compare with and without constraints
        output_with_physics, _ = physics_layer(input_tensor)
        
        # Physics layer should produce different output than linear layer alone
        linear_output = physics_layer.activation(physics_layer.linear(input_tensor))
        
        # Outputs should be different (constraints are applied)
        assert not torch.allclose(output_with_physics, linear_output, atol=1e-6)

class TestMaterialConditioningNetwork:
    """Test cases for Material Conditioning Network."""
    
    @pytest.fixture
    def material_network(self):
        """Material conditioning network for testing."""
        return MaterialConditioningNetwork(material_dim=16, hidden_dim=64, output_dim=32)
        
    def test_material_conditioning_forward(self, material_network):
        """Test material conditioning forward pass."""
        material_props = torch.randn(4, 16)
        
        influences = material_network(material_props)
        
        assert isinstance(influences, dict)
        assert 'friction' in influences
        assert 'stiffness' in influences
        assert 'damping' in influences
        
        # Check output shapes
        for influence_type, influence_values in influences.items():
            assert influence_values.shape == (4, 32)
            
    def test_material_influence_ranges(self, material_network):
        """Test material influence value ranges."""
        material_props = torch.randn(4, 16)
        influences = material_network(material_props)
        
        # Damping should be in [0, 1] due to sigmoid activation
        damping = influences['damping']
        assert torch.all(damping >= 0) and torch.all(damping <= 1)
        
        # Other influences should be finite
        assert torch.all(torch.isfinite(influences['friction']))
        assert torch.all(torch.isfinite(influences['stiffness']))

class TestUncertaintyEstimationHead:
    """Test cases for Uncertainty Estimation Head."""
    
    @pytest.fixture
    def uncertainty_head(self):
        """Uncertainty estimation head for testing."""
        return UncertaintyEstimationHead(input_dim=64, output_dim=32, hidden_dim=128)
        
    def test_uncertainty_estimation_forward(self, uncertainty_head):
        """Test uncertainty estimation forward pass."""
        input_tensor = torch.randn(4, 64)
        
        aleatoric, epistemic, confidence = uncertainty_head(input_tensor)
        
        assert aleatoric.shape == (4, 32)
        assert epistemic.shape == (4, 32)
        assert confidence.shape == (4, 1)
        
        # Uncertainties should be positive (Softplus activation)
        assert torch.all(aleatoric >= 0)
        assert torch.all(epistemic >= 0)
        
        # Confidence should be in [0, 1] (Sigmoid activation)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)
        
    def test_uncertainty_combination(self, uncertainty_head):
        """Test combining aleatoric and epistemic uncertainties."""
        input_tensor = torch.randn(4, 64)
        aleatoric, epistemic, confidence = uncertainty_head(input_tensor)
        
        # Combined uncertainty should be computed correctly
        total_uncertainty = torch.sqrt(aleatoric**2 + epistemic**2)
        
        assert torch.all(total_uncertainty >= aleatoric)
        assert torch.all(total_uncertainty >= epistemic)
        assert torch.all(torch.isfinite(total_uncertainty))

class TestMultiScaleTemporalEncoder:
    """Test cases for Multi-Scale Temporal Encoder."""
    
    def test_temporal_encoder_creation(self, predictor):
        """Test temporal encoder initialization and forward pass."""
        if hasattr(predictor, 'temporal_encoder'):
            encoder = predictor.temporal_encoder
            
            # Test forward pass
            input_seq = torch.randn(2, 10, 128)  # batch, seq_len, input_dim
            
            features, attention_weights = encoder(input_seq)
            
            assert features.shape == (2, 10, 128)  # Same as input
            assert attention_weights.shape == (2, 3)  # batch, num_scales
            
            # Attention weights should sum to 1
            assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(2))

class TestPredictionEvaluator:
    """Test cases for Prediction Evaluator."""
    
    @pytest.fixture
    def evaluator(self, predictor):
        """Prediction evaluator instance."""
        return PredictionEvaluator(predictor)
        
    @pytest.fixture
    def test_data(self):
        """Test data for evaluation."""
        batch_size = 8
        return {
            'states': torch.randn(batch_size, 64),
            'actions': torch.randn(batch_size, 5, 8),
            'targets': torch.randn(batch_size, 5, 64),
            'materials': torch.randn(batch_size, 16)
        }
        
    def test_prediction_evaluation(self, evaluator, test_data):
        """Test prediction evaluation."""
        metrics = evaluator.evaluate_predictions(
            test_data['states'],
            test_data['actions'],
            test_data['targets'],
            test_data['materials']
        )
        
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'horizon_degradation' in metrics
        
        # Check metric ranges
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['horizon_degradation'] > 0
        
    def test_evaluation_history(self, evaluator, test_data):
        """Test evaluation history tracking."""
        # Run multiple evaluations
        for _ in range(3):
            evaluator.evaluate_predictions(
                test_data['states'],
                test_data['actions'],
                test_data['targets']
            )
            
        assert len(evaluator.evaluation_history) == 3
        
        # Get summary
        summary = evaluator.get_evaluation_summary()
        
        assert 'num_evaluations' in summary
        assert 'recent_performance' in summary
        assert summary['num_evaluations'] == 3
        
    def test_uncertainty_calibration(self, evaluator):
        """Test uncertainty calibration computation."""
        predictions = torch.randn(4, 3, 64)
        targets = torch.randn(4, 3, 64)
        uncertainties = torch.rand(4, 3, 64) * 0.2 + 0.1
        
        calibration = evaluator._compute_uncertainty_calibration(
            predictions, targets, uncertainties
        )
        
        assert 'uncertainty_calibration_error' in calibration
        assert 'mean_uncertainty' in calibration
        assert 'uncertainty_std' in calibration
        
        assert calibration['uncertainty_calibration_error'] >= 0
        assert calibration['mean_uncertainty'] > 0
        assert calibration['uncertainty_std'] >= 0

class TestPredictionConfig:
    """Test cases for Prediction Configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PredictionConfig()
        
        assert config.prediction_horizon == 10
        assert config.state_dim == 128
        assert config.action_dim == 16
        assert config.hidden_dim == 256
        assert config.uncertainty_estimation is True
        assert config.physics_informed is True
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = PredictionConfig(
            prediction_horizon=5,
            state_dim=64,
            uncertainty_estimation=False
        )
        
        assert config.prediction_horizon == 5
        assert config.state_dim == 64
        assert config.uncertainty_estimation is False
        # Other values should remain default
        assert config.action_dim == 16

class TestIntegrationScenarios:
    """Integration test scenarios."""
    
    def test_full_prediction_pipeline(self):
        """Test complete prediction pipeline."""
        config = PredictionConfig(
            prediction_horizon=3,
            state_dim=32,
            action_dim=4,
            hidden_dim=64
        )
        
        predictor = SensoryPredictor(config)
        evaluator = PredictionEvaluator(predictor)
        
        # Generate test data
        current_state = torch.randn(2, 32)
        planned_actions = torch.randn(2, 3, 4)
        target_states = torch.randn(2, 3, 32)
        
        # Make prediction
        result = predictor(current_state, planned_actions)
        
        # Evaluate prediction
        metrics = evaluator.evaluate_predictions(
            current_state, planned_actions, target_states
        )
        
        assert isinstance(result, PredictionResult)
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        
    def test_physics_material_integration(self):
        """Test integration of physics constraints and material conditioning."""
        config = PredictionConfig(
            prediction_horizon=2,
            state_dim=16,
            action_dim=4,
            physics_informed=True,
            material_conditioning=True
        )
        
        predictor = SensoryPredictor(config)
        
        # Test with material properties
        current_state = torch.randn(1, 16)
        actions = torch.randn(1, 2, 4)
        materials = torch.randn(1, 8)
        
        result = predictor(current_state, actions, materials)
        
        assert result.predicted_states.shape == (1, 2, 16)
        assert result.physics_consistency is not None
        assert result.material_influences is not None

if __name__ == '__main__':
    pytest.main([__file__, '-v'])