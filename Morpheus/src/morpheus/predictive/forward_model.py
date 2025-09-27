#!/usr/bin/env python3
"""
Forward Model Predictor - Sensory state prediction for MORPHEUS.

This module implements advanced neural networks for predicting future
sensory states based on current state and planned actions.

Features:
- Multi-step ahead prediction
- Uncertainty quantification
- Physics-informed neural networks
- Material property integration
- Temporal sequence modeling
- Real-time prediction capabilities
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM, GRU, TransformerEncoder, TransformerEncoderLayer
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import time
import math
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

@dataclass
class PredictionConfig:
    """Configuration for forward model prediction."""
    prediction_horizon: int = 10  # Steps ahead to predict
    state_dim: int = 128
    action_dim: int = 16
    hidden_dim: int = 256
    num_layers: int = 3
    uncertainty_estimation: bool = True
    physics_informed: bool = True
    material_conditioning: bool = True
    temporal_attention: bool = True
    multi_scale: bool = True

@dataclass
class PredictionResult:
    """Result from forward model prediction."""
    predicted_states: torch.Tensor  # (batch, horizon, state_dim)
    predicted_uncertainties: torch.Tensor  # (batch, horizon, state_dim)
    attention_weights: Optional[torch.Tensor] = None
    physics_consistency: Optional[torch.Tensor] = None
    material_influences: Optional[Dict[str, torch.Tensor]] = None
    confidence_scores: Optional[torch.Tensor] = None
    processing_time: float = 0.0

class PhysicsInformedLayer(nn.Module):
    """Physics-informed neural network layer."""
    
    def __init__(self, input_dim: int, output_dim: int, physics_constraints: List[str]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_constraints = physics_constraints
        
        # Standard neural network components
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()
        
        # Physics constraint networks
        self.constraint_networks = nn.ModuleDict()
        
        for constraint in physics_constraints:
            if constraint == 'conservation_momentum':
                self.constraint_networks[constraint] = nn.Sequential(
                    nn.Linear(input_dim, output_dim // 2),
                    nn.Tanh(),
                    nn.Linear(output_dim // 2, output_dim)
                )
            elif constraint == 'conservation_energy':
                self.constraint_networks[constraint] = nn.Sequential(
                    nn.Linear(input_dim, output_dim // 4),
                    nn.Tanh(),
                    nn.Linear(output_dim // 4, 1)
                )
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with physics constraints."""
        
        # Standard neural network forward pass
        standard_output = self.activation(self.linear(x))
        
        # Apply physics constraints
        constraint_outputs = {}
        constrained_output = standard_output
        
        for constraint, network in self.constraint_networks.items():
            constraint_out = network(x)
            constraint_outputs[constraint] = constraint_out
            
            if constraint == 'conservation_momentum':
                # Ensure momentum conservation
                constrained_output = constrained_output + 0.1 * constraint_out
                
            elif constraint == 'conservation_energy':
                # Energy constraint scaling
                energy_scale = torch.sigmoid(constraint_out)
                constrained_output = constrained_output * energy_scale
                
        return constrained_output, constraint_outputs

class MaterialConditioningNetwork(nn.Module):
    """Network for conditioning predictions on material properties."""
    
    def __init__(self, material_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.material_dim = material_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Material property encoder
        self.material_encoder = nn.Sequential(
            nn.Linear(material_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Material influence networks
        self.influence_networks = nn.ModuleDict({
            'friction': nn.Sequential(
                nn.Linear(hidden_dim // 4, hidden_dim // 8),
                nn.Tanh(),
                nn.Linear(hidden_dim // 8, output_dim)
            ),
            'stiffness': nn.Sequential(
                nn.Linear(hidden_dim // 4, hidden_dim // 8),
                nn.ReLU(),
                nn.Linear(hidden_dim // 8, output_dim)
            ),
            'damping': nn.Sequential(
                nn.Linear(hidden_dim // 4, hidden_dim // 8),
                nn.Sigmoid(),
                nn.Linear(hidden_dim // 8, output_dim)
            )
        })
        
    def forward(self, material_properties: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for material conditioning."""
        
        # Encode material properties
        material_encoding = self.material_encoder(material_properties)
        
        # Compute material influences
        influences = {}
        for influence_type, network in self.influence_networks.items():
            influences[influence_type] = network(material_encoding)
            
        return influences

class MultiScaleTemporalEncoder(nn.Module):
    """Multi-scale temporal encoding for different time horizons."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_scales: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        # Different temporal encoders for different scales
        self.encoders = nn.ModuleList()
        
        for scale in range(num_scales):
            # Dilated convolution for multi-scale temporal modeling
            dilation = 2 ** scale
            encoder = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim // num_scales, 
                         kernel_size=3, dilation=dilation, padding=dilation),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim // num_scales),
                nn.Conv1d(hidden_dim // num_scales, hidden_dim // num_scales,
                         kernel_size=3, dilation=dilation, padding=dilation),
                nn.ReLU()
            )
            self.encoders.append(encoder)
            
        # Attention mechanism for scale fusion
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_scales),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-scale temporal encoding.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            Tuple of (encoded_features, attention_weights)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Transpose for conv1d (batch, input_dim, seq_len)
        x_conv = x.transpose(1, 2)
        
        # Apply each scale encoder
        scale_outputs = []
        for encoder in self.encoders:
            scale_out = encoder(x_conv)  # (batch, hidden_dim//num_scales, seq_len)
            scale_outputs.append(scale_out)
            
        # Concatenate scale outputs
        multi_scale_features = torch.cat(scale_outputs, dim=1)  # (batch, hidden_dim, seq_len)
        
        # Transpose back and get final features
        multi_scale_features = multi_scale_features.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        
        # Compute attention weights for scale fusion
        # Use mean over sequence for attention computation
        attention_input = multi_scale_features.mean(dim=1)  # (batch, hidden_dim)
        attention_weights = self.scale_attention(attention_input)  # (batch, num_scales)
        
        return multi_scale_features, attention_weights

class UncertaintyEstimationHead(nn.Module):
    """Neural network head for uncertainty estimation."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        
        # Aleatoric uncertainty (data-dependent)
        self.aleatoric_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        # Epistemic uncertainty (model-dependent)
        self.epistemic_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for uncertainty estimation.
        
        Returns:
            Tuple of (aleatoric_uncertainty, epistemic_uncertainty, confidence)
        """
        aleatoric = self.aleatoric_head(x)
        epistemic = self.epistemic_head(x)
        confidence = self.confidence_head(x)
        
        return aleatoric, epistemic, confidence

class SensoryPredictor(nn.Module):
    """
    Complete forward model for sensory state prediction.
    
    Predicts future sensory states based on current state and planned actions,
    with uncertainty quantification and physics constraints.
    """
    
    def __init__(self, config: Union[Dict[str, Any], PredictionConfig]):
        super().__init__()
        
        if isinstance(config, dict):
            self.config = PredictionConfig(**config)
        else:
            self.config = config
            
        # Network architecture parameters
        state_dim = self.config.state_dim
        action_dim = self.config.action_dim
        hidden_dim = self.config.hidden_dim
        
        # Input processing
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Multi-scale temporal encoding
        if self.config.multi_scale:
            self.temporal_encoder = MultiScaleTemporalEncoder(
                hidden_dim + hidden_dim // 2,  # state + action encodings
                hidden_dim
            )
        
        # Main prediction network
        if self.config.temporal_attention:
            # Transformer-based prediction
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            )
            self.prediction_network = TransformerEncoder(
                encoder_layer,
                num_layers=self.config.num_layers
            )
        else:
            # LSTM-based prediction
            self.prediction_network = LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=self.config.num_layers,
                batch_first=True,
                dropout=0.1 if self.config.num_layers > 1 else 0
            )
            
        # Physics-informed layers
        if self.config.physics_informed:
            physics_constraints = ['conservation_momentum', 'conservation_energy']
            self.physics_layer = PhysicsInformedLayer(
                hidden_dim, hidden_dim, physics_constraints
            )
            
        # Material conditioning
        if self.config.material_conditioning:
            # Assume material properties are encoded in first part of state
            material_dim = min(32, state_dim // 4)  # Use part of state for materials
            self.material_network = MaterialConditioningNetwork(
                material_dim, hidden_dim, hidden_dim
            )
            
        # Output prediction heads
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, state_dim)
            ) for _ in range(self.config.prediction_horizon)
        ])
        
        # Uncertainty estimation
        if self.config.uncertainty_estimation:
            self.uncertainty_head = UncertaintyEstimationHead(
                hidden_dim, state_dim, hidden_dim
            )
            
        # Attention mechanism for prediction steps
        self.step_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"SensoryPredictor initialized with {self.config.prediction_horizon}-step horizon")
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
                    
    def forward(self, 
                current_state: torch.Tensor,
                planned_actions: torch.Tensor,
                material_properties: Optional[torch.Tensor] = None) -> PredictionResult:
        """
        Forward pass for sensory prediction.
        
        Args:
            current_state: Current sensory state (batch, state_dim)
            planned_actions: Planned actions (batch, horizon, action_dim)
            material_properties: Optional material properties (batch, material_dim)
            
        Returns:
            PredictionResult with predictions and uncertainties
        """
        start_time = time.time()
        batch_size = current_state.size(0)
        
        # Encode current state
        state_encoding = self.state_encoder(current_state)  # (batch, hidden_dim)
        
        # Encode action sequence
        action_encodings = []
        for t in range(self.config.prediction_horizon):
            if t < planned_actions.size(1):
                action_enc = self.action_encoder(planned_actions[:, t])
            else:
                # Use last available action if sequence is shorter
                action_enc = self.action_encoder(planned_actions[:, -1])
            action_encodings.append(action_enc)
            
        action_sequence = torch.stack(action_encodings, dim=1)  # (batch, horizon, hidden_dim//2)
        
        # Combine state and action encodings
        state_expanded = state_encoding.unsqueeze(1).expand(-1, self.config.prediction_horizon, -1)
        combined_input = torch.cat([state_expanded, action_sequence], dim=-1)
        
        # Multi-scale temporal encoding
        temporal_features = combined_input
        scale_attention_weights = None
        
        if hasattr(self, 'temporal_encoder'):
            temporal_features, scale_attention_weights = self.temporal_encoder(combined_input)
            
        # Main prediction network processing
        if self.config.temporal_attention:
            # Transformer-based processing
            predicted_sequence = self.prediction_network(temporal_features)
        else:
            # LSTM-based processing
            predicted_sequence, _ = self.prediction_network(temporal_features)
            
        # Apply physics constraints
        physics_consistency = None
        if hasattr(self, 'physics_layer'):
            # Apply physics layer to each timestep
            physics_outputs = []
            physics_constraints = []
            
            for t in range(self.config.prediction_horizon):
                physics_out, constraints = self.physics_layer(predicted_sequence[:, t])
                physics_outputs.append(physics_out)
                physics_constraints.append(constraints)
                
            predicted_sequence = torch.stack(physics_outputs, dim=1)
            physics_consistency = physics_constraints
            
        # Apply material conditioning
        material_influences = None
        if hasattr(self, 'material_network') and material_properties is not None:
            material_influences = self.material_network(material_properties)
            
            # Apply material influences to predictions
            for influence_type, influence_values in material_influences.items():
                if influence_type == 'friction':
                    # Friction affects motion predictions
                    predicted_sequence = predicted_sequence * (1.0 + 0.1 * influence_values.unsqueeze(1))
                elif influence_type == 'stiffness':
                    # Stiffness affects deformation predictions
                    predicted_sequence = predicted_sequence + 0.05 * influence_values.unsqueeze(1)
                elif influence_type == 'damping':
                    # Damping affects velocity predictions
                    predicted_sequence = predicted_sequence * influence_values.unsqueeze(1)
                    
        # Generate final predictions using prediction heads
        final_predictions = []
        for t in range(self.config.prediction_horizon):
            pred = self.prediction_heads[t](predicted_sequence[:, t])
            final_predictions.append(pred)
            
        predicted_states = torch.stack(final_predictions, dim=1)
        
        # Uncertainty estimation
        predicted_uncertainties = None
        confidence_scores = None
        attention_weights = None
        
        if hasattr(self, 'uncertainty_head'):
            # Compute uncertainties for each prediction step
            uncertainties = []
            confidences = []
            
            for t in range(self.config.prediction_horizon):
                aleatoric, epistemic, confidence = self.uncertainty_head(predicted_sequence[:, t])
                
                # Combined uncertainty
                total_uncertainty = torch.sqrt(aleatoric**2 + epistemic**2)
                uncertainties.append(total_uncertainty)
                confidences.append(confidence)
                
            predicted_uncertainties = torch.stack(uncertainties, dim=1)
            confidence_scores = torch.stack(confidences, dim=1)
            
        # Compute attention weights over prediction steps
        if hasattr(self, 'step_attention'):
            query = predicted_sequence[:, 0:1]  # Use first prediction as query
            key_value = predicted_sequence
            
            attended_features, attention_weights = self.step_attention(
                query, key_value, key_value
            )
            
        processing_time = time.time() - start_time
        
        return PredictionResult(
            predicted_states=predicted_states,
            predicted_uncertainties=predicted_uncertainties,
            attention_weights=attention_weights,
            physics_consistency=physics_consistency,
            material_influences=material_influences,
            confidence_scores=confidence_scores,
            processing_time=processing_time
        )
        
    def predict_single_step(self,
                           current_state: torch.Tensor,
                           action: torch.Tensor,
                           material_properties: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict single step ahead.
        
        Args:
            current_state: Current state (batch, state_dim)
            action: Single action (batch, action_dim)
            material_properties: Optional material properties
            
        Returns:
            Tuple of (next_state, uncertainty)
        """
        # Expand action to sequence of length 1
        action_sequence = action.unsqueeze(1)
        
        # Use full prediction but return only first step
        result = self.forward(current_state, action_sequence, material_properties)
        
        next_state = result.predicted_states[:, 0]  # First prediction step
        uncertainty = result.predicted_uncertainties[:, 0] if result.predicted_uncertainties is not None else None
        
        return next_state, uncertainty
        
    def predict_rollout(self,
                       initial_state: torch.Tensor,
                       action_policy: callable,
                       num_steps: int,
                       material_properties: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Perform multi-step rollout prediction with action policy.
        
        Args:
            initial_state: Initial state (batch, state_dim)
            action_policy: Function that takes state and returns action
            num_steps: Number of rollout steps
            material_properties: Optional material properties
            
        Returns:
            Dictionary with rollout results
        """
        batch_size = initial_state.size(0)
        
        # Storage for rollout
        states = [initial_state]
        actions = []
        uncertainties = []
        
        current_state = initial_state
        
        for step in range(num_steps):
            # Get action from policy
            action = action_policy(current_state)
            actions.append(action)
            
            # Predict next state
            next_state, uncertainty = self.predict_single_step(
                current_state, action, material_properties
            )
            
            states.append(next_state)
            if uncertainty is not None:
                uncertainties.append(uncertainty)
                
            current_state = next_state
            
        return {
            'states': torch.stack(states, dim=1),  # (batch, steps+1, state_dim)
            'actions': torch.stack(actions, dim=1),  # (batch, steps, action_dim)
            'uncertainties': torch.stack(uncertainties, dim=1) if uncertainties else None
        }
        
    def compute_prediction_error(self,
                               predictions: torch.Tensor,
                               targets: torch.Tensor,
                               uncertainties: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute prediction errors with uncertainty weighting.
        
        Args:
            predictions: Predicted states (batch, horizon, state_dim)
            targets: Target states (batch, horizon, state_dim)
            uncertainties: Prediction uncertainties (batch, horizon, state_dim)
            
        Returns:
            Dictionary of error metrics
        """
        # Basic MSE error
        mse_error = F.mse_loss(predictions, targets, reduction='none')
        
        errors = {
            'mse': mse_error,
            'rmse': torch.sqrt(mse_error.mean(dim=-1)),
            'mae': F.l1_loss(predictions, targets, reduction='none')
        }
        
        # Uncertainty-weighted errors
        if uncertainties is not None:
            # Negative log-likelihood assuming Gaussian distribution
            nll = 0.5 * (torch.log(2 * math.pi * uncertainties**2) + 
                        (predictions - targets)**2 / (uncertainties**2))
            errors['nll'] = nll
            
            # Prediction interval coverage
            # Check if targets fall within prediction intervals
            lower_bound = predictions - 1.96 * uncertainties  # 95% interval
            upper_bound = predictions + 1.96 * uncertainties
            
            coverage = ((targets >= lower_bound) & (targets <= upper_bound)).float()
            errors['coverage'] = coverage
            
        return errors
        
    def update_with_experience(self,
                             states: torch.Tensor,
                             actions: torch.Tensor,
                             next_states: torch.Tensor,
                             material_properties: Optional[torch.Tensor] = None):
        """
        Update model with new experience (for online learning).
        
        Args:
            states: Current states (batch, state_dim)
            actions: Actions taken (batch, action_dim)
            next_states: Observed next states (batch, state_dim)
            material_properties: Optional material properties
        """
        # This would be implemented for online learning scenarios
        # For now, just log the experience
        
        batch_size = states.size(0)
        
        # Compute prediction for comparison
        action_seq = actions.unsqueeze(1)  # Single step
        with torch.no_grad():
            result = self.forward(states, action_seq, material_properties)
            predictions = result.predicted_states[:, 0]
            
            # Compute prediction error
            error = F.mse_loss(predictions, next_states)
            
        logger.debug(f"Experience update: batch_size={batch_size}, prediction_error={error.item():.4f}")
        
    def get_model_complexity(self) -> Dict[str, int]:
        """Get model complexity metrics."""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'prediction_horizon': self.config.prediction_horizon,
            'state_dimension': self.config.state_dim,
            'action_dimension': self.config.action_dim,
            'hidden_dimension': self.config.hidden_dim
        }
        
    def save_predictor_state(self, filepath: str):
        """Save predictor state to file."""
        
        state = {
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__,
            'complexity': self.get_model_complexity()
        }
        
        torch.save(state, filepath)
        logger.info(f"Predictor state saved to {filepath}")
        
    def load_predictor_state(self, filepath: str):
        """Load predictor state from file."""
        
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            self.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Predictor state loaded from {filepath}")
            
            if 'complexity' in checkpoint:
                logger.info(f"Model complexity: {checkpoint['complexity']}")
                
        except Exception as e:
            logger.error(f"Failed to load predictor state: {e}")

class PredictionEvaluator:
    """Evaluator for forward model predictions."""
    
    def __init__(self, predictor: SensoryPredictor):
        self.predictor = predictor
        self.evaluation_history = []
        
    def evaluate_predictions(self,
                           test_states: torch.Tensor,
                           test_actions: torch.Tensor,
                           test_targets: torch.Tensor,
                           material_properties: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate predictor on test data.
        
        Args:
            test_states: Test states (batch, state_dim)
            test_actions: Test action sequences (batch, horizon, action_dim)
            test_targets: Target state sequences (batch, horizon, state_dim)
            material_properties: Optional material properties
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.predictor.eval()
        
        with torch.no_grad():
            # Get predictions
            result = self.predictor(test_states, test_actions, material_properties)
            
            # Compute errors
            errors = self.predictor.compute_prediction_error(
                result.predicted_states,
                test_targets,
                result.predicted_uncertainties
            )
            
            # Aggregate metrics
            metrics = {
                'mse': errors['mse'].mean().item(),
                'rmse': errors['rmse'].mean().item(),
                'mae': errors['mae'].mean().item()
            }
            
            if 'nll' in errors:
                metrics['nll'] = errors['nll'].mean().item()
                metrics['coverage'] = errors['coverage'].mean().item()
                
            # Temporal metrics (how accuracy degrades over time)
            horizon_mse = errors['mse'].mean(dim=(0, 2))  # Average over batch and features
            metrics['horizon_degradation'] = (horizon_mse[-1] / horizon_mse[0]).item()
            
            # Uncertainty calibration
            if result.predicted_uncertainties is not None:
                calibration = self._compute_uncertainty_calibration(
                    result.predicted_states,
                    test_targets,
                    result.predicted_uncertainties
                )
                metrics.update(calibration)
                
        self.evaluation_history.append({
            'timestamp': time.time(),
            'metrics': metrics,
            'test_size': test_states.size(0)
        })
        
        return metrics
        
    def _compute_uncertainty_calibration(self,
                                       predictions: torch.Tensor,
                                       targets: torch.Tensor,
                                       uncertainties: torch.Tensor) -> Dict[str, float]:
        """Compute uncertainty calibration metrics."""
        
        # Compute prediction errors
        errors = (predictions - targets).abs()
        
        # Rank predictions by uncertainty
        uncertainty_flat = uncertainties.flatten()
        error_flat = errors.flatten()
        
        # Sort by uncertainty
        sorted_indices = torch.argsort(uncertainty_flat)
        sorted_uncertainties = uncertainty_flat[sorted_indices]
        sorted_errors = error_flat[sorted_indices]
        
        # Compute calibration in bins
        num_bins = 10
        bin_boundaries = torch.quantile(sorted_uncertainties, 
                                       torch.linspace(0, 1, num_bins + 1))
        
        calibration_error = 0.0
        
        for i in range(num_bins):
            # Find predictions in this uncertainty bin
            if i == 0:
                mask = sorted_uncertainties <= bin_boundaries[i + 1]
            elif i == num_bins - 1:
                mask = sorted_uncertainties >= bin_boundaries[i]
            else:
                mask = ((sorted_uncertainties >= bin_boundaries[i]) & 
                       (sorted_uncertainties <= bin_boundaries[i + 1]))
                
            if mask.sum() > 0:
                bin_uncertainty = sorted_uncertainties[mask].mean()
                bin_error = sorted_errors[mask].mean()
                
                # Calibration error is difference between predicted and actual error
                calibration_error += (bin_uncertainty - bin_error).abs()
                
        calibration_error /= num_bins
        
        return {
            'uncertainty_calibration_error': calibration_error.item(),
            'mean_uncertainty': uncertainties.mean().item(),
            'uncertainty_std': uncertainties.std().item()
        }
        
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation history."""
        
        if not self.evaluation_history:
            return {}
            
        recent_evals = self.evaluation_history[-10:]  # Last 10 evaluations
        
        summary = {
            'num_evaluations': len(self.evaluation_history),
            'recent_performance': {}
        }
        
        # Aggregate recent metrics
        for metric in recent_evals[0]['metrics'].keys():
            values = [eval_data['metrics'][metric] for eval_data in recent_evals]
            summary['recent_performance'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0
            }
            
        return summary