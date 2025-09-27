#!/usr/bin/env python3
"""
MORPHEUS Predictive Module.

This module implements forward models and predictive capabilities for MORPHEUS,
enabling future state prediction and uncertainty quantification.

Components:
- SensoryPredictor: Main forward model for sensory state prediction
- PhysicsInformedLayer: Physics-constrained neural network layers
- MaterialConditioningNetwork: Material property conditioning
- UncertaintyEstimationHead: Uncertainty quantification
- PredictionEvaluator: Model evaluation and metrics

Key Features:
- Multi-step ahead prediction with configurable horizons
- Physics-informed neural networks with conservation laws
- Material property integration for realistic predictions
- Uncertainty quantification (aleatoric and epistemic)
- Temporal attention mechanisms
- Real-time prediction capabilities
"""

from .forward_model import (
    SensoryPredictor,
    PredictionConfig,
    PredictionResult,
    PhysicsInformedLayer,
    MaterialConditioningNetwork,
    UncertaintyEstimationHead,
    PredictionEvaluator
)

__all__ = [
    'SensoryPredictor',
    'PredictionConfig',
    'PredictionResult', 
    'PhysicsInformedLayer',
    'MaterialConditioningNetwork',
    'UncertaintyEstimationHead',
    'PredictionEvaluator'
]

__version__ = '0.1.0'