#!/usr/bin/env python3
"""
MORPHEUS Dream Simulation Module.

This module implements the dream-based optimization engine for MORPHEUS,
allowing for experience replay, strategy discovery, and neural optimization.

Components:
- DreamOrchestrator: Main dream session coordination
- NeuralStrategyOptimizer: Neural network for strategy optimization
- ExperienceReplay: Individual experience processing
- DreamSession: Complete session results and metrics

Key Features:
- Parallel dream processing with multi-threading
- Neural network guided strategy discovery
- Experience variation generation
- Strategy consolidation and ranking
- Performance metrics and monitoring
"""

from .dream_orchestrator import (
    DreamOrchestrator,
    DreamConfig,
    DreamSession,
    ExperienceReplay,
    NeuralStrategyOptimizer
)

__all__ = [
    'DreamOrchestrator',
    'DreamConfig', 
    'DreamSession',
    'ExperienceReplay',
    'NeuralStrategyOptimizer'
]

__version__ = '0.1.0'