#!/usr/bin/env python3
"""
MORPHEUS Integration Module.

This module provides integration bridges for connecting MORPHEUS
with external systems and data sources.

Components:
- MaterialBridge: Bridge to GASM-Robotics material system
- GASMBridge: Complete integration with GASM-Robotics
- SynchronizationMetrics: Performance monitoring for integrations

Key Features:
- Real-time state synchronization with GASM
- Material property database integration
- Physics simulation coordination
- Visual and audio data pipeline
- Coordinate system transformations
- Performance monitoring and metrics
"""

from .material_bridge import MaterialBridge, MaterialProperties
from .gasm_bridge import GASMBridge, GASMState, MorpheusCommand, SynchronizationMetrics

__all__ = [
    'MaterialBridge',
    'MaterialProperties',
    'GASMBridge',
    'GASMState', 
    'MorpheusCommand',
    'SynchronizationMetrics'
]

__version__ = '0.1.0'