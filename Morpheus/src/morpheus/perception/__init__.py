#!/usr/bin/env python3
"""
MORPHEUS Perception Module.

This module handles multi-modal sensory processing including tactile,
audio, and visual perception with material property integration.

Components:
- TactileProcessor: Advanced tactile processing with material properties
- AudioProcessor: 3D spatial audio processing with material inference
- SensoryFusionNetwork: Multi-modal neural fusion network
- FusionNetworkManager: Manager for fusion network operations

Key Features:
- Material-aware sensory processing
- 3D spatial audio with HRTF modeling
- Neural attention-based fusion
- Uncertainty quantification
- Real-time processing optimization
- Cross-modal learning and adaptation
"""

from .tactile_processor import TactileProcessor, TactileSignature, ContactPoint
from .audio_spatial import AudioProcessor, AudioSignature, AudioSource
from .sensory_fusion import (
    SensoryFusionNetwork, 
    FusionNetworkManager,
    FusionResult,
    ModalityConfig
)

__all__ = [
    'TactileProcessor',
    'TactileSignature', 
    'ContactPoint',
    'AudioProcessor',
    'AudioSignature',
    'AudioSource',
    'SensoryFusionNetwork',
    'FusionNetworkManager', 
    'FusionResult',
    'ModalityConfig'
]

__version__ = '0.1.0'