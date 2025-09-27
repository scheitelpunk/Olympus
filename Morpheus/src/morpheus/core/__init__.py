"""Core MORPHEUS components.

This module contains the fundamental building blocks of the MORPHEUS system,
including configuration management, type definitions, and the main orchestrator.
"""

# Core types (always available)
from .types import (
    SensoryExperience, TactileSignature, AudioSignature, VisualSignature,
    MaterialProperties, ContactPoint, Vector3D, MaterialType, ActionType,
    LearnedStrategy, DreamSessionConfig, SystemMetrics, MaterialInteraction
)

# Optional config imports (require pydantic)
try:
    from .config import (
        load_config, get_config, ConfigManager, MorpheusConfig,
        DatabaseConfig, TactileConfig, AudioConfig, VisualConfig
    )
    _HAS_CONFIG = True
except ImportError:
    _HAS_CONFIG = False

# Build __all__ dynamically
__all__ = [
    # Core types (always available)
    'SensoryExperience',
    'TactileSignature',
    'AudioSignature', 
    'VisualSignature',
    'MaterialProperties',
    'ContactPoint',
    'Vector3D',
    'MaterialType',
    'ActionType',
    'LearnedStrategy',
    'DreamSessionConfig',
    'SystemMetrics',
    'MaterialInteraction'
]

# Add config exports if available
if _HAS_CONFIG:
    __all__.extend([
        'load_config',
        'get_config', 
        'ConfigManager',
        'MorpheusConfig',
        'DatabaseConfig',
        'TactileConfig',
        'AudioConfig', 
        'VisualConfig'
    ])