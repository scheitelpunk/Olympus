"""MORPHEUS: Multi-modal Optimization through Replay, Prediction, and Haptic-Environmental Understanding System.

A sophisticated system that combines multi-modal sensory perception with dream-based
optimization to create learning robots through experience replay and improvement.

Key Components:
- Multi-modal perception (tactile, audio, visual)
- Material-aware processing using GASM-Robotics integration
- Dream-based experience replay and optimization
- PostgreSQL-based persistent storage
- Neural network fusion and prediction

Example:
    >>> from morpheus.core.config import load_config
    >>> from morpheus.integration.material_bridge import MaterialBridge
    >>> 
    >>> config = load_config('config.yaml')
    >>> bridge = MaterialBridge('/path/to/GASM-Robotics')
    >>> 
    >>> # Get material properties
    >>> steel_props = bridge.get_material('steel')
    >>> 
    >>> # Process tactile data with materials
    >>> from morpheus.perception.tactile_processor import TactileProcessor, TactileProcessorConfig
    >>> processor = TactileProcessor(TactileProcessorConfig(), bridge)
"""

__version__ = "0.1.0"
__author__ = "MORPHEUS Development Team"
__email__ = "morpheus@example.com"

# Core types (always available)
from .core.types import (
    SensoryExperience, TactileSignature, AudioSignature, VisualSignature,
    MaterialProperties, ContactPoint, Vector3D, MaterialType, ActionType,
    LearnedStrategy, DreamSessionConfig, SystemMetrics
)

# Optional imports (require dependencies)
try:
    from .core.config import load_config, get_config, ConfigManager
    _HAS_CONFIG = True
except ImportError:
    _HAS_CONFIG = False

try:
    from .integration.material_bridge import MaterialBridge
    _HAS_MATERIAL_BRIDGE = True
except ImportError:
    _HAS_MATERIAL_BRIDGE = False

try:
    from .storage.postgres_storage import PostgreSQLStorage
    _HAS_STORAGE = True
except ImportError:
    _HAS_STORAGE = False

try:
    from .perception.tactile_processor import TactileProcessor, TactileProcessorConfig
    _HAS_TACTILE = True
except ImportError:
    _HAS_TACTILE = False

# Version info
VERSION_INFO = {
    'major': 0,
    'minor': 1,
    'patch': 0,
    'release': 'alpha'
}


def get_version() -> str:
    """Get version string."""
    return __version__


def get_version_info() -> dict:
    """Get detailed version information."""
    return VERSION_INFO.copy()


# Build __all__ list dynamically based on available imports
__all__ = [
    # Version (always available)
    '__version__',
    'get_version',
    'get_version_info',
    
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
]

# Add conditional imports to __all__
if _HAS_CONFIG:
    __all__.extend(['load_config', 'get_config', 'ConfigManager'])

if _HAS_MATERIAL_BRIDGE:
    __all__.append('MaterialBridge')

if _HAS_STORAGE:
    __all__.append('PostgreSQLStorage')

if _HAS_TACTILE:
    __all__.extend(['TactileProcessor', 'TactileProcessorConfig'])

# Package constants
SUPPORTED_PYTHON_VERSIONS = ['3.8', '3.9', '3.10', '3.11']
REQUIRED_GASM_VERSION = '>=2.0.0'
MIN_POSTGRESQL_VERSION = '12.0'

# Logging setup
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())