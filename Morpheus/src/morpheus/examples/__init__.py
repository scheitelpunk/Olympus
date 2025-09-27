"""MORPHEUS example demonstrations and tutorials.

This package contains comprehensive examples showcasing MORPHEUS system capabilities:

- basic_perception: Simple material-based perception demo
- material_exploration: Interactive material learning and prediction
- dream_cycle_demo: Complete dream cycle with strategy optimization
- full_integration: Comprehensive system integration demonstration

All examples are designed to work with the existing MORPHEUS infrastructure
including PostgreSQL storage, GASM-Robotics materials, and neural networks.

Usage:
    python -m morpheus.examples.basic_perception
    python -m morpheus.examples.material_exploration
    python -m morpheus.examples.dream_cycle_demo  
    python -m morpheus.examples.full_integration

Requirements:
    - PostgreSQL database running (see docker-compose.yml)
    - GASM-Robotics directory with materials configuration
    - MORPHEUS configuration file (configs/default_config.yaml)
"""

from pathlib import Path
import sys
import logging

# Configure basic logging for examples
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Utility functions for examples
def find_morpheus_root() -> Path:
    """Find MORPHEUS project root directory."""
    current = Path(__file__).parent
    
    # Look for MORPHEUS root indicators
    for parent in [current] + list(current.parents):
        if (parent / 'src' / 'morpheus').exists() and (parent / 'requirements.txt').exists():
            return parent
    
    # Fallback to current directory
    return current

def ensure_morpheus_path():
    """Ensure MORPHEUS is in Python path for examples."""
    morpheus_root = find_morpheus_root()
    morpheus_src = morpheus_root / 'src'
    
    if str(morpheus_src) not in sys.path:
        sys.path.insert(0, str(morpheus_src))

def get_example_config():
    """Get default configuration for examples."""
    morpheus_root = find_morpheus_root()
    
    return {
        'config_path': str(morpheus_root / 'configs' / 'default_config.yaml'),
        'gasm_path': str(morpheus_root / 'GASM-Robotics'),
        'project_root': str(morpheus_root)
    }

# Automatically ensure path is set when importing examples
ensure_morpheus_path()

__all__ = [
    'find_morpheus_root',
    'ensure_morpheus_path', 
    'get_example_config'
]