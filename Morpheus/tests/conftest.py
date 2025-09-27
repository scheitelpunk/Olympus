"""Pytest configuration and fixtures for MORPHEUS tests."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import yaml

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_gasm_config(temp_dir):
    """Create mock GASM configuration."""
    config_dir = temp_dir / "assets" / "configs"
    config_dir.mkdir(parents=True)
    
    config_data = {
        'materials': {
            'steel': {
                'color': [0.7, 0.7, 0.7, 1.0],
                'friction': 0.8,
                'restitution': 0.3,
                'density': 7850,
                'young_modulus': 200e9,
                'poisson_ratio': 0.3
            },
            'rubber': {
                'color': [0.2, 0.2, 0.2, 1.0],
                'friction': 1.2,
                'restitution': 0.8,
                'density': 1200,
                'young_modulus': 1e6,
                'poisson_ratio': 0.48
            }
        }
    }
    
    config_path = config_dir / "simulation_params.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    return temp_dir

@pytest.fixture
def mock_pybullet():
    """Mock PyBullet for testing."""
    mock_pb = MagicMock()
    mock_pb.getContactPoints.return_value = [
        (0, 1, 2, 0, 0, [0, 0, 0.5], [0, 0, 0.5], [0, 0, 1], 5.0, 0, 0.1)
    ]
    mock_pb.getSimulationTime.return_value = 1.0
    return mock_pb

@pytest.fixture
def mock_db_config():
    """Mock database configuration."""
    return {
        'host': 'localhost',
        'port': 5432,
        'database': 'test_morpheus',
        'user': 'test_user',
        'password': 'test_pass'
    }

@pytest.fixture
def sample_tactile_data():
    """Generate sample tactile data."""
    return {
        'timestamp': 1.0,
        'material': 'steel',
        'contact_force': 5.0,
        'contact_area': 0.001,
        'pressure': 5000.0,
        'hardness': 0.8,
        'texture': 'smooth',
        'temperature': 22.0,
        'vibration': np.random.randn(32),
        'grip_quality': 0.7
    }

@pytest.fixture
def sample_experience():
    """Generate sample experience data."""
    return {
        'session_id': '123e4567-e89b-12d3-a456-426614174000',
        'timestamp': 1.0,
        'tactile_embedding': np.random.randn(64).tolist(),
        'audio_embedding': np.random.randn(32).tolist(),
        'visual_embedding': np.random.randn(128).tolist(),
        'fused_embedding': np.random.randn(128).tolist(),
        'primary_material': 'steel',
        'action_type': 'grasp',
        'success': True,
        'reward': 1.0
    }