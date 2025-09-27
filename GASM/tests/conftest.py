"""
Pytest Configuration and Shared Test Fixtures for GASM-Roboting Project

This module provides comprehensive fixtures and configuration for testing
the spatial agent system, including mock objects, test data generators,
and environment setup for various testing scenarios.

Author: GASM-Roboting Test Suite
Version: 1.0.0
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, MagicMock, patch
import logging
import json
import time

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test constants
TEST_TOLERANCE = 1e-10
INTEGRATION_TOLERANCE = 1e-6
PERFORMANCE_TOLERANCE = 0.1  # 10% tolerance for timing tests

# Test data directories
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEMP_TEST_DIR = Path(tempfile.gettempdir()) / "gasm_tests"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment at start of test session."""
    # Create test directories
    TEST_DATA_DIR.mkdir(exist_ok=True)
    TEMP_TEST_DIR.mkdir(exist_ok=True)
    
    # Set environment variables for testing
    import os
    os.environ["GASM_TEST_MODE"] = "true"
    os.environ["PYBULLET_GUI"] = "false"  # Disable GUI for tests
    
    logger.info("Test environment initialized")
    
    yield
    
    # Cleanup after all tests
    if TEMP_TEST_DIR.exists():
        shutil.rmtree(TEMP_TEST_DIR)
    logger.info("Test environment cleaned up")


@pytest.fixture
def temp_dir():
    """Provide temporary directory for test files."""
    temp_path = TEMP_TEST_DIR / f"test_{int(time.time() * 1000)}"
    temp_path.mkdir(parents=True, exist_ok=True)
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_poses():
    """Provide sample SE(3) poses for testing."""
    poses = {
        'identity': np.eye(4),
        'translation_only': np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ]),
        'rotation_only': np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]),
        'combined': np.array([
            [0.866, -0.5, 0, 1],
            [0.5, 0.866, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ])
    }
    return poses


@pytest.fixture
def sample_quaternions():
    """Provide sample quaternions for testing."""
    return {
        'identity': np.array([0, 0, 0, 1]),
        'x_90': np.array([np.sin(np.pi/4), 0, 0, np.cos(np.pi/4)]),
        'y_90': np.array([0, np.sin(np.pi/4), 0, np.cos(np.pi/4)]),
        'z_90': np.array([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]),
        'arbitrary': np.array([0.1, 0.2, 0.3, 0.926])  # Normalized
    }


@pytest.fixture
def sample_spatial_constraints():
    """Provide sample spatial constraints for testing."""
    from src.spatial_agent.gasm_bridge import SpatialConstraint, ConstraintType
    
    constraints = [
        SpatialConstraint(
            type=ConstraintType.ABOVE,
            subject="cube_red",
            target="cube_blue",
            parameters={"vertical_offset": 0.05},
            priority=0.8
        ),
        SpatialConstraint(
            type=ConstraintType.DISTANCE,
            subject="robot_arm",
            target="target_object",
            parameters={"distance": 0.15, "axis": "euclidean"},
            priority=0.9
        ),
        SpatialConstraint(
            type=ConstraintType.ANGLE,
            subject="gripper",
            target="workpiece",
            parameters={"angle": 45.0, "axis": "z", "units": "degrees"},
            priority=0.7
        )
    ]
    return constraints


@pytest.fixture
def mock_vision_system():
    """Mock vision system for testing without dependencies."""
    mock_vision = Mock()
    
    # Mock detection results
    from src.spatial_agent.vision import Detection
    sample_detections = [
        Detection(
            label="cube",
            confidence=0.85,
            bbox=(50, 50, 150, 150),
            position_3d=(0.1, 0.2, 0.05),
            center_2d=(100, 100)
        ),
        Detection(
            label="sphere",
            confidence=0.92,
            bbox=(200, 100, 280, 180),
            position_3d=(-0.1, 0.1, 0.04),
            center_2d=(240, 140)
        )
    ]
    
    mock_vision.detect.return_value = sample_detections
    mock_vision.detect_in_pybullet_scene.return_value = sample_detections
    mock_vision.get_detection_summary.return_value = {
        "count": len(sample_detections),
        "avg_confidence": 0.885,
        "objects": {"cube": {"count": 1}, "sphere": {"count": 1}},
        "has_3d_positions": True
    }
    
    return mock_vision


@pytest.fixture
def mock_pybullet():
    """Mock PyBullet physics simulation."""
    with patch('pybullet.connect') as mock_connect:
        mock_connect.return_value = 0  # Connection ID
        
        with patch('pybullet.loadURDF') as mock_load:
            mock_load.return_value = 1  # Object ID
            
            with patch('pybullet.getBasePositionAndOrientation') as mock_get_pose:
                mock_get_pose.return_value = ([0, 0, 0], [0, 0, 0, 1])
                
                with patch('pybullet.resetBasePositionAndOrientation') as mock_set_pose:
                    mock_set_pose.return_value = None
                    
                    yield {
                        'connect': mock_connect,
                        'loadURDF': mock_load,
                        'getBasePositionAndOrientation': mock_get_pose,
                        'resetBasePositionAndOrientation': mock_set_pose
                    }


@pytest.fixture
def sample_gasm_responses():
    """Provide sample GASM responses for testing."""
    from src.spatial_agent.gasm_bridge import GASMResponse, SpatialConstraint, ConstraintType, SE3Pose
    
    responses = {
        'success_simple': GASMResponse(
            success=True,
            constraints=[
                SpatialConstraint(
                    type=ConstraintType.ABOVE,
                    subject="object_a",
                    target="object_b",
                    parameters={"vertical_offset": 0.05},
                    priority=0.8
                )
            ],
            target_poses={
                "object_a": SE3Pose(
                    position=[0.0, 0.0, 0.15],
                    orientation=[1.0, 0.0, 0.0, 0.0],
                    confidence=0.9
                )
            },
            confidence=0.85,
            execution_time=0.025
        ),
        'error_response': GASMResponse(
            success=False,
            constraints=[],
            target_poses={},
            confidence=0.0,
            execution_time=0.001,
            error_message="Failed to parse instruction"
        )
    }
    return responses


@pytest.fixture
def performance_monitor():
    """Monitor test performance for timing-sensitive tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.measurements = {}
        
        def start(self, label="default"):
            self.start_time = time.perf_counter()
            return self
        
        def stop(self, label="default", expected_max=None):
            if self.start_time is None:
                raise ValueError("Must call start() first")
            
            elapsed = time.perf_counter() - self.start_time
            self.measurements[label] = elapsed
            
            if expected_max and elapsed > expected_max:
                pytest.fail(f"Performance test failed: {label} took {elapsed:.3f}s, "
                          f"expected < {expected_max:.3f}s")
            
            return elapsed
        
        def get_measurement(self, label="default"):
            return self.measurements.get(label)
    
    return PerformanceMonitor()


@pytest.fixture
def test_config():
    """Provide test configuration parameters."""
    return {
        'tolerance': {
            'position': 1e-6,
            'orientation': 1e-6,
            'se3': 1e-10,
            'integration': 1e-6
        },
        'performance': {
            'max_detection_time': 2.0,
            'max_planning_time': 5.0,
            'max_constraint_time': 0.1
        },
        'simulation': {
            'physics_timestep': 1/240,
            'gui_enabled': False,
            'real_time': False
        },
        'vision': {
            'image_width': 640,
            'image_height': 480,
            'fov': 60.0,
            'confidence_threshold': 0.3
        }
    }


@pytest.fixture
def sample_images():
    """Generate sample images for vision testing."""
    # Create synthetic test images
    images = {}
    
    # RGB image with basic shapes
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add red rectangle (cube)
    rgb_image[100:200, 50:150, 0] = 255  # Red channel
    # Add blue circle (sphere)
    y, x = np.ogrid[:480, :640]
    mask = (x - 240)**2 + (y - 140)**2 <= 40**2
    rgb_image[mask, 2] = 255  # Blue channel
    
    images['rgb_shapes'] = rgb_image
    
    # Grayscale image
    gray_image = np.mean(rgb_image, axis=2).astype(np.uint8)
    images['grayscale'] = gray_image
    
    # Empty image
    images['empty'] = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Noisy image
    noisy_image = rgb_image.copy()
    noise = np.random.normal(0, 10, noisy_image.shape).astype(np.int16)
    noisy_image = np.clip(noisy_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    images['noisy'] = noisy_image
    
    return images


@pytest.fixture
def metric_validator():
    """Utility for validating metric calculations."""
    class MetricValidator:
        @staticmethod
        def validate_pose_error(error_dict: Dict[str, Any], tolerance: float = 1e-6):
            """Validate pose error metrics have expected structure and values."""
            required_keys = [
                'translation_error', 'rotation_error', 'geodesic_distance',
                'frobenius_norm', 'se3_norm'
            ]
            
            for key in required_keys:
                assert key in error_dict, f"Missing error metric: {key}"
                assert isinstance(error_dict[key], (int, float)), f"Invalid type for {key}"
                assert error_dict[key] >= 0, f"Negative error for {key}: {error_dict[key]}"
        
        @staticmethod
        def validate_constraint_satisfaction(metrics: Dict[str, Any]):
            """Validate constraint satisfaction metrics."""
            assert 'satisfied' in metrics
            assert 'violation_magnitude' in metrics
            assert isinstance(metrics['satisfied'], bool)
            assert metrics['violation_magnitude'] >= 0
        
        @staticmethod
        def validate_detection_results(detections: List, min_confidence: float = 0.0):
            """Validate detection results structure and values."""
            for det in detections:
                assert hasattr(det, 'label')
                assert hasattr(det, 'confidence')
                assert hasattr(det, 'bbox')
                assert det.confidence >= min_confidence
                assert len(det.bbox) == 4  # x1, y1, x2, y2
    
    return MetricValidator()


@pytest.fixture
def trajectory_generator():
    """Generate test trajectories for motion planning tests."""
    def generate_linear_trajectory(start_pose, end_pose, num_points=10):
        """Generate linear interpolation trajectory."""
        from src.spatial_agent.utils_se3 import SE3Utils
        
        trajectory = []
        for i in range(num_points):
            t = i / (num_points - 1)
            interpolated = SE3Utils.interpolate_poses(start_pose, end_pose, t)
            trajectory.append(interpolated)
        
        return np.array(trajectory)
    
    def generate_circular_trajectory(center, radius, num_points=20):
        """Generate circular trajectory around center point."""
        from src.spatial_agent.utils_se3 import create_pose
        
        trajectory = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            
            pose = create_pose([x, y, z])
            trajectory.append(pose)
        
        return np.array(trajectory)
    
    return {
        'linear': generate_linear_trajectory,
        'circular': generate_circular_trajectory
    }


@pytest.fixture
def constraint_generator():
    """Generate various spatial constraints for testing."""
    from src.spatial_agent.gasm_bridge import SpatialConstraint, ConstraintType
    
    def create_above_constraint(subject, target, offset=0.05):
        return SpatialConstraint(
            type=ConstraintType.ABOVE,
            subject=subject,
            target=target,
            parameters={"vertical_offset": offset},
            priority=0.8
        )
    
    def create_distance_constraint(subject, target, distance=0.1):
        return SpatialConstraint(
            type=ConstraintType.DISTANCE,
            subject=subject,
            target=target,
            parameters={"distance": distance, "axis": "euclidean"},
            priority=0.7
        )
    
    def create_angle_constraint(subject, target, angle=90.0):
        return SpatialConstraint(
            type=ConstraintType.ANGLE,
            subject=subject,
            target=target,
            parameters={"angle": angle, "axis": "z", "units": "degrees"},
            priority=0.6
        )
    
    return {
        'above': create_above_constraint,
        'distance': create_distance_constraint,
        'angle': create_angle_constraint
    }


@pytest.fixture
def integration_environment():
    """Set up complete integration test environment."""
    class IntegrationEnvironment:
        def __init__(self):
            self.temp_dir = TEMP_TEST_DIR / f"integration_{int(time.time())}"
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.components = {}
            self.cleanup_functions = []
        
        def setup_gasm_bridge(self, config=None):
            """Set up GASM bridge component."""
            from src.spatial_agent.gasm_bridge import GASMBridge
            self.components['gasm'] = GASMBridge(config)
            return self.components['gasm']
        
        def setup_vision_system(self, config=None):
            """Set up vision system component."""
            from src.spatial_agent.vision import VisionSystem
            self.components['vision'] = VisionSystem(config)
            return self.components['vision']
        
        def setup_metrics_system(self):
            """Set up metrics evaluation system."""
            # This would set up the actual metrics system
            # For now, return a mock
            self.components['metrics'] = Mock()
            return self.components['metrics']
        
        def cleanup(self):
            """Clean up integration environment."""
            for cleanup_fn in self.cleanup_functions:
                try:
                    cleanup_fn()
                except Exception as e:
                    logger.warning(f"Cleanup error: {e}")
            
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
    
    env = IntegrationEnvironment()
    yield env
    env.cleanup()


@pytest.fixture
def benchmark_data():
    """Provide benchmark datasets for performance testing."""
    return {
        'small_scene': {
            'objects': 5,
            'constraints': 3,
            'expected_detection_time': 0.5,
            'expected_planning_time': 1.0
        },
        'medium_scene': {
            'objects': 15,
            'constraints': 8,
            'expected_detection_time': 1.0,
            'expected_planning_time': 3.0
        },
        'large_scene': {
            'objects': 50,
            'constraints': 20,
            'expected_detection_time': 3.0,
            'expected_planning_time': 10.0
        }
    }


@pytest.fixture(autouse=True)
def reset_numpy_random():
    """Reset numpy random state for reproducible tests."""
    np.random.seed(42)


@pytest.fixture(autouse=True)
def capture_test_logs(caplog):
    """Ensure test logs are captured at debug level."""
    caplog.set_level(logging.DEBUG)


@pytest.fixture
def error_injection():
    """Utility for injecting errors during testing."""
    class ErrorInjector:
        def __init__(self):
            self.patches = []
        
        def inject_import_error(self, module_name):
            """Inject ImportError for specified module."""
            def mock_import(name, *args, **kwargs):
                if name == module_name:
                    raise ImportError(f"No module named '{module_name}'")
                return __import__(name, *args, **kwargs)
            
            patcher = patch('builtins.__import__', side_effect=mock_import)
            self.patches.append(patcher)
            return patcher.start()
        
        def inject_runtime_error(self, target, error_type=RuntimeError, message="Injected error"):
            """Inject runtime error for target function."""
            def error_side_effect(*args, **kwargs):
                raise error_type(message)
            
            patcher = patch(target, side_effect=error_side_effect)
            self.patches.append(patcher)
            return patcher.start()
        
        def cleanup(self):
            """Clean up all patches."""
            for patcher in self.patches:
                patcher.stop()
            self.patches.clear()
    
    injector = ErrorInjector()
    yield injector
    injector.cleanup()


# Test markers for categorizing tests
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "simulation: mark test as requiring simulation environment"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add unit marker to test files starting with test_
        if "test_" in item.nodeid and "integration" not in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to performance tests
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker to tests that might take longer
        if any(keyword in item.name for keyword in ["simulation", "training", "large_scale"]):
            item.add_marker(pytest.mark.slow)


# Utility functions for test data generation
def generate_random_poses(n: int, translation_range: float = 1.0, 
                         rotation_range: float = np.pi) -> np.ndarray:
    """Generate random SE(3) poses for testing."""
    from spatial_agent.utils_se3 import SE3Utils
    poses = []
    for _ in range(n):
        pose = SE3Utils.random_se3_pose(translation_range, rotation_range)
        poses.append(pose)
    return np.array(poses)


def create_test_scene_config(num_objects: int = 5) -> Dict[str, Any]:
    """Create test scene configuration."""
    return {
        'objects': [f'test_object_{i}' for i in range(num_objects)],
        'workspace_bounds': {
            'x': [-1.0, 1.0],
            'y': [-1.0, 1.0], 
            'z': [0.0, 2.0]
        },
        'constraints': [],
        'physics_enabled': False
    }


# Export commonly used test utilities
__all__ = [
    'TEST_TOLERANCE',
    'INTEGRATION_TOLERANCE', 
    'PERFORMANCE_TOLERANCE',
    'generate_random_poses',
    'create_test_scene_config'
]