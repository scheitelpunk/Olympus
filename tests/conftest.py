"""
Global pytest configuration and fixtures for OLYMPUS test suite.

This file provides shared test fixtures, configuration, and utilities
for all OLYMPUS tests, ensuring consistent test setup and teardown.
"""

import asyncio
import logging
import os
import pytest
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Generator, Optional
from unittest.mock import AsyncMock, MagicMock, Mock

# Add src to Python path for importing OLYMPUS modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import OLYMPUS modules
from olympus.ethical_core.asimov_kernel import AsimovKernel, ActionContext, ActionType
from olympus.safety_layer.action_filter import ActionFilter, PhysicsLimits, SpatialLimits
from olympus.core.olympus_orchestrator import OlympusOrchestrator, ActionRequest, Priority
from olympus.modules.prometheus.self_repair import SelfRepairSystem


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def asimov_kernel():
    """Create AsimovKernel instance for testing."""
    return AsimovKernel(log_level=logging.ERROR)  # Reduce log noise in tests


@pytest.fixture
def action_filter():
    """Create ActionFilter instance for testing."""
    physics_limits = PhysicsLimits(
        max_force=20.0,
        max_speed=1.0,
        max_acceleration=2.0,
        max_jerk=10.0,
        max_torque=5.0
    )
    
    spatial_limits = SpatialLimits(
        workspace_bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)),
        min_obstacle_distance=0.1,
        max_reach_distance=1.5
    )
    
    return ActionFilter(
        physics_limits=physics_limits,
        spatial_limits=spatial_limits,
        strict_mode=True
    )


@pytest.fixture
async def orchestrator(temp_dir):
    """Create OlympusOrchestrator instance for testing."""
    config_path = temp_dir / "test_config.yaml"
    config_path.write_text("""
logging:
  level: ERROR
system:
  name: "test_olympus"
  max_workers: 2
""")
    
    orchestrator = OlympusOrchestrator(str(config_path))
    await orchestrator.initialize_system()
    yield orchestrator
    await orchestrator.shutdown(graceful=True)


@pytest.fixture
def self_repair_system():
    """Create SelfRepairSystem instance for testing."""
    audit_system = AsyncMock()
    component_manager = AsyncMock()
    repair_constraints = Mock()
    
    return SelfRepairSystem(
        audit_system=audit_system,
        component_manager=component_manager,
        repair_constraints=repair_constraints
    )


@pytest.fixture
def safe_action_context():
    """Create a safe ActionContext for testing."""
    return ActionContext(
        action_type=ActionType.INFORMATION,
        description="Safe information retrieval",
        target="test_system",
        risk_level="low",
        human_present=True,
        emergency_context=False
    )


@pytest.fixture
def unsafe_action_context():
    """Create an unsafe ActionContext for testing."""
    return ActionContext(
        action_type=ActionType.PHYSICAL,
        description="High force operation with potential harm",
        target="human_workspace",
        risk_level="critical",
        human_present=False,
        emergency_context=False
    )


@pytest.fixture
def emergency_action_context():
    """Create an emergency ActionContext for testing."""
    return ActionContext(
        action_type=ActionType.EMERGENCY_STOP,
        description="Emergency stop activation",
        target="all_systems",
        risk_level="critical",
        human_present=True,
        emergency_context=True
    )


@pytest.fixture
def valid_action_dict():
    """Create a valid action dictionary for filtering tests."""
    return {
        'action_type': 'move',
        'target_position': [0.5, 0.3, 0.8],
        'current_position': [0.0, 0.0, 0.5],
        'velocity': [0.1, 0.1, 0.1],
        'force': [5.0, 0.0, 0.0],
        'acceleration': [0.5, 0.0, 0.0],
        'tool': 'gripper',
        'environment': {
            'lighting': 80,
            'temperature': 22,
            'vibration_level': 1,
            'hazardous_materials': False
        },
        'system_status': {
            'battery_level': 85,
            'error_count': 0,
            'sensors_operational': True
        },
        'humans_detected': []
    }


@pytest.fixture
def dangerous_action_dict():
    """Create a dangerous action dictionary for filtering tests."""
    return {
        'action_type': 'cut',
        'target_position': [2.0, 2.0, 2.0],  # Outside workspace
        'current_position': [0.0, 0.0, 0.5],
        'velocity': [2.0, 0.0, 0.0],  # Exceeds max speed
        'force': [50.0, 0.0, 0.0],  # Exceeds max force
        'acceleration': [5.0, 0.0, 0.0],  # Exceeds max acceleration
        'tool': 'plasma',  # Dangerous tool
        'environment': {
            'lighting': 10,  # Too dark
            'temperature': 60,  # Too hot
            'vibration_level': 8,  # High vibration
            'hazardous_materials': True
        },
        'system_status': {
            'battery_level': 15,  # Low battery
            'error_count': 10,  # Multiple errors
            'sensors_operational': False  # Critical sensor failure
        },
        'humans_detected': [
            {
                'distance': 0.05,  # Too close
                'min_safe_distance': 1.0
            }
        ]
    }


@pytest.fixture
def mock_fault():
    """Create a mock fault for repair testing."""
    fault = Mock()
    fault.fault_id = "test_fault_001"
    fault.component = "test_component"
    fault.symptoms = ["high_cpu_usage", "slow_response_time"]
    fault.severity = "high"
    fault.timestamp = time.time()
    return fault


@pytest.fixture
def performance_metrics():
    """Create performance test metrics collector."""
    class PerformanceMetrics:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.memory_start = None
            self.memory_end = None
            self.metrics = {}
        
        def start_measurement(self):
            self.start_time = time.time()
            # In a real implementation, we'd measure actual memory usage
            self.memory_start = 0
        
        def end_measurement(self):
            self.end_time = time.time()
            self.memory_end = 0
        
        def get_duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0
        
        def get_memory_usage(self):
            if self.memory_start is not None and self.memory_end is not None:
                return self.memory_end - self.memory_start
            return 0
        
        def record_metric(self, name: str, value: Any):
            self.metrics[name] = value
    
    return PerformanceMetrics()


@pytest.fixture
def coverage_tracker():
    """Create coverage tracking utilities."""
    class CoverageTracker:
        def __init__(self):
            self.covered_functions = set()
            self.total_functions = set()
        
        def mark_function_covered(self, function_name: str):
            self.covered_functions.add(function_name)
            self.total_functions.add(function_name)
        
        def mark_function_exists(self, function_name: str):
            self.total_functions.add(function_name)
        
        def get_coverage_percentage(self) -> float:
            if not self.total_functions:
                return 0.0
            return (len(self.covered_functions) / len(self.total_functions)) * 100
        
        def get_uncovered_functions(self) -> set:
            return self.total_functions - self.covered_functions
    
    return CoverageTracker()


# Test configuration constants
TEST_CONFIG = {
    'ASIMOV_KERNEL': {
        'INTEGRITY_CHECK_INTERVAL': 0.01,  # Faster for testing
        'MAX_EVALUATION_TIME': 5.0  # seconds
    },
    'ACTION_FILTER': {
        'STRICT_MODE': True,
        'MAX_FILTER_TIME': 1.0  # seconds
    },
    'ORCHESTRATOR': {
        'MAX_STARTUP_TIME': 10.0,  # seconds
        'MAX_SHUTDOWN_TIME': 5.0   # seconds
    },
    'SELF_REPAIR': {
        'MAX_REPAIR_TIME': 30.0,  # seconds
        'AUTO_REPAIR_ENABLED': False  # Disabled for testing
    }
}


@pytest.fixture
def test_config():
    """Provide test configuration constants."""
    return TEST_CONFIG


# Test data generators
class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def generate_action_contexts(count: int = 10) -> list:
        """Generate multiple action contexts for testing."""
        contexts = []
        for i in range(count):
            contexts.append(ActionContext(
                action_type=ActionType.INFORMATION,
                description=f"Test action {i}",
                risk_level="low",
                human_present=i % 2 == 0
            ))
        return contexts
    
    @staticmethod
    def generate_action_requests(count: int = 10) -> list:
        """Generate multiple action requests for testing."""
        requests = []
        for i in range(count):
            requests.append(ActionRequest(
                id=f"test_request_{i}",
                module="test_module",
                action=f"test_action_{i}",
                parameters={"param": i},
                priority=Priority.NORMAL,
                requester=f"test_user_{i}"
            ))
        return requests


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()


# Pytest hooks for test collection and reporting
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "safety: mark test as safety-critical test"
    )
    config.addinivalue_line(
        "markers", "ethical: mark test as ethical compliance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Add safety marker for safety-critical tests
        if "asimov" in str(item.fspath) or "safety" in str(item.fspath):
            item.add_marker(pytest.mark.safety)
            item.add_marker(pytest.mark.ethical)


# Async test helpers
async def async_wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1) -> bool:
    """Wait for a condition to become true with timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        await asyncio.sleep(interval)
    return False


def create_mock_component(name: str, methods: list = None) -> Mock:
    """Create a mock component with standard methods."""
    if methods is None:
        methods = ["initialize", "shutdown", "get_status", "execute_action"]
    
    mock = Mock()
    mock.name = name
    for method in methods:
        setattr(mock, method, AsyncMock())
    
    return mock