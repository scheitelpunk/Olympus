"""
Pytest Configuration and Fixtures for GASM-Roboting API Tests
"""

import pytest
import asyncio
from typing import AsyncGenerator, Dict, Any
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import test app and configuration
from ..main import app
from ..middleware.auth import auth_config, token_manager, user_store
from ..middleware.rate_limit import RateLimitConfig
from . import TEST_CONFIG, TEST_USERS


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_app():
    """Create a test instance of the FastAPI application."""
    # Override settings for testing
    app.dependency_overrides = {}
    
    # Use test configuration
    auth_config.jwt_secret = TEST_CONFIG["test_jwt_secret"]
    
    return app


@pytest.fixture
def client(test_app):
    """Create a test client for synchronous tests."""
    with TestClient(test_app) as test_client:
        yield test_client


@pytest.fixture
async def async_client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for asynchronous tests."""
    async with AsyncClient(app=test_app, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture
def auth_headers() -> Dict[str, Dict[str, str]]:
    """Generate authentication headers for different user types."""
    headers = {}
    
    for user_type, user_data in TEST_USERS.items():
        # Create token for test user
        token = token_manager.create_access_token(
            user_id=user_data["id"],
            scopes=[scope for scope in user_data["scopes"]],
            session_id=f"test_session_{user_type}"
        )
        
        headers[user_type] = {
            "Authorization": f"Bearer {token.access_token}",
            "Content-Type": "application/json"
        }
    
    # API key header
    headers["api_key"] = {
        "X-API-Key": TEST_CONFIG["test_api_keys"]["valid_key"],
        "Content-Type": "application/json"
    }
    
    return headers


@pytest.fixture
def test_users():
    """Provide test user data."""
    # Add test users to the user store
    from ..models.auth import User, UserRole, PermissionScope
    
    for user_type, user_data in TEST_USERS.items():
        user = User(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            role=UserRole(user_data["role"]),
            scopes=[PermissionScope(scope) for scope in user_data["scopes"]],
            is_active=True,
            is_verified=True,
            created_at="2024-01-01T00:00:00Z",
            login_count=0
        )
        user_store.users[user.id] = user
    
    yield TEST_USERS
    
    # Cleanup
    for user_data in TEST_USERS.values():
        user_store.users.pop(user_data["id"], None)


@pytest.fixture
def test_spatial_data():
    """Provide test spatial data (poses, constraints, etc.)."""
    from . import TEST_POSES, TEST_CONSTRAINTS, TEST_SPATIAL_INSTRUCTIONS
    
    return {
        "poses": TEST_POSES,
        "constraints": TEST_CONSTRAINTS,
        "instructions": TEST_SPATIAL_INSTRUCTIONS
    }


@pytest.fixture
def mock_gasm_bridge():
    """Mock GASM bridge for testing."""
    class MockGASMBridge:
        def __init__(self):
            self.is_initialized = True
            self.fallback_mode = False
            
        def process(self, text: str) -> Dict[str, Any]:
            """Mock GASM processing."""
            return {
                "success": True,
                "constraints": [
                    {
                        "type": "above",
                        "subject": "object_a",
                        "target": "object_b",
                        "parameters": {"vertical_offset": 0.05},
                        "priority": 0.8
                    }
                ],
                "target_poses": {
                    "object_a": {
                        "position": [0.0, 0.0, 0.15],
                        "orientation": [1.0, 0.0, 0.0, 0.0],
                        "confidence": 0.9
                    }
                },
                "confidence": 0.85,
                "execution_time": 0.1,
                "debug_info": {"mode": "mock"}
            }
            
        def get_supported_constraints(self):
            return ["above", "below", "distance", "angle", "aligned"]
    
    return MockGASMBridge()


@pytest.fixture
def mock_motion_planner():
    """Mock motion planner for testing."""
    class MockPlanningResult:
        def __init__(self, success=True):
            self.success = success
            self.next_pose = type('Pose', (), {
                'x': 0.1, 'y': 0.1, 'z': 0.1,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            })()
            self.step_size = 0.05
            self.constraints_violated = []
            self.obstacles_detected = []
            self.reasoning = "Mock planning successful"
            self.debug_info = {"planner": "mock"}
    
    class MockMotionPlanner:
        def __init__(self):
            self.obstacles = []
            self.constraints = []
            self.strategy = "constrained"
            
        def set_strategy(self, strategy):
            self.strategy = strategy
            
        def add_obstacle(self, obstacle):
            self.obstacles.append(obstacle)
            
        def add_constraint(self, constraint):
            self.constraints.append(constraint)
            
        def clear_obstacles(self):
            self.obstacles.clear()
            
        def clear_constraints(self):
            self.constraints.clear()
            
        def plan_step(self, current_pose, target_pose, constraints=None):
            return MockPlanningResult()
    
    return MockMotionPlanner()


@pytest.fixture
def mock_metrics_calculator():
    """Mock metrics calculator for testing."""
    class MockPoseError:
        def __init__(self):
            self.position_error = 0.01
            self.rotation_error = 0.05
            self.total_error = 0.06
            self.is_converged = True
            self.position_vector = [0.01, 0.0, 0.0]
            self.rotation_axis = [0.0, 0.0, 1.0]
    
    class MockConstraintScore:
        def __init__(self):
            from ..models.spatial import ConstraintType
            self.constraint_type = ConstraintType.DISTANCE
            self.score = 0.95
            self.violation_magnitude = 0.005
            self.is_satisfied = True
            self.tolerance_used = 0.01
            self.additional_info = {}
    
    class MockMetricsCalculator:
        def __init__(self):
            pass
            
        def pose_error(self, current_pose, target_pose):
            return MockPoseError()
            
        def constraint_score(self, state, constraints):
            return {
                "test_constraint": MockConstraintScore()
            }
            
        def is_done(self, pose_errors, constraint_scores, custom_thresholds=None):
            return True, {
                "pose_converged": True,
                "constraints_converged": True,
                "stable": True,
                "overall_converged": True
            }
            
        def accumulate_statistics(self, pose_error, constraint_scores):
            return {
                "position_errors": {"mean": 0.01, "std": 0.005},
                "rotation_errors": {"mean": 0.05, "std": 0.02}
            }
    
    return MockMetricsCalculator()


@pytest.fixture(autouse=True)
def setup_test_environment(
    monkeypatch,
    mock_gasm_bridge,
    mock_motion_planner, 
    mock_metrics_calculator
):
    """Setup test environment with mocked components."""
    # Mock spatial components
    monkeypatch.setattr(
        "src.api.v1.spatial_endpoints.gasm_bridge", 
        mock_gasm_bridge
    )
    monkeypatch.setattr(
        "src.api.v1.spatial_endpoints.motion_planner",
        mock_motion_planner
    )
    monkeypatch.setattr(
        "src.api.v1.spatial_endpoints.metrics_calculator",
        mock_metrics_calculator
    )
    monkeypatch.setattr(
        "src.api.v1.spatial_endpoints.SPATIAL_COMPONENTS_AVAILABLE",
        True
    )
    
    # Mock GASM components
    monkeypatch.setattr(
        "src.api.v1.gasm_endpoints.gasm_bridge",
        mock_gasm_bridge
    )
    monkeypatch.setattr(
        "src.api.v1.gasm_endpoints.GASM_BRIDGE_AVAILABLE",
        True
    )
    
    # Mock rate limiting for tests
    rate_config = RateLimitConfig()
    rate_config.default_limit = 1000  # High limit for tests
    rate_config.exempt_ips = ["testserver", "127.0.0.1"]
    
    monkeypatch.setattr(
        "src.api.middleware.rate_limit.RateLimitConfig",
        lambda: rate_config
    )


@pytest.fixture
def sample_requests():
    """Sample API request payloads for testing."""
    return {
        "text_processing": {
            "text": "place the red block above the blue cube",
            "enable_geometry": True,
            "return_embeddings": False,
            "return_geometry": False,
            "max_length": 512
        },
        "pose_request": {
            "target_pose": {
                "x": 0.1,
                "y": 0.1, 
                "z": 0.1,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.5,
                "frame_id": "world"
            },
            "pose_format": "dict_format",
            "timeout": 30.0
        },
        "motion_plan": {
            "current_pose": {
                "x": 0.0, "y": 0.0, "z": 0.0,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0
            },
            "target_pose": {
                "x": 0.1, "y": 0.1, "z": 0.1,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.5
            },
            "constraints": [],
            "obstacles": [],
            "strategy": "constrained"
        },
        "gasm_process": {
            "instruction": "place the red block above the blue cube",
            "context": {"workspace": "table_top"},
            "return_debug_info": True
        },
        "spatial_instruction": {
            "instruction": "move the robot arm to position (0.1, 0.1, 0.1)",
            "current_state": {"position": [0.0, 0.0, 0.0]},
            "constraints": [],
            "options": {"include_debug": False}
        }
    }


# Test utility functions
def assert_response_structure(response_data: Dict[str, Any], expected_keys: List[str]):
    """Assert that response has expected structure."""
    for key in expected_keys:
        assert key in response_data, f"Expected key '{key}' not found in response"


def assert_error_response(response_data: Dict[str, Any], expected_status_code: int = None):
    """Assert that response is a properly formatted error."""
    assert "success" in response_data
    assert response_data["success"] is False
    assert "error" in response_data
    assert "timestamp" in response_data
    
    if expected_status_code:
        # This would need to be checked at the HTTP response level
        pass


def assert_success_response(response_data: Dict[str, Any]):
    """Assert that response is a properly formatted success."""
    assert "success" in response_data
    assert response_data["success"] is True
    assert "timestamp" in response_data