"""
GASM-Roboting API Test Suite

Comprehensive test suite for all API components including endpoints,
middleware, models, and integration tests.
"""

import pytest
import asyncio
from typing import AsyncGenerator
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Test configuration
TEST_CONFIG = {
    "test_database_url": "sqlite:///./test.db",
    "test_redis_url": None,  # Use in-memory for tests
    "test_jwt_secret": "test_jwt_secret_key_for_testing_only",
    "test_api_keys": {
        "valid_key": "gasm_test_key_12345",
        "expired_key": "gasm_expired_key_67890"
    }
}

# Test users
TEST_USERS = {
    "admin": {
        "id": "test_admin_001",
        "username": "test_admin",
        "email": "admin@test.gasm-roboting.dev",
        "password": "test_admin_password",
        "role": "admin",
        "scopes": ["admin", "read", "write", "spatial:control", "gasm:process"]
    },
    "user": {
        "id": "test_user_001", 
        "username": "test_user",
        "email": "user@test.gasm-roboting.dev",
        "password": "test_user_password",
        "role": "user",
        "scopes": ["read", "write"]
    },
    "viewer": {
        "id": "test_viewer_001",
        "username": "test_viewer",
        "email": "viewer@test.gasm-roboting.dev", 
        "password": "test_viewer_password",
        "role": "viewer",
        "scopes": ["read"]
    }
}

# Test spatial data
TEST_POSES = {
    "origin": {
        "x": 0.0, "y": 0.0, "z": 0.0,
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0
    },
    "target_1": {
        "x": 0.1, "y": 0.1, "z": 0.1,
        "roll": 0.0, "pitch": 0.0, "yaw": 0.5
    },
    "quaternion_pose": {
        "position": [0.0, 0.0, 0.1],
        "orientation": [1.0, 0.0, 0.0, 0.0],  # Identity quaternion
        "frame_id": "world"
    }
}

TEST_CONSTRAINTS = {
    "distance": {
        "type": "distance",
        "subject": "object_a",
        "target": "object_b", 
        "parameters": {"target_distance": 0.1, "tolerance": 0.01},
        "priority": 0.8
    },
    "above": {
        "type": "above",
        "subject": "red_block",
        "target": "blue_cube",
        "parameters": {"z": 0.05},
        "priority": 0.9
    }
}

TEST_SPATIAL_INSTRUCTIONS = [
    "place the red block above the blue cube",
    "keep the objects 10cm apart",
    "rotate the part 45 degrees clockwise",
    "align the components along the x-axis",
    "position the tool between the workpieces"
]

# Fixtures will be defined in conftest.py
__all__ = [
    "TEST_CONFIG",
    "TEST_USERS", 
    "TEST_POSES",
    "TEST_CONSTRAINTS",
    "TEST_SPATIAL_INSTRUCTIONS"
]