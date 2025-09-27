#!/usr/bin/env python3
"""
Test suite for GASM Bridge functionality.

Tests integration with GASM-Robotics system including
state synchronization, command transmission, and performance monitoring.
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from morpheus.integration import (
    GASMBridge,
    GASMState,
    MorpheusCommand,
    SynchronizationMetrics
)

class TestGASMBridge:
    """Test cases for GASM Bridge."""
    
    @pytest.fixture
    def bridge_config(self):
        """Standard GASM bridge configuration."""
        return {
            'sync_frequency': 50.0,
            'enable_visual': True,
            'enable_audio': True,
            'enable_physics': True,
            'coordinate_frame': 'world',
            'max_history': 100
        }
        
    @pytest.fixture
    def gasm_bridge(self, bridge_config):
        """GASM bridge instance for testing."""
        # Mock GASM availability to test simulation mode
        with patch('morpheus.integration.gasm_bridge.GASM_AVAILABLE', False):
            bridge = GASMBridge(bridge_config)
            return bridge
            
    def test_initialization_simulation_mode(self, gasm_bridge, bridge_config):
        """Test bridge initialization in simulation mode."""
        assert gasm_bridge.config == bridge_config
        assert gasm_bridge.gasm_available is False
        assert gasm_bridge.sync_frequency == bridge_config['sync_frequency']
        
        # Should have mock components
        assert gasm_bridge.spatial_agent is not None
        assert gasm_bridge.vision_system is not None
        assert gasm_bridge.path_planner is not None
        
    @patch('morpheus.integration.gasm_bridge.GASM_AVAILABLE', True)
    def test_initialization_gasm_mode(self, bridge_config):
        """Test bridge initialization with GASM available."""
        # This test would require actual GASM installation
        # For now, just test the configuration path
        with patch('morpheus.integration.gasm_bridge.SpatialAgent') as mock_agent:
            mock_agent.return_value = Mock()
            
            bridge = GASMBridge(bridge_config)
            assert bridge.gasm_available is True
            
    def test_state_synchronization_start_stop(self, gasm_bridge):
        """Test starting and stopping synchronization."""
        assert not gasm_bridge._running
        
        # Start synchronization
        gasm_bridge.start_synchronization()
        assert gasm_bridge._running
        assert gasm_bridge._sync_thread is not None
        
        # Give it a moment to start
        time.sleep(0.1)
        
        # Stop synchronization
        gasm_bridge.stop_synchronization()
        assert not gasm_bridge._running
        
    def test_gasm_state_retrieval(self, gasm_bridge):
        """Test getting GASM state in simulation mode."""
        state = gasm_bridge._get_gasm_state()
        
        assert isinstance(state, GASMState)
        assert state.robot_pose is not None
        assert state.robot_pose.shape == (4, 4)  # SE3 matrix
        assert state.joint_positions is not None
        assert state.joint_velocities is not None
        assert isinstance(state.contact_forces, list)
        assert isinstance(state.contact_points, list)
        assert isinstance(state.materials_in_contact, list)
        
    def test_command_sending(self, gasm_bridge):
        """Test sending commands to GASM."""
        command = MorpheusCommand(
            target_pose=np.eye(4),
            target_joints=np.zeros(7),
            gripper_command=0.5
        )
        
        # Should not raise exception in simulation mode
        gasm_bridge.send_command(command)
        assert gasm_bridge.last_command == command
        
    def test_robot_pose_retrieval(self, gasm_bridge):
        """Test getting robot pose."""
        # Start sync to populate state
        gasm_bridge.start_synchronization()
        time.sleep(0.1)
        
        pose = gasm_bridge.get_robot_pose()
        
        if pose is not None:
            assert pose.shape == (4, 4)
            assert np.allclose(pose[3, :3], [0, 0, 0])  # Last row
            assert pose[3, 3] == 1  # Homogeneous coordinate
            
        gasm_bridge.stop_synchronization()
        
    def test_contact_information(self, gasm_bridge):
        """Test contact information retrieval."""
        gasm_bridge.start_synchronization()
        time.sleep(0.1)
        
        contact_info = gasm_bridge.get_contact_information()
        
        assert isinstance(contact_info, dict)
        assert 'forces' in contact_info
        assert 'points' in contact_info
        assert 'materials' in contact_info
        assert 'has_contact' in contact_info
        
        assert isinstance(contact_info['forces'], list)
        assert isinstance(contact_info['points'], list)
        assert isinstance(contact_info['materials'], list)
        assert isinstance(contact_info['has_contact'], bool)
        
        gasm_bridge.stop_synchronization()
        
    def test_visual_features(self, gasm_bridge):
        """Test visual feature extraction."""
        # Mock vision system
        gasm_bridge.vision_system = Mock()
        gasm_bridge.vision_system.extract_features.return_value = np.random.randn(128)
        
        gasm_bridge.start_synchronization()
        time.sleep(0.1)
        
        features = gasm_bridge.get_visual_features()
        
        if features is not None:
            assert isinstance(features, np.ndarray)
            assert features.shape == (128,)
            
        gasm_bridge.stop_synchronization()
        
    def test_audio_sources(self, gasm_bridge):
        """Test audio source retrieval."""
        gasm_bridge.start_synchronization()
        time.sleep(0.1)
        
        audio_sources = gasm_bridge.get_audio_sources()
        
        assert isinstance(audio_sources, list)
        # In simulation mode, might be empty or have mock sources
        
        gasm_bridge.stop_synchronization()
        
    def test_coordinate_transformation(self, gasm_bridge):
        """Test coordinate transformation."""
        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        # In simulation mode, should return original points
        transformed = gasm_bridge.transform_coordinates(points, 'robot', 'world')
        
        assert transformed.shape == points.shape
        # In simulation mode, might be identity transform
        
    def test_material_properties(self, gasm_bridge):
        """Test material property management."""
        # Set material properties
        props = {
            'friction': 0.8,
            'restitution': 0.3,
            'density': 7800
        }
        
        # Should not raise exception
        gasm_bridge.set_material_properties('steel', props)
        
        # Get material properties
        retrieved_props = gasm_bridge.get_material_properties('steel')
        
        assert isinstance(retrieved_props, dict)
        assert 'friction' in retrieved_props
        assert 'restitution' in retrieved_props
        assert 'density' in retrieved_props
        
    def test_synchronization_metrics(self, gasm_bridge):
        """Test synchronization metrics collection."""
        gasm_bridge.start_synchronization()
        time.sleep(0.2)  # Let it run for a bit
        
        metrics = gasm_bridge.get_synchronization_metrics()
        
        assert isinstance(metrics, SynchronizationMetrics)
        assert metrics.sync_frequency >= 0
        assert metrics.last_sync_time > 0
        assert metrics.message_latency >= 0
        
        gasm_bridge.stop_synchronization()
        
    def test_state_history(self, gasm_bridge):
        """Test state history tracking."""
        gasm_bridge.start_synchronization()
        time.sleep(0.2)
        
        history = gasm_bridge.get_state_history(num_states=10)
        
        assert isinstance(history, list)
        assert len(history) <= 10
        
        # All states should be GASMState instances
        for state in history:
            assert isinstance(state, GASMState)
            
        gasm_bridge.stop_synchronization()
        
    def test_planning_integration(self, gasm_bridge):
        """Test planning system integration."""
        # Test plan execution
        plan = [np.eye(4) for _ in range(3)]  # Simple plan
        
        # Should not raise exception
        gasm_bridge.execute_plan(plan, execution_speed=1.0)
        
        # Test planning status
        status = gasm_bridge.get_planning_status()
        
        assert isinstance(status, dict)
        assert 'status' in status
        assert 'progress' in status
        
    def test_visual_snapshot(self, gasm_bridge):
        """Test visual snapshot capture."""
        # Mock vision system
        gasm_bridge.vision_system = Mock()
        gasm_bridge.vision_system.capture_snapshot.return_value = {
            'image_shape': (480, 640, 3),
            'timestamp': time.time(),
            'detected_objects': []
        }
        
        snapshot = gasm_bridge.capture_visual_snapshot()
        
        if snapshot is not None:
            assert isinstance(snapshot, dict)
            assert 'image_shape' in snapshot
            assert 'timestamp' in snapshot
            
    def test_simulation_reset(self, gasm_bridge):
        """Test simulation reset functionality."""
        # Start sync and populate some history
        gasm_bridge.start_synchronization()
        time.sleep(0.1)
        gasm_bridge.stop_synchronization()
        
        # Should have some history
        initial_history_len = len(gasm_bridge.get_state_history())
        
        # Reset simulation
        gasm_bridge.reset_simulation()
        
        # History should be cleared
        final_history_len = len(gasm_bridge.get_state_history())
        assert final_history_len == 0
        
    def test_session_data_saving(self, gasm_bridge, tmp_path):
        """Test session data saving."""
        # Generate some history
        gasm_bridge.start_synchronization()
        time.sleep(0.1)
        gasm_bridge.stop_synchronization()
        
        save_path = tmp_path / "session_data.json"
        
        # Save session data
        gasm_bridge.save_session_data(str(save_path))
        
        assert save_path.exists()
        
        # Load and verify content
        import json
        with open(save_path, 'r') as f:
            session_data = json.load(f)
            
        assert 'config' in session_data
        assert 'state_history' in session_data
        assert 'sync_metrics' in session_data
        
    def test_state_size_estimation(self, gasm_bridge):
        """Test state size estimation for performance tracking."""
        state = GASMState(
            timestamp=time.time(),
            robot_pose=np.eye(4),
            joint_positions=np.zeros(7),
            joint_velocities=np.zeros(7),
            contact_forces=[np.array([1, 2, 3])],
            contact_points=[np.array([0.1, 0.2, 0.3])],
            materials_in_contact=['steel'],
            visual_features=np.random.randn(128)
        )
        
        size_mb = gasm_bridge._estimate_state_size(state)
        
        assert isinstance(size_mb, float)
        assert size_mb > 0
        
    def test_numpy_serialization(self, gasm_bridge):
        """Test numpy array serialization for JSON."""
        # Test different numpy types
        test_objects = [
            np.array([1, 2, 3]),
            np.int64(42),
            np.float32(3.14),
            "regular_string"
        ]
        
        for obj in test_objects:
            serialized = gasm_bridge._numpy_serializer(obj)
            # Should not raise exception and return JSON-serializable type
            assert serialized is not None

class TestGASMState:
    """Test cases for GASM State data structure."""
    
    def test_gasm_state_creation(self):
        """Test GASM state creation."""
        state = GASMState(
            timestamp=time.time(),
            robot_pose=np.eye(4),
            joint_positions=np.array([0, 0.5, 0, -1.57, 0, 1.57, 0]),
            joint_velocities=np.zeros(7),
            contact_forces=[np.array([0, 0, 5])],
            contact_points=[np.array([0.1, 0, 0])],
            materials_in_contact=['aluminum'],
            visual_features=np.random.randn(256)
        )
        
        assert state.timestamp > 0
        assert state.robot_pose.shape == (4, 4)
        assert len(state.joint_positions) == 7
        assert len(state.joint_velocities) == 7
        assert len(state.contact_forces) == 1
        assert len(state.contact_points) == 1
        assert state.materials_in_contact == ['aluminum']
        assert state.visual_features.shape == (256,)

class TestMorpheusCommand:
    """Test cases for Morpheus Command structure."""
    
    def test_command_creation(self):
        """Test command creation with various parameters."""
        command = MorpheusCommand(
            target_pose=np.eye(4),
            target_joints=np.array([0, 0.5, 0, -1.57, 0, 1.57, 0]),
            force_commands=np.array([0, 0, 10]),
            gripper_command=0.8,
            planning_goal={'position': [1, 0, 0.5]}
        )
        
        assert command.target_pose.shape == (4, 4)
        assert len(command.target_joints) == 7
        assert len(command.force_commands) == 3
        assert command.gripper_command == 0.8
        assert command.planning_goal['position'] == [1, 0, 0.5]
        
    def test_partial_command(self):
        """Test command with only some parameters set."""
        command = MorpheusCommand(
            target_joints=np.zeros(7),
            gripper_command=0.0
        )
        
        assert command.target_pose is None
        assert command.target_joints is not None
        assert command.force_commands is None
        assert command.gripper_command == 0.0

class TestSynchronizationMetrics:
    """Test cases for Synchronization Metrics."""
    
    def test_metrics_initialization(self):
        """Test metrics structure initialization."""
        metrics = SynchronizationMetrics(
            last_sync_time=time.time(),
            sync_frequency=50.0,
            message_latency=0.02,
            data_transfer_rate=1.5,
            dropped_messages=0,
            sync_errors=0
        )
        
        assert metrics.last_sync_time > 0
        assert metrics.sync_frequency == 50.0
        assert metrics.message_latency == 0.02
        assert metrics.data_transfer_rate == 1.5
        assert metrics.dropped_messages == 0
        assert metrics.sync_errors == 0

class TestMockComponents:
    """Test cases for mock components used in simulation mode."""
    
    def test_mock_spatial_agent(self):
        """Test mock spatial agent functionality."""
        from morpheus.integration.gasm_bridge import MockSpatialAgent
        
        agent = MockSpatialAgent()
        
        # Test pose retrieval
        pose = agent.get_current_pose()
        assert pose.shape == (4, 4)
        
        # Test joint states
        joints = agent.get_joint_positions()
        assert len(joints) == 7
        
        velocities = agent.get_joint_velocities()
        assert len(velocities) == 7
        
        # Test contact information
        contact_info = agent.get_contact_information()
        assert isinstance(contact_info, dict)
        assert 'forces' in contact_info
        assert 'points' in contact_info
        assert 'materials' in contact_info
        
        # Test audio sources
        audio_sources = agent.get_audio_sources()
        assert isinstance(audio_sources, list)
        
    def test_mock_vision_system(self):
        """Test mock vision system functionality."""
        from morpheus.integration.gasm_bridge import MockVisionSystem
        
        vision = MockVisionSystem()
        
        # Test feature extraction
        features = vision.extract_features()
        assert features.shape == (128,)
        
        # Test snapshot capture
        snapshot = vision.capture_snapshot()
        assert isinstance(snapshot, dict)
        assert 'image_shape' in snapshot
        assert 'timestamp' in snapshot
        
    def test_mock_path_planner(self):
        """Test mock path planner functionality."""
        from morpheus.integration.gasm_bridge import MockPathPlanner
        
        planner = MockPathPlanner()
        
        # Test state retrieval
        state = planner.get_current_state()
        assert isinstance(state, dict)
        assert 'status' in state
        assert 'progress' in state
        
        # Test goal setting
        planner.set_goal({'position': [1, 0, 0]})
        assert planner.status == 'planning'
        
        # Test plan execution
        plan = [np.eye(4) for _ in range(3)]
        planner.execute_plan(plan, 1.0)
        assert planner.status == 'executing'

class TestPerformanceAndReliability:
    """Test cases for performance and reliability features."""
    
    def test_high_frequency_synchronization(self, gasm_bridge):
        """Test synchronization at high frequencies."""
        # Set high sync frequency
        gasm_bridge.sync_frequency = 200.0
        
        gasm_bridge.start_synchronization()
        time.sleep(0.5)  # Run for 500ms
        
        metrics = gasm_bridge.get_synchronization_metrics()
        
        # Should achieve reasonable sync frequency
        assert metrics.sync_frequency > 10.0  # At least 10 Hz
        
        gasm_bridge.stop_synchronization()
        
    def test_error_handling_in_sync_loop(self, gasm_bridge):
        """Test error handling during synchronization."""
        # Mock the _get_gasm_state to raise exception
        original_method = gasm_bridge._get_gasm_state
        gasm_bridge._get_gasm_state = Mock(side_effect=Exception("Test error"))
        
        gasm_bridge.start_synchronization()
        time.sleep(0.2)
        
        metrics = gasm_bridge.get_synchronization_metrics()
        
        # Should have recorded sync errors
        assert metrics.sync_errors > 0
        
        # Restore original method
        gasm_bridge._get_gasm_state = original_method
        gasm_bridge.stop_synchronization()
        
    def test_memory_management(self, gasm_bridge):
        """Test memory management with long-running synchronization."""
        # Set small history limit
        gasm_bridge.max_history = 5
        
        gasm_bridge.start_synchronization()
        time.sleep(0.3)  # Generate several states
        
        history = gasm_bridge.get_state_history()
        
        # Should not exceed max history
        assert len(history) <= gasm_bridge.max_history
        
        gasm_bridge.stop_synchronization()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])