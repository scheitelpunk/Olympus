#!/usr/bin/env python3
"""
GASM Bridge - Integration with GASM-Robotics for MORPHEUS.

This module provides seamless integration between MORPHEUS and the
GASM-Robotics system, enabling:
- Material property synchronization
- Physics simulation coordination
- Visual feature extraction
- Real-time state synchronization
- Spatial coordinate transformation

Features:
- Bidirectional communication with GASM
- Real-time physics state synchronization
- Material database integration
- Visual processing pipeline
- Multi-threading for performance
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
from pathlib import Path
import importlib.util

# Try to import GASM components if available
try:
    # Add GASM-Roboting to path if it exists
    gasm_path = Path(__file__).parent.parent.parent.parent / "GASM-Robotics"
    if gasm_path.exists():
        sys.path.insert(0, str(gasm_path / "src"))
    
    # Import GASM components
    from spatial_agent import SpatialAgent
    from spatial_agent.core import SE3Transform
    from spatial_agent.perception import VisionSystem
    from spatial_agent.planning import PathPlanner
    GASM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GASM-Robotics not available: {e}")
    GASM_AVAILABLE = False
    # Define dummy classes for type hints
    class SpatialAgent: pass
    class SE3Transform: pass
    class VisionSystem: pass
    class PathPlanner: pass

logger = logging.getLogger(__name__)

@dataclass
class GASMState:
    """Complete state from GASM-Robotics system."""
    timestamp: float
    robot_pose: np.ndarray  # SE3 pose matrix
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    contact_forces: List[np.ndarray]
    contact_points: List[np.ndarray]
    materials_in_contact: List[str]
    visual_features: Optional[np.ndarray] = None
    audio_sources: Optional[List[Dict[str, Any]]] = None
    planning_state: Optional[Dict[str, Any]] = None

@dataclass
class MorpheusCommand:
    """Commands to send to GASM-Robotics system."""
    target_pose: Optional[np.ndarray] = None
    target_joints: Optional[np.ndarray] = None
    target_velocity: Optional[np.ndarray] = None
    force_commands: Optional[np.ndarray] = None
    gripper_command: Optional[float] = None
    planning_goal: Optional[Dict[str, Any]] = None
    simulation_params: Optional[Dict[str, Any]] = None

@dataclass
class SynchronizationMetrics:
    """Metrics for GASM-MORPHEUS synchronization."""
    last_sync_time: float
    sync_frequency: float
    message_latency: float
    data_transfer_rate: float  # MB/s
    dropped_messages: int
    sync_errors: int

class GASMBridge:
    """
    Bridge between MORPHEUS and GASM-Robotics systems.
    
    Provides bidirectional communication, state synchronization,
    and coordinate system management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GASM bridge.
        
        Args:
            config: Configuration dictionary with GASM settings
        """
        self.config = config
        self.gasm_available = GASM_AVAILABLE
        
        # Bridge parameters
        self.sync_frequency = config.get('sync_frequency', 100.0)  # Hz
        self.enable_visual = config.get('enable_visual', True)
        self.enable_audio = config.get('enable_audio', True)
        self.enable_physics = config.get('enable_physics', True)
        self.coordinate_frame = config.get('coordinate_frame', 'world')
        
        # State management
        self.current_gasm_state = None
        self.last_command = None
        self.state_history = []
        self.max_history = config.get('max_history', 1000)
        
        # Thread safety
        self._lock = threading.Lock()
        self._sync_thread = None
        self._running = False
        
        # Performance metrics
        self.sync_metrics = SynchronizationMetrics(
            last_sync_time=0,
            sync_frequency=0,
            message_latency=0,
            data_transfer_rate=0,
            dropped_messages=0,
            sync_errors=0
        )
        
        # GASM components (if available)
        self.spatial_agent = None
        self.vision_system = None
        self.path_planner = None
        
        # Initialize GASM connection
        if self.gasm_available:
            self._init_gasm_components()
        else:
            logger.warning("GASM-Robotics not available, running in simulation mode")
            self._init_simulation_mode()
            
        logger.info(f"GASMBridge initialized (GASM available: {self.gasm_available})")
        
    def _init_gasm_components(self):
        """Initialize GASM-Robotics components."""
        
        try:
            # Initialize spatial agent
            gasm_config = self.config.get('spatial_agent', {})
            self.spatial_agent = SpatialAgent(gasm_config)
            
            # Initialize vision system
            if self.enable_visual:
                vision_config = self.config.get('vision', {})
                self.vision_system = VisionSystem(vision_config)
                
            # Initialize path planner
            planning_config = self.config.get('planning', {})
            self.path_planner = PathPlanner(planning_config)
            
            logger.info("GASM components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GASM components: {e}")
            self.gasm_available = False
            self._init_simulation_mode()
            
    def _init_simulation_mode(self):
        """Initialize simulation mode when GASM is not available."""
        
        # Create mock objects for simulation
        self.spatial_agent = MockSpatialAgent()
        self.vision_system = MockVisionSystem()
        self.path_planner = MockPathPlanner()
        
        logger.info("Initialized in simulation mode")
        
    def start_synchronization(self):
        """Start real-time synchronization with GASM."""
        
        if self._running:
            return
            
        self._running = True
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        
        logger.info(f"Started GASM synchronization at {self.sync_frequency} Hz")
        
    def stop_synchronization(self):
        """Stop synchronization with GASM."""
        
        self._running = False
        
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=1.0)
            
        logger.info("Stopped GASM synchronization")
        
    def _sync_loop(self):
        """Main synchronization loop."""
        
        sync_interval = 1.0 / self.sync_frequency
        
        while self._running:
            start_time = time.time()
            
            try:
                # Get current state from GASM
                gasm_state = self._get_gasm_state()
                
                if gasm_state:
                    with self._lock:
                        self.current_gasm_state = gasm_state
                        
                        # Update state history
                        self.state_history.append(gasm_state)
                        if len(self.state_history) > self.max_history:
                            self.state_history.pop(0)
                            
                    # Update sync metrics
                    self._update_sync_metrics(start_time)
                    
                # Send any pending commands
                if self.last_command:
                    self._send_command_to_gasm(self.last_command)
                    
            except Exception as e:
                logger.error(f"Synchronization error: {e}")
                self.sync_metrics.sync_errors += 1
                
            # Maintain sync frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, sync_interval - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def _get_gasm_state(self) -> Optional[GASMState]:
        """Get current state from GASM system."""
        
        try:
            if self.spatial_agent:
                # Get robot pose and joint states
                robot_pose = self.spatial_agent.get_current_pose()
                joint_pos = self.spatial_agent.get_joint_positions()
                joint_vel = self.spatial_agent.get_joint_velocities()
                
                # Get contact information
                contact_info = self.spatial_agent.get_contact_information()
                contact_forces = contact_info.get('forces', [])
                contact_points = contact_info.get('points', [])
                materials = contact_info.get('materials', [])
                
                # Get visual features if enabled
                visual_features = None
                if self.enable_visual and self.vision_system:
                    visual_features = self.vision_system.extract_features()
                    
                # Get audio information if enabled
                audio_sources = None
                if self.enable_audio:
                    audio_sources = self.spatial_agent.get_audio_sources()
                    
                # Get planning state
                planning_state = None
                if self.path_planner:
                    planning_state = self.path_planner.get_current_state()
                    
                return GASMState(
                    timestamp=time.time(),
                    robot_pose=robot_pose,
                    joint_positions=joint_pos,
                    joint_velocities=joint_vel,
                    contact_forces=contact_forces,
                    contact_points=contact_points,
                    materials_in_contact=materials,
                    visual_features=visual_features,
                    audio_sources=audio_sources,
                    planning_state=planning_state
                )
                
        except Exception as e:
            logger.error(f"Failed to get GASM state: {e}")
            
        return None
        
    def _send_command_to_gasm(self, command: MorpheusCommand):
        """Send command to GASM system."""
        
        try:
            if self.spatial_agent:
                # Send pose commands
                if command.target_pose is not None:
                    self.spatial_agent.set_target_pose(command.target_pose)
                    
                # Send joint commands
                if command.target_joints is not None:
                    self.spatial_agent.set_joint_targets(command.target_joints)
                    
                # Send velocity commands
                if command.target_velocity is not None:
                    self.spatial_agent.set_target_velocity(command.target_velocity)
                    
                # Send force commands
                if command.force_commands is not None:
                    self.spatial_agent.set_force_commands(command.force_commands)
                    
                # Send gripper commands
                if command.gripper_command is not None:
                    self.spatial_agent.set_gripper_position(command.gripper_command)
                    
                # Send planning goals
                if command.planning_goal is not None and self.path_planner:
                    self.path_planner.set_goal(command.planning_goal)
                    
                # Update simulation parameters
                if command.simulation_params is not None:
                    self.spatial_agent.update_simulation_params(command.simulation_params)
                    
        except Exception as e:
            logger.error(f"Failed to send command to GASM: {e}")
            
    def _update_sync_metrics(self, start_time: float):
        """Update synchronization metrics."""
        
        current_time = time.time()
        
        # Update sync frequency
        if self.sync_metrics.last_sync_time > 0:
            time_diff = current_time - self.sync_metrics.last_sync_time
            if time_diff > 0:
                freq = 1.0 / time_diff
                # Exponential moving average
                alpha = 0.1
                self.sync_metrics.sync_frequency = (alpha * freq + 
                                                  (1 - alpha) * self.sync_metrics.sync_frequency)
                                                  
        # Update latency
        self.sync_metrics.message_latency = current_time - start_time
        self.sync_metrics.last_sync_time = current_time
        
        # Estimate data transfer rate (simplified)
        if self.current_gasm_state:
            # Estimate size of state data
            estimated_size = self._estimate_state_size(self.current_gasm_state)
            self.sync_metrics.data_transfer_rate = estimated_size / max(0.001, self.sync_metrics.message_latency)
            
    def _estimate_state_size(self, state: GASMState) -> float:
        """Estimate size of state data in MB."""
        
        size_bytes = 0
        
        # Basic state data
        size_bytes += state.robot_pose.nbytes if state.robot_pose is not None else 0
        size_bytes += state.joint_positions.nbytes if state.joint_positions is not None else 0
        size_bytes += state.joint_velocities.nbytes if state.joint_velocities is not None else 0
        
        # Contact data
        for forces in state.contact_forces:
            size_bytes += forces.nbytes if forces is not None else 0
        for points in state.contact_points:
            size_bytes += points.nbytes if points is not None else 0
            
        # Visual features
        if state.visual_features is not None:
            size_bytes += state.visual_features.nbytes
            
        # Convert to MB
        return size_bytes / (1024 * 1024)
        
    def get_current_state(self) -> Optional[GASMState]:
        """Get current GASM state."""
        
        with self._lock:
            return self.current_gasm_state
            
    def send_command(self, command: MorpheusCommand):
        """Send command to GASM system."""
        
        with self._lock:
            self.last_command = command
            
        logger.debug(f"Queued command for GASM: {command}")
        
    def get_robot_pose(self) -> Optional[np.ndarray]:
        """Get current robot pose as SE3 matrix."""
        
        state = self.get_current_state()
        if state and state.robot_pose is not None:
            return state.robot_pose
            
        return None
        
    def get_contact_information(self) -> Dict[str, Any]:
        """Get current contact information."""
        
        state = self.get_current_state()
        
        if state:
            return {
                'forces': state.contact_forces,
                'points': state.contact_points,
                'materials': state.materials_in_contact,
                'has_contact': len(state.contact_forces) > 0
            }
            
        return {
            'forces': [],
            'points': [],
            'materials': [],
            'has_contact': False
        }
        
    def get_visual_features(self) -> Optional[np.ndarray]:
        """Get current visual features."""
        
        state = self.get_current_state()
        if state:
            return state.visual_features
            
        return None
        
    def get_audio_sources(self) -> List[Dict[str, Any]]:
        """Get current audio sources."""
        
        state = self.get_current_state()
        if state and state.audio_sources:
            return state.audio_sources
            
        return []
        
    def transform_coordinates(self, 
                            points: np.ndarray, 
                            from_frame: str, 
                            to_frame: str) -> np.ndarray:
        """Transform coordinates between reference frames."""
        
        try:
            if self.spatial_agent and hasattr(self.spatial_agent, 'transform_coordinates'):
                return self.spatial_agent.transform_coordinates(points, from_frame, to_frame)
            else:
                # Simple identity transform for simulation
                logger.warning(f"Coordinate transform not available, returning original points")
                return points
                
        except Exception as e:
            logger.error(f"Coordinate transformation failed: {e}")
            return points
            
    def set_material_properties(self, material_name: str, properties: Dict[str, float]):
        """Update material properties in GASM simulation."""
        
        try:
            if self.spatial_agent and hasattr(self.spatial_agent, 'set_material_properties'):
                self.spatial_agent.set_material_properties(material_name, properties)
            else:
                logger.debug(f"Material properties update not available: {material_name}")
                
        except Exception as e:
            logger.error(f"Failed to update material properties: {e}")
            
    def get_material_properties(self, material_name: str) -> Dict[str, float]:
        """Get material properties from GASM."""
        
        try:
            if self.spatial_agent and hasattr(self.spatial_agent, 'get_material_properties'):
                return self.spatial_agent.get_material_properties(material_name)
            else:
                # Return default properties
                return {
                    'friction': 0.5,
                    'restitution': 0.3,
                    'density': 1000,
                    'young_modulus': 1e9,
                    'poisson_ratio': 0.3
                }
                
        except Exception as e:
            logger.error(f"Failed to get material properties: {e}")
            return {}
            
    def execute_plan(self, plan: List[np.ndarray], execution_speed: float = 1.0):
        """Execute a planned trajectory."""
        
        try:
            if self.path_planner:
                self.path_planner.execute_plan(plan, execution_speed)
            else:
                logger.warning("Path planner not available")
                
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            
    def get_planning_status(self) -> Dict[str, Any]:
        """Get current planning status."""
        
        state = self.get_current_state()
        
        if state and state.planning_state:
            return state.planning_state
            
        return {
            'status': 'unknown',
            'progress': 0.0,
            'current_waypoint': 0,
            'total_waypoints': 0
        }
        
    def capture_visual_snapshot(self) -> Optional[Dict[str, Any]]:
        """Capture visual snapshot from GASM vision system."""
        
        try:
            if self.vision_system and hasattr(self.vision_system, 'capture_snapshot'):
                return self.vision_system.capture_snapshot()
            else:
                logger.warning("Vision system not available")
                return None
                
        except Exception as e:
            logger.error(f"Visual snapshot failed: {e}")
            return None
            
    def get_synchronization_metrics(self) -> SynchronizationMetrics:
        """Get synchronization performance metrics."""
        
        with self._lock:
            return self.sync_metrics
            
    def get_state_history(self, num_states: int = 100) -> List[GASMState]:
        """Get recent state history."""
        
        with self._lock:
            return self.state_history[-num_states:]
            
    def reset_simulation(self, reset_params: Optional[Dict[str, Any]] = None):
        """Reset GASM simulation."""
        
        try:
            if self.spatial_agent and hasattr(self.spatial_agent, 'reset'):
                self.spatial_agent.reset(reset_params)
                
                # Clear state history
                with self._lock:
                    self.state_history.clear()
                    self.current_gasm_state = None
                    
                logger.info("GASM simulation reset")
                
        except Exception as e:
            logger.error(f"Simulation reset failed: {e}")
            
    def save_session_data(self, filepath: str):
        """Save session data for analysis."""
        
        try:
            session_data = {
                'config': self.config,
                'state_history': [self._state_to_dict(state) for state in self.get_state_history()],
                'sync_metrics': {
                    'sync_frequency': self.sync_metrics.sync_frequency,
                    'average_latency': self.sync_metrics.message_latency,
                    'data_transfer_rate': self.sync_metrics.data_transfer_rate,
                    'sync_errors': self.sync_metrics.sync_errors
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2, default=self._numpy_serializer)
                
            logger.info(f"Session data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
            
    def _state_to_dict(self, state: GASMState) -> Dict[str, Any]:
        """Convert GASMState to dictionary for serialization."""
        
        return {
            'timestamp': state.timestamp,
            'robot_pose': state.robot_pose.tolist() if state.robot_pose is not None else None,
            'joint_positions': state.joint_positions.tolist() if state.joint_positions is not None else None,
            'joint_velocities': state.joint_velocities.tolist() if state.joint_velocities is not None else None,
            'contact_forces': [f.tolist() for f in state.contact_forces],
            'contact_points': [p.tolist() for p in state.contact_points],
            'materials_in_contact': state.materials_in_contact,
            'has_visual_features': state.visual_features is not None,
            'num_audio_sources': len(state.audio_sources) if state.audio_sources else 0
        }
        
    def _numpy_serializer(self, obj):
        """JSON serializer for numpy objects."""
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
            
        return str(obj)

# Mock classes for simulation mode
class MockSpatialAgent:
    """Mock spatial agent for testing without GASM."""
    
    def __init__(self):
        self.pose = np.eye(4)
        self.joint_positions = np.zeros(7)
        self.joint_velocities = np.zeros(7)
        
    def get_current_pose(self):
        # Add small random motion
        noise = np.random.normal(0, 0.001, (4, 4))
        noise[3, :] = 0  # Keep homogeneous row
        noise[:, 3] = 0  # Keep translation column
        self.pose += noise
        return self.pose
        
    def get_joint_positions(self):
        # Add small random joint motion
        self.joint_positions += np.random.normal(0, 0.01, 7)
        return self.joint_positions
        
    def get_joint_velocities(self):
        self.joint_velocities = np.random.normal(0, 0.1, 7)
        return self.joint_velocities
        
    def get_contact_information(self):
        # Simulate occasional contact
        if np.random.random() > 0.8:
            return {
                'forces': [np.random.normal(0, 1, 3)],
                'points': [np.random.normal(0, 0.1, 3)],
                'materials': ['steel']
            }
        else:
            return {'forces': [], 'points': [], 'materials': []}
            
    def get_audio_sources(self):
        # Simulate occasional audio sources
        if np.random.random() > 0.9:
            return [{
                'position': np.random.normal(0, 1, 3),
                'frequency': np.random.uniform(100, 8000),
                'amplitude': np.random.uniform(0, 1),
                'material': 'steel'
            }]
        return []
        
    def set_target_pose(self, pose):
        pass
        
    def set_joint_targets(self, joints):
        pass
        
    def set_target_velocity(self, velocity):
        pass
        
    def set_force_commands(self, forces):
        pass
        
    def set_gripper_position(self, position):
        pass

class MockVisionSystem:
    """Mock vision system for testing."""
    
    def extract_features(self):
        # Return random visual features
        return np.random.normal(0, 1, 128)
        
    def capture_snapshot(self):
        return {
            'image_shape': (480, 640, 3),
            'detected_objects': [],
            'timestamp': time.time()
        }

class MockPathPlanner:
    """Mock path planner for testing."""
    
    def __init__(self):
        self.status = 'idle'
        self.progress = 0.0
        
    def get_current_state(self):
        return {
            'status': self.status,
            'progress': self.progress,
            'current_waypoint': 0,
            'total_waypoints': 0
        }
        
    def set_goal(self, goal):
        self.status = 'planning'
        
    def execute_plan(self, plan, speed):
        self.status = 'executing'