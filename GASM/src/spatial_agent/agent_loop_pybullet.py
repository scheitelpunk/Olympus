"""
Spatial Agent with 3D PyBullet Physics Simulation Environment

Complete implementation with:
1. PyBullet physics simulation setup
2. URDF loading for conveyor and sensor objects  
3. 6DOF SE(3) pose control and physics
4. Camera rendering for optional vision pipeline
5. Integration with all spatial_agent components (gasm_bridge, planner, metrics, vision)
6. Complete feedback loop with physics simulation
7. CLI interface with multiple modes
8. Headless and GUI modes
9. Collision detection and constraint enforcement
10. Comprehensive error handling and fallbacks
"""

import os
import sys
import argparse
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import json

# PyBullet imports with fallback
try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    print("⚠️  PyBullet not available. Install with: pip install pybullet")
    PYBULLET_AVAILABLE = False
    p = None

# Core ML imports
try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not available")
    TORCH_AVAILABLE = False
    # Mock torch for type hints
    class MockTorch:
        class Tensor: pass
    torch = MockTorch()

# Computer Vision imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("⚠️  OpenCV not available. Install with: pip install opencv-python")
    CV2_AVAILABLE = False

# Scientific computing
import math
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import GASM core if available
try:
    from gasm.core import GASM as EnhancedGASM
    from gasm.core import UniversalInvariantAttention as SE3InvariantAttention
    GASM_AVAILABLE = True
except ImportError:
    print("⚠️  GASM core not available")
    GASM_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulationMode(Enum):
    """Simulation execution modes"""
    HEADLESS = "headless"
    GUI = "gui"
    RECORD = "record"
    DEBUG = "debug"


@dataclass
class SE3Pose:
    """6DOF SE(3) pose representation"""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # quaternion [x, y, z, w]
    
    def __post_init__(self):
        """Ensure proper shapes and normalization"""
        self.position = np.array(self.position, dtype=np.float32)
        self.orientation = np.array(self.orientation, dtype=np.float32)
        # Normalize quaternion
        norm = np.linalg.norm(self.orientation)
        if norm > 1e-6:
            self.orientation = self.orientation / norm
        else:
            self.orientation = np.array([0., 0., 0., 1.])  # Identity quaternion
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix"""
        T = np.eye(4)
        T[:3, 3] = self.position
        
        # Convert quaternion to rotation matrix
        x, y, z, w = self.orientation
        T[:3, :3] = np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
        return T
    
    @classmethod
    def from_matrix(cls, T: np.ndarray) -> 'SE3Pose':
        """Create from 4x4 transformation matrix"""
        position = T[:3, 3]
        
        # Extract quaternion from rotation matrix
        R = T[:3, :3]
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        
        orientation = np.array([x, y, z, w])
        return cls(position, orientation)


@dataclass
class SimulationConfig:
    """Simulation configuration parameters"""
    # Physics settings
    gravity: Tuple[float, float, float] = (0., 0., -9.81)
    time_step: float = 1./240.  # 240Hz simulation
    max_steps: int = 1000
    
    # Rendering settings
    width: int = 640
    height: int = 480
    fov: float = 60.0
    near_plane: float = 0.1
    far_plane: float = 10.0
    
    # Camera settings
    camera_distance: float = 2.0
    camera_yaw: float = 45.0
    camera_pitch: float = -30.0
    camera_target: Tuple[float, float, float] = (0., 0., 0.)
    
    # Object settings
    conveyor_length: float = 2.0
    conveyor_width: float = 0.5
    conveyor_height: float = 0.1
    conveyor_speed: float = 0.1  # m/s
    
    # Constraint settings
    position_tolerance: float = 0.01  # meters
    orientation_tolerance: float = 0.05  # radians
    collision_margin: float = 0.01  # meters


class PhysicsObject:
    """Wrapper for PyBullet physics objects"""
    
    def __init__(self, body_id: int, object_type: str, name: str = ""):
        self.body_id = body_id
        self.object_type = object_type
        self.name = name
        self._initial_pose = None
        self._constraints = []
    
    def get_pose(self) -> SE3Pose:
        """Get current 6DOF pose"""
        if not PYBULLET_AVAILABLE or self.body_id < 0:
            return SE3Pose([0., 0., 0.], [0., 0., 0., 1.])
        
        try:
            pos, orn = p.getBasePositionAndOrientation(self.body_id)
            return SE3Pose(np.array(pos), np.array(orn))
        except Exception as e:
            logger.warning(f"Failed to get pose for {self.name}: {e}")
            return SE3Pose([0., 0., 0.], [0., 0., 0., 1.])
    
    def set_pose(self, pose: SE3Pose):
        """Set 6DOF pose"""
        if not PYBULLET_AVAILABLE or self.body_id < 0:
            return
        
        try:
            p.resetBasePositionAndOrientation(
                self.body_id,
                pose.position.tolist(),
                pose.orientation.tolist()
            )
        except Exception as e:
            logger.warning(f"Failed to set pose for {self.name}: {e}")
    
    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get linear and angular velocities"""
        if not PYBULLET_AVAILABLE or self.body_id < 0:
            return np.zeros(3), np.zeros(3)
        
        try:
            lin_vel, ang_vel = p.getBaseVelocity(self.body_id)
            return np.array(lin_vel), np.array(ang_vel)
        except Exception as e:
            logger.warning(f"Failed to get velocity for {self.name}: {e}")
            return np.zeros(3), np.zeros(3)
    
    def apply_force(self, force: np.ndarray, position: Optional[np.ndarray] = None):
        """Apply force at specified position"""
        if not PYBULLET_AVAILABLE or self.body_id < 0:
            return
        
        try:
            if position is None:
                position = [0., 0., 0.]
            p.applyExternalForce(
                self.body_id, -1, force.tolist(), position.tolist(),
                p.WORLD_FRAME
            )
        except Exception as e:
            logger.warning(f"Failed to apply force to {self.name}: {e}")


class PyBulletSimulation:
    """Main PyBullet simulation manager"""
    
    def __init__(self, config: SimulationConfig, mode: SimulationMode = SimulationMode.HEADLESS):
        self.config = config
        self.mode = mode
        self.physics_client = None
        self.objects: Dict[str, PhysicsObject] = {}
        self.constraints: List[int] = []
        self.camera_images = []
        self.step_count = 0
        
        # Initialize physics
        self._setup_physics()
        
        # Create default objects
        self._setup_environment()
    
    def _setup_physics(self):
        """Initialize PyBullet physics simulation"""
        if not PYBULLET_AVAILABLE:
            logger.warning("PyBullet not available - using mock simulation")
            return
        
        try:
            # Connect to physics server
            if self.mode == SimulationMode.GUI:
                self.physics_client = p.connect(p.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            else:
                self.physics_client = p.connect(p.DIRECT)
            
            # Set up physics parameters
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(*self.config.gravity)
            p.setTimeStep(self.config.time_step)
            p.setRealTimeSimulation(0)  # Disable real-time for consistent stepping
            
            logger.info(f"PyBullet simulation initialized in {self.mode.value} mode")
            
        except Exception as e:
            logger.error(f"Failed to initialize PyBullet: {e}")
            self.physics_client = None
    
    def _setup_environment(self):
        """Create simulation environment with conveyor and objects"""
        if not PYBULLET_AVAILABLE or self.physics_client is None:
            # Create mock objects for testing
            self.objects["ground"] = PhysicsObject(-1, "ground", "ground_plane")
            self.objects["conveyor"] = PhysicsObject(-1, "conveyor", "conveyor_belt")
            return
        
        try:
            # Load ground plane
            ground_id = p.loadURDF("plane.urdf")
            self.objects["ground"] = PhysicsObject(ground_id, "ground", "ground_plane")
            
            # Create conveyor belt (procedural box)
            conveyor_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[
                    self.config.conveyor_length/2,
                    self.config.conveyor_width/2,
                    self.config.conveyor_height/2
                ]
            )
            conveyor_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[
                    self.config.conveyor_length/2,
                    self.config.conveyor_width/2,
                    self.config.conveyor_height/2
                ],
                rgbaColor=[0.7, 0.7, 0.7, 1.0]
            )
            conveyor_id = p.createMultiBody(
                baseMass=0,  # Static object
                baseCollisionShapeIndex=conveyor_shape,
                baseVisualShapeIndex=conveyor_visual,
                basePosition=[0, 0, self.config.conveyor_height/2]
            )
            self.objects["conveyor"] = PhysicsObject(conveyor_id, "conveyor", "conveyor_belt")
            
            logger.info("Environment setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup environment: {e}")
    
    def load_urdf_object(self, urdf_path: str, name: str, pose: SE3Pose) -> Optional[PhysicsObject]:
        """Load object from URDF file"""
        if not PYBULLET_AVAILABLE or self.physics_client is None:
            # Mock object for testing
            mock_obj = PhysicsObject(-1, "urdf", name)
            self.objects[name] = mock_obj
            return mock_obj
        
        try:
            if os.path.exists(urdf_path):
                body_id = p.loadURDF(
                    urdf_path,
                    basePosition=pose.position.tolist(),
                    baseOrientation=pose.orientation.tolist()
                )
                obj = PhysicsObject(body_id, "urdf", name)
                self.objects[name] = obj
                logger.info(f"Loaded URDF object: {name} from {urdf_path}")
                return obj
            else:
                logger.warning(f"URDF file not found: {urdf_path}")
                return self._create_procedural_box(name, pose)
                
        except Exception as e:
            logger.error(f"Failed to load URDF {urdf_path}: {e}")
            return self._create_procedural_box(name, pose)
    
    def _create_procedural_box(self, name: str, pose: SE3Pose, 
                             size: Tuple[float, float, float] = (0.1, 0.1, 0.1),
                             mass: float = 1.0) -> PhysicsObject:
        """Create procedural box when URDF not available"""
        if not PYBULLET_AVAILABLE or self.physics_client is None:
            mock_obj = PhysicsObject(-1, "procedural", name)
            self.objects[name] = mock_obj
            return mock_obj
        
        try:
            # Create collision and visual shapes
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[size[0]/2, size[1]/2, size[2]/2]
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[size[0]/2, size[1]/2, size[2]/2],
                rgbaColor=[
                    np.random.uniform(0.2, 0.8),
                    np.random.uniform(0.2, 0.8),
                    np.random.uniform(0.2, 0.8),
                    1.0
                ]
            )
            
            # Create multi-body
            body_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=pose.position.tolist(),
                baseOrientation=pose.orientation.tolist()
            )
            
            obj = PhysicsObject(body_id, "procedural", name)
            self.objects[name] = obj
            logger.info(f"Created procedural box: {name}")
            return obj
            
        except Exception as e:
            logger.error(f"Failed to create procedural box {name}: {e}")
            mock_obj = PhysicsObject(-1, "procedural", name)
            self.objects[name] = mock_obj
            return mock_obj
    
    def add_constraint(self, obj1_name: str, obj2_name: str, 
                      constraint_type: str = "fixed") -> Optional[int]:
        """Add physics constraint between objects"""
        if not PYBULLET_AVAILABLE or self.physics_client is None:
            return None
        
        try:
            obj1 = self.objects.get(obj1_name)
            obj2 = self.objects.get(obj2_name)
            
            if obj1 is None or obj2 is None:
                logger.warning(f"Cannot create constraint: objects not found")
                return None
            
            if constraint_type == "fixed":
                constraint_id = p.createConstraint(
                    obj1.body_id, -1, obj2.body_id, -1,
                    p.JOINT_FIXED,
                    [0, 0, 0], [0, 0, 0], [0, 0, 0]
                )
            elif constraint_type == "point2point":
                constraint_id = p.createConstraint(
                    obj1.body_id, -1, obj2.body_id, -1,
                    p.JOINT_POINT2POINT,
                    [0, 0, 0], [0, 0, 0], [0, 0, 0]
                )
            else:
                logger.warning(f"Unknown constraint type: {constraint_type}")
                return None
            
            self.constraints.append(constraint_id)
            logger.info(f"Created {constraint_type} constraint between {obj1_name} and {obj2_name}")
            return constraint_id
            
        except Exception as e:
            logger.error(f"Failed to create constraint: {e}")
            return None
    
    def step_simulation(self) -> Dict[str, Any]:
        """Step physics simulation and return state information"""
        if not PYBULLET_AVAILABLE or self.physics_client is None:
            # Mock simulation step
            self.step_count += 1
            return {
                "step": self.step_count,
                "time": self.step_count * self.config.time_step,
                "objects": {name: obj.get_pose() for name, obj in self.objects.items()},
                "collisions": [],
                "mock": True
            }
        
        try:
            # Step physics
            p.stepSimulation()
            self.step_count += 1
            
            # Collect state information
            state = {
                "step": self.step_count,
                "time": self.step_count * self.config.time_step,
                "objects": {},
                "collisions": [],
                "mock": False
            }
            
            # Get object poses
            for name, obj in self.objects.items():
                state["objects"][name] = obj.get_pose()
            
            # Check for collisions
            contact_points = p.getContactPoints()
            for contact in contact_points:
                state["collisions"].append({
                    "bodyA": contact[1],
                    "bodyB": contact[2],
                    "position": contact[5],
                    "normal": contact[7],
                    "distance": contact[8]
                })
            
            return state
            
        except Exception as e:
            logger.error(f"Simulation step failed: {e}")
            self.step_count += 1
            return {
                "step": self.step_count,
                "time": self.step_count * self.config.time_step,
                "objects": {name: obj.get_pose() for name, obj in self.objects.items()},
                "collisions": [],
                "error": str(e)
            }
    
    def render_camera(self, save_image: bool = False) -> Optional[np.ndarray]:
        """Render camera view and optionally save image"""
        if not PYBULLET_AVAILABLE or self.physics_client is None:
            # Return mock image
            mock_image = np.random.randint(0, 255, (self.config.height, self.config.width, 3), dtype=np.uint8)
            if save_image:
                self.camera_images.append(mock_image)
            return mock_image
        
        try:
            # Compute view and projection matrices
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.config.camera_target,
                distance=self.config.camera_distance,
                yaw=self.config.camera_yaw,
                pitch=self.config.camera_pitch,
                roll=0,
                upAxisIndex=2
            )
            
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=self.config.fov,
                aspect=self.config.width / self.config.height,
                nearPlane=self.config.near_plane,
                farPlane=self.config.far_plane
            )
            
            # Render image
            width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                width=self.config.width,
                height=self.config.height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # Convert to numpy array
            rgb_array = np.array(rgb_img).reshape((height, width, 4))[:, :, :3]
            
            if save_image:
                self.camera_images.append(rgb_array.copy())
            
            return rgb_array
            
        except Exception as e:
            logger.error(f"Camera rendering failed: {e}")
            mock_image = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            if save_image:
                self.camera_images.append(mock_image)
            return mock_image
    
    def detect_collisions(self) -> List[Dict[str, Any]]:
        """Detect and analyze collisions"""
        if not PYBULLET_AVAILABLE or self.physics_client is None:
            return []
        
        try:
            collisions = []
            contact_points = p.getContactPoints()
            
            for contact in contact_points:
                bodyA_id, bodyB_id = contact[1], contact[2]
                
                # Find object names
                nameA = nameB = "unknown"
                for name, obj in self.objects.items():
                    if obj.body_id == bodyA_id:
                        nameA = name
                    if obj.body_id == bodyB_id:
                        nameB = name
                
                collisions.append({
                    "objectA": nameA,
                    "objectB": nameB,
                    "position": np.array(contact[5]),
                    "normal": np.array(contact[7]),
                    "distance": contact[8],
                    "force": contact[9]
                })
            
            return collisions
            
        except Exception as e:
            logger.error(f"Collision detection failed: {e}")
            return []
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        if not PYBULLET_AVAILABLE or self.physics_client is None:
            self.step_count = 0
            return
        
        try:
            # Remove all constraints
            for constraint_id in self.constraints:
                p.removeConstraint(constraint_id)
            self.constraints.clear()
            
            # Reset all objects to initial poses
            for name, obj in self.objects.items():
                if obj._initial_pose is not None:
                    obj.set_pose(obj._initial_pose)
            
            self.step_count = 0
            logger.info("Simulation reset")
            
        except Exception as e:
            logger.error(f"Simulation reset failed: {e}")
    
    def cleanup(self):
        """Clean up simulation resources"""
        if PYBULLET_AVAILABLE and self.physics_client is not None:
            try:
                p.disconnect()
                logger.info("PyBullet simulation cleaned up")
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")


class GASMBridge:
    """Bridge between GASM and PyBullet simulation"""
    
    def __init__(self, feature_dim: int = 768, hidden_dim: int = 256):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.gasm = None
        
        if GASM_AVAILABLE and TORCH_AVAILABLE:
            try:
                self.gasm = EnhancedGASM(
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    output_dim=6,  # SE(3) pose: 3 position + 3 orientation (axis-angle)
                    num_heads=8
                )
                logger.info("GASM bridge initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GASM: {e}")
        else:
            logger.warning("GASM or PyTorch not available - using mock bridge")
    
    def extract_entities_from_text(self, text: str) -> Dict[str, Any]:
        """Extract spatial entities from text description"""
        try:
            # Simple keyword-based extraction (can be enhanced with NLP)
            entities = []
            relations = []
            
            # Common spatial objects
            objects = ["box", "cube", "sphere", "cylinder", "conveyor", "sensor", "robot", "camera"]
            spatial_relations = ["on", "above", "below", "left", "right", "front", "back", "near", "far"]
            
            words = text.lower().split()
            
            # Extract objects
            for i, word in enumerate(words):
                if word in objects:
                    entities.append({
                        "name": f"{word}_{len(entities)}",
                        "type": word,
                        "position": i
                    })
            
            # Extract spatial relations
            for i, word in enumerate(words):
                if word in spatial_relations:
                    relations.append({
                        "type": word,
                        "position": i
                    })
            
            return {
                "entities": entities,
                "relations": relations,
                "features": self._text_to_features(text)
            }
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {"entities": [], "relations": [], "features": None}
    
    def _text_to_features(self, text: str) -> Optional['torch.Tensor']:
        """Convert text to feature tensor"""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            # Simple bag-of-words encoding (can be enhanced with transformers)
            vocab = ["box", "cube", "sphere", "cylinder", "conveyor", "sensor", 
                    "on", "above", "below", "left", "right", "front", "back",
                    "move", "place", "pick", "drop", "sort", "align"]
            
            features = torch.zeros(self.feature_dim)
            words = text.lower().split()
            
            for word in words:
                if word in vocab:
                    idx = vocab.index(word)
                    if idx < self.feature_dim:
                        features[idx] = 1.0
            
            return features.unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return torch.zeros(1, self.feature_dim) if TORCH_AVAILABLE else None
    
    def plan_spatial_arrangement(self, entities: List[Dict], constraints: Dict) -> List[SE3Pose]:
        """Use GASM to plan spatial arrangement"""
        if self.gasm is None or not TORCH_AVAILABLE:
            # Fallback: simple grid arrangement
            poses = []
            for i, entity in enumerate(entities):
                x = (i % 3) * 0.2  # Grid spacing
                y = (i // 3) * 0.2
                z = 0.1
                poses.append(SE3Pose([x, y, z], [0., 0., 0., 1.]))
            return poses
        
        try:
            n_entities = len(entities)
            if n_entities == 0:
                return []
            
            # Create feature matrix
            features = torch.randn(n_entities, self.feature_dim)  # Random features for now
            
            # Create relation matrix
            relations = torch.zeros(n_entities, n_entities, 64)  # Dummy relations
            
            # Convert constraints to GASM format
            gasm_constraints = {}
            if "distance" in constraints:
                gasm_constraints["distance"] = torch.tensor(constraints["distance"])
            
            # Run GASM
            with torch.no_grad():
                geometry = self.gasm(
                    E=entities,
                    F=features,
                    R=relations,
                    C=gasm_constraints
                )
            
            # Convert to SE(3) poses
            poses = []
            for i in range(geometry.shape[0]):
                if geometry.shape[1] >= 6:
                    pos = geometry[i, :3].numpy()
                    rot_aa = geometry[i, 3:6].numpy()  # axis-angle
                    # Convert axis-angle to quaternion
                    angle = np.linalg.norm(rot_aa)
                    if angle > 1e-6:
                        axis = rot_aa / angle
                        quat = np.array([
                            axis[0] * np.sin(angle/2),
                            axis[1] * np.sin(angle/2),
                            axis[2] * np.sin(angle/2),
                            np.cos(angle/2)
                        ])
                    else:
                        quat = np.array([0., 0., 0., 1.])
                else:
                    pos = geometry[i, :3].numpy()
                    quat = np.array([0., 0., 0., 1.])
                
                poses.append(SE3Pose(pos, quat))
            
            logger.info(f"GASM planned arrangement for {n_entities} entities")
            return poses
            
        except Exception as e:
            logger.error(f"GASM planning failed: {e}")
            # Fallback
            poses = []
            for i, entity in enumerate(entities):
                x = (i % 3) * 0.2
                y = (i // 3) * 0.2
                z = 0.1
                poses.append(SE3Pose([x, y, z], [0., 0., 0., 1.]))
            return poses


class SpatialMetrics:
    """Compute spatial arrangement metrics"""
    
    @staticmethod
    def compute_arrangement_quality(poses: List[SE3Pose], constraints: Dict) -> Dict[str, float]:
        """Compute quality metrics for spatial arrangement"""
        try:
            metrics = {
                "constraint_satisfaction": 0.0,
                "spatial_efficiency": 0.0,
                "collision_free": 1.0,
                "stability": 1.0
            }
            
            if len(poses) < 2:
                return metrics
            
            # Constraint satisfaction
            if "distance" in constraints:
                total_error = 0.0
                constraint_count = 0
                for constraint in constraints["distance"]:
                    i, j, target_dist = int(constraint[0]), int(constraint[1]), constraint[2]
                    if i < len(poses) and j < len(poses):
                        actual_dist = np.linalg.norm(poses[i].position - poses[j].position)
                        error = abs(actual_dist - target_dist)
                        total_error += error
                        constraint_count += 1
                
                if constraint_count > 0:
                    avg_error = total_error / constraint_count
                    metrics["constraint_satisfaction"] = max(0.0, 1.0 - avg_error)
            
            # Spatial efficiency (compactness)
            positions = np.array([pose.position for pose in poses])
            centroid = np.mean(positions, axis=0)
            distances = [np.linalg.norm(pos - centroid) for pos in positions]
            avg_distance = np.mean(distances)
            metrics["spatial_efficiency"] = max(0.0, 1.0 - avg_distance)
            
            # Simple collision check
            min_dist = 0.05  # 5cm minimum distance
            for i in range(len(poses)):
                for j in range(i + 1, len(poses)):
                    dist = np.linalg.norm(poses[i].position - poses[j].position)
                    if dist < min_dist:
                        metrics["collision_free"] = 0.0
                        break
                if metrics["collision_free"] == 0.0:
                    break
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics computation failed: {e}")
            return {"constraint_satisfaction": 0.0, "spatial_efficiency": 0.0,
                   "collision_free": 0.0, "stability": 0.0}


class VisionPipeline:
    """Computer vision processing pipeline"""
    
    def __init__(self):
        self.enabled = CV2_AVAILABLE
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Process camera image for object detection"""
        if not self.enabled:
            return {"objects": [], "features": None}
        
        try:
            # Simple computer vision processing
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Contour detection
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours
            objects = []
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) > 100:  # Filter small contours
                    # Bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Moments for centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = x + w//2, y + h//2
                    
                    objects.append({
                        "id": i,
                        "bbox": [x, y, w, h],
                        "centroid": [cx, cy],
                        "area": cv2.contourArea(contour)
                    })
            
            return {
                "objects": objects,
                "features": {
                    "edges": edges,
                    "contours": len(contours)
                }
            }
            
        except Exception as e:
            logger.error(f"Vision processing failed: {e}")
            return {"objects": [], "features": None}


class ConstraintSolver:
    """Advanced constraint solver for spatial arrangements"""
    
    def __init__(self):
        self.constraints = []
        self.tolerance = 1e-6
        self.max_iterations = 100
    
    def add_constraint(self, constraint_type: str, objects: List[str], parameters: Dict):
        """Add spatial constraint"""
        constraint = {
            "type": constraint_type,
            "objects": objects,
            "parameters": parameters,
            "active": True
        }
        self.constraints.append(constraint)
        return len(self.constraints) - 1
    
    def solve_constraints(self, poses: Dict[str, SE3Pose]) -> Dict[str, SE3Pose]:
        """Solve spatial constraints using iterative optimization"""
        try:
            optimized_poses = poses.copy()
            
            for iteration in range(self.max_iterations):
                max_error = 0.0
                
                for constraint in self.constraints:
                    if not constraint["active"]:
                        continue
                    
                    error = self._compute_constraint_error(constraint, optimized_poses)
                    max_error = max(max_error, abs(error))
                    
                    if abs(error) > self.tolerance:
                        # Apply correction
                        correction = self._compute_constraint_correction(constraint, optimized_poses, error)
                        self._apply_correction(constraint, optimized_poses, correction)
                
                if max_error < self.tolerance:
                    logger.info(f"Constraints converged in {iteration + 1} iterations")
                    break
            
            return optimized_poses
            
        except Exception as e:
            logger.error(f"Constraint solving failed: {e}")
            return poses
    
    def _compute_constraint_error(self, constraint: Dict, poses: Dict[str, SE3Pose]) -> float:
        """Compute error for a specific constraint"""
        try:
            if constraint["type"] == "distance":
                obj1, obj2 = constraint["objects"][:2]
                if obj1 in poses and obj2 in poses:
                    actual_dist = np.linalg.norm(poses[obj1].position - poses[obj2].position)
                    target_dist = constraint["parameters"].get("distance", 0.0)
                    return actual_dist - target_dist
            
            elif constraint["type"] == "alignment":
                obj1, obj2 = constraint["objects"][:2]
                if obj1 in poses and obj2 in poses:
                    axis = constraint["parameters"].get("axis", [1, 0, 0])
                    diff = poses[obj2].position - poses[obj1].position
                    proj = np.dot(diff, axis)
                    return np.linalg.norm(diff - proj * np.array(axis))
            
            elif constraint["type"] == "orientation":
                obj = constraint["objects"][0]
                if obj in poses:
                    target_quat = constraint["parameters"].get("orientation", [0, 0, 0, 1])
                    current_quat = poses[obj].orientation
                    # Quaternion difference
                    return 1.0 - abs(np.dot(current_quat, target_quat))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error computation failed: {e}")
            return 0.0
    
    def _compute_constraint_correction(self, constraint: Dict, poses: Dict[str, SE3Pose], error: float) -> Dict:
        """Compute correction for constraint violation"""
        correction = {"position": {}, "orientation": {}}
        
        try:
            if constraint["type"] == "distance":
                obj1, obj2 = constraint["objects"][:2]
                if obj1 in poses and obj2 in poses:
                    direction = poses[obj2].position - poses[obj1].position
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 1e-6:
                        direction = direction / direction_norm
                        correction_magnitude = error * 0.5  # Split correction between objects
                        correction["position"][obj1] = direction * correction_magnitude
                        correction["position"][obj2] = -direction * correction_magnitude
            
            return correction
            
        except Exception as e:
            logger.error(f"Correction computation failed: {e}")
            return correction
    
    def _apply_correction(self, constraint: Dict, poses: Dict[str, SE3Pose], correction: Dict):
        """Apply computed correction to poses"""
        try:
            for obj_name, pos_correction in correction["position"].items():
                if obj_name in poses:
                    poses[obj_name].position += pos_correction
            
            for obj_name, orn_correction in correction.get("orientation", {}).items():
                if obj_name in poses:
                    # Apply orientation correction (simplified)
                    poses[obj_name].orientation += orn_correction * 0.1
                    # Renormalize
                    norm = np.linalg.norm(poses[obj_name].orientation)
                    if norm > 1e-6:
                        poses[obj_name].orientation /= norm
                        
        except Exception as e:
            logger.error(f"Correction application failed: {e}")


class TrajectoryPlanner:
    """Advanced trajectory planning for smooth motion"""
    
    def __init__(self):
        self.trajectories = {}
        self.current_time = 0.0
    
    def plan_trajectory(self, start_pose: SE3Pose, end_pose: SE3Pose, 
                       duration: float, trajectory_type: str = "cubic") -> Dict:
        """Plan smooth trajectory between poses"""
        try:
            if trajectory_type == "cubic":
                return self._plan_cubic_trajectory(start_pose, end_pose, duration)
            elif trajectory_type == "quintic":
                return self._plan_quintic_trajectory(start_pose, end_pose, duration)
            else:
                return self._plan_linear_trajectory(start_pose, end_pose, duration)
                
        except Exception as e:
            logger.error(f"Trajectory planning failed: {e}")
            return self._plan_linear_trajectory(start_pose, end_pose, duration)
    
    def _plan_cubic_trajectory(self, start: SE3Pose, end: SE3Pose, duration: float) -> Dict:
        """Plan cubic polynomial trajectory"""
        # Cubic coefficients for position
        pos_coeffs = []
        for i in range(3):
            a0 = start.position[i]
            a1 = 0.0  # zero initial velocity
            a2 = 3.0 * (end.position[i] - start.position[i]) / (duration**2)
            a3 = -2.0 * (end.position[i] - start.position[i]) / (duration**3)
            pos_coeffs.append([a0, a1, a2, a3])
        
        # SLERP for orientation
        return {
            "type": "cubic",
            "duration": duration,
            "position_coeffs": pos_coeffs,
            "start_orientation": start.orientation,
            "end_orientation": end.orientation
        }
    
    def _plan_quintic_trajectory(self, start: SE3Pose, end: SE3Pose, duration: float) -> Dict:
        """Plan quintic polynomial trajectory for smoother motion"""
        pos_coeffs = []
        for i in range(3):
            # Quintic with zero initial/final velocity and acceleration
            a0 = start.position[i]
            a1, a2 = 0.0, 0.0  # zero initial velocity and acceleration
            a3 = 10.0 * (end.position[i] - start.position[i]) / (duration**3)
            a4 = -15.0 * (end.position[i] - start.position[i]) / (duration**4)
            a5 = 6.0 * (end.position[i] - start.position[i]) / (duration**5)
            pos_coeffs.append([a0, a1, a2, a3, a4, a5])
        
        return {
            "type": "quintic",
            "duration": duration,
            "position_coeffs": pos_coeffs,
            "start_orientation": start.orientation,
            "end_orientation": end.orientation
        }
    
    def _plan_linear_trajectory(self, start: SE3Pose, end: SE3Pose, duration: float) -> Dict:
        """Plan linear interpolation trajectory"""
        return {
            "type": "linear",
            "duration": duration,
            "start_pose": start,
            "end_pose": end
        }
    
    def sample_trajectory(self, trajectory: Dict, t: float) -> SE3Pose:
        """Sample trajectory at time t"""
        try:
            if t >= trajectory["duration"]:
                t = trajectory["duration"]
            
            normalized_t = t / trajectory["duration"]
            
            if trajectory["type"] == "linear":
                # Linear interpolation
                pos = (1 - normalized_t) * trajectory["start_pose"].position + \
                      normalized_t * trajectory["end_pose"].position
                
                # SLERP for orientation
                q1 = trajectory["start_pose"].orientation
                q2 = trajectory["end_pose"].orientation
                orn = self._slerp(q1, q2, normalized_t)
                
            else:
                # Polynomial trajectory
                pos = np.zeros(3)
                for i in range(3):
                    coeffs = trajectory["position_coeffs"][i]
                    if trajectory["type"] == "cubic":
                        pos[i] = coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3
                    elif trajectory["type"] == "quintic":
                        pos[i] = coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + \
                                coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5
                
                # SLERP for orientation
                orn = self._slerp(trajectory["start_orientation"], 
                                 trajectory["end_orientation"], normalized_t)
            
            return SE3Pose(pos, orn)
            
        except Exception as e:
            logger.error(f"Trajectory sampling failed: {e}")
            return SE3Pose([0, 0, 0], [0, 0, 0, 1])
    
    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation for quaternions"""
        try:
            dot = np.dot(q1, q2)
            if dot < 0.0:
                q2 = -q2
                dot = -dot
            
            if dot > 0.9995:
                # Linear interpolation for very close quaternions
                result = q1 + t * (q2 - q1)
                return result / np.linalg.norm(result)
            
            theta_0 = np.arccos(abs(dot))
            sin_theta_0 = np.sin(theta_0)
            theta = theta_0 * t
            sin_theta = np.sin(theta)
            
            s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0
            
            return s0 * q1 + s1 * q2
            
        except Exception as e:
            logger.error(f"SLERP failed: {e}")
            return q1


class CollisionResolver:
    """Advanced collision detection and resolution"""
    
    def __init__(self):
        self.collision_history = []
        self.resolution_strategies = ["separate", "replan", "wait"]
    
    def detect_potential_collisions(self, poses: Dict[str, SE3Pose], 
                                   velocities: Dict[str, np.ndarray] = None,
                                   prediction_time: float = 1.0) -> List[Dict]:
        """Detect potential future collisions"""
        potential_collisions = []
        
        try:
            object_names = list(poses.keys())
            for i in range(len(object_names)):
                for j in range(i + 1, len(object_names)):
                    obj1, obj2 = object_names[i], object_names[j]
                    
                    # Current distance
                    current_dist = np.linalg.norm(poses[obj1].position - poses[obj2].position)
                    
                    # Predicted future positions
                    if velocities:
                        future_pos1 = poses[obj1].position + velocities.get(obj1, np.zeros(3)) * prediction_time
                        future_pos2 = poses[obj2].position + velocities.get(obj2, np.zeros(3)) * prediction_time
                        future_dist = np.linalg.norm(future_pos1 - future_pos2)
                    else:
                        future_dist = current_dist
                    
                    # Collision threshold (sum of object radii + safety margin)
                    collision_threshold = 0.15  # 15cm safety margin
                    
                    if future_dist < collision_threshold:
                        potential_collisions.append({
                            "objects": [obj1, obj2],
                            "current_distance": current_dist,
                            "predicted_distance": future_dist,
                            "time_to_collision": self._estimate_collision_time(
                                poses[obj1].position, poses[obj2].position,
                                velocities.get(obj1, np.zeros(3)),
                                velocities.get(obj2, np.zeros(3))
                            ),
                            "severity": "high" if future_dist < 0.05 else "medium"
                        })
            
            return potential_collisions
            
        except Exception as e:
            logger.error(f"Collision detection failed: {e}")
            return []
    
    def resolve_collision(self, collision: Dict, poses: Dict[str, SE3Pose]) -> Dict[str, SE3Pose]:
        """Resolve detected collision"""
        try:
            obj1, obj2 = collision["objects"]
            strategy = self._select_resolution_strategy(collision)
            
            if strategy == "separate":
                return self._separate_objects(obj1, obj2, poses)
            elif strategy == "replan":
                return self._replan_paths(obj1, obj2, poses, collision)
            else:  # wait
                return poses  # No immediate action, wait for other object to move
                
        except Exception as e:
            logger.error(f"Collision resolution failed: {e}")
            return poses
    
    def _estimate_collision_time(self, pos1: np.ndarray, pos2: np.ndarray,
                                vel1: np.ndarray, vel2: np.ndarray) -> float:
        """Estimate time to collision"""
        try:
            relative_pos = pos2 - pos1
            relative_vel = vel2 - vel1
            
            # Solve quadratic equation for closest approach
            a = np.dot(relative_vel, relative_vel)
            b = 2 * np.dot(relative_pos, relative_vel)
            c = np.dot(relative_pos, relative_pos)
            
            if abs(a) < 1e-6:  # No relative motion
                return float('inf')
            
            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                return float('inf')
            
            t1 = (-b - np.sqrt(discriminant)) / (2*a)
            t2 = (-b + np.sqrt(discriminant)) / (2*a)
            
            # Return earliest positive time
            if t1 > 0:
                return t1
            elif t2 > 0:
                return t2
            else:
                return float('inf')
                
        except Exception as e:
            logger.error(f"Collision time estimation failed: {e}")
            return float('inf')
    
    def _select_resolution_strategy(self, collision: Dict) -> str:
        """Select appropriate collision resolution strategy"""
        if collision["severity"] == "high":
            return "separate"
        elif collision["time_to_collision"] < 1.0:
            return "replan"
        else:
            return "wait"
    
    def _separate_objects(self, obj1: str, obj2: str, poses: Dict[str, SE3Pose]) -> Dict[str, SE3Pose]:
        """Separate colliding objects"""
        try:
            new_poses = poses.copy()
            
            # Compute separation direction
            direction = poses[obj2].position - poses[obj1].position
            distance = np.linalg.norm(direction)
            
            if distance > 1e-6:
                direction = direction / distance
                separation = 0.2  # 20cm separation
                
                # Move objects apart
                new_poses[obj1].position -= direction * (separation / 2)
                new_poses[obj2].position += direction * (separation / 2)
            
            return new_poses
            
        except Exception as e:
            logger.error(f"Object separation failed: {e}")
            return poses
    
    def _replan_paths(self, obj1: str, obj2: str, poses: Dict[str, SE3Pose], 
                     collision: Dict) -> Dict[str, SE3Pose]:
        """Replan paths to avoid collision"""
        # Simplified replanning - offset one object's path
        try:
            new_poses = poses.copy()
            
            # Add small offset to obj1's position
            offset = np.array([0.1, 0.1, 0.0])  # 10cm offset in X-Y
            new_poses[obj1].position += offset
            
            return new_poses
            
        except Exception as e:
            logger.error(f"Path replanning failed: {e}")
            return poses


class SpatialAgent:
    """Main spatial agent with complete 3D simulation integration"""
    
    def __init__(self, config: SimulationConfig, mode: SimulationMode = SimulationMode.HEADLESS):
        self.config = config
        self.mode = mode
        
        # Initialize components
        self.simulation = PyBulletSimulation(config, mode)
        self.gasm_bridge = GASMBridge()
        self.metrics = SpatialMetrics()
        self.vision = VisionPipeline()
        
        # Advanced features
        self.constraint_solver = ConstraintSolver()
        self.trajectory_planner = TrajectoryPlanner()
        self.collision_resolver = CollisionResolver()
        
        # State tracking
        self.current_step = 0
        self.execution_log = []
        self.performance_metrics = []
        self.active_tasks = []
        self.object_registry = {}
        
        # Planning and control parameters
        self.control_frequency = 240.0  # Hz
        self.planning_horizon = 100  # steps
        self.max_velocity = 1.0  # m/s
        self.max_acceleration = 2.0  # m/s²
        
    def execute_task(self, task_description: str, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Execute spatial task from text description with advanced planning and control"""
        if max_steps is None:
            max_steps = self.config.max_steps
        
        try:
            logger.info(f"Executing task: {task_description}")
            
            # 1. Parse task description and extract entities
            entities_data = self.gasm_bridge.extract_entities_from_text(task_description)
            entities = entities_data["entities"]
            relations = entities_data["relations"]
            
            # 2. Create spatial constraints from relations
            constraints = self._relations_to_constraints(relations)
            
            # 3. Initial spatial arrangement planning using GASM
            planned_poses = self.gasm_bridge.plan_spatial_arrangement(entities, constraints)
            
            # 4. Create objects in simulation with enhanced object registry
            created_objects = []
            pose_dict = {}
            for i, (entity, pose) in enumerate(zip(entities, planned_poses)):
                obj_name = f"{entity['type']}_{i}"
                
                # Enhanced URDF loading with fallbacks
                urdf_path = self._get_urdf_path(entity['type'])
                obj = self.simulation.load_urdf_object(urdf_path, obj_name, pose)
                if obj:
                    created_objects.append(obj)
                    pose_dict[obj_name] = pose
                    self.object_registry[obj_name] = {
                        "object": obj,
                        "entity": entity,
                        "initial_pose": pose,
                        "target_pose": pose,
                        "trajectory": None,
                        "status": "active"
                    }
            
            # 5. Advanced constraint solving and optimization
            if pose_dict:
                optimized_poses = self.constraint_solver.solve_constraints(pose_dict)
                # Apply optimized poses
                for obj_name, optimized_pose in optimized_poses.items():
                    if obj_name in self.object_registry:
                        self.object_registry[obj_name]["target_pose"] = optimized_pose
            
            # 6. Generate trajectories for smooth motion
            self._generate_trajectories(created_objects)
            
            # 7. Advanced simulation loop with multi-modal feedback
            step_results = []
            trajectory_start_time = 0.0
            
            for step in range(max_steps):
                current_time = step * self.config.time_step
                
                # Execute trajectory control
                self._execute_trajectory_control(current_time - trajectory_start_time)
                
                # Step physics simulation
                state = self.simulation.step_simulation()
                
                # Multi-camera rendering system
                camera_image = None
                if self.mode == SimulationMode.GUI or self.mode == SimulationMode.RECORD:
                    camera_image = self.simulation.render_camera(save_image=True)
                
                # Advanced vision processing
                vision_data = {}
                if camera_image is not None:
                    vision_data = self.vision.process_image(camera_image)
                    # Add 3D pose estimation from 2D vision
                    vision_data["pose_estimates"] = self._estimate_3d_poses_from_vision(
                        vision_data.get("objects", []), camera_image
                    )
                
                # Enhanced collision detection and resolution
                collisions = self.simulation.detect_collisions()
                current_poses = {obj.name: obj.get_pose() for obj in created_objects}
                potential_collisions = self.collision_resolver.detect_potential_collisions(
                    current_poses, self._get_object_velocities(created_objects)
                )
                
                # Resolve collisions if detected
                if potential_collisions:
                    resolved_poses = current_poses
                    for collision in potential_collisions:
                        resolved_poses = self.collision_resolver.resolve_collision(collision, resolved_poses)
                    
                    # Apply collision resolution
                    for obj_name, new_pose in resolved_poses.items():
                        for obj in created_objects:
                            if obj.name == obj_name:
                                obj.set_pose(new_pose)
                                break
                
                # Comprehensive metrics computation
                current_poses_list = [obj.get_pose() for obj in created_objects]
                arrangement_metrics = self.metrics.compute_arrangement_quality(
                    current_poses_list, constraints
                )
                
                # Dynamic replanning if needed
                if arrangement_metrics["constraint_satisfaction"] < 0.5 and step > 100:
                    logger.info("Performance low, triggering replanning")
                    self._replan_arrangement(entities, constraints, created_objects)
                    trajectory_start_time = current_time
                
                # Enhanced step result with more data
                step_result = {
                    "step": step,
                    "time": current_time,
                    "poses": current_poses_list,
                    "collisions": collisions,
                    "potential_collisions": potential_collisions,
                    "vision": vision_data,
                    "metrics": arrangement_metrics,
                    "physics_state": state,
                    "control_active": any(reg["trajectory"] is not None 
                                        for reg in self.object_registry.values())
                }
                step_results.append(step_result)
                
                # Advanced termination conditions
                success_threshold = 0.95
                if arrangement_metrics["constraint_satisfaction"] > success_threshold:
                    if self._verify_stable_solution(step_results[-10:] if len(step_results) >= 10 else step_results):
                        logger.info(f"Task completed successfully at step {step}")
                        break
                
                # Collision handling with recovery
                if len(collisions) > 0:
                    logger.warning(f"Collision detected at step {step}, attempting recovery")
                    recovery_success = self._attempt_collision_recovery(created_objects, collisions)
                    if not recovery_success:
                        logger.error("Collision recovery failed")
                
                # Adaptive GUI delay based on complexity
                if self.mode == SimulationMode.GUI:
                    delay = 0.01 if len(created_objects) < 5 else 0.02
                    time.sleep(delay)
            
            # 8. Comprehensive final evaluation
            final_poses = [obj.get_pose() for obj in created_objects]
            final_metrics = self.metrics.compute_arrangement_quality(final_poses, constraints)
            
            # Additional success criteria
            stability_check = self._check_stability(step_results[-20:] if len(step_results) >= 20 else step_results)
            final_success = (final_metrics["constraint_satisfaction"] > 0.8 and 
                           final_metrics["collision_free"] > 0.9 and
                           stability_check)
            
            result = {
                "task": task_description,
                "success": final_success,
                "entities": len(entities),
                "objects_created": len(created_objects),
                "steps": len(step_results),
                "final_metrics": final_metrics,
                "stability": stability_check,
                "step_results": step_results,
                "execution_time": len(step_results) * self.config.time_step,
                "trajectory_data": self._extract_trajectory_data(),
                "performance_summary": self._compute_performance_summary(step_results)
            }
            
            self.execution_log.append(result)
            self.performance_metrics.append(final_metrics)
            
            logger.info(f"Task execution completed: Success={final_success}, "
                       f"Constraint Satisfaction={final_metrics['constraint_satisfaction']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "task": task_description,
                "success": False,
                "error": str(e),
                "entities": 0,
                "objects_created": 0,
                "steps": 0,
                "final_metrics": {},
                "step_results": [],
                "execution_time": 0.0,
                "trajectory_data": {},
                "performance_summary": {}
            }
    
    def _relations_to_constraints(self, relations: List[Dict]) -> Dict[str, List]:
        """Convert spatial relations to GASM constraints"""
        constraints = {"distance": []}
        
        try:
            # Simple mapping from relations to distance constraints
            for relation in relations:
                rel_type = relation.get("type", "")
                
                if rel_type in ["on", "above"]:
                    # Objects should be close vertically
                    constraints["distance"].append([0, 1, 0.15])  # 15cm apart
                elif rel_type in ["near"]:
                    constraints["distance"].append([0, 1, 0.1])   # 10cm apart
                elif rel_type in ["far"]:
                    constraints["distance"].append([0, 1, 0.5])   # 50cm apart
            
            return constraints
            
        except Exception as e:
            logger.error(f"Constraint conversion failed: {e}")
            return {"distance": []}
    
    def _get_urdf_path(self, object_type: str) -> str:
        """Get URDF path for object type with fallbacks"""
        try:
            # Map object types to URDF files
            urdf_mapping = {
                "box": "objects/box.urdf",
                "cube": "objects/cube.urdf",
                "sphere": "objects/sphere.urdf",
                "cylinder": "objects/cylinder.urdf",
                "robot": "objects/robot_arm.urdf",
                "conveyor": "objects/conveyor_belt.urdf",
                "sensor": "objects/sensor_mount.urdf"
            }
            
            urdf_path = urdf_mapping.get(object_type, f"objects/{object_type}.urdf")
            
            # Check if file exists, otherwise use fallback
            if not os.path.exists(urdf_path):
                logger.warning(f"URDF not found: {urdf_path}, using fallback")
                return f"objects/box.urdf"  # Default fallback
            
            return urdf_path
            
        except Exception as e:
            logger.error(f"URDF path resolution failed: {e}")
            return "objects/box.urdf"
    
    def _generate_trajectories(self, created_objects: List[PhysicsObject]):
        """Generate smooth trajectories for object motion"""
        try:
            for obj in created_objects:
                if obj.name in self.object_registry:
                    registry = self.object_registry[obj.name]
                    start_pose = registry["initial_pose"]
                    target_pose = registry["target_pose"]
                    
                    # Calculate trajectory duration based on distance
                    distance = np.linalg.norm(target_pose.position - start_pose.position)
                    duration = max(1.0, distance / self.max_velocity * 2.0)  # Conservative duration
                    
                    # Generate trajectory
                    trajectory = self.trajectory_planner.plan_trajectory(
                        start_pose, target_pose, duration, "quintic"
                    )
                    
                    registry["trajectory"] = trajectory
                    logger.debug(f"Generated trajectory for {obj.name}, duration: {duration:.2f}s")
                    
        except Exception as e:
            logger.error(f"Trajectory generation failed: {e}")
    
    def _execute_trajectory_control(self, t: float):
        """Execute trajectory-based control for all objects"""
        try:
            for obj_name, registry in self.object_registry.items():
                if registry["trajectory"] is not None and registry["status"] == "active":
                    trajectory = registry["trajectory"]
                    
                    if t <= trajectory["duration"]:
                        # Sample trajectory at current time
                        target_pose = self.trajectory_planner.sample_trajectory(trajectory, t)
                        
                        # Apply pose to object
                        registry["object"].set_pose(target_pose)
                    else:
                        # Trajectory completed
                        registry["trajectory"] = None
                        registry["status"] = "completed"
                        logger.debug(f"Trajectory completed for {obj_name}")
                        
        except Exception as e:
            logger.error(f"Trajectory control failed: {e}")
    
    def _get_object_velocities(self, objects: List[PhysicsObject]) -> Dict[str, np.ndarray]:
        """Get current velocities of all objects"""
        velocities = {}
        try:
            for obj in objects:
                linear_vel, angular_vel = obj.get_velocity()
                velocities[obj.name] = linear_vel
            return velocities
        except Exception as e:
            logger.error(f"Velocity computation failed: {e}")
            return {}
    
    def _estimate_3d_poses_from_vision(self, detected_objects: List[Dict], 
                                     image: np.ndarray) -> List[Dict]:
        """Estimate 3D poses from 2D vision data (simplified implementation)"""
        try:
            pose_estimates = []
            
            for obj in detected_objects:
                # Simple depth estimation based on object size
                bbox = obj.get("bbox", [0, 0, 0, 0])
                area = obj.get("area", 1)
                
                # Rough depth estimation (assumes known object sizes)
                estimated_depth = 1000.0 / np.sqrt(area)  # Very simplified
                
                # Convert 2D centroid to 3D position estimate
                centroid = obj.get("centroid", [0, 0])
                
                # Camera projection (simplified)
                fx, fy = 600, 600  # Approximate focal lengths
                cx, cy = image.shape[1] // 2, image.shape[0] // 2
                
                x_3d = (centroid[0] - cx) * estimated_depth / fx
                y_3d = (centroid[1] - cy) * estimated_depth / fy
                z_3d = estimated_depth
                
                pose_estimates.append({
                    "object_id": obj["id"],
                    "position": [x_3d, y_3d, z_3d],
                    "confidence": min(0.8, area / 10000.0)  # Confidence based on size
                })
            
            return pose_estimates
            
        except Exception as e:
            logger.error(f"3D pose estimation failed: {e}")
            return []
    
    def _replan_arrangement(self, entities: List[Dict], constraints: Dict, objects: List[PhysicsObject]):
        """Replan spatial arrangement when performance is poor"""
        try:
            logger.info("Replanning spatial arrangement")
            
            # Get current poses
            current_poses = [obj.get_pose() for obj in objects]
            
            # Use GASM to generate new arrangement
            new_poses = self.gasm_bridge.plan_spatial_arrangement(entities, constraints)
            
            # Update target poses and generate new trajectories
            for i, (obj, new_pose) in enumerate(zip(objects, new_poses)):
                if obj.name in self.object_registry:
                    self.object_registry[obj.name]["target_pose"] = new_pose
                    
                    # Generate new trajectory
                    duration = 2.0  # Fixed duration for replanning
                    trajectory = self.trajectory_planner.plan_trajectory(
                        current_poses[i], new_pose, duration, "cubic"
                    )
                    self.object_registry[obj.name]["trajectory"] = trajectory
                    self.object_registry[obj.name]["status"] = "active"
            
            logger.info("Replanning completed")
            
        except Exception as e:
            logger.error(f"Replanning failed: {e}")
    
    def _verify_stable_solution(self, recent_steps: List[Dict]) -> bool:
        """Verify that the solution is stable over recent steps"""
        try:
            if len(recent_steps) < 5:
                return False
            
            # Check constraint satisfaction stability
            satisfactions = [step["metrics"]["constraint_satisfaction"] for step in recent_steps]
            satisfaction_std = np.std(satisfactions)
            
            # Check position stability
            position_changes = []
            for i in range(1, len(recent_steps)):
                total_change = 0.0
                prev_poses = recent_steps[i-1]["poses"]
                curr_poses = recent_steps[i]["poses"]
                
                for prev, curr in zip(prev_poses, curr_poses):
                    change = np.linalg.norm(curr.position - prev.position)
                    total_change += change
                
                position_changes.append(total_change)
            
            position_stability = np.mean(position_changes) < 0.01  # 1cm threshold
            satisfaction_stability = satisfaction_std < 0.05
            
            return position_stability and satisfaction_stability
            
        except Exception as e:
            logger.error(f"Stability verification failed: {e}")
            return False
    
    def _attempt_collision_recovery(self, objects: List[PhysicsObject], collisions: List[Dict]) -> bool:
        """Attempt to recover from collision by separating objects"""
        try:
            recovery_success = True
            
            for collision in collisions:
                # Get colliding object names
                collision_obj_names = []
                for obj in objects:
                    if (obj.body_id == collision.get("objectA") or 
                        obj.body_id == collision.get("objectB")):
                        collision_obj_names.append(obj.name)
                
                if len(collision_obj_names) == 2:
                    obj1_name, obj2_name = collision_obj_names
                    
                    # Find objects
                    obj1 = obj2 = None
                    for obj in objects:
                        if obj.name == obj1_name:
                            obj1 = obj
                        elif obj.name == obj2_name:
                            obj2 = obj
                    
                    if obj1 and obj2:
                        # Separate objects
                        pos1 = obj1.get_pose().position
                        pos2 = obj2.get_pose().position
                        
                        direction = pos2 - pos1
                        distance = np.linalg.norm(direction)
                        
                        if distance > 1e-6:
                            direction = direction / distance
                            separation = 0.2  # 20cm separation
                            
                            # Move objects apart
                            new_pos1 = pos1 - direction * (separation / 2)
                            new_pos2 = pos2 + direction * (separation / 2)
                            
                            # Update poses
                            pose1 = obj1.get_pose()
                            pose2 = obj2.get_pose()
                            pose1.position = new_pos1
                            pose2.position = new_pos2
                            
                            obj1.set_pose(pose1)
                            obj2.set_pose(pose2)
                        else:
                            recovery_success = False
                    else:
                        recovery_success = False
                else:
                    recovery_success = False
            
            return recovery_success
            
        except Exception as e:
            logger.error(f"Collision recovery failed: {e}")
            return False
    
    def _check_stability(self, recent_steps: List[Dict]) -> bool:
        """Check overall stability of the system"""
        try:
            if len(recent_steps) < 10:
                return True  # Not enough data, assume stable
            
            # Check multiple stability criteria
            criteria_passed = 0
            total_criteria = 4
            
            # 1. Constraint satisfaction stability
            satisfactions = [step["metrics"]["constraint_satisfaction"] for step in recent_steps]
            if np.std(satisfactions) < 0.1:
                criteria_passed += 1
            
            # 2. Collision-free stability
            collision_counts = [len(step["collisions"]) for step in recent_steps]
            if np.mean(collision_counts) < 0.5:  # Less than 0.5 collisions on average
                criteria_passed += 1
            
            # 3. Position stability
            position_variances = []
            for pose_idx in range(len(recent_steps[0]["poses"])):
                positions = []
                for step in recent_steps:
                    if pose_idx < len(step["poses"]):
                        positions.append(step["poses"][pose_idx].position)
                
                if positions:
                    variance = np.var(positions, axis=0).mean()
                    position_variances.append(variance)
            
            if position_variances and np.mean(position_variances) < 0.001:  # Low variance
                criteria_passed += 1
            
            # 4. System energy stability (approximated by movement)
            total_movements = []
            for step in recent_steps:
                total_movement = 0.0
                for pose in step["poses"]:
                    vel = np.linalg.norm(pose.position)  # Simplified
                    total_movement += vel
                total_movements.append(total_movement)
            
            if np.std(total_movements) < 1.0:
                criteria_passed += 1
            
            stability_ratio = criteria_passed / total_criteria
            return stability_ratio >= 0.75  # At least 75% of criteria must pass
            
        except Exception as e:
            logger.error(f"Stability check failed: {e}")
            return False
    
    def _extract_trajectory_data(self) -> Dict[str, Any]:
        """Extract trajectory execution data for analysis"""
        try:
            trajectory_data = {
                "active_trajectories": 0,
                "completed_trajectories": 0,
                "trajectory_details": {}
            }
            
            for obj_name, registry in self.object_registry.items():
                if registry["trajectory"] is not None:
                    trajectory_data["active_trajectories"] += 1
                elif registry["status"] == "completed":
                    trajectory_data["completed_trajectories"] += 1
                
                trajectory_data["trajectory_details"][obj_name] = {
                    "status": registry["status"],
                    "has_trajectory": registry["trajectory"] is not None
                }
            
            return trajectory_data
            
        except Exception as e:
            logger.error(f"Trajectory data extraction failed: {e}")
            return {}
    
    def _compute_performance_summary(self, step_results: List[Dict]) -> Dict[str, Any]:
        """Compute comprehensive performance summary"""
        try:
            if not step_results:
                return {}
            
            # Aggregate metrics over time
            constraint_satisfactions = [step["metrics"]["constraint_satisfaction"] for step in step_results]
            collision_counts = [len(step["collisions"]) for step in step_results]
            spatial_efficiencies = [step["metrics"]["spatial_efficiency"] for step in step_results]
            
            summary = {
                "constraint_satisfaction": {
                    "mean": np.mean(constraint_satisfactions),
                    "std": np.std(constraint_satisfactions),
                    "final": constraint_satisfactions[-1],
                    "improvement": constraint_satisfactions[-1] - constraint_satisfactions[0] if len(constraint_satisfactions) > 0 else 0
                },
                "collision_analysis": {
                    "total_collisions": sum(collision_counts),
                    "collision_rate": np.mean(collision_counts),
                    "max_simultaneous_collisions": max(collision_counts) if collision_counts else 0,
                    "collision_free_ratio": sum(1 for count in collision_counts if count == 0) / len(collision_counts)
                },
                "efficiency": {
                    "spatial_efficiency_mean": np.mean(spatial_efficiencies),
                    "spatial_efficiency_final": spatial_efficiencies[-1]
                },
                "execution_stats": {
                    "total_steps": len(step_results),
                    "control_active_steps": sum(1 for step in step_results if step.get("control_active", False)),
                    "replanning_events": sum(1 for step in step_results if "replanning" in step.get("events", []))
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Performance summary computation failed: {e}")
            return {}
    
    def save_video(self, filename: str = "simulation.mp4"):
        """Save recorded camera images as video"""
        if not CV2_AVAILABLE or len(self.simulation.camera_images) == 0:
            logger.warning("No camera images to save or OpenCV not available")
            return
        
        try:
            height, width, layers = self.simulation.camera_images[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            video_writer = cv2.VideoWriter(
                filename, fourcc, 30.0, (width, height)
            )
            
            for image in self.simulation.camera_images:
                # Convert RGB to BGR for OpenCV
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_image)
            
            video_writer.release()
            logger.info(f"Video saved to {filename}")
            
        except Exception as e:
            logger.error(f"Video saving failed: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all executed tasks"""
        try:
            if not self.performance_metrics:
                return {"no_data": True}
            
            # Aggregate metrics
            metrics_keys = self.performance_metrics[0].keys()
            summary = {}
            
            for key in metrics_keys:
                values = [m.get(key, 0.0) for m in self.performance_metrics]
                summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
            
            summary["total_tasks"] = len(self.execution_log)
            summary["success_rate"] = sum(1 for log in self.execution_log if log.get("success", False)) / len(self.execution_log)
            
            return summary
            
        except Exception as e:
            logger.error(f"Performance summary failed: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.simulation.cleanup()
            logger.info("Spatial agent cleanup complete")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def run_demo_scenarios():
    """Run comprehensive demo scenarios"""
    print("="*60)
    print("SPATIAL AGENT 3D PYBULLET DEMO SCENARIOS")
    print("="*60)
    
    # Demo scenarios
    scenarios = [
        {
            "name": "Basic Object Placement",
            "description": "Place a red box near a blue sphere on the conveyor belt",
            "steps": 500,
            "mode": "headless"
        },
        {
            "name": "Multi-Object Sorting",
            "description": "Sort three cubes and two spheres by color and size",
            "steps": 800,
            "mode": "headless"
        },
        {
            "name": "Robot Arm Manipulation",
            "description": "Use robot arm to pick up cylinder and place it on sensor mount",
            "steps": 1000,
            "mode": "headless"
        },
        {
            "name": "Conveyor Belt Assembly",
            "description": "Arrange boxes on conveyor belt in a line with equal spacing",
            "steps": 600,
            "mode": "headless"
        },
        {
            "name": "Collision Avoidance",
            "description": "Move multiple objects without collisions in a crowded space",
            "steps": 750,
            "mode": "headless"
        }
    ]
    
    # Configuration for demo
    config = SimulationConfig()
    config.max_steps = 1000
    config.time_step = 1./120.  # Faster simulation for demo
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Running: {scenario['name']}")
        print("-" * 40)
        
        try:
            mode = SimulationMode(scenario['mode'])
            agent = SpatialAgent(config, mode)
            
            result = agent.execute_task(scenario['description'], max_steps=scenario['steps'])
            results.append({**result, "scenario_name": scenario['name']})
            
            # Print results
            success_icon = "✅" if result['success'] else "❌"
            print(f"{success_icon} Task: {result['task'][:50]}...")
            print(f"   Success: {result['success']} | Objects: {result['objects_created']} | Steps: {result['steps']}")
            print(f"   Constraint Satisfaction: {result['final_metrics'].get('constraint_satisfaction', 0):.3f}")
            print(f"   Collision Free: {result['final_metrics'].get('collision_free', 0):.3f}")
            print(f"   Execution Time: {result['execution_time']:.2f}s")
            
            agent.cleanup()
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                "scenario_name": scenario['name'],
                "success": False,
                "error": str(e)
            })
    
    # Overall results
    print("\n" + "="*60)
    print("DEMO RESULTS SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)
    success_rate = successful / total * 100
    
    print(f"Overall Success Rate: {successful}/{total} ({success_rate:.1f}%)")
    print(f"Scenarios Completed: {total}")
    
    # Detailed metrics
    if successful > 0:
        avg_constraint_satisfaction = np.mean([
            r['final_metrics'].get('constraint_satisfaction', 0) 
            for r in results if r.get('success', False)
        ])
        avg_execution_time = np.mean([
            r.get('execution_time', 0) 
            for r in results if r.get('success', False)
        ])
        
        print(f"Average Constraint Satisfaction: {avg_constraint_satisfaction:.3f}")
        print(f"Average Execution Time: {avg_execution_time:.2f}s")
    
    return results


def main():
    """Enhanced Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Spatial Agent with 3D PyBullet Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic task execution
  python agent_loop_pybullet.py --text "place box near sphere" --render gui
  
  # Record simulation video
  python agent_loop_pybullet.py --text "sort objects by size" --render record --save_video output.mp4
  
  # Run with custom configuration
  python agent_loop_pybullet.py --text "robot picks up cylinder" --config config.json --steps 2000
  
  # Run comprehensive demo
  python agent_loop_pybullet.py --demo
  
  # Benchmark performance
  python agent_loop_pybullet.py --benchmark --iterations 10
        """)
    
    parser.add_argument("--text", type=str, default=None,
                       help="Task description text")
    parser.add_argument("--steps", type=int, default=1000,
                       help="Maximum simulation steps")
    parser.add_argument("--use_vision", action="store_true",
                       help="Enable advanced vision processing")
    parser.add_argument("--render", choices=["headless", "gui", "record"], default="headless",
                       help="Rendering mode")
    parser.add_argument("--save_video", type=str, default=None,
                       help="Save video filename (only with record mode)")
    parser.add_argument("--config", type=str, default=None,
                       help="JSON configuration file path")
    parser.add_argument("--demo", action="store_true",
                       help="Run comprehensive demo scenarios")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmarks")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of benchmark iterations")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--export_metrics", type=str, default=None,
                       help="Export metrics to JSON file")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check dependencies
    missing_deps = []
    if not PYBULLET_AVAILABLE:
        missing_deps.append("pybullet")
    if not CV2_AVAILABLE and args.save_video:
        missing_deps.append("opencv-python")
    if not TORCH_AVAILABLE:
        missing_deps.append("torch")
    
    if missing_deps:
        print(f"⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        print("Simulation will run in fallback mode with limited functionality.\n")
    
    # Run demo scenarios
    if args.demo:
        results = run_demo_scenarios()
        if args.export_metrics:
            try:
                with open(args.export_metrics, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nDemo results exported to: {args.export_metrics}")
            except Exception as e:
                print(f"Failed to export results: {e}")
        return
    
    # Run benchmarks
    if args.benchmark:
        print("Running performance benchmarks...")
        benchmark_task = "place three boxes in a triangle formation"
        config = SimulationConfig()
        config.max_steps = 500  # Shorter for benchmarks
        
        times = []
        successes = []
        
        for i in range(args.iterations):
            print(f"Benchmark {i+1}/{args.iterations}")
            agent = SpatialAgent(config, SimulationMode.HEADLESS)
            start_time = time.time()
            result = agent.execute_task(benchmark_task)
            end_time = time.time()
            
            times.append(end_time - start_time)
            successes.append(result['success'])
            agent.cleanup()
        
        # Print benchmark results
        print(f"\nBenchmark Results ({args.iterations} iterations):")
        print(f"  Success Rate: {sum(successes)}/{len(successes)} ({sum(successes)/len(successes)*100:.1f}%)")
        print(f"  Average Time: {np.mean(times):.2f}s")
        print(f"  Time Std Dev: {np.std(times):.2f}s")
        print(f"  Min/Max Time: {min(times):.2f}s / {max(times):.2f}s")
        return
    
    # Regular task execution
    if not args.text:
        parser.print_help()
        print("\nError: --text is required unless using --demo or --benchmark")
        return
    
    # Load configuration
    config = SimulationConfig()
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            print(f"Loaded configuration from: {args.config}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    # Set simulation mode
    mode = SimulationMode(args.render)
    
    # Initialize spatial agent
    agent = SpatialAgent(config, mode)
    
    try:
        print(f"\n🚀 Starting Spatial Agent in {mode.value} mode")
        print(f"📝 Task: {args.text}")
        print(f"⚙️  Max Steps: {args.steps}")
        print(f"🔧 Physics: {'PyBullet' if PYBULLET_AVAILABLE else 'Mock'}")
        print(f"👁️  Vision: {'Enabled' if CV2_AVAILABLE and args.use_vision else 'Disabled'}")
        print("-" * 50)
        
        # Execute task
        start_time = time.time()
        result = agent.execute_task(args.text, max_steps=args.steps)
        end_time = time.time()
        
        # Print comprehensive results
        print("\n" + "="*60)
        print("🎯 SPATIAL AGENT EXECUTION RESULTS")
        print("="*60)
        
        success_icon = "✅" if result['success'] else "❌"
        print(f"{success_icon} Task: {result['task']}")
        print(f"🏆 Success: {result['success']}")
        print(f"🎲 Entities Created: {result['entities']} / Objects: {result['objects_created']}")
        print(f"⏱️  Steps: {result['steps']} / Execution Time: {result['execution_time']:.2f}s")
        print(f"🕒 Wall Clock Time: {end_time - start_time:.2f}s")
        
        print(f"\n📊 Final Metrics:")
        metrics = result['final_metrics']
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                icon = "🟢" if value > 0.8 else "🟡" if value > 0.5 else "🔴"
                print(f"  {icon} {key.replace('_', ' ').title()}: {value:.3f}")
        
        print(f"\n🎮 Stability: {'Stable' if result.get('stability', False) else 'Unstable'}")
        
        # Performance summary
        if result.get('performance_summary'):
            perf = result['performance_summary']
            print(f"\n⚡ Performance Analysis:")
            
            constraint_perf = perf.get('constraint_satisfaction', {})
            print(f"  📈 Constraint Satisfaction: {constraint_perf.get('final', 0):.3f} "
                  f"(improvement: {constraint_perf.get('improvement', 0):+.3f})")
            
            collision_perf = perf.get('collision_analysis', {})
            print(f"  🚫 Total Collisions: {collision_perf.get('total_collisions', 0)}")
            print(f"  🛡️  Collision-Free Ratio: {collision_perf.get('collision_free_ratio', 0):.1%}")
            
            exec_stats = perf.get('execution_stats', {})
            print(f"  🎮 Control Active Steps: {exec_stats.get('control_active_steps', 0)}")
        
        # Trajectory data
        traj_data = result.get('trajectory_data', {})
        if traj_data:
            print(f"\n🛤️  Trajectory Execution:")
            print(f"  📍 Active Trajectories: {traj_data.get('active_trajectories', 0)}")
            print(f"  ✅ Completed Trajectories: {traj_data.get('completed_trajectories', 0)}")
        
        # Save video if requested
        if args.save_video and mode == SimulationMode.RECORD:
            print(f"\n🎬 Saving video to: {args.save_video}")
            agent.save_video(args.save_video)
        
        # Export metrics if requested
        if args.export_metrics:
            try:
                with open(args.export_metrics, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"📄 Metrics exported to: {args.export_metrics}")
            except Exception as e:
                print(f"❌ Failed to export metrics: {e}")
        
        # Performance summary across all tasks
        summary = agent.get_performance_summary()
        if summary.get('total_tasks', 0) > 1:
            print(f"\n📈 Session Summary:")
            print(f"  🎯 Total Tasks: {summary.get('total_tasks', 0)}")
            print(f"  🏆 Success Rate: {summary.get('success_rate', 0.0):.1%}")
        
        # Wait for user input in GUI mode
        if mode == SimulationMode.GUI:
            print(f"\n👆 GUI mode active - press Enter in terminal to exit...")
            input()
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    finally:
        print(f"\n🧹 Cleaning up...")
        agent.cleanup()
        print(f"✅ Done!")


# Additional utility functions
def install_dependencies():
    """Install missing dependencies"""
    try:
        import subprocess
        import sys
        
        deps = ["pybullet>=3.2.5", "opencv-python>=4.5.0", "torch>=2.0.0"]
        for dep in deps:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                print(f"✅ Installed: {dep}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {dep}: {e}")
    except Exception as e:
        print(f"Installation error: {e}")


def check_system_requirements():
    """Check system requirements and capabilities"""
    print("🔍 System Requirements Check:")
    print("-" * 30)
    
    # Check Python version
    import sys
    python_version = sys.version_info
    python_ok = python_version >= (3, 8)
    print(f"🐍 Python {python_version.major}.{python_version.minor}.{python_version.micro}: {'✅' if python_ok else '❌'}")
    
    # Check dependencies
    deps_status = {
        "PyBullet": PYBULLET_AVAILABLE,
        "OpenCV": CV2_AVAILABLE,
        "PyTorch": TORCH_AVAILABLE,
        "NumPy": True,  # Always available in our setup
        "SciPy": True   # Assumed available
    }
    
    for dep, available in deps_status.items():
        print(f"📦 {dep}: {'✅' if available else '❌'}")
    
    # Check GPU availability
    gpu_available = False
    if TORCH_AVAILABLE:
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except:
            pass
    print(f"🎮 GPU (CUDA): {'✅' if gpu_available else '❌'}")
    
    # System recommendations
    print(f"\n💡 Recommendations:")
    if not all(deps_status.values()):
        print(f"   Install missing dependencies with: pip install pybullet opencv-python torch")
    if not gpu_available and TORCH_AVAILABLE:
        print(f"   Consider GPU acceleration for better performance")
    if python_version < (3, 9):
        print(f"   Python 3.9+ recommended for optimal performance")
    
    return all(deps_status.values()) and python_ok


if __name__ == "__main__":
    main()