# GASM API Reference Documentation

## Overview

This comprehensive API reference provides detailed documentation for all public interfaces, classes, methods, and functions in the GASM (Geometric Assembly State Machine) system. The API is organized by module and includes complete parameter descriptions, return types, examples, and usage patterns.

## Table of Contents

1. [Core Modules](#core-modules)
2. [GASM Bridge API](#gasm-bridge-api)
3. [SE(3) Utilities API](#se3-utilities-api)
4. [Motion Planning API](#motion-planning-api)
5. [Metrics and Performance API](#metrics-and-performance-api)
6. [Configuration API](#configuration-api)
7. [Error Handling](#error-handling)
8. [Type Definitions](#type-definitions)

## Core Modules

### gasm_bridge

The primary interface module for GASM system integration.

#### create_bridge()

```python
def create_bridge(config: Optional[Dict[str, Any]] = None) -> GASMBridge
```

Factory function to create a GASM bridge instance.

**Parameters:**
- `config` (dict, optional): Configuration dictionary for GASM initialization
  - `device` (str): Computation device ('cpu', 'cuda', 'auto'). Default: 'auto'
  - `fallback_mode` (bool): Enable fallback processing. Default: True
  - `cache_enabled` (bool): Enable result caching. Default: True
  - `timeout_seconds` (int): Processing timeout. Default: 30
  - `precision` (str): Numerical precision ('float32', 'float64'). Default: 'float32'

**Returns:**
- `GASMBridge`: Initialized GASM bridge instance

**Example:**
```python
from gasm_bridge import create_bridge

# Basic initialization
bridge = create_bridge()

# Custom configuration
config = {
    'device': 'cuda',
    'fallback_mode': True,
    'timeout_seconds': 60
}
bridge = create_bridge(config)
```

---

## GASM Bridge API

### GASMBridge

Main class for processing natural language spatial instructions.

#### __init__()

```python
def __init__(self, config: Optional[Dict[str, Any]] = None)
```

Initialize GASM bridge with optional configuration.

**Parameters:**
- `config` (dict, optional): Configuration parameters

#### process()

```python
def process(self, text: str) -> Dict[str, Any]
```

Process natural language spatial instruction.

**Parameters:**
- `text` (str): Natural language spatial instruction

**Returns:**
- `dict`: Processing result with the following structure:
  ```python
  {
      'success': bool,                    # Processing success status
      'constraints': List[Dict],          # Generated spatial constraints
      'target_poses': Dict[str, Dict],    # Target poses for entities
      'confidence': float,                # Confidence score [0.0, 1.0]
      'execution_time': float,            # Processing time in seconds
      'error_message': Optional[str],     # Error description if failed
      'debug_info': Optional[Dict]        # Debug information
  }
  ```

**Raises:**
- `ValueError`: If text is empty or invalid
- `TimeoutError`: If processing exceeds timeout
- `GASMProcessingError`: If internal processing fails

**Example:**
```python
result = bridge.process("place the red block above the blue cube")

if result['success']:
    print(f"Generated {len(result['constraints'])} constraints")
    for entity, pose in result['target_poses'].items():
        position = pose['position']
        print(f"{entity}: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
else:
    print(f"Processing failed: {result['error_message']}")
```

#### validate_pose()

```python
def validate_pose(self, pose_dict: Dict[str, Any]) -> bool
```

Validate SE(3) pose format and values.

**Parameters:**
- `pose_dict` (dict): Pose dictionary with required keys:
  - `position` (List[float]): 3D position [x, y, z]
  - `orientation` (List[float]): Quaternion [x, y, z, w] or rotation matrix
  - `frame_id` (str, optional): Reference frame. Default: 'world'
  - `confidence` (float, optional): Confidence score. Default: 1.0

**Returns:**
- `bool`: True if pose is valid

**Example:**
```python
pose = {
    'position': [0.1, 0.2, 0.3],
    'orientation': [0.0, 0.0, 0.0, 1.0],
    'frame_id': 'world',
    'confidence': 0.95
}

if bridge.validate_pose(pose):
    print("Pose is valid")
else:
    print("Invalid pose format")
```

#### get_supported_constraints()

```python
def get_supported_constraints() -> List[str]
```

Get list of supported spatial constraint types.

**Returns:**
- `List[str]`: List of constraint type names

**Example:**
```python
constraints = bridge.get_supported_constraints()
print("Supported constraints:", constraints)
# Output: ['above', 'below', 'left', 'right', 'near', 'far', 'distance', 'angle']
```

#### get_sample_responses()

```python
def get_sample_responses() -> Dict[str, Dict[str, Any]]
```

Get sample responses for common spatial relationships.

**Returns:**
- `Dict[str, Dict]`: Dictionary mapping instruction types to sample responses

**Example:**
```python
samples = bridge.get_sample_responses()
for instruction_type, response in samples.items():
    print(f"{instruction_type}: {response['success']}")
```

### SpatialConstraint

Data class representing a geometric constraint between objects.

#### __init__()

```python
def __init__(
    self, 
    type: ConstraintType,
    subject: str,
    target: Optional[str] = None,
    parameters: Dict[str, Any] = None,
    priority: float = 0.5,
    tolerance: Dict[str, float] = None
)
```

**Parameters:**
- `type` (ConstraintType): Type of spatial constraint
- `subject` (str): Primary object in the relationship
- `target` (str, optional): Secondary object in the relationship
- `parameters` (dict, optional): Additional constraint parameters
- `priority` (float): Constraint priority [0.0, 1.0]. Default: 0.5
- `tolerance` (dict, optional): Tolerance values for constraint satisfaction

**Example:**
```python
from gasm_bridge import SpatialConstraint, ConstraintType

constraint = SpatialConstraint(
    type=ConstraintType.ABOVE,
    subject="red_block",
    target="blue_cube",
    parameters={"vertical_offset": 0.05},
    priority=0.8,
    tolerance={"position": 0.01, "orientation": 0.1}
)
```

### SE3Pose

Data class for 6-DOF pose representation.

#### __init__()

```python
def __init__(
    self,
    position: List[float],
    orientation: List[float],
    frame_id: str = "world",
    confidence: float = 1.0
)
```

**Parameters:**
- `position` (List[float]): 3D position [x, y, z] in meters
- `orientation` (List[float]): Quaternion [x, y, z, w] or rotation matrix (9 elements)
- `frame_id` (str): Reference frame identifier. Default: 'world'
- `confidence` (float): Confidence score [0.0, 1.0]. Default: 1.0

#### to_homogeneous_matrix()

```python
def to_homogeneous_matrix(self) -> np.ndarray
```

Convert pose to 4x4 homogeneous transformation matrix.

**Returns:**
- `np.ndarray`: 4x4 transformation matrix

**Example:**
```python
pose = SE3Pose(
    position=[1.0, 2.0, 3.0],
    orientation=[0.0, 0.0, 0.0, 1.0]
)

T = pose.to_homogeneous_matrix()
print(f"Transformation matrix shape: {T.shape}")  # (4, 4)
```

---

## SE(3) Utilities API

### SE3Utils

Comprehensive utilities for SE(3) group operations.

#### validate_rotation_matrix()

```python
@staticmethod
def validate_rotation_matrix(
    R_matrix: np.ndarray, 
    tolerance: float = 1e-6
) -> bool
```

Validate if a matrix is a proper rotation matrix.

**Parameters:**
- `R_matrix` (np.ndarray): 3x3 matrix to validate
- `tolerance` (float): Numerical tolerance. Default: 1e-6

**Returns:**
- `bool`: True if valid rotation matrix

**Raises:**
- `SE3ValidationError`: If matrix is invalid

**Example:**
```python
from utils_se3 import SE3Utils
import numpy as np

R = np.eye(3)  # Identity matrix
try:
    is_valid = SE3Utils.validate_rotation_matrix(R)
    print(f"Matrix is valid: {is_valid}")
except SE3ValidationError as e:
    print(f"Validation error: {e}")
```

#### homogeneous_matrix()

```python
@staticmethod
def homogeneous_matrix(
    rotation: np.ndarray, 
    translation: np.ndarray
) -> np.ndarray
```

Create homogeneous transformation matrix.

**Parameters:**
- `rotation` (np.ndarray): 3x3 rotation matrix or 4-element quaternion
- `translation` (np.ndarray): 3-element translation vector

**Returns:**
- `np.ndarray`: 4x4 homogeneous transformation matrix

**Example:**
```python
R = np.eye(3)
t = np.array([1, 2, 3])
T = SE3Utils.homogeneous_matrix(R, t)

print("Position:", T[:3, 3])  # [1 2 3]
print("Rotation:", T[:3, :3])  # Identity
```

#### se3_exp_map()

```python
@staticmethod
def se3_exp_map(xi: np.ndarray) -> np.ndarray
```

Exponential map from se(3) to SE(3).

**Parameters:**
- `xi` (np.ndarray): 6-element se(3) vector [translation; rotation]

**Returns:**
- `np.ndarray`: 4x4 SE(3) transformation matrix

**Example:**
```python
# Small motion: 0.1m forward, 0.1 rad rotation about z-axis
xi = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.1])
T = SE3Utils.se3_exp_map(xi)

print("Resulting transformation:")
print(T)
```

#### se3_log_map()

```python
@staticmethod
def se3_log_map(T: np.ndarray) -> np.ndarray
```

Logarithm map from SE(3) to se(3).

**Parameters:**
- `T` (np.ndarray): 4x4 SE(3) transformation matrix

**Returns:**
- `np.ndarray`: 6-element se(3) vector [translation; rotation]

**Example:**
```python
# Round-trip test
T_original = SE3Utils.homogeneous_matrix(np.eye(3), np.array([1, 2, 3]))
xi = SE3Utils.se3_log_map(T_original)
T_reconstructed = SE3Utils.se3_exp_map(xi)

print("Round-trip error:", np.max(np.abs(T_original - T_reconstructed)))
```

#### geodesic_distance_SE3()

```python
@staticmethod
def geodesic_distance_SE3(
    T1: np.ndarray, 
    T2: np.ndarray,
    translation_weight: float = 1.0
) -> float
```

Compute geodesic distance between SE(3) transformations.

**Parameters:**
- `T1` (np.ndarray): First transformation matrix
- `T2` (np.ndarray): Second transformation matrix  
- `translation_weight` (float): Weight for translation vs rotation. Default: 1.0

**Returns:**
- `float`: Geodesic distance

**Example:**
```python
T1 = np.eye(4)
T2 = SE3Utils.homogeneous_matrix(np.eye(3), np.array([1, 0, 0]))

distance = SE3Utils.geodesic_distance_SE3(T1, T2)
print(f"Distance: {distance:.3f}")  # Should be 1.0 (pure translation)
```

#### interpolate_poses()

```python
@staticmethod
def interpolate_poses(
    T1: np.ndarray, 
    T2: np.ndarray, 
    t: float
) -> np.ndarray
```

Geodesic interpolation between SE(3) poses.

**Parameters:**
- `T1` (np.ndarray): Start transformation
- `T2` (np.ndarray): End transformation
- `t` (float): Interpolation parameter [0, 1]

**Returns:**
- `np.ndarray`: Interpolated transformation matrix

**Example:**
```python
T_start = np.eye(4)
T_end = SE3Utils.homogeneous_matrix(np.eye(3), np.array([2, 0, 0]))

# Interpolate at midpoint
T_mid = SE3Utils.interpolate_poses(T_start, T_end, 0.5)
print("Midpoint position:", T_mid[:3, 3])  # Should be [1, 0, 0]
```

### Convenience Functions

#### create_pose()

```python
def create_pose(
    position: np.ndarray, 
    orientation: Union[np.ndarray, str] = None,
    orientation_type: str = 'quaternion'
) -> np.ndarray
```

Create SE(3) pose from position and orientation.

**Parameters:**
- `position` (np.ndarray): 3-element position vector
- `orientation` (np.ndarray, optional): Orientation representation
- `orientation_type` (str): Type of orientation ('quaternion', 'euler', 'matrix')

**Returns:**
- `np.ndarray`: 4x4 transformation matrix

**Example:**
```python
from utils_se3 import create_pose

# Position only
T1 = create_pose([1, 2, 3])

# With quaternion
T2 = create_pose([0, 0, 0], [0, 0, 0, 1], 'quaternion')

# With Euler angles  
T3 = create_pose([1, 1, 1], [0, 0, np.pi/4], 'euler')
```

#### pose_to_dict()

```python
def pose_to_dict(T: np.ndarray) -> dict
```

Convert SE(3) pose to dictionary representation.

**Parameters:**
- `T` (np.ndarray): 4x4 transformation matrix

**Returns:**
- `dict`: Dictionary with multiple pose representations

**Example:**
```python
from utils_se3 import pose_to_dict

T = create_pose([1, 2, 3], [0, 0, np.pi/4], 'euler')
pose_dict = pose_to_dict(T)

print("Position:", pose_dict['position'])
print("Quaternion:", pose_dict['quaternion'])
print("Euler XYZ:", pose_dict['euler_xyz'])
```

---

## Motion Planning API

### MotionPlanner

Rule-based motion planner with multiple strategies.

#### __init__()

```python
def __init__(self, config: Optional[PlanningConfig] = None)
```

Initialize motion planner with configuration.

**Parameters:**
- `config` (PlanningConfig, optional): Planning configuration

#### plan_step()

```python
def plan_step(
    self, 
    current_pose: Pose, 
    target_pose: Pose,
    constraints: Optional[List[Constraint]] = None
) -> PlanningResult
```

Compute next step toward target pose.

**Parameters:**
- `current_pose` (Pose): Current robot pose
- `target_pose` (Pose): Desired target pose
- `constraints` (List[Constraint], optional): Motion constraints

**Returns:**
- `PlanningResult`: Planning result with next pose and metadata

**Example:**
```python
from planner import create_default_planner, Pose

planner = create_default_planner()

current = Pose(x=0, y=0, z=1, yaw=0)
target = Pose(x=1, y=1, z=1, yaw=np.pi/2)

result = planner.plan_step(current, target)

if result.success:
    next_pose = result.next_pose
    print(f"Next position: ({next_pose.x:.2f}, {next_pose.y:.2f}, {next_pose.z:.2f})")
else:
    print("Planning failed:", result.reasoning)
```

#### set_strategy()

```python
def set_strategy(self, strategy: PlanningStrategy)
```

Change planning strategy.

**Parameters:**
- `strategy` (PlanningStrategy): Planning strategy enum value

**Example:**
```python
from planner import PlanningStrategy

planner.set_strategy(PlanningStrategy.SAFE)
```

#### add_constraint()

```python
def add_constraint(self, constraint: Constraint) -> int
```

Add motion constraint.

**Parameters:**
- `constraint` (Constraint): Motion constraint to add

**Returns:**
- `int`: Constraint index

#### add_obstacle()

```python
def add_obstacle(self, obstacle: Obstacle) -> int
```

Add collision obstacle.

**Parameters:**
- `obstacle` (Obstacle): Obstacle to add

**Returns:**
- `int`: Obstacle index

### PlanningConfig

Configuration for motion planning parameters.

```python
@dataclass
class PlanningConfig:
    max_position_step: float = 0.1      # Maximum position step (meters)
    max_rotation_step: float = 0.2      # Maximum rotation step (radians)
    position_gain: float = 1.0          # Position control gain
    rotation_gain: float = 0.5          # Rotation control gain
    constraint_gain: float = 2.0        # Constraint importance weight
    safety_margin: float = 0.05         # Safety margin (meters)
    position_tolerance: float = 0.01    # Position convergence tolerance
    rotation_tolerance: float = 0.05    # Rotation convergence tolerance
    max_iterations: int = 1000          # Maximum planning iterations
    enable_obstacle_avoidance: bool = True
    enable_constraint_checking: bool = True
```

### PlanningResult

Result of planning step operation.

```python
@dataclass
class PlanningResult:
    success: bool                           # Planning success
    next_pose: Optional[Pose] = None        # Next pose to execute
    step_size: float = 0.0                  # Step magnitude
    constraints_violated: List[str] = []    # Violated constraint names
    obstacles_detected: List[int] = []      # Detected obstacle indices
    reasoning: str = ""                     # Human-readable explanation
    debug_info: Dict[str, Any] = {}        # Debug information
```

### Factory Functions

#### create_default_planner()

```python
def create_default_planner() -> MotionPlanner
```

Create motion planner with default configuration.

#### create_safe_planner()

```python
def create_safe_planner() -> MotionPlanner
```

Create conservative planner for safety-critical applications.

#### create_fast_planner()

```python
def create_fast_planner() -> MotionPlanner
```

Create fast planner for open environments.

---

## Metrics and Performance API

### PerformanceMetrics

Comprehensive performance tracking for GASM operations.

```python
@dataclass
class PerformanceMetrics:
    total_time: float = 0.0                    # Total processing time
    preprocessing_time: float = 0.0            # Text preprocessing time
    inference_time: float = 0.0               # Neural network inference
    postprocessing_time: float = 0.0          # Result postprocessing
    constraint_solving_time: float = 0.0      # Constraint optimization
    
    peak_memory_usage: float = 0.0            # Peak memory (MB)
    average_memory_usage: float = 0.0         # Average memory (MB)
    memory_efficiency: float = 0.0            # Memory utilization ratio
    
    instructions_per_second: float = 0.0      # Throughput metric
    success_rate: float = 0.0                 # Success percentage
    average_confidence: float = 0.0           # Average confidence score
    
    cpu_utilization: float = 0.0              # CPU usage percentage
    gpu_utilization: float = 0.0              # GPU usage percentage
    
    convergence_iterations: int = 0           # Optimization iterations
    numerical_stability_score: float = 0.0    # Stability metric
```

### MetricsCollector

Collect and analyze GASM performance metrics.

#### __init__()

```python
def __init__(self, enable_detailed=True)
```

#### start_collection()

```python
def start_collection(self)
```

Start metrics collection session.

#### end_collection()

```python
def end_collection(self) -> PerformanceMetrics
```

End collection and compute metrics.

#### get_summary()

```python
def get_summary(self) -> Dict[str, float]
```

Get performance summary statistics.

**Example:**
```python
from metrics import MetricsCollector

collector = MetricsCollector()

collector.start_collection()

# Your GASM operations here
bridge = create_bridge()
result = bridge.process("place box above table")

metrics = collector.end_collection()

print(f"Processing time: {metrics.total_time:.3f}s")
print(f"Peak memory: {metrics.peak_memory_usage:.1f}MB")
print(f"Success rate: {metrics.success_rate:.1%}")
```

### PoseError

Detailed pose error analysis for spatial accuracy assessment.

```python
@dataclass
class PoseError:
    position_error: float = 0.0                # Euclidean position error
    rotation_error: float = 0.0               # Angular error (radians)
    position_vector: torch.Tensor = None      # Position error vector
    rotation_axis: torch.Tensor = None        # Rotation error axis
    total_error: float = 0.0                  # Combined weighted error
    is_converged: bool = False                # Convergence status
```

### calculate_pose_error()

```python
def calculate_pose_error(
    target_pose: Dict[str, Any], 
    actual_pose: Dict[str, Any]
) -> PoseError
```

Calculate comprehensive pose error metrics.

**Parameters:**
- `target_pose` (dict): Desired pose
- `actual_pose` (dict): Achieved pose

**Returns:**
- `PoseError`: Detailed error analysis

---

## Configuration API

### GASMConfig

Central configuration management for GASM system.

#### __init__()

```python
def __init__(self, config_file: Optional[str] = None)
```

Initialize configuration from file or defaults.

#### load_from_file()

```python
def load_from_file(self, config_file: str)
```

Load configuration from YAML or JSON file.

#### save_to_file()

```python
def save_to_file(self, config_file: str)
```

Save current configuration to file.

#### get()

```python
def get(self, key_path: str, default=None) -> Any
```

Get configuration value using dot notation.

**Parameters:**
- `key_path` (str): Configuration key path (e.g., 'system.device')
- `default`: Default value if key not found

**Returns:**
- Value at key path or default

#### set()

```python
def set(self, key_path: str, value: Any)
```

Set configuration value using dot notation.

**Example:**
```python
from gasm_config import GASMConfig

config = GASMConfig()

# Set values
config.set('system.device', 'cuda')
config.set('planning.max_iterations', 500)

# Get values
device = config.get('system.device', 'cpu')
iterations = config.get('planning.max_iterations', 1000)

# Save configuration
config.save_to_file('my_gasm_config.yaml')
```

### Configuration Schema

Default configuration structure:

```yaml
system:
  device: 'auto'              # 'cpu', 'cuda', 'auto'
  precision: 'float32'        # 'float16', 'float32', 'float64'
  cache_enabled: true
  log_level: 'INFO'

gasm:
  fallback_mode: true
  timeout_seconds: 30
  confidence_threshold: 0.7
  max_retries: 3

planning:
  strategy: 'adaptive'        # 'direct', 'constrained', 'safe', 'adaptive'
  max_iterations: 1000
  tolerance: 0.01
  safety_margin: 0.05

neural_networks:
  hidden_dim: 256
  num_heads: 8
  dropout: 0.1
  batch_size: 32

optimization:
  learning_rate: 0.001
  convergence_threshold: 1e-6
  max_constraint_iterations: 100

performance:
  enable_profiling: false
  memory_monitoring: true
  cache_size: 1000
```

---

## Error Handling

### Exception Hierarchy

```python
class GASMError(Exception):
    """Base exception class for GASM-related errors"""
    pass

class GASMConfigurationError(GASMError):
    """Configuration-related errors"""
    pass

class GASMProcessingError(GASMError):
    """Processing and computation errors"""
    pass

class GASMValidationError(GASMError):
    """Input validation errors"""
    pass

class SE3ValidationError(GASMError):
    """SE(3) mathematical validation errors"""
    pass

class ConstraintConflictError(GASMError):
    """Constraint system conflicts"""
    pass

class PlanningError(GASMError):
    """Motion planning errors"""
    pass
```

### Error Handling Examples

```python
from gasm_bridge import create_bridge, GASMProcessingError
from utils_se3 import SE3ValidationError

try:
    bridge = create_bridge({'device': 'invalid_device'})
except GASMConfigurationError as e:
    print(f"Configuration error: {e}")

try:
    result = bridge.process("")  # Empty input
except GASMValidationError as e:
    print(f"Input validation error: {e}")

try:
    invalid_matrix = np.array([[1, 2], [3, 4]])  # Not 4x4
    SE3Utils.validate_homogeneous_matrix(invalid_matrix)
except SE3ValidationError as e:
    print(f"SE(3) validation error: {e}")
```

---

## Type Definitions

### Type Aliases

```python
from typing import List, Dict, Any, Union, Optional, Tuple

# Basic types
Position3D = List[float]          # [x, y, z]
Quaternion = List[float]          # [x, y, z, w]
EulerAngles = List[float]         # [roll, pitch, yaw]
HomogeneousMatrix = np.ndarray    # 4x4 transformation matrix
SE3Vector = np.ndarray            # 6-element se(3) vector

# Composite types
PoseDict = Dict[str, Union[Position3D, Quaternion, str, float]]
ConstraintDict = Dict[str, Any]
ResultDict = Dict[str, Any]

# Configuration types
DeviceType = Union['cpu', 'cuda', 'auto']
PrecisionType = Union['float16', 'float32', 'float64']
PlanningStrategyType = Union['direct', 'constrained', 'safe', 'adaptive']
```

### Enumerations

```python
from enum import Enum

class ConstraintType(Enum):
    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"
    NEAR = "near"
    FAR = "far"
    DISTANCE = "distance"
    ANGLE = "angle"
    ALIGNED = "aligned"
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    TOUCHING = "touching"
    INSIDE = "inside"
    OUTSIDE = "outside"
    BETWEEN = "between"

class PlanningStrategy(Enum):
    DIRECT = "direct"
    CONSTRAINED = "constrained"
    SAFE = "safe"
    ADAPTIVE = "adaptive"
```

### Protocol Definitions

```python
from typing import Protocol

class SpatialProcessor(Protocol):
    """Protocol for spatial instruction processors"""
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process spatial instruction"""
        ...
    
    def validate_pose(self, pose: Dict[str, Any]) -> bool:
        """Validate pose format"""
        ...

class ConstraintSolver(Protocol):
    """Protocol for constraint solvers"""
    
    def solve_constraints(
        self, 
        initial_poses: List[np.ndarray],
        constraints: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """Solve spatial constraints"""
        ...
    
    def check_convergence(self, poses: List[np.ndarray]) -> bool:
        """Check if solution has converged"""
        ...
```

### Version Information

```python
def get_version_info() -> Dict[str, str]:
    """Get GASM version and dependency information"""
    return {
        'gasm_version': '2.0.0',
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__
    }

# Example usage
version_info = get_version_info()
print(f"GASM Version: {version_info['gasm_version']}")
```

This API reference provides comprehensive documentation for all public interfaces in the GASM system. Each function and class includes detailed parameter descriptions, return types, examples, and usage patterns to facilitate integration and development.