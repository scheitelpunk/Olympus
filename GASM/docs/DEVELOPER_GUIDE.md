# GASM Developer Guide

## Overview

This guide provides comprehensive instructions for developers who want to extend, modify, or contribute to the GASM (Geometric Assembly State Machine) system. It covers the development environment setup, coding standards, architecture patterns, and extension mechanisms.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Code Architecture and Patterns](#code-architecture-and-patterns)
3. [Extending GASM Components](#extending-gasm-components)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Testing and Quality Assurance](#testing-and-quality-assurance)
6. [Performance Optimization](#performance-optimization)
7. [Debugging and Profiling](#debugging-and-profiling)
8. [Deployment and Distribution](#deployment-and-distribution)

## Development Environment Setup

### Prerequisites

```bash
# System requirements
Python >= 3.8
CUDA >= 11.0 (optional, for GPU acceleration)
Git >= 2.0

# Hardware recommendations
RAM: >= 8GB (16GB recommended)
Storage: >= 10GB free space
GPU: NVIDIA GPU with >= 4GB VRAM (optional)
```

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-org/GASM-Roboting.git
cd GASM-Roboting

# Create virtual environment
python -m venv gasm-dev
source gasm-dev/bin/activate  # On Windows: gasm-dev\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install GASM in development mode
pip install -e .

# Verify installation
python -c "from gasm_bridge import create_bridge; print('GASM installed successfully')"
```

### Development Dependencies

```txt
# requirements-dev.txt
torch>=1.10.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
pytest>=6.0.0
pytest-cov>=3.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.910
sphinx>=4.0.0
sphinx-rtd-theme>=1.0.0
jupyter>=1.0.0
ipykernel>=6.0.0
pre-commit>=2.15.0

# Optional dependencies
torch-geometric>=2.0.0
geomstats>=2.4.0
pybullet>=3.2.0
opencv-python>=4.5.0
```

### IDE Configuration

#### VS Code Setup

```json
// .vscode/settings.json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".coverage": true
    }
}
```

#### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.910
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: ["--profile", "black"]
```

## Code Architecture and Patterns

### Directory Structure

```
GASM-Roboting/
├── src/
│   ├── gasm/                     # Core GASM package
│   │   ├── __init__.py
│   │   ├── core.py               # Main GASM neural networks
│   │   ├── coordination/         # Multi-agent coordination
│   │   ├── simulation/           # Physics simulation
│   │   └── utils.py              # Utility functions
│   └── spatial_agent/            # Spatial reasoning agent
│       ├── __init__.py
│       ├── gasm_bridge.py        # GASM integration bridge
│       ├── planner.py            # Motion planning
│       ├── utils_se3.py          # SE(3) mathematics
│       └── metrics.py            # Performance metrics
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── e2e/                      # End-to-end tests
├── docs/                         # Documentation
├── examples/                     # Example code
├── scripts/                      # Utility scripts
└── assets/                       # Static assets
```

### Design Patterns

#### 1. Factory Pattern for Component Creation

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ComponentFactory(ABC):
    """Abstract factory for GASM components"""
    
    @abstractmethod
    def create_bridge(self, config: Optional[Dict[str, Any]] = None):
        pass
    
    @abstractmethod
    def create_planner(self, config: Optional[Dict[str, Any]] = None):
        pass
    
    @abstractmethod
    def create_neural_network(self, config: Optional[Dict[str, Any]] = None):
        pass

class DefaultComponentFactory(ComponentFactory):
    """Default factory implementation"""
    
    def create_bridge(self, config: Optional[Dict[str, Any]] = None):
        from gasm_bridge import GASMBridge
        return GASMBridge(config)
    
    def create_planner(self, config: Optional[Dict[str, Any]] = None):
        from planner import MotionPlanner, PlanningConfig
        planning_config = PlanningConfig(**config) if config else PlanningConfig()
        return MotionPlanner(planning_config)
    
    def create_neural_network(self, config: Optional[Dict[str, Any]] = None):
        from gasm_core import EnhancedGASM
        return EnhancedGASM(**(config or {}))

class GPUComponentFactory(ComponentFactory):
    """GPU-optimized factory implementation"""
    
    def create_bridge(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['device'] = 'cuda'
        config['precision'] = 'float16'  # Use half precision for speed
        from gasm_bridge import GASMBridge
        return GASMBridge(config)
    
    def create_neural_network(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['device'] = 'cuda'
        config['use_mixed_precision'] = True
        from gasm_core import EnhancedGASM
        return EnhancedGASM(**config)

# Usage
factory = GPUComponentFactory() if torch.cuda.is_available() else DefaultComponentFactory()
bridge = factory.create_bridge()
planner = factory.create_planner()
```

#### 2. Observer Pattern for Event Handling

```python
from typing import List, Callable, Any
from abc import ABC, abstractmethod

class Observer(ABC):
    """Observer interface for event handling"""
    
    @abstractmethod
    def update(self, event_type: str, data: Any) -> None:
        pass

class Subject:
    """Subject class that notifies observers"""
    
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)
    
    def notify(self, event_type: str, data: Any) -> None:
        for observer in self._observers:
            observer.update(event_type, data)

class GASMEventHandler(Observer):
    """Event handler for GASM processing events"""
    
    def __init__(self, name: str):
        self.name = name
        self.event_log = []
    
    def update(self, event_type: str, data: Any) -> None:
        self.event_log.append({
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        })
        
        if event_type == 'constraint_violation':
            self._handle_constraint_violation(data)
        elif event_type == 'planning_failure':
            self._handle_planning_failure(data)
    
    def _handle_constraint_violation(self, data):
        logger.warning(f"Constraint violation detected: {data}")
    
    def _handle_planning_failure(self, data):
        logger.error(f"Planning failure: {data}")

class MetricsCollector(Observer):
    """Collect performance metrics"""
    
    def __init__(self):
        self.metrics = []
    
    def update(self, event_type: str, data: Any) -> None:
        if event_type == 'processing_complete':
            self.metrics.append({
                'success': data.get('success', False),
                'confidence': data.get('confidence', 0.0),
                'execution_time': data.get('execution_time', 0.0),
                'timestamp': time.time()
            })
```

#### 3. Strategy Pattern for Planning Algorithms

```python
from abc import ABC, abstractmethod
from enum import Enum

class PlanningStrategy(ABC):
    """Abstract base class for planning strategies"""
    
    @abstractmethod
    def plan(self, current_pose, target_pose, constraints):
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

class DirectPlanningStrategy(PlanningStrategy):
    """Direct path planning strategy"""
    
    def plan(self, current_pose, target_pose, constraints):
        # Direct gradient descent implementation
        direction = self._calculate_direction(current_pose, target_pose)
        return self._apply_step_limits(direction)
    
    def get_name(self) -> str:
        return "direct"
    
    def _calculate_direction(self, current, target):
        return target.to_array() - current.to_array()
    
    def _apply_step_limits(self, step):
        # Apply safety limits
        return np.clip(step, -0.1, 0.1)

class RRTStarPlanningStrategy(PlanningStrategy):
    """RRT* planning strategy for complex environments"""
    
    def __init__(self, max_iterations=1000, step_size=0.05):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.tree = []
    
    def plan(self, current_pose, target_pose, constraints):
        # RRT* implementation
        self.tree = [current_pose]
        
        for _ in range(self.max_iterations):
            # Sample random point
            random_pose = self._sample_random_pose()
            
            # Find nearest node in tree
            nearest_node = self._find_nearest_node(random_pose)
            
            # Extend tree toward random pose
            new_node = self._extend_tree(nearest_node, random_pose)
            
            if new_node and self._is_collision_free(new_node, constraints):
                self.tree.append(new_node)
                
                # Check if we reached the target
                if self._distance_to_target(new_node, target_pose) < 0.1:
                    return self._extract_path(new_node, target_pose)
        
        return None  # Planning failed
    
    def get_name(self) -> str:
        return "rrt_star"

class PlanningContext:
    """Context class for strategy pattern"""
    
    def __init__(self, strategy: PlanningStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: PlanningStrategy):
        self._strategy = strategy
    
    def execute_planning(self, current_pose, target_pose, constraints):
        return self._strategy.plan(current_pose, target_pose, constraints)
    
    def get_strategy_name(self):
        return self._strategy.get_name()

# Usage
context = PlanningContext(DirectPlanningStrategy())
result = context.execute_planning(current, target, constraints)

# Switch strategy based on environment complexity
if environment_complexity > 0.8:
    context.set_strategy(RRTStarPlanningStrategy())
```

#### 4. Plugin Architecture

```python
from abc import ABC, abstractmethod
import importlib
from typing import Dict, Any, List

class GASMPlugin(ABC):
    """Base class for GASM plugins"""
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        pass

class VisionProcessingPlugin(GASMPlugin):
    """Plugin for computer vision processing"""
    
    def get_name(self) -> str:
        return "vision_processing"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        self.camera_config = config.get('camera', {})
        self.processing_config = config.get('processing', {})
        # Initialize camera and CV models
        return True
    
    def process(self, data: Any) -> Any:
        # Process camera data and extract object poses
        image = data.get('image')
        if image is not None:
            # Run object detection
            objects = self._detect_objects(image)
            # Estimate poses
            poses = self._estimate_poses(objects)
            return {'detected_objects': objects, 'poses': poses}
        return {}
    
    def cleanup(self) -> None:
        # Release camera resources
        pass

class PluginManager:
    """Manager for GASM plugins"""
    
    def __init__(self):
        self.plugins: Dict[str, GASMPlugin] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_plugin(self, plugin: GASMPlugin, config: Dict[str, Any] = None):
        """Register a plugin instance"""
        plugin_name = plugin.get_name()
        
        if plugin.initialize(config or {}):
            self.plugins[plugin_name] = plugin
            self.plugin_configs[plugin_name] = config or {}
            logger.info(f"Plugin '{plugin_name}' registered successfully")
        else:
            logger.error(f"Failed to initialize plugin '{plugin_name}'")
    
    def load_plugin_from_file(self, module_path: str, class_name: str, 
                             config: Dict[str, Any] = None):
        """Load plugin from file"""
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            plugin_instance = plugin_class()
            self.register_plugin(plugin_instance, config)
        except Exception as e:
            logger.error(f"Failed to load plugin from {module_path}: {e}")
    
    def get_plugin(self, plugin_name: str) -> GASMPlugin:
        """Get plugin by name"""
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins"""
        return list(self.plugins.keys())
    
    def process_with_plugin(self, plugin_name: str, data: Any) -> Any:
        """Process data with specific plugin"""
        plugin = self.plugins.get(plugin_name)
        if plugin:
            return plugin.process(data)
        else:
            raise ValueError(f"Plugin '{plugin_name}' not found")
    
    def cleanup_all(self):
        """Cleanup all plugins"""
        for plugin in self.plugins.values():
            plugin.cleanup()

# Usage
plugin_manager = PluginManager()
plugin_manager.register_plugin(VisionProcessingPlugin(), {
    'camera': {'device_id': 0, 'resolution': [640, 480]},
    'processing': {'model_path': 'models/object_detection.pth'}
})

# Process data with plugin
image_data = {'image': captured_image}
result = plugin_manager.process_with_plugin('vision_processing', image_data)
```

## Extending GASM Components

### Creating Custom Constraint Types

```python
from gasm_bridge import ConstraintType, SpatialConstraint
from typing import Dict, Any
import torch
import numpy as np

class CustomConstraintType(ConstraintType):
    """Extended constraint types for specific applications"""
    FLUID_FLOW = "fluid_flow"
    ELECTROMAGNETIC = "electromagnetic"
    THERMAL_GRADIENT = "thermal_gradient"

class FluidFlowConstraint(SpatialConstraint):
    """Constraint for fluid dynamics in assembly tasks"""
    
    def __init__(self, subject: str, target: str = None, **kwargs):
        super().__init__(
            type=CustomConstraintType.FLUID_FLOW,
            subject=subject,
            target=target,
            **kwargs
        )
        
        # Fluid-specific parameters
        self.viscosity = self.parameters.get('viscosity', 1.0)
        self.flow_rate = self.parameters.get('flow_rate', 0.1)
        self.pressure_gradient = self.parameters.get('pressure_gradient', [0, 0, -9.81])
    
    def evaluate_constraint(self, positions: torch.Tensor, velocities: torch.Tensor = None) -> torch.Tensor:
        """Evaluate fluid flow constraint energy"""
        if velocities is None:
            return torch.tensor(0.0)
        
        # Simplified Navier-Stokes-inspired constraint
        subject_idx = self._get_entity_index(self.subject, positions)
        
        if subject_idx is not None:
            velocity = velocities[subject_idx]
            position = positions[subject_idx]
            
            # Pressure force
            pressure_force = torch.tensor(self.pressure_gradient) * position[2]  # Height-dependent
            
            # Viscous force (simplified)
            viscous_force = -self.viscosity * velocity
            
            # Total force
            total_force = pressure_force + viscous_force
            
            # Energy is force magnitude
            energy = torch.norm(total_force)
            
            return energy
        
        return torch.tensor(0.0)
    
    def compute_gradient(self, positions: torch.Tensor, velocities: torch.Tensor = None) -> torch.Tensor:
        """Compute gradient for optimization"""
        gradient = torch.zeros_like(positions)
        
        subject_idx = self._get_entity_index(self.subject, positions)
        
        if subject_idx is not None and velocities is not None:
            # Gradient with respect to position
            gradient[subject_idx, 2] = self.pressure_gradient[2]  # Z-direction pressure
            
            # Gradient with respect to velocity (if optimizing velocities)
            if hasattr(self, 'velocity_gradient'):
                self.velocity_gradient[subject_idx] = -self.viscosity * torch.ones(3)
        
        return gradient

# Register custom constraint
from gasm_core import ConstraintRegistry
ConstraintRegistry.register_constraint_type(CustomConstraintType.FLUID_FLOW, FluidFlowConstraint)
```

### Extending Neural Network Architecture

```python
import torch
import torch.nn as nn
from gasm_core import EnhancedGASM, SE3InvariantAttention

class DomainSpecificGASM(EnhancedGASM):
    """GASM extended for specific domain applications"""
    
    def __init__(self, feature_dim=256, domain="manufacturing", **kwargs):
        super().__init__(feature_dim=feature_dim, **kwargs)
        
        self.domain = domain
        
        # Domain-specific layers
        self.domain_encoder = self._create_domain_encoder(domain, feature_dim)
        self.domain_decoder = self._create_domain_decoder(domain, feature_dim)
        
        # Additional attention mechanisms
        self.cross_domain_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Domain-specific constraint weights
        self.constraint_weights = nn.ParameterDict({
            'safety': nn.Parameter(torch.ones(1) * 2.0),  # High priority for safety
            'efficiency': nn.Parameter(torch.ones(1) * 1.0),  # Medium priority
            'precision': nn.Parameter(torch.ones(1) * 1.5)   # High priority for precision
        })
    
    def _create_domain_encoder(self, domain: str, feature_dim: int) -> nn.Module:
        """Create domain-specific encoder"""
        if domain == "manufacturing":
            return nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim)
            )
        elif domain == "medical":
            # Medical domain needs higher precision and smoother activations
            return nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.Tanh(),  # Smoother than ReLU
                nn.Dropout(0.05),  # Lower dropout for stability
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim)
            )
        else:
            return nn.Identity()
    
    def _create_domain_decoder(self, domain: str, feature_dim: int) -> nn.Module:
        """Create domain-specific decoder"""
        if domain == "manufacturing":
            return nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, 6)  # SE(3) output
            )
        elif domain == "medical":
            return nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.Tanh(),
                nn.Linear(feature_dim // 2, 6),
                nn.Tanh()  # Bound outputs for safety
            )
        else:
            return nn.Linear(feature_dim, 6)
    
    def forward(self, poses, features, constraints=None, edge_index=None, **kwargs):
        """Forward pass with domain-specific processing"""
        batch_size = poses.size(0)
        
        # Base GASM processing
        base_features = super().forward(poses, features, constraints, edge_index, **kwargs)
        
        # Domain-specific encoding
        domain_features = self.domain_encoder(base_features)
        
        # Cross-domain attention
        attended_features, attention_weights = self.cross_domain_attention(
            domain_features, domain_features, domain_features
        )
        
        # Apply constraint weights based on domain
        if constraints is not None:
            weighted_constraints = self._apply_constraint_weights(constraints)
            # Integrate weighted constraints into features
            attended_features = attended_features + weighted_constraints
        
        # Domain-specific decoding
        output_poses = self.domain_decoder(attended_features)
        
        return output_poses
    
    def _apply_constraint_weights(self, constraints):
        """Apply domain-specific constraint weights"""
        weighted_constraints = torch.zeros_like(constraints)
        
        # This is a simplified example - actual implementation would
        # depend on constraint representation
        if hasattr(constraints, 'constraint_types'):
            for i, constraint_type in enumerate(constraints.constraint_types):
                if constraint_type in self.constraint_weights:
                    weight = self.constraint_weights[constraint_type]
                    weighted_constraints[i] *= weight
        
        return weighted_constraints

class HierarchicalGASM(nn.Module):
    """Hierarchical GASM for multi-scale reasoning"""
    
    def __init__(self, feature_dim=256, num_levels=3):
        super().__init__()
        
        self.num_levels = num_levels
        
        # Create GASM modules for each hierarchical level
        self.gasm_levels = nn.ModuleList([
            EnhancedGASM(feature_dim=feature_dim // (2**i))
            for i in range(num_levels)
        ])
        
        # Cross-level communication
        self.level_projections = nn.ModuleList([
            nn.Linear(feature_dim // (2**i), feature_dim // (2**(i+1)))
            for i in range(num_levels - 1)
        ])
        
        self.level_aggregation = nn.Linear(
            sum(feature_dim // (2**i) for i in range(num_levels)),
            feature_dim
        )
    
    def forward(self, poses, features, constraints=None, edge_index=None):
        """Hierarchical forward pass"""
        level_outputs = []
        current_features = features
        
        # Process through each level
        for i, (gasm_level, projection) in enumerate(zip(self.gasm_levels, self.level_projections)):
            # Process at current level
            level_output = gasm_level(poses, current_features, constraints, edge_index)
            level_outputs.append(level_output)
            
            # Project to next level
            if i < len(self.level_projections):
                current_features = projection(current_features)
        
        # Process final level
        final_output = self.gasm_levels[-1](poses, current_features, constraints, edge_index)
        level_outputs.append(final_output)
        
        # Aggregate across levels
        concatenated = torch.cat(level_outputs, dim=-1)
        aggregated_output = self.level_aggregation(concatenated)
        
        return aggregated_output
```

### Creating Custom Planners

```python
from planner import MotionPlanner, PlanningResult, Pose, Constraint
import torch
import numpy as np
from typing import List, Dict, Any

class LearningEnhancedPlanner(MotionPlanner):
    """Motion planner enhanced with learning capabilities"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Neural network for learning planning policies
        self.policy_network = self._create_policy_network()
        self.value_network = self._create_value_network()
        
        # Experience replay buffer
        self.experience_buffer = []
        self.buffer_size = 10000
        
        # Training parameters
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(
            list(self.policy_network.parameters()) + 
            list(self.value_network.parameters()),
            lr=self.learning_rate
        )
        
        # Exploration parameters
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
    
    def _create_policy_network(self):
        """Create policy network for action selection"""
        return nn.Sequential(
            nn.Linear(18, 256),  # State: current pose (6) + target pose (6) + constraints (6)
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6),   # Action: SE(3) step
            nn.Tanh()           # Bounded actions
        )
    
    def _create_value_network(self):
        """Create value network for state evaluation"""
        return nn.Sequential(
            nn.Linear(18, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)   # State value
        )
    
    def plan_step(self, current_pose: Pose, target_pose: Pose, 
                  constraints: List[Constraint] = None) -> PlanningResult:
        """Enhanced planning with learning"""
        
        # Encode state
        state = self._encode_state(current_pose, target_pose, constraints or [])
        
        # Choose action (epsilon-greedy)
        if np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.uniform(-0.1, 0.1, size=6)
        else:
            # Exploit: use policy network
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action = self.policy_network(state_tensor).numpy()
        
        # Apply action to get next pose
        next_array = current_pose.to_array() + action
        next_pose = Pose.from_array(next_array)
        
        # Evaluate the step
        reward = self._calculate_reward(current_pose, next_pose, target_pose, constraints or [])
        
        # Store experience
        self._store_experience(state, action, reward, next_pose, target_pose)
        
        # Train networks periodically
        if len(self.experience_buffer) > 100:
            self._train_networks()
        
        # Update exploration rate
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Check constraints and collisions
        violated_constraints = self._check_constraints(next_pose, constraints or [])
        obstacles_detected = self._check_collisions(next_pose)
        
        success = len(violated_constraints) == 0 and len(obstacles_detected) == 0
        step_size = np.linalg.norm(action[:3])
        
        return PlanningResult(
            success=success,
            next_pose=next_pose if success else None,
            step_size=step_size,
            constraints_violated=violated_constraints,
            obstacles_detected=obstacles_detected,
            reasoning=f"Learning-enhanced planning (ε={self.epsilon:.3f})",
            debug_info={
                'action': action.tolist(),
                'reward': reward,
                'state_value': self._evaluate_state(state)
            }
        )
    
    def _encode_state(self, current: Pose, target: Pose, constraints: List[Constraint]) -> np.ndarray:
        """Encode state for neural networks"""
        current_array = current.to_array()
        target_array = target.to_array()
        
        # Encode constraints (simplified)
        constraint_features = np.zeros(6)
        for i, constraint in enumerate(constraints[:6]):  # Limit to 6 constraints
            if constraint.type == "above":
                constraint_features[i] = 1.0
            elif constraint.type == "distance":
                constraint_features[i] = constraint.parameters.get('distance', 0.1)
            # Add more constraint encodings as needed
        
        return np.concatenate([current_array, target_array, constraint_features])
    
    def _calculate_reward(self, current: Pose, next_pose: Pose, target: Pose, 
                         constraints: List[Constraint]) -> float:
        """Calculate reward for reinforcement learning"""
        # Distance to target (negative reward for being far)
        current_distance = current.distance_to(target)
        next_distance = next_pose.distance_to(target)
        distance_reward = current_distance - next_distance  # Positive if getting closer
        
        # Constraint satisfaction reward
        constraint_reward = 0.0
        violated_constraints = self._check_constraints(next_pose, constraints)
        constraint_reward = -len(violated_constraints) * 0.5  # Penalty for violations
        
        # Collision penalty
        collision_penalty = -len(self._check_collisions(next_pose)) * 1.0
        
        # Step efficiency (small penalty for large steps)
        step_penalty = -np.linalg.norm(next_pose.to_array() - current.to_array()) * 0.1
        
        # Goal reached bonus
        goal_bonus = 10.0 if next_distance < 0.01 else 0.0
        
        total_reward = (distance_reward + constraint_reward + 
                       collision_penalty + step_penalty + goal_bonus)
        
        return total_reward
    
    def _store_experience(self, state, action, reward, next_pose, target_pose):
        """Store experience in replay buffer"""
        next_state = self._encode_state(next_pose, target_pose, [])
        
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': next_pose.distance_to(target_pose) < 0.01
        }
        
        self.experience_buffer.append(experience)
        
        # Keep buffer size limited
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
    
    def _train_networks(self, batch_size=32):
        """Train policy and value networks"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample random batch
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Prepare tensors
        states = torch.tensor([exp['state'] for exp in batch], dtype=torch.float32)
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.float32)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        next_states = torch.tensor([exp['next_state'] for exp in batch], dtype=torch.float32)
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.float32)
        
        # Train value network
        current_values = self.value_network(states).squeeze()
        next_values = self.value_network(next_states).squeeze()
        target_values = rewards + 0.99 * next_values * (1 - dones)
        
        value_loss = nn.MSELoss()(current_values, target_values.detach())
        
        # Train policy network (policy gradient)
        predicted_actions = self.policy_network(states)
        advantages = (target_values - current_values).detach()
        
        policy_loss = -torch.mean(torch.sum(predicted_actions * actions, dim=1) * advantages)
        
        # Combined loss
        total_loss = value_loss + policy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_network.parameters()) + 
            list(self.value_network.parameters()), 
            max_norm=1.0
        )
        self.optimizer.step()
    
    def _evaluate_state(self, state) -> float:
        """Evaluate state value"""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            value = self.value_network(state_tensor).item()
        return value
    
    def save_models(self, path_prefix: str):
        """Save trained models"""
        torch.save(self.policy_network.state_dict(), f"{path_prefix}_policy.pth")
        torch.save(self.value_network.state_dict(), f"{path_prefix}_value.pth")
    
    def load_models(self, path_prefix: str):
        """Load trained models"""
        self.policy_network.load_state_dict(torch.load(f"{path_prefix}_policy.pth"))
        self.value_network.load_state_dict(torch.load(f"{path_prefix}_value.pth"))
```

## Contributing Guidelines

### Code Style and Standards

```python
# Python code style guidelines (following PEP 8 with extensions)

# 1. Import organization
"""Standard imports first, then third-party, then local imports"""
import os
import sys
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from gasm_bridge import create_bridge
from utils_se3 import SE3Utils

# 2. Function and class documentation
def process_spatial_instruction(
    text: str, 
    entities: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process natural language spatial instruction through GASM.
    
    This function converts natural language descriptions of spatial relationships
    into geometric constraints and target poses that can be executed by robots.
    
    Args:
        text: Natural language spatial instruction (e.g., "place box above table")
        entities: List of known entities in the scene. If None, entities will be
                 automatically extracted from the text.
        context: Additional context information such as workspace bounds,
                safety constraints, or domain-specific parameters.
    
    Returns:
        Dictionary containing:
            - success: Boolean indicating whether processing succeeded
            - constraints: List of geometric constraints
            - target_poses: Dictionary mapping entities to SE(3) poses
            - confidence: Confidence score [0.0, 1.0]
            - execution_time: Processing time in seconds
            - error_message: Error description if processing failed
    
    Raises:
        ValueError: If text is empty or invalid
        GASMProcessingError: If constraint generation fails
    
    Example:
        >>> result = process_spatial_instruction("place red block above blue cube")
        >>> if result['success']:
        ...     print(f"Generated {len(result['constraints'])} constraints")
        ...     for entity, pose in result['target_poses'].items():
        ...         print(f"{entity}: {pose['position']}")
    """
    pass

class GASMComponent:
    """
    Base class for GASM system components.
    
    This class provides common functionality for all GASM components including
    configuration management, logging, and lifecycle methods.
    
    Attributes:
        config: Component configuration dictionary
        logger: Logger instance for this component
        is_initialized: Whether component has been initialized
    
    Example:
        >>> class CustomComponent(GASMComponent):
        ...     def process(self, data):
        ...         self.logger.info(f"Processing {len(data)} items")
        ...         return self._custom_processing(data)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GASM component.
        
        Args:
            config: Configuration dictionary. If None, default configuration
                   will be used.
        """
        pass

# 3. Type hints and validation
from typing import Union, Tuple, TypeVar, Generic

T = TypeVar('T')
PoseType = Union[np.ndarray, Dict[str, Any], 'SE3Pose']

def validate_pose_input(pose: PoseType) -> np.ndarray:
    """Validate and normalize pose input to numpy array format."""
    if isinstance(pose, np.ndarray):
        if pose.shape != (4, 4):
            raise ValueError(f"Pose array must be 4x4, got {pose.shape}")
        return pose
    elif isinstance(pose, dict):
        if 'position' not in pose or 'orientation' not in pose:
            raise ValueError("Pose dict must contain 'position' and 'orientation'")
        return create_pose_from_dict(pose)
    else:
        raise TypeError(f"Unsupported pose type: {type(pose)}")

# 4. Error handling patterns
class GASMError(Exception):
    """Base class for GASM-related errors."""
    pass

class GASMConfigurationError(GASMError):
    """Raised when configuration is invalid."""
    pass

class GASMProcessingError(GASMError):
    """Raised when processing fails."""
    pass

def safe_processing_wrapper(func):
    """Decorator for safe processing with error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except GASMError:
            # Re-raise GASM-specific errors
            raise
        except Exception as e:
            # Wrap other exceptions
            raise GASMProcessingError(f"Unexpected error in {func.__name__}: {e}") from e
    return wrapper
```

### Testing Standards

```python
# Testing guidelines and examples

import unittest
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

class TestGASMBridge(unittest.TestCase):
    """Test suite for GASM bridge functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.bridge = create_bridge({
            'device': 'cpu',
            'fallback_mode': True,
            'timeout_seconds': 5
        })
        
        # Test data
        self.test_instructions = [
            "place box above table",
            "move robot left of sensor",
            "align objects along x-axis"
        ]
        
        self.valid_pose = {
            'position': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'frame_id': 'world',
            'confidence': 1.0
        }
    
    def tearDown(self):
        """Clean up after each test method"""
        if hasattr(self.bridge, 'cleanup'):
            self.bridge.cleanup()
    
    def test_basic_processing(self):
        """Test basic instruction processing functionality"""
        instruction = "place object above surface"
        result = self.bridge.process(instruction)
        
        # Basic structure validation
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('constraints', result)
        self.assertIn('confidence', result)
        self.assertIn('execution_time', result)
        
        # Type validation
        self.assertIsInstance(result['success'], bool)
        self.assertIsInstance(result['constraints'], list)
        self.assertIsInstance(result['confidence'], (int, float))
        self.assertIsInstance(result['execution_time'], (int, float))
    
    def test_constraint_generation(self):
        """Test constraint generation for various instruction types"""
        test_cases = [
            ("box above table", "above"),
            ("robot near sensor", "near"),
            ("objects 10cm apart", "distance")
        ]
        
        for instruction, expected_constraint_type in test_cases:
            with self.subTest(instruction=instruction):
                result = self.bridge.process(instruction)
                
                if result['success'] and result['constraints']:
                    constraint_types = [c['type'] for c in result['constraints']]
                    self.assertIn(expected_constraint_type, constraint_types)
    
    def test_pose_validation(self):
        """Test pose validation functionality"""
        # Valid pose
        self.assertTrue(self.bridge.validate_pose(self.valid_pose))
        
        # Invalid poses
        invalid_poses = [
            {'position': [0.0, 0.0]},  # Missing Z coordinate
            {'position': [0.0, 0.0, 0.0]},  # Missing orientation
            {'orientation': [0.0, 0.0, 0.0, 1.0]},  # Missing position
            {'position': [0.0, 0.0, 0.0], 'orientation': [1.0, 0.0, 0.0]},  # Wrong quaternion size
        ]
        
        for invalid_pose in invalid_poses:
            with self.subTest(pose=invalid_pose):
                self.assertFalse(self.bridge.validate_pose(invalid_pose))
    
    def test_error_handling(self):
        """Test error handling for various edge cases"""
        error_cases = [
            "",  # Empty string
            None,  # None input
            "sdfsdfsd invalid text 12345",  # Gibberish
            "a" * 10000,  # Very long text
        ]
        
        for error_input in error_cases:
            with self.subTest(input=error_input):
                result = self.bridge.process(error_input)
                
                # Should not crash and should return valid structure
                self.assertIsInstance(result, dict)
                self.assertIn('success', result)
                
                # For invalid inputs, success might be False
                if not result['success']:
                    self.assertIn('error_message', result)
    
    @patch('gasm_bridge.time.time')
    def test_execution_time_tracking(self, mock_time):
        """Test that execution time is tracked correctly"""
        # Mock time to return predictable values
        mock_time.side_effect = [0.0, 1.5]  # 1.5 second execution
        
        result = self.bridge.process("test instruction")
        
        self.assertAlmostEqual(result['execution_time'], 1.5, places=2)
    
    def test_concurrent_processing(self):
        """Test concurrent processing safety"""
        import threading
        
        results = []
        errors = []
        
        def process_instruction(instruction):
            try:
                result = self.bridge.process(f"place object {instruction}")
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=process_instruction, args=(f"instruction_{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(errors), 0, f"Errors in concurrent processing: {errors}")
        self.assertEqual(len(results), 10)
        
        # All results should be valid
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)

# Pytest-style tests
class TestSE3Utils:
    """Test SE(3) utility functions"""
    
    @pytest.fixture
    def sample_poses(self):
        """Fixture providing sample SE(3) poses"""
        return {
            'identity': np.eye(4),
            'translation': create_pose([1, 2, 3], orientation=None),
            'rotation': create_pose([0, 0, 0], [0, 0, np.pi/4], 'euler'),
            'combined': create_pose([1, 2, 3], [0.1, 0.2, 0.3], 'euler')
        }
    
    def test_homogeneous_matrix_creation(self, sample_poses):
        """Test homogeneous matrix creation and validation"""
        for name, pose in sample_poses.items():
            # Should be 4x4
            assert pose.shape == (4, 4)
            
            # Bottom row should be [0, 0, 0, 1]
            np.testing.assert_array_equal(pose[3, :], [0, 0, 0, 1])
            
            # Rotation part should be orthogonal
            R = pose[:3, :3]
            should_be_identity = R.T @ R
            np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-10)
            
            # Determinant should be 1
            assert np.isclose(np.linalg.det(R), 1.0)
    
    @pytest.mark.parametrize("translation,rotation_type,rotation", [
        ([1, 2, 3], 'euler', [0, 0, 0]),
        ([0, 0, 0], 'euler', [np.pi/2, 0, 0]),
        ([1, 1, 1], 'quaternion', [0, 0, 0, 1]),
        ([-1, -2, -3], 'quaternion', [0, 0, np.sin(np.pi/8), np.cos(np.pi/8)]),
    ])
    def test_pose_creation_parametrized(self, translation, rotation_type, rotation):
        """Parametrized test for pose creation"""
        pose = create_pose(translation, rotation, rotation_type)
        
        # Verify translation
        np.testing.assert_array_equal(pose[:3, 3], translation)
        
        # Verify it's a valid SE(3) matrix
        SE3Utils.validate_homogeneous_matrix(pose)
    
    def test_pose_operations(self, sample_poses):
        """Test pose composition and inverse operations"""
        T1 = sample_poses['translation']
        T2 = sample_poses['rotation']
        
        # Test composition
        T_composed = SE3Utils.pose_composition(T1, T2)
        SE3Utils.validate_homogeneous_matrix(T_composed)
        
        # Test inverse
        T1_inv = SE3Utils.pose_inverse(T1)
        SE3Utils.validate_homogeneous_matrix(T1_inv)
        
        # T * T^-1 should be identity
        identity_check = SE3Utils.pose_composition(T1, T1_inv)
        np.testing.assert_allclose(identity_check, np.eye(4), atol=1e-10)
    
    def test_geodesic_distance(self, sample_poses):
        """Test geodesic distance calculations"""
        identity = sample_poses['identity']
        translation = sample_poses['translation']
        
        # Distance from identity to itself should be 0
        assert SE3Utils.geodesic_distance_SE3(identity, identity) == 0.0
        
        # Distance should be symmetric
        dist1 = SE3Utils.geodesic_distance_SE3(identity, translation)
        dist2 = SE3Utils.geodesic_distance_SE3(translation, identity)
        assert np.isclose(dist1, dist2)
        
        # Distance should be positive
        assert dist1 > 0.0

# Performance tests
class TestGASMPerformance:
    """Performance benchmarks for GASM components"""
    
    @pytest.mark.slow
    def test_processing_speed_benchmark(self):
        """Benchmark processing speed"""
        bridge = create_bridge({'device': 'cpu'})
        
        instructions = [
            "place box above table",
            "move robot left of sensor", 
            "align objects along axis"
        ] * 100  # 300 instructions total
        
        start_time = time.time()
        
        for instruction in instructions:
            result = bridge.process(instruction)
            assert result['success'] or 'error_message' in result
        
        total_time = time.time() - start_time
        avg_time = total_time / len(instructions)
        
        # Performance requirements
        assert avg_time < 0.1, f"Average processing time {avg_time:.3f}s exceeds 0.1s limit"
        assert total_time < 30.0, f"Total processing time {total_time:.3f}s exceeds 30s limit"
        
        print(f"Performance: {avg_time*1000:.2f}ms average, {len(instructions)/total_time:.1f} instructions/sec")
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_acceleration(self):
        """Test GPU acceleration benefits"""
        instructions = ["place object above surface"] * 50
        
        # CPU processing
        cpu_bridge = create_bridge({'device': 'cpu'})
        start_time = time.time()
        for instruction in instructions:
            cpu_bridge.process(instruction)
        cpu_time = time.time() - start_time
        
        # GPU processing
        gpu_bridge = create_bridge({'device': 'cuda'})
        start_time = time.time()
        for instruction in instructions:
            gpu_bridge.process(instruction)
        gpu_time = time.time() - start_time
        
        # GPU should be faster (or at least not significantly slower)
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.2f}x")
        
        # Allow for some overhead in small batches
        assert speedup > 0.5, f"GPU processing is significantly slower: {speedup:.2f}x"

# Integration tests
@pytest.mark.integration
class TestGASMIntegration:
    """Integration tests for complete GASM workflows"""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end spatial reasoning workflow"""
        # Initialize system
        bridge = create_bridge()
        planner = create_default_planner()
        
        # Process instruction
        instruction = "place red block above blue cube"
        result = bridge.process(instruction)
        
        assert result['success'], f"Processing failed: {result.get('error_message')}"
        assert len(result['constraints']) > 0
        assert len(result['target_poses']) > 0
        
        # Use constraints in planning
        for entity, pose_data in result['target_poses'].items():
            target_pose = Pose(
                x=pose_data['position'][0],
                y=pose_data['position'][1], 
                z=pose_data['position'][2]
            )
            
            current_pose = Pose(x=0, y=0, z=0)
            
            planning_result = planner.plan_step(current_pose, target_pose)
            
            # Planning should succeed or provide meaningful feedback
            assert planning_result.success or len(planning_result.constraints_violated) > 0
    
    def test_robotic_integration_simulation(self):
        """Test integration with simulated robotic system"""
        # Mock robot controller
        robot_controller = Mock()
        robot_controller.move_to_pose.return_value = True
        robot_controller.get_current_pose.return_value = {
            'position': [0, 0, 0],
            'orientation': [0, 0, 0, 1]
        }
        
        # Integration layer
        integration = GASMRobotIntegration(create_bridge(), robot_controller)
        
        # Execute instruction
        success = integration.execute_instruction("move tool to position")
        
        # Verify interaction with robot
        assert robot_controller.move_to_pose.called
        assert robot_controller.get_current_pose.called
```

### Documentation Standards

```python
# Documentation guidelines

# 1. Module docstrings
"""
GASM Core Module - Neural Geometric Assembly State Machine

This module implements the core neural network architectures for geometric
assembly and spatial reasoning. It provides SE(3)-invariant attention
mechanisms, constraint-based optimization, and learning-based planning.

The main components are:
    - EnhancedGASM: Primary neural network for spatial reasoning
    - SE3InvariantAttention: Attention mechanism respecting SE(3) symmetries  
    - ConstraintHandler: Energy-based constraint satisfaction
    - GeometricLayer: Lie group operations and manifold computations

Example:
    Basic usage for spatial reasoning:
    
    >>> from gasm_core import EnhancedGASM
    >>> model = EnhancedGASM(feature_dim=256)
    >>> poses = torch.randn(10, 6)  # 10 poses in se(3)
    >>> features = torch.randn(10, 256)  # Associated features
    >>> output_poses = model(poses, features)
    >>> print(f"Output shape: {output_poses.shape}")  # [10, 6]

Mathematical Foundation:
    The system operates on the SE(3) manifold (Special Euclidean group in 3D)
    which represents rigid body transformations. Key mathematical concepts:
    
    - SE(3) group: {T = [R t; 0 1] | R ∈ SO(3), t ∈ ℝ³}
    - se(3) algebra: 6-dimensional tangent space at identity
    - Exponential map: se(3) → SE(3) for geodesic interpolation
    - Invariant operations: preserve geometric relationships under transformations

Author: GASM Development Team
Version: 2.0.0
License: MIT

References:
    [1] Cohen, T. & Welling, M. "Group Equivariant Convolutional Networks" (2016)
    [2] Finzi, M. et al. "A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups" (2021)
    [3] Sohl-Dickstein, J. et al. "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (2015)
"""

# 2. Class docstrings
class EnhancedGASM(nn.Module):
    """
    Enhanced Geometric Assembly State Machine with SE(3)-invariant processing.
    
    This neural network implements spatial reasoning for robotic assembly tasks
    using geometric deep learning principles. The architecture maintains
    SE(3) equivariance, ensuring that geometric relationships are preserved
    under rigid body transformations.
    
    Architecture:
        The network consists of multiple layers:
        1. Input embedding: Maps poses and features to high-dimensional space
        2. SE(3)-invariant attention: Computes relationships between objects
        3. Constraint integration: Incorporates geometric constraints
        4. Geometric layers: Perform Lie group operations
        5. Output projection: Generates target poses in se(3)
    
    Attributes:
        feature_dim (int): Dimensionality of internal feature representations
        num_attention_heads (int): Number of attention heads for multi-head attention
        device (torch.device): Computation device (CPU or CUDA)
        se3_group (SpecialEuclidean): Geometric computation backend
        constraint_weights (nn.ParameterDict): Learnable constraint importance weights
        
    Args:
        feature_dim (int, optional): Internal feature dimension. Defaults to 256.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        device (str, optional): Computation device. Defaults to 'auto'.
        use_constraints (bool, optional): Enable constraint processing. Defaults to True.
        
    Raises:
        ValueError: If feature_dim is not divisible by num_heads
        RuntimeError: If CUDA is requested but not available
        
    Example:
        Create and use GASM for spatial reasoning:
        
        >>> # Initialize model
        >>> gasm = EnhancedGASM(feature_dim=512, num_heads=16)
        >>> 
        >>> # Prepare input data
        >>> batch_size, num_objects = 4, 10
        >>> poses = torch.randn(batch_size, num_objects, 6)  # se(3) poses
        >>> features = torch.randn(batch_size, num_objects, 512)  # Object features
        >>> 
        >>> # Forward pass
        >>> output_poses = gasm(poses, features)
        >>> print(f"Input shape: {poses.shape}")   # [4, 10, 6]
        >>> print(f"Output shape: {output_poses.shape}")  # [4, 10, 6]
        >>> 
        >>> # With constraints
        >>> constraints = {
        ...     'above': torch.tensor([[0, 1, 0.1]]),  # Object 0 above object 1 by 0.1m
        ...     'distance': torch.tensor([[2, 3, 0.05]])  # Objects 2,3 distance 0.05m
        ... }
        >>> constrained_poses = gasm(poses, features, constraints=constraints)
        
    Note:
        This implementation assumes input poses are in se(3) algebra coordinates
        (6-dimensional vectors representing translation and rotation). For SE(3)
        matrix inputs, use SE3Utils.se3_log_map() for conversion.
        
    Mathematical Details:
        The SE(3)-invariant attention mechanism computes:
        
        Attention(Q, K, V) = softmax(QK^T / √d_k + G(poses))V
        
        where G(poses) encodes geometric relationships using geodesic distances
        on the SE(3) manifold, ensuring invariance to rigid transformations.
    """
    
    def __init__(self, feature_dim=256, num_heads=8, dropout=0.1, device='auto', use_constraints=True):
        super().__init__()
        # Implementation...
        
    def forward(self, poses, features, constraints=None, edge_index=None):
        """
        Forward pass through the GASM network.
        
        Processes object poses and features through SE(3)-invariant attention
        and geometric constraint satisfaction to generate optimized target poses.
        
        Args:
            poses (torch.Tensor): Input poses in se(3) coordinates.
                Shape: [batch_size, num_objects, 6] or [num_objects, 6]
                The 6 dimensions represent [translation_x, translation_y, translation_z,
                rotation_x, rotation_y, rotation_z] in axis-angle representation.
                
            features (torch.Tensor): Object feature embeddings.
                Shape: [batch_size, num_objects, feature_dim] or [num_objects, feature_dim]
                Features can include visual descriptors, semantic labels, or physical properties.
                
            constraints (dict, optional): Geometric constraints to satisfy.
                Dictionary with constraint types as keys:
                - 'above': Tensor of shape [num_constraints, 3] with [obj1_idx, obj2_idx, height]
                - 'distance': Tensor of shape [num_constraints, 3] with [obj1_idx, obj2_idx, distance]
                - 'angle': Tensor of shape [num_constraints, 4] with [obj1_idx, obj2_idx, axis, angle]
                Defaults to None (no constraints).
                
            edge_index (torch.Tensor, optional): Graph connectivity for attention.
                Shape: [2, num_edges]. Each column defines an edge between objects.
                If None, uses fully-connected attention.
                
        Returns:
            torch.Tensor: Optimized target poses in se(3) coordinates.
                Shape: Same as input poses. Values represent target configurations
                that satisfy the geometric constraints while minimizing assembly energy.
                
        Raises:
            RuntimeError: If input tensors have incompatible shapes
            ValueError: If constraint indices exceed number of objects
            
        Example:
            Process a simple assembly scenario:
            
            >>> # Two objects: block and base
            >>> poses = torch.tensor([[
            ...     [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],  # Block at height 0.1m
            ...     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # Base at ground level
            ... ]])
            >>> 
            >>> features = torch.randn(1, 2, 256)  # Random features
            >>> 
            >>> # Constraint: block should be above base by 0.2m
            >>> constraints = {
            ...     'above': torch.tensor([[0, 1, 0.2]])
            ... }
            >>> 
            >>> target_poses = gasm(poses, features, constraints)
            >>> print(f"Target block height: {target_poses[0, 0, 2]:.3f}")  # Should be ~0.2
            
        Note:
            The network maintains SE(3) equivariance, meaning that applying a rigid
            transformation to all input poses will result in the same transformation
            being applied to all output poses:
            
            GASM(T ∘ poses) = T ∘ GASM(poses)
            
            where T is any rigid transformation and ∘ denotes composition.
        """
        # Implementation...
        
    def compute_attention_weights(self, poses, features):
        """
        Compute SE(3)-invariant attention weights between objects.
        
        This method calculates attention weights that respect the geometric
        structure of SE(3), ensuring that relationships between objects
        remain consistent under rigid transformations.
        
        Args:
            poses (torch.Tensor): Object poses in se(3). Shape: [batch_size, num_objects, 6]
            features (torch.Tensor): Object features. Shape: [batch_size, num_objects, feature_dim]
            
        Returns:
            torch.Tensor: Attention weights. Shape: [batch_size, num_heads, num_objects, num_objects]
            
        Mathematical Details:
            Attention weights are computed using geodesic distances on SE(3):
            
            w_ij = exp(-d_SE3(pose_i, pose_j) / σ) · softmax(q_i^T k_j / √d_k)
            
            where d_SE3 is the geodesic distance on SE(3) manifold and σ is a
            learnable temperature parameter.
        """
        # Implementation...
```

## Quality Assurance

### Continuous Integration

```yaml
# .github/workflows/ci.yml
name: GASM CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']
        
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        pip install -e .
        
    - name: Lint with flake8
      run: |
        flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
        
    - name: Type check with mypy
      run: |
        mypy src/
        
    - name: Test with pytest
      run: |
        pytest tests/ --cov=src/ --cov-report=xml --cov-report=html
        
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        
  integration-test:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.9
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .
        
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
        
    - name: Run end-to-end tests
      run: |
        pytest tests/e2e/ -v --timeout=300
        
  performance-test:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.9
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .
        
    - name: Run performance benchmarks
      run: |
        pytest tests/performance/ -v --benchmark-only
        
    - name: Generate performance report
      run: |
        python scripts/generate_performance_report.py
        
  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security scan
      uses: pypa/gh-action-pip-audit@v1.0.0
      with:
        inputs: requirements.txt requirements-dev.txt
        
  build-docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.9
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install sphinx sphinx-rtd-theme
        
    - name: Build documentation
      run: |
        cd docs/
        make html
        
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

This comprehensive developer guide provides all the necessary information for contributing to and extending the GASM system. It covers environment setup, code standards, architectural patterns, testing requirements, and quality assurance processes.