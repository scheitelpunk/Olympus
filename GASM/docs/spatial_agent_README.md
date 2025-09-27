# Spatial Agent with 3D PyBullet Simulation

## Overview

The Spatial Agent provides comprehensive 3D spatial reasoning and physics simulation capabilities using PyBullet and GASM (Geometric Attention for Spatial Modeling). It enables natural language task execution in realistic 3D environments with physics-accurate object interactions.

## Features

### 🏗️ **Core Components**
- **PyBullet Integration**: Full 3D physics simulation with collision detection
- **GASM Bridge**: Geometric attention for spatial reasoning and planning  
- **SE(3) Pose Control**: Complete 6DOF pose representation and manipulation
- **URDF Loading**: Support for complex 3D models with physics properties
- **Vision Pipeline**: Computer vision processing for scene analysis
- **Constraint Handling**: Physics-based constraint satisfaction

### 🎯 **Simulation Capabilities**
- **Real-time Physics**: 240Hz simulation with customizable parameters
- **Multi-object Scenes**: Handle complex arrangements of multiple objects
- **Collision Detection**: Automatic collision detection and response
- **Camera Rendering**: Configurable camera views and video recording
- **Headless/GUI Modes**: Run with or without visual interface

### 🧠 **AI Integration**
- **Natural Language**: Parse spatial tasks from text descriptions
- **GASM Planning**: Use geometric attention for spatial arrangement planning
- **Adaptive Learning**: Neural patterns for improved performance over time
- **Constraint Reasoning**: Automatic constraint generation from spatial relations

## Installation

### Prerequisites

```bash
# Core dependencies
pip install torch torchvision
pip install numpy scipy

# Physics simulation
pip install pybullet

# Computer vision (optional)
pip install opencv-python

# GASM requirements
pip install geomstats torch-geometric
```

### Setup

```bash
# Clone the repository
cd GASM-Roboting/

# Install the spatial agent
pip install -e src/spatial_agent/
```

## Quick Start

### Basic Usage

```python
from src.spatial_agent import SpatialAgent, SimulationConfig, SimulationMode

# Create simulation configuration
config = SimulationConfig(
    max_steps=1000,
    time_step=1./240.,
    width=640,
    height=480
)

# Initialize spatial agent
agent = SpatialAgent(config, SimulationMode.HEADLESS)

# Execute spatial task
result = agent.execute_task("Place a red box next to a blue cube")

print(f"Success: {result['success']}")
print(f"Final metrics: {result['final_metrics']}")

# Cleanup
agent.cleanup()
```

### Command Line Interface

```bash
# Basic task execution
python src/spatial_agent/agent_loop_pybullet.py --text "Stack three boxes vertically" --steps 500

# GUI mode with vision
python src/spatial_agent/agent_loop_pybullet.py \
    --text "Arrange objects in a circle" \
    --render gui \
    --use_vision \
    --steps 1000

# Record video
python src/spatial_agent/agent_loop_pybullet.py \
    --text "Sort objects by size" \
    --render record \
    --save_video output.mp4 \
    --steps 2000

# Custom configuration
python src/spatial_agent/agent_loop_pybullet.py \
    --text "Complex assembly task" \
    --config src/spatial_agent/config.json \
    --render gui
```

## Architecture

### Core Classes

#### `SpatialAgent`
Main orchestrator that coordinates all components:
- Task parsing and execution
- Simulation management 
- Performance tracking
- Result aggregation

#### `PyBulletSimulation`
Physics simulation manager:
- Physics world setup
- Object creation and manipulation
- Constraint management
- Collision detection
- Camera rendering

#### `GASMBridge`
Integration with GASM neural networks:
- Text-to-spatial entity extraction
- Spatial arrangement planning
- SE(3) pose optimization
- Constraint satisfaction

#### `SE3Pose`
6DOF pose representation:
- Position and orientation handling
- Quaternion normalization
- Transformation matrices
- Pose interpolation

#### `PhysicsObject`
Wrapper for PyBullet objects:
- Pose management
- Force application
- Velocity control
- Constraint attachment

### Data Flow

```
Text Input → Entity Extraction → GASM Planning → Object Creation → Physics Simulation → Metrics Evaluation
     ↑                                                                        ↓
     └── Vision Feedback ← Camera Rendering ← Constraint Enforcement ←───────┘
```

## Configuration

### Simulation Parameters

```json
{
    "physics": {
        "gravity": [0.0, 0.0, -9.81],
        "time_step": 0.004166666666666667,
        "max_steps": 1000
    },
    "rendering": {
        "width": 640,
        "height": 480,
        "fov": 60.0,
        "near_plane": 0.1,
        "far_plane": 10.0
    },
    "camera": {
        "distance": 2.0,
        "yaw": 45.0,
        "pitch": -30.0,
        "target": [0.0, 0.0, 0.0]
    },
    "gasm": {
        "feature_dim": 768,
        "hidden_dim": 256,
        "num_heads": 8,
        "max_iterations": 10
    }
}
```

### URDF Objects

The system supports loading custom URDF files:

```xml
<?xml version="1.0"?>
<robot name="box">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.00167" ixy="0" ixz="0" iyy="0.00167" iyz="0" izz="0.00167"/>
    </inertial>
  </link>
</robot>
```

## Examples

### Example 1: Simple Placement

```python
# Place objects with spatial relations
agent.execute_task("Put the red box on top of the blue cube")
```

### Example 2: Complex Arrangement

```python
# Multi-object arrangement with constraints
result = agent.execute_task(
    "Arrange five boxes in a pentagon pattern with equal spacing"
)
```

### Example 3: Physics Interaction

```python
# Physics-based task
agent.execute_task("Drop spheres and let them roll into a container")
```

### Example 4: Vision-Guided Assembly

```python
# Use vision for feedback
config = SimulationConfig()
agent = SpatialAgent(config, SimulationMode.GUI)

result = agent.execute_task("Sort objects by color using vision feedback")
```

## Performance Metrics

The system tracks multiple performance indicators:

- **Constraint Satisfaction**: How well spatial constraints are met
- **Spatial Efficiency**: Compactness and organization of arrangements  
- **Collision Avoidance**: Percentage of collision-free configurations
- **Execution Time**: Task completion speed
- **Success Rate**: Overall task success percentage

## Testing

### Run Test Suite

```bash
python src/spatial_agent/test_simulation.py
```

### Individual Tests

```python
# Test basic functionality
python -c "
from src.spatial_agent.test_simulation import test_basic_simulation
test_basic_simulation()
"

# Test physics integration
python -c "
from src.spatial_agent.test_simulation import test_physics_integration  
test_physics_integration()
"
```

## Advanced Usage

### Custom Constraints

```python
# Define custom spatial constraints
constraints = {
    "distance": [
        [0, 1, 0.2],  # Objects 0 and 1 should be 20cm apart
        [1, 2, 0.15]  # Objects 1 and 2 should be 15cm apart
    ],
    "collision": [0.05],  # Minimum 5cm separation
    "angle": [
        [0, 1, 2, 90.0]  # 90 degree angle between objects 0-1-2
    ]
}

result = agent.gasm_bridge.plan_spatial_arrangement(entities, constraints)
```

### Video Recording

```python
# Record simulation video
agent = SpatialAgent(config, SimulationMode.RECORD)
result = agent.execute_task("Complex assembly sequence")
agent.save_video("assembly_demo.mp4")
```

### Performance Analysis

```python
# Execute multiple tasks and analyze performance
tasks = [
    "Stack boxes",
    "Arrange in circle", 
    "Sort by size",
    "Create pyramid"
]

for task in tasks:
    result = agent.execute_task(task)
    
# Get comprehensive performance summary
summary = agent.get_performance_summary()
print(f"Success rate: {summary['success_rate']:.1%}")
print(f"Avg constraint satisfaction: {summary['constraint_satisfaction']['mean']:.3f}")
```

## Troubleshooting

### Common Issues

1. **PyBullet Installation**
   ```bash
   # If PyBullet fails to install
   pip install --upgrade pip
   pip install pybullet --no-cache-dir
   ```

2. **GASM Dependencies**
   ```bash
   # Install geometric computing libraries
   pip install geomstats torch-geometric
   ```

3. **Graphics Issues**
   ```bash
   # For headless servers
   export DISPLAY=:99
   Xvfb :99 -screen 0 1024x768x24 &
   ```

4. **Memory Issues**
   ```python
   # Reduce simulation complexity
   config = SimulationConfig(
       max_steps=200,  # Reduce steps
       width=320,      # Lower resolution
       height=240
   )
   ```

### Performance Optimization

1. **Headless Mode**: Use `SimulationMode.HEADLESS` for faster execution
2. **Step Reduction**: Lower `max_steps` for simple tasks  
3. **Feature Dimensions**: Reduce GASM `feature_dim` and `hidden_dim`
4. **Batch Processing**: Group similar tasks together

## Contributing

1. **Code Style**: Follow PEP 8 guidelines
2. **Testing**: Add tests for new functionality
3. **Documentation**: Update docstrings and README
4. **Performance**: Profile and optimize critical paths

## License

This project is part of the GASM-Roboting framework and follows the same licensing terms.

## References

- [PyBullet Documentation](https://pybullet.org/wordpress/)
- [GASM: Geometric Attention for Spatial Modeling](../gasm_core.py)
- [SE(3) Group Theory](https://en.wikipedia.org/wiki/SE(3))
- [URDF Format Specification](http://wiki.ros.org/urdf)