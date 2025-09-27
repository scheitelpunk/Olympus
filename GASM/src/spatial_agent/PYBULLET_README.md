# 3D Spatial Agent with PyBullet Physics Integration

A complete implementation of an intelligent spatial reasoning agent that operates in a 3D physics-based simulation environment using PyBullet. This system integrates advanced trajectory planning, collision detection, computer vision, and GASM (Geometric Attention for Spatial Reasoning) for autonomous spatial task execution.

## 🚀 Features

### Core Capabilities
- **Full PyBullet physics simulation** with realistic dynamics
- **6DOF SE(3) pose control** for precise spatial manipulation  
- **URDF loading and procedural object generation**
- **Advanced trajectory planning** (linear, cubic, quintic polynomials)
- **Real-time collision detection and resolution**
- **Multi-modal vision pipeline** with 2D-to-3D pose estimation
- **Constraint-based spatial reasoning** with iterative optimization
- **Comprehensive feedback loops** integrating all components

### Advanced Features
- **GASM integration** for intelligent spatial arrangement planning
- **Dynamic replanning** based on performance feedback
- **Stability verification** and quality metrics
- **Video recording** and simulation export
- **Interactive GUI and headless modes**
- **Comprehensive test suite** with benchmarking
- **Extensible configuration system**

## 📋 System Requirements

### Required Dependencies
```bash
# Core physics and math
pybullet>=3.2.5          # 3D physics simulation
numpy>=1.21.0            # Numerical computing
scipy>=1.7.0             # Scientific computing

# Computer vision (optional but recommended)
opencv-python>=4.5.0     # Vision processing
pillow>=8.0.0            # Image handling

# Machine learning (optional for GASM)
torch>=2.0.0             # Neural networks
torch-geometric>=2.4.0   # Geometric deep learning

# 3D processing (optional)
trimesh>=3.21.0          # 3D mesh processing
networkx>=2.8.0          # Graph algorithms
```

### System Requirements
- Python 3.8+ (Python 3.9+ recommended)
- 4GB+ RAM (8GB+ recommended for complex scenes)
- OpenGL 3.3+ support for GUI rendering
- GPU support optional but recommended for neural components

## 🛠️ Installation

### Quick Install
```bash
# Navigate to spatial agent directory
cd src/spatial_agent/

# Install all dependencies
pip install pybullet>=3.2.5 opencv-python>=4.5.0 torch>=2.0.0 trimesh>=3.21.0 networkx>=2.8.0

# Or use the automated installer
python run_spatial_agent.py --install-deps
```

### Verify Installation
```bash
# Check system requirements
python run_spatial_agent.py --diagnostics

# Run comprehensive tests  
python run_spatial_agent.py --test
```

## 🎮 Usage

### Command Line Interface

#### Basic Task Execution
```bash
# Execute spatial task in GUI mode
python agent_loop_pybullet.py --text "place red box near blue sphere" --render gui

# Record simulation video
python agent_loop_pybullet.py --text "sort three objects by size" --render record --save_video output.mp4

# Use custom configuration
python agent_loop_pybullet.py --text "robot arm picks up cylinder" --config config.json --steps 2000
```

#### Demo and Benchmarking
```bash
# Run comprehensive demo scenarios
python agent_loop_pybullet.py --demo

# Performance benchmarks
python agent_loop_pybullet.py --benchmark --iterations 10

# Export metrics to JSON
python agent_loop_pybullet.py --text "arrange objects" --export_metrics results.json
```

### Launcher Script (Recommended)
```bash
# Interactive mode (default)
python run_spatial_agent.py

# Quick task execution
python run_spatial_agent.py --quick "place box on conveyor belt"

# Run demo scenarios
python run_spatial_agent.py --demo

# System diagnostics
python run_spatial_agent.py --diagnostics
```

### Interactive Mode
```bash
python run_spatial_agent.py --interactive

# Interactive commands:
🤖 Spatial Agent > place red box near blue sphere
🤖 Spatial Agent > demo                    # Run demos
🤖 Spatial Agent > benchmark              # Run benchmarks  
🤖 Spatial Agent > gui                    # Switch to GUI mode
🤖 Spatial Agent > help                   # Show help
🤖 Spatial Agent > quit                   # Exit
```

## 📚 API Usage

### Basic Python API
```python
from agent_loop_pybullet import SpatialAgent, SimulationConfig, SimulationMode

# Initialize agent
config = SimulationConfig()
agent = SpatialAgent(config, SimulationMode.GUI)

# Execute spatial task
result = agent.execute_task("arrange three boxes in a triangle formation")

print(f"Success: {result['success']}")
print(f"Constraint Satisfaction: {result['final_metrics']['constraint_satisfaction']:.3f}")

# Cleanup
agent.cleanup()
```

### Advanced Configuration
```python
# Custom simulation configuration
config = SimulationConfig()
config.max_steps = 2000
config.time_step = 1./240.  # 240Hz physics
config.width, config.height = 1280, 720  # HD rendering
config.gravity = (0., 0., -9.81)

# Advanced constraint settings
config.position_tolerance = 0.005  # 5mm precision
config.collision_margin = 0.02     # 2cm safety margin

agent = SpatialAgent(config, SimulationMode.RECORD)
```

### Component Access
```python
# Access individual components
simulation = agent.simulation          # PyBullet physics
gasm_bridge = agent.gasm_bridge       # GASM spatial reasoning  
vision = agent.vision                 # Vision pipeline
trajectory_planner = agent.trajectory_planner  # Motion planning
constraint_solver = agent.constraint_solver    # Constraint optimization
collision_resolver = agent.collision_resolver  # Collision handling

# Direct object manipulation
urdf_path = "objects/robot_arm.urdf"
pose = SE3Pose([0, 0, 0.5], [0, 0, 0, 1])
robot = simulation.load_urdf_object(urdf_path, "robot_arm", pose)
```

## 🏗️ Architecture

### System Components
```
SpatialAgent (Main Controller)
├── PyBulletSimulation (Physics Engine)
│   ├── Environment Setup
│   ├── URDF Loading
│   ├── Procedural Generation
│   └── Physics Stepping
├── GASMBridge (Spatial Reasoning)
│   ├── Text Parsing
│   ├── Entity Extraction  
│   └── Arrangement Planning
├── TrajectoryPlanner (Motion Planning)
│   ├── Polynomial Trajectories
│   ├── SLERP Interpolation
│   └── Smooth Motion Control
├── ConstraintSolver (Optimization)
│   ├── Distance Constraints
│   ├── Alignment Constraints
│   └── Iterative Optimization
├── CollisionResolver (Safety)
│   ├── Predictive Detection
│   ├── Resolution Strategies
│   └── Recovery Mechanisms
├── VisionPipeline (Perception)
│   ├── 2D Object Detection
│   ├── 3D Pose Estimation
│   └── Visual Feedback
└── SpatialMetrics (Evaluation)
    ├── Quality Assessment
    ├── Performance Tracking
    └── Success Metrics
```

### Data Flow
1. **Task Input** → Text parsing and entity extraction
2. **Spatial Planning** → GASM generates initial arrangement
3. **Constraint Solving** → Optimization for feasible poses
4. **Trajectory Generation** → Smooth motion planning
5. **Physics Simulation** → Real-time dynamics execution
6. **Feedback Loop** → Vision, collision detection, replanning
7. **Quality Assessment** → Performance metrics and success evaluation

## 🎯 Task Examples

### Basic Spatial Tasks
```bash
# Object placement
"place the red box near the blue sphere"
"put the cylinder on top of the cube"
"arrange objects in a line on the conveyor belt"

# Spatial relationships  
"sort three cubes by size from largest to smallest"
"align all spheres along the x-axis with equal spacing"
"create a pyramid with boxes as the base and sphere on top"

# Complex arrangements
"arrange objects in a circle around the sensor mount"
"create a balanced stack without any collisions"  
"organize tools on the conveyor belt for assembly line"
```

### Robot Manipulation Tasks
```bash
# Robot arm operations
"use robot arm to pick up the cylinder and place it on the sensor"
"robot arm sorts objects into two groups based on color"
"robotic assembly of components on the conveyor belt"

# Precision tasks
"robot places small parts with 1mm accuracy"
"use robot to thread objects through narrow passages"
"robotic quality inspection with vision feedback"
```

### Multi-Object Scenarios  
```bash
# Sorting and organization
"sort mixed objects by color, size, and shape"
"organize tools in optimal configuration for efficiency"
"separate defective parts from good ones using vision"

# Collision avoidance
"move five objects to new positions without any collisions"
"rearrange workspace while maintaining all safety constraints"
"optimize layout to minimize total movement distance"
```

## 🔧 Configuration

### Main Configuration File (`config.json`)
```json
{
  "physics": {
    "time_step": 0.004166666666666667,
    "max_steps": 1000,
    "solver_iterations": 150,
    "gravity": [0.0, 0.0, -9.81]
  },
  "rendering": {
    "width": 640, "height": 480,
    "fov": 60.0,
    "enable_shadows": true
  },
  "trajectory": {
    "default_duration": 2.0,
    "max_velocity": 1.0,
    "trajectory_type": "quintic"
  },
  "collision": {
    "detection_margin": 0.05,
    "resolution_strategy": "separate"
  }
}
```

### Performance Tuning
```json
{
  "performance": {
    "target_fps": 60,
    "quality_vs_speed": "balanced",
    "adaptive_time_step": false
  },
  "physics": {
    "solver_iterations": 150,  // Higher = more accurate
    "contact_erp": 0.2,        // Contact error reduction
    "friction": 0.8            // Surface friction
  }
}
```

## 📊 Performance & Metrics

### Quality Metrics
- **Constraint Satisfaction**: How well spatial constraints are met (0-1)
- **Collision Free**: Percentage of simulation without collisions (0-1)  
- **Spatial Efficiency**: Compactness and organization quality (0-1)
- **Stability**: System stability over time (0-1)

### Performance Tracking
```python
# Access performance data
summary = agent.get_performance_summary()
print(f"Success Rate: {summary['success_rate']:.1%}")
print(f"Average Constraint Satisfaction: {summary['constraint_satisfaction']['mean']:.3f}")
print(f"Total Collisions: {summary['collision_analysis']['total_collisions']}")
```

### Benchmarking
```bash
# Run standardized benchmarks
python run_spatial_agent.py --benchmark --iterations 20

# Results include:
# - Success rates across task types
# - Average execution times  
# - Constraint satisfaction statistics
# - Collision avoidance performance
# - System resource usage
```

## 🧪 Testing

### Comprehensive Test Suite
```bash
# Run all tests
python test_pybullet_agent.py

# Test categories:
# - SE(3) pose mathematics
# - Physics simulation integration  
# - Constraint solving algorithms
# - Trajectory planning accuracy
# - Collision detection/resolution
# - Vision pipeline functionality
# - GASM spatial reasoning
# - Integration tests
```

### Custom Testing
```python
import unittest
from test_pybullet_agent import TestSpatialAgentIntegration

# Run specific test class
suite = unittest.TestLoader().loadTestsFromTestCase(TestSpatialAgentIntegration)
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
```

## 🎥 Simulation Recording

### Video Export
```bash
# Record simulation as video
python agent_loop_pybullet.py --render record --save_video demo.mp4 --text "sort objects"

# Custom resolution and framerate
python agent_loop_pybullet.py --render record --config high_quality_config.json
```

### Data Export
```bash
# Export comprehensive metrics
python agent_loop_pybullet.py --text "task" --export_metrics data.json

# Exported data includes:
# - Step-by-step simulation states
# - Object trajectories and poses
# - Performance metrics over time
# - Collision events and resolutions
# - Vision processing results
```

## 🔍 Debugging and Visualization

### Debug Modes
```json
{
  "debug": {
    "show_wireframes": true,
    "show_coordinate_frames": true,
    "show_contact_points": true,
    "show_constraints": true,
    "show_trajectories": true,
    "physics_debug_mode": true
  }
}
```

### Logging Configuration
```json
{
  "logging": {
    "level": "DEBUG",
    "log_to_file": true,
    "log_file": "spatial_agent.log",
    "log_physics_warnings": true,
    "log_performance": true,
    "log_trajectories": true
  }
}
```

## 🚨 Troubleshooting

### Common Issues

#### PyBullet Installation Issues
```bash
# Linux
sudo apt-get install python3-dev
pip install pybullet

# macOS  
brew install cmake
pip install pybullet

# Windows
pip install pybullet
```

#### OpenGL/Rendering Issues
```bash
# Linux - install OpenGL drivers
sudo apt-get install mesa-utils

# Test OpenGL
glxgears

# Fallback to headless mode
python agent_loop_pybullet.py --render headless
```

#### Memory Issues with Large Scenes
```python
# Reduce simulation complexity
config = SimulationConfig()
config.max_steps = 500        # Fewer simulation steps
config.time_step = 1./60.     # Lower frequency
config.width, config.height = 320, 240  # Lower resolution
```

### Performance Optimization
```python
# High-performance configuration  
config = SimulationConfig()
config.time_step = 1./120.    # 120Hz physics
config.solver_iterations = 100  # Fewer solver iterations
config.target_fps = 30        # Lower render framerate

# Disable expensive features
config.enable_shadows = False
config.log_performance = False
config.vision_enabled = False
```

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository_url>
cd spatial_agent

# Install development dependencies  
pip install -e .
pip install pytest black isort mypy

# Run tests
python -m pytest test_pybullet_agent.py -v

# Code formatting
black agent_loop_pybullet.py
isort agent_loop_pybullet.py
```

### Adding New Features
1. **URDF Objects**: Add new `.urdf` files to `objects/` directory
2. **Constraint Types**: Extend `ConstraintSolver` class with new constraint types
3. **Trajectory Types**: Add new trajectory planning algorithms to `TrajectoryPlanner`
4. **Vision Processing**: Enhance `VisionPipeline` with advanced computer vision
5. **Spatial Reasoning**: Extend `GASMBridge` with additional spatial intelligence

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **PyBullet Team** for the excellent physics simulation framework
- **OpenAI** for inspiration from robotic manipulation research  
- **Stanford AI Lab** for spatial reasoning methodologies
- **Geometric Deep Learning Community** for GASM-related techniques

---

## 📞 Support

For issues, questions, or contributions:
- Open an issue on the repository
- Review the troubleshooting section
- Run diagnostics: `python run_spatial_agent.py --diagnostics`
- Check the comprehensive test suite for examples

**Happy Spatial Reasoning! 🤖🎯**