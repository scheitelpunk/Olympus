# 2D Spatial Agent - GASM Demonstration

A complete, self-contained demonstration of **Geometric Algebra and Spatial Mathematics (GASM)** with real-time visualization and natural language interfaces.

## Overview

The 2D Spatial Agent implements a feedback loop for spatial reasoning:

```
Plan → Execute → Observe → Evaluate → Iterate
```

- **Plan**: Convert natural language to geometric constraints
- **Execute**: Apply GASM optimization to find optimal spatial arrangements  
- **Observe**: Update scene state and check for collisions
- **Evaluate**: Compute fitness scores and check convergence
- **Iterate**: Continue until convergence or maximum steps reached

## Features

✅ **Natural Language Interface**: "box above robot", "sensor near conveyor"  
✅ **Real-time Visualization**: Live matplotlib updates showing optimization progress  
✅ **2D Scene**: Simple conveyor belt and sensor with moveable entities  
✅ **Spatial Constraints**: Above, below, near, far, distance, angle relationships  
✅ **GASM Integration**: Uses SE(3)-invariant attention and geometric reasoning  
✅ **Convergence Detection**: Automatic stopping when solution is found  
✅ **GIF Export**: Save optimization animations for demonstrations  
✅ **CLI Interface**: Complete command-line tool with extensive options  

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install minimal requirements for spatial agent
pip install torch matplotlib numpy pillow imageio
```

### Basic Usage

```bash
# Simple spatial arrangement
python src/spatial_agent/agent_loop_2d.py --text "box above robot"

# Multiple constraints with animation
python src/spatial_agent/agent_loop_2d.py --text "robot near sensor, box left of robot" --save_video

# Fast execution without visualization
python src/spatial_agent/agent_loop_2d.py --text "box above conveyor" --no_visualization --steps 20
```

## Command Line Interface

```
python src/spatial_agent/agent_loop_2d.py [OPTIONS]

Required Arguments:
  --text TEXT              Natural language spatial description

Optional Arguments:
  --steps N               Maximum optimization steps (default: 50)
  --seed N                Random seed for reproducibility  
  --save_video           Save optimization as GIF animation
  --no_visualization     Disable real-time display (faster)
  --scene_size W H       Scene dimensions (default: 10.0 8.0)
  --convergence_threshold T  Convergence threshold (default: 1e-3)
  --verbose              Enable verbose logging
```

## Supported Spatial Relationships

The agent understands various spatial relationships:

### Directional
- `"A above B"` - A should be positioned above B
- `"A below B"` - A should be positioned below B  
- `"A left of B"` - A should be to the left of B
- `"A right of B"` - A should be to the right of B

### Proximity
- `"A near B"` - A should be close to B
- `"A far from B"` - A should be distant from B

### Complex Examples
- `"box above robot and sensor near box"`
- `"robot left of conveyor, box right of robot"`
- `"sensor far from conveyor, box above sensor"`

## Scene Objects

The 2D scene contains:

### Static Objects
- **Conveyor Belt**: Gray rectangle at position (2.0, 2.0)
- **Sensor**: Red circle at position (5.0, 5.0)

### Moveable Entities  
- **Box**: Blue square that can be optimized
- **Robot**: Green triangle that can be optimized

## Architecture

### Core Components

1. **TextToConstraints**: Parses natural language into geometric constraints
2. **Scene2D**: Manages 2D environment with collision detection
3. **VisualizationEngine**: Real-time matplotlib visualization with animations  
4. **SpatialAgent2D**: Main agent implementing the optimization loop
5. **GASM Integration**: Uses enhanced GASM model for spatial reasoning

### Optimization Flow

```python
# 1. Parse text to constraints
constraints = agent._plan(text_description)

# 2. Iterative optimization loop
for iteration in range(max_iterations):
    # Execute: Apply GASM optimization
    new_positions = agent._execute(constraints, iteration)
    
    # Observe: Update scene state
    agent._observe(new_positions)
    
    # Evaluate: Check fitness and convergence  
    evaluation = agent._evaluate(constraints, new_positions)
    
    if evaluation['converged']:
        break
```

## Examples

### Example 1: Simple Positioning
```bash
python src/spatial_agent/agent_loop_2d.py --text "box above robot" --seed 42
```

Output:
```
🎯 OPTIMIZATION RESULTS
✅ Success: True  
🔄 Iterations: 12
📊 Final Score: 0.0834
📍 FINAL POSITIONS:
  box: (3.45, 4.12)
  robot: (3.41, 2.18)
```

### Example 2: Complex Constraints with Animation
```bash
python src/spatial_agent/agent_loop_2d.py \
  --text "robot near sensor, box left of robot" \
  --save_video --steps 30
```

Creates `spatial_agent_demo.gif` showing the optimization process.

## Testing & Validation

### Comprehensive Test Suite
```bash
# Run complete test suite
python src/spatial_agent/test_agent_2d_complete.py
```

### Interactive Demonstrations
```bash
# Run all demos
python src/spatial_agent/demo_complete.py

# Run specific demo
python src/spatial_agent/demo_complete.py --demo 1  # Basic constraints
python src/spatial_agent/demo_complete.py --demo 3  # Visualization
python src/spatial_agent/demo_complete.py --demo 7  # Performance benchmark
```

### Quick Tests
```bash
# Basic functionality test
python src/spatial_agent/test_agent_2d.py --unittest

# Basic demo  
python src/spatial_agent/test_agent_2d.py --demo

# Test without dependencies (graceful fallbacks)
python src/spatial_agent/agent_loop_2d.py --text "box above robot" --no_visualization
```

## Performance

### Typical Results
- **Convergence**: Usually within 10-30 iterations
- **Speed**: ~2-5 iterations per second with visualization
- **Accuracy**: Constraint violations typically < 0.1 units

### Optimization Tips
- Use `--no_visualization` for faster execution
- Lower `--steps` for quicker results
- Set `--seed` for reproducible experiments

## Integration with GASM Core

The spatial agent integrates with the main GASM system:

- **SE(3) Invariance**: Maintains geometric properties under transformations
- **Attention Mechanisms**: Uses SE(3)-invariant attention for spatial reasoning
- **Constraint Handling**: Energy-based constraint satisfaction
- **Robustness**: Graceful fallbacks when GASM components are unavailable

## Extending the Agent

### Adding New Constraints
```python
# In TextToConstraints class
def _parse_custom_constraint(self, text, entity_map):
    """Parse custom spatial relationship"""
    constraints = []
    # Add parsing logic
    return constraints

# Register in __init__
self.constraint_parsers['custom'] = self._parse_custom_constraint
```

### Adding New Entities
```python
# In Scene2D.__init__  
self.entities['new_entity'] = {
    'position': [x, y], 
    'size': size,
    'type': 'shape_type'
}
```

### Custom Visualization
```python
# In VisualizationEngine._create_entity_artists
if entity['type'] == 'custom_shape':
    artist = create_custom_artist(pos, size, color)
```

## Troubleshooting

### Common Issues

**ImportError with GASM core**:
- The agent includes fallback implementations  
- Core functionality works without full GASM

**Visualization not appearing**:
- Check matplotlib backend: `python -c "import matplotlib; print(matplotlib.get_backend())"`
- Try `--no_visualization` flag

**Slow convergence**:
- Increase `--steps` parameter
- Adjust `--convergence_threshold`
- Check constraint complexity

**Animation saving fails**:
- Install imageio: `pip install imageio imageio-ffmpeg`
- Check write permissions in current directory

### Debug Mode
```bash
python src/spatial_agent/agent_loop_2d.py --text "..." --verbose
```

## Technical Details

### State Representation
Each entity has a pose (x, y, θ) in 2D space, extended to (x, y, 0) for 3D compatibility with GASM.

### Constraint Encoding  
Spatial relationships are converted to optimization constraints:
- Positional: `y_A > y_B + offset` for "A above B"
- Distance: `||pos_A - pos_B|| = target_distance`

### Optimization Strategy
1. **GASM Forward Pass**: SE(3)-invariant attention with geometric reasoning
2. **Constraint Application**: Energy-based constraint satisfaction
3. **Boundary Clamping**: Ensure entities stay within scene bounds
4. **Collision Avoidance**: Penalty for overlapping with static objects

## Future Enhancements

🔮 **Planned Features**:
- [ ] 3D spatial reasoning with full SE(3) poses
- [ ] Dynamic objects (moving conveyor, rotating sensors)  
- [ ] Multi-step task planning
- [ ] Learning from successful arrangements
- [ ] Natural language explanation of solutions
- [ ] Web interface with interactive controls

## License

Part of the GASM-Roboting project. See main repository for license details.

---

**Ready to explore spatial reasoning with GASM? Start with a simple example and watch your entities find their optimal arrangements!** 🤖✨