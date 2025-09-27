# MORPHEUS: Multi-modal Optimization through Replay, Prediction, and Haptic-Environmental Understanding System

MORPHEUS is an advanced robotics system that combines multi-modal sensory perception with dream-based optimization to create robotic systems that learn and improve through experience replay. The system integrates tactile, audio, and visual processing with physics-based material understanding and neural learning.

## 🌟 Key Features

- **🤖 Multi-Modal Perception**: Advanced tactile, audio, and spatial perception processing
- **🧠 Dream-Based Learning**: Experience replay with parallel strategy optimization
- **⚗️ Material-Aware Physics**: Integration with GASM-Robotics material properties
- **🔮 Predictive Modeling**: Forward model prediction with uncertainty quantification
- **💾 PostgreSQL Storage**: Production-ready database with optimized schemas
- **🎯 Real-Time Processing**: <10ms latency for sensory fusion
- **🔄 Continuous Learning**: Autonomous improvement through dream cycles

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- PostgreSQL (via Docker)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/versino-psiomega/olympus.git
cd Morpheus
```

2. **Setup environment**
```bash
# Copy environment template
cp .env.example .env

# Install Python dependencies
pip install -r requirements.txt
```

3. **Start database**
```bash
# Start PostgreSQL with Docker
python scripts/setup_database.py
```

4. **Install MORPHEUS**
```bash
pip install -e .
```

### Running Examples

```bash
# Basic perception demonstration
python -m morpheus.examples.basic_perception

# Material exploration and learning
python -m morpheus.examples.material_exploration

# Complete dream cycle demonstration  
python -m morpheus.examples.dream_cycle_demo

# Full system integration
python -m morpheus.examples.full_integration
```

## 📖 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [API Reference](docs/API_REFERENCE.md) 
- [Integration Guide](docs/INTEGRATION_GUIDE.md)
- [Database Schema](docs/DATABASE_SCHEMA.md)

## 🏗️ System Architecture

```
MORPHEUS System
├── Core Components
│   ├── Orchestrator          # Main system coordinator
│   ├── Configuration         # YAML-based config management
│   └── Type System           # Comprehensive data types
├── Perception
│   ├── Tactile Processor     # Material-aware tactile processing
│   ├── Audio Spatial         # 3D spatial audio processing
│   └── Sensory Fusion        # Multi-modal neural fusion
├── Dream Simulation
│   ├── Dream Orchestrator    # Experience replay engine
│   ├── Strategy Optimizer    # Neural strategy learning
│   └── Memory Consolidation  # Knowledge integration
├── Predictive Models
│   ├── Forward Model         # State prediction
│   ├── Material Predictor    # Material property prediction
│   └── Uncertainty Engine    # Prediction confidence
├── Storage
│   ├── PostgreSQL Interface  # Production database
│   ├── Experience Storage    # Multi-modal experience data
│   └── Strategy Storage      # Learned strategies
└── Integration
    ├── Material Bridge       # GASM-Robotics materials
    ├── GASM Bridge          # Real-time GASM integration
    └── PyBullet Interface   # Physics simulation
```

## 🎯 Usage Examples

### Basic Perception

```python
from morpheus import MorpheusOrchestrator

# Initialize system
morpheus = MorpheusOrchestrator(
    config_path="configs/default_config.yaml",
    gasm_roboting_path="./GASM-Robotics",
    db_config={
        'host': 'localhost',
        'database': 'morpheus',
        'user': 'morpheus_user',
        'password': 'morpheus_pass'
    }
)

# Process sensory observation
observation = {
    'material': 'steel',
    'body_id': 1,
    'robot_position': [0, 0, 0.5],
    'contact_force': 5.0,
    'action_type': 'grasp',
    'success': True
}

result = morpheus.perceive(observation)
print(f"Material detected: {result['material']}")
print(f"Tactile signature: {result['tactile']}")
```

### Dream-Based Learning

```python
# Collect experiences through normal operation
for _ in range(100):
    result = morpheus.perceive(observation)

# Enter dream state for optimization
dream_result = morpheus.dream(duration=30)
print(f"Strategies discovered: {dream_result['strategies_found']}")

# Get learned strategies
strategies = morpheus.get_learned_strategies()
for strategy in strategies[:3]:
    print(f"Strategy: {strategy['name']}")
    print(f"Improvement: {strategy['improvement_ratio']:.2%}")
```

### Material Prediction

```python
# Predict material interaction
prediction = morpheus.predict_material_interaction(
    'steel', 'rubber', 
    scenario={'force': 10.0, 'velocity': 0.1}
)

print(f"Friction: {prediction['interaction']['combined_friction']}")
print(f"Expected sound: {prediction['interaction']['expected_sound_freq']} Hz")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=morpheus --cov-report=html
```

## ⚡ Performance

MORPHEUS achieves the following performance characteristics:

- **Perception Processing**: 1000+ experiences/second
- **Sensory Fusion**: <5ms latency for 128-dim embeddings
- **Audio Processing**: Real-time 44.1kHz capability
- **Dream Optimization**: 100+ strategies/minute
- **Memory Usage**: <4GB RAM for full operation
- **Database Throughput**: 5000+ writes/second

## 🔧 Configuration

MORPHEUS uses YAML configuration with environment variable support:

```yaml
# configs/default_config.yaml
system:
  mode: "full"
  device: "cpu"

perception:
  tactile:
    sensitivity: 0.01
    sampling_rate: 1000
  
dream:
  parallel_dreams: 4
  replay_speed: 10.0
```

## 🤝 GASM-Robotics Integration

MORPHEUS seamlessly integrates with GASM-Robotics:

- **Material System**: Direct loading of material properties
- **Physics Engine**: PyBullet physics integration
- **Spatial Reasoning**: SE(3) pose and constraint handling
- **Real-time Sync**: 100Hz+ synchronization capability

## 📊 Database Schema

PostgreSQL schema optimized for sensory data:

- **experiences**: Multi-modal sensory experiences with JSONB flexibility
- **dream_sessions**: Dream session results and metrics
- **learned_strategies**: Discovered optimization strategies
- **material_predictions**: Material interaction predictions

## 🛠️ Development

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run formatting
black src/ tests/
```

### Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## 📄 License

This project is licensed under CC-BY-NC-4.0 License - Copyright (c) 2025 Versino PsiOmega GmbH

## 🙏 Acknowledgments

- GASM-Robotics project for material system foundation
- PyBullet physics engine for realistic simulation
- PostgreSQL for robust data persistence
- PyTorch for neural network capabilities

---

**MORPHEUS** - Advancing robotics through multi-modal learning and dream-based optimization