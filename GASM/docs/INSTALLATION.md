# GASM-Roboting Installation Guide

This guide provides comprehensive instructions for installing GASM-Roboting in different configurations.

## Quick Start

### Option 1: Install from PyPI (Recommended)

```bash
# Basic installation
pip install gasm-roboting

# Full installation with all features
pip install gasm-roboting[all]

# Specific feature sets
pip install gasm-roboting[geometric,vision,robotics]
```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/gasm-roboting/gasm-roboting.git
cd gasm-roboting

# Install in development mode
pip install -e ".[dev,all]"

# Or use the automated setup script
python scripts/setup_environment.py --mode development --all
```

## Installation Options

### Core Installation
Minimal installation with basic functionality:
```bash
pip install gasm-roboting
```

Includes:
- Core GASM algorithms
- SE(3) attention mechanisms
- Basic spatial reasoning
- 2D visualization

### Feature-Specific Installations

#### Geometric Deep Learning
```bash
pip install gasm-roboting[geometric]
```
Adds:
- PyTorch Geometric for graph neural networks
- Geomstats for differential geometry
- Advanced geometric computations

#### Natural Language Processing
```bash
pip install gasm-roboting[nlp]
```
Adds:
- Transformers library
- SpaCy for text processing
- Language model integration

#### Computer Vision
```bash
pip install gasm-roboting[vision]
```
Adds:
- OpenCV for image processing
- Advanced visualization tools
- Vision-based spatial reasoning

#### Robotics & Physics
```bash
pip install gasm-roboting[robotics]
```
Adds:
- PyBullet physics simulation
- 3D robotic environments
- SE(3) pose control

#### Web Interface
```bash
pip install gasm-roboting[web]
```
Adds:
- FastAPI web server
- Gradio interactive interface
- RESTful API endpoints

#### Development Tools
```bash
pip install gasm-roboting[dev]
```
Adds:
- Testing frameworks (pytest)
- Code formatting (black, isort)
- Type checking (mypy)
- Documentation tools (Sphinx)

### Complete Installation
All features and dependencies:
```bash
pip install gasm-roboting[all]
```

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 4 GB RAM
- 2 GB disk space

### Recommended Requirements
- Python 3.9 or higher
- 8 GB RAM
- GPU with CUDA support (for large models)
- 5 GB disk space

### Operating System Support
- **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 10+
- **macOS**: 10.15+ (Catalina)
- **Windows**: Windows 10/11 (WSL2 recommended)

## GPU Support

### CUDA Installation
For GPU acceleration with geometric deep learning:

```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install GASM with geometric features
pip install gasm-roboting[geometric,vision]
```

### Verify GPU Support
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## Automated Setup

Use the built-in setup script for guided installation:

```bash
# Download and run setup script
curl -O https://raw.githubusercontent.com/gasm-roboting/gasm-roboting/main/scripts/setup_environment.py
python setup_environment.py --help

# Interactive setup
python setup_environment.py --mode production --gpu --vision --robotics

# Development setup
python setup_environment.py --mode development --extras all --verbose
```

### Setup Script Options
- `--mode {development,production,minimal}`: Installation mode
- `--extras EXTRAS`: Comma-separated feature list
- `--gpu`: Enable GPU optimizations
- `--vision`: Include computer vision
- `--robotics`: Include physics simulation
- `--check-only`: Verify current installation
- `--verbose`: Detailed output

## Verification

### Test Installation
```bash
# Run package tests
python -c "import gasm; gasm.quick_start()"

# Check dependencies
python -c "import gasm; print(gasm.check_dependencies())"

# Test command-line tools
gasm-setup --check-only
gasm-agent-2d --help
gasm-agent-3d --help
```

### Example Usage
```python
from gasm import GASM, SE3InvariantAttention
import torch

# Create a simple GASM model
model = GASM(
    feature_dim=768,
    hidden_dim=256,
    output_dim=3
)

# Test with dummy data
entities = ["robot", "conveyor", "sensor"]
features = torch.randn(3, 768)  # 3 entities, 768 features
relations = torch.randn(3, 3, 16)  # 3x3 relation matrix

# Run forward pass
result = model(entities, features, relations)
print(f"Output shape: {result.shape}")  # Should be (3, 3)
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Missing core dependencies
pip install torch numpy scipy matplotlib

# Missing optional dependencies
pip install gasm-roboting[geometric]  # for torch-geometric issues
pip install gasm-roboting[vision]     # for OpenCV issues
```

#### CUDA Issues
```bash
# Verify CUDA installation
nvidia-smi

# Install compatible PyTorch version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### Permission Errors
```bash
# Use virtual environment
python -m venv gasm_env
source gasm_env/bin/activate  # Linux/Mac
# gasm_env\Scripts\activate   # Windows

pip install gasm-roboting[all]
```

#### Memory Issues
```bash
# Install minimal version first
pip install gasm-roboting

# Then add features incrementally
pip install gasm-roboting[geometric]
pip install gasm-roboting[vision]
```

### Platform-Specific Notes

#### Linux
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install python3-dev python3-pip build-essential
sudo apt install libgl1-mesa-glx libglib2.0-0  # For OpenCV

# For PyBullet (optional)
sudo apt install libosmesa6-dev
```

#### macOS
```bash
# Using Homebrew
brew install python
brew install libomp  # For some ML libraries

# For M1/M2 Macs, use conda for better compatibility
conda install pytorch torchvision -c pytorch
pip install gasm-roboting
```

#### Windows
```bash
# Use WSL2 for best compatibility
wsl --install
# Then follow Linux instructions

# Or use Anaconda on Windows
conda install pytorch torchvision -c pytorch
pip install gasm-roboting
```

## Development Installation

For contributing to GASM-Roboting:

```bash
# Clone repository
git clone https://github.com/gasm-roboting/gasm-roboting.git
cd gasm-roboting

# Create development environment
python -m venv venv
source venv/bin/activate

# Install in editable mode with all development tools
pip install -e ".[dev,all]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Build documentation
cd docs && make html
```

## Docker Installation

Use Docker for containerized deployment:

```bash
# Pull pre-built image
docker pull gasm-roboting/gasm:latest

# Run container
docker run -p 8000:8000 gasm-roboting/gasm:latest

# Or build from source
git clone https://github.com/gasm-roboting/gasm-roboting.git
cd gasm-roboting
docker build -t gasm-roboting .
docker run -p 8000:8000 gasm-roboting
```

## Support

- **Documentation**: https://gasm-roboting.readthedocs.io/
- **Issues**: https://github.com/gasm-roboting/gasm-roboting/issues
- **Discussions**: https://github.com/gasm-roboting/gasm-roboting/discussions
- **Email**: dev@gasm-roboting.ai

For installation problems, please include:
1. Python version (`python --version`)
2. Operating system
3. Installation command used
4. Complete error message
5. Output of `pip list` or `conda list`