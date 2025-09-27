# Changelog

All notable changes to the GASM-Roboting project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Modern Python packaging with pyproject.toml
- Comprehensive setup.py with optional dependencies
- CLI entry points for agent loops and utilities
- Environment setup script with dependency management
- Package distribution configuration

### Changed
- Reorganized project structure for proper packaging
- Updated requirements.txt with categorized dependencies
- Improved dependency management with optional extras

## [0.1.0] - 2025-01-09

### Added
- Initial release of GASM-Roboting framework
- SE(3)-invariant attention mechanisms
- Geometric constraint handling
- 2D and 3D spatial agent implementations
- PyBullet physics integration
- FastAPI web interface
- Computer vision integration
- Comprehensive test suite
- Documentation and examples

### Core Features
- **GASM Core**: Mathematically correct geometric attention
- **Spatial Agents**: 2D and 3D robotic agents with physics
- **SE(3) Invariance**: Proper geometric transformations
- **Constraint Handling**: Energy-based constraint satisfaction
- **Vision System**: Computer vision for spatial reasoning
- **API Interface**: RESTful API with FastAPI
- **CLI Tools**: Command-line interfaces for agents

### Dependencies
- PyTorch >= 2.0.0 for deep learning
- PyTorch Geometric >= 2.4.0 for graph neural networks  
- Geomstats >= 2.7.0 for differential geometry
- PyBullet >= 3.2.0 for physics simulation
- FastAPI >= 0.100.0 for web API
- OpenCV >= 4.7.0 for computer vision
- NumPy, SciPy, Matplotlib for scientific computing

### Package Structure
```
gasm-roboting/
├── src/gasm/           # Core GASM implementation
├── src/spatial_agent/  # Spatial reasoning agents
├── src/api/            # Web API interface
├── scripts/            # Utility scripts
├── tests/              # Test suite
├── docs/               # Documentation
└── examples/           # Usage examples
```

### Command Line Tools
- `gasm-agent-2d`: 2D spatial agent with matplotlib visualization
- `gasm-agent-3d`: 3D physics agent with PyBullet
- `gasm-demo`: Interactive demonstrations
- `gasm-server`: FastAPI web server
- `gasm-setup`: Environment setup and dependency management

### Installation Options
- **Minimal**: Core functionality only
- **Geometric**: With geometric deep learning libraries
- **Vision**: With computer vision support
- **Robotics**: With physics simulation
- **Web**: With API and web interface
- **Development**: With testing and development tools
- **All**: Complete installation with all features

### Supported Platforms
- Linux (Ubuntu 18.04+, CentOS 7+)
- macOS (10.15+)  
- Windows (WSL2 recommended)

### Python Support
- Python 3.8+
- PyTorch 2.0+ compatibility
- CUDA support (optional)

---

## Development Notes

### Version Numbering
- Major: Breaking changes to public API
- Minor: New features, backward compatible
- Patch: Bug fixes, backward compatible

### Release Process
1. Update version in `src/gasm/__init__.py`
2. Update this CHANGELOG.md
3. Create git tag: `git tag v0.1.0`
4. Build package: `python -m build`
5. Upload to PyPI: `python -m twine upload dist/*`

### Contributing
See CONTRIBUTING.md for development guidelines.

### License
MIT License - see LICENSE file for details.