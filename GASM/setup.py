#!/usr/bin/env python3
"""
GASM-Roboting Package Setup
==========================

Geometric Agent Swarm Models with Robotic Integration

A comprehensive package for geometric deep learning with SE(3)-invariant 
attention mechanisms, spatial reasoning, and robotic applications.
"""

import sys
import os
from pathlib import Path
from setuptools import setup, find_packages

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Read version from package
try:
    from src.gasm import __version__
except ImportError:
    __version__ = "0.1.0"

# Read long description from README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = """
    GASM-Roboting: Geometric Agent Swarm Models with Robotic Integration
    
    A comprehensive framework for geometric deep learning with SE(3)-invariant 
    attention mechanisms, spatial reasoning, and robotic applications.
    """

# Core dependencies
install_requires = [
    "torch>=2.0.0",
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "matplotlib>=3.5.0",
    "pillow>=8.0.0",
    "psutil>=5.9.0",
    "setuptools>=61.0",
]

# Optional dependencies for different features
extras_require = {
    "geometric": [
        "torch-geometric>=2.4.0",
        "geomstats>=2.7.0",
    ],
    "nlp": [
        "transformers>=4.21.0",
        "spacy>=3.7.0",
    ],
    "web": [
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "gradio>=4.16.0",
        "spaces>=0.19.0",
    ],
    "visualization": [
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "imageio>=2.19.0",
        "imageio-ffmpeg>=0.4.0",
    ],
    "robotics": [
        "pybullet>=3.2.0",
        "opencv-python>=4.7.0",
    ],
    "vision": [
        "opencv-python>=4.7.0",
        "pillow>=8.0.0",
        "scikit-image>=0.19.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=1.0.0",
        "pre-commit>=2.20.0",
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "build>=0.8.0",
        "twine>=4.0.0",
    ],
}

# Complete installation with all optional dependencies
extras_require["all"] = list(set(
    dep for deps in extras_require.values() for dep in deps
))

setup(
    name="gasm-roboting",
    version=__version__,
    author="Versino PsiOmega GmbH",
    author_email="dev@versino-psiomega.com",
    description="Geometric Agent Swarm Models with Robotic Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/versino-psiomega/olympus",
    project_urls={
        "Bug Reports": "https://github.com/versino-psiomega/olympus/issues",
        "Source": "https://github.com/versino-psiomega/olympus",
        "Documentation": "https://versino-psiomega.github.io/olympus/gasm/",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "spatial_agent": ["*.json", "objects/*.urdf"],
        "gasm": ["*.json"],
        "": ["*.md", "*.txt", "*.yaml", "*.yml"],
    },
    entry_points={
        "console_scripts": [
            "gasm-agent-2d=spatial_agent.agent_loop_2d:main",
            "gasm-agent-3d=spatial_agent.agent_loop_pybullet:main",
            "gasm-demo=spatial_agent.demo:main",
            "gasm-server=api.main:main",
            "gasm-setup=scripts.setup_environment:main",
        ],
    },
    zip_safe=False,
    keywords=[
        "geometric-deep-learning",
        "se3-invariance", 
        "spatial-reasoning",
        "robotics",
        "agent-swarms",
        "attention-mechanisms",
        "differential-geometry",
        "pytorch",
        "transformer-models"
    ],
)