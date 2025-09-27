#!/usr/bin/env python3
"""
Setup script for OLYMPUS - Ethical Autonomous Intelligence Ecosystem
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements file
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="olympus-ai",
    version="1.0.0",
    author="Versino PsiOmega GmbH",
    author_email="dev@versino-psiomega.com",
    description="The world's first fully autonomous, self-aware, self-healing, collectively intelligent robotic ecosystem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/versino-psiomega/olympus",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "monitoring": [
            "grafana-api>=1.0.3",
            "elasticsearch>=8.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "olympus=olympus.cli:main",
            "olympus-monitor=olympus.monitoring:main",
            "olympus-sim=olympus.simulation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "olympus": [
            "configs/*.yaml",
            "configs/*.json",
            "assets/*",
            "templates/*",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/versino-psiomega/olympus/issues",
        "Source": "https://github.com/versino-psiomega/olympus",
        "Documentation": "https://versino-psiomega.github.io/olympus/",
    },
    keywords="robotics artificial-intelligence ethics autonomous-systems collective-intelligence",
)