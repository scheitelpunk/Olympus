from setuptools import setup, find_packages

setup(
    name="morpheus",
    version="0.1.0",
    description="Multi-modal Optimization through Replay, Prediction, and Haptic-Environmental Understanding System",
    author="MORPHEUS Development Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "torch>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "pybullet>=3.2.5",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "scikit-learn>=1.3.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "sqlalchemy>=2.0.0",
        "alembic>=1.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "pytest-cov>=4.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "morpheus=morpheus.cli:main",
        ],
    },
)