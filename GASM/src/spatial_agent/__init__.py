"""
Spatial Agent Package - Comprehensive Spatial Reasoning and Control

This package provides comprehensive spatial agent capabilities including:
- Pose error calculation with SE(3) manifold geometry
- Constraint satisfaction scoring and evaluation
- Convergence detection with statistical analysis
- Progress tracking and debugging utilities

Main Components:
- SpatialMetricsCalculator: Core metrics computation engine
- PoseError: Detailed pose error representation
- ConstraintScore: Constraint satisfaction analysis
- ToleranceConfig: Configurable tolerance thresholds

The package supports various spatial reasoning tasks including:
- Robot motion planning and control
- Spatial constraint satisfaction
- Multi-agent coordination
- Geometric optimization

Author: Claude Code Implementation Agent
Version: 1.0.0
"""

# Optional imports to handle missing dependencies gracefully
try:
    from .metrics import (
        # Core classes
        SpatialMetricsCalculator,
        PoseError,
        ConstraintScore,
        ConstraintType,
        ToleranceConfig,
        ProgressMetrics,
        
        # Convenience functions
        calculate_pose_error,
        evaluate_constraints,
        check_convergence
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("⚠️ Metrics module not available - some functionality disabled")

# Vision system import
try:
    from .vision import VisionSystem, VisionConfig, Detection, create_vision_system, detect_objects
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("⚠️ Vision module not available")

# Build __all__ list based on available modules
__all__ = []

if METRICS_AVAILABLE:
    __all__.extend([
        'SpatialMetricsCalculator',
        'PoseError',
        'ConstraintScore', 
        'ConstraintType',
        'ToleranceConfig',
        'ProgressMetrics',
        'calculate_pose_error',
        'evaluate_constraints',
        'check_convergence'
    ])

if VISION_AVAILABLE:
    __all__.extend([
        'VisionSystem',
        'VisionConfig',
        'Detection',
        'create_vision_system',
        'detect_objects'
    ])

__version__ = "1.0.0"
__author__ = "Claude Code Implementation Agent"