"""
GASM (Geometric Attention with Spatial Manifolds) Package

This package provides geometric attention mechanisms for neural networks
with proper SE(3) invariance and manifold-aware processing.
"""

import logging

# Package metadata
__version__ = "1.0.0"
__author__ = "Versino PsiOmega GmbH"
__description__ = "Geometric Attention with Spatial Manifolds for PyTorch"

logger = logging.getLogger(__name__)

# Try to import core components with graceful error handling
try:
    from .core import (
        MathematicallyCorrectGASM,
        EnhancedGASM,
        SE3InvariantAttention,
        UniversalInvariantAttention,
        EfficientCurvatureComputation,
        ConstraintHandler,
        EnhancedBatchProcessor,
        ErrorRecoveryWrapper,
        robust_function,
    )
    
    # Make GASM alias to enhanced version
    GASM = EnhancedGASM
    _core_available = True
    
except ImportError as e:
    logger.warning(f"Could not import core GASM components: {e}")
    _core_available = False
    
    # Create placeholder classes
    class _PlaceholderGASM:
        def __init__(self, *args, **kwargs):
            raise ImportError("GASM core components not available. Please check dependencies.")
    
    MathematicallyCorrectGASM = _PlaceholderGASM
    EnhancedGASM = _PlaceholderGASM
    GASM = _PlaceholderGASM
    SE3InvariantAttention = _PlaceholderGASM
    UniversalInvariantAttention = _PlaceholderGASM
    EfficientCurvatureComputation = _PlaceholderGASM
    ConstraintHandler = _PlaceholderGASM
    EnhancedBatchProcessor = _PlaceholderGASM
    ErrorRecoveryWrapper = _PlaceholderGASM
    robust_function = lambda *args, **kwargs: lambda f: f

# Try to import utility functions
try:
    from .utils import (
        check_se3_invariance,
        generate_random_se3_transform,
        apply_se3_transform,
        compute_geodesic_distance,
        validate_edge_index,
        create_fallback_edges,
        safe_tensor_operation,
        compute_attention_entropy,
        extract_geometric_features,
    )
    _utils_available = True
    
except ImportError as e:
    logger.warning(f"Could not import utility functions: {e}")
    _utils_available = False
    
    # Create placeholder functions
    check_se3_invariance = lambda *args, **kwargs: False
    generate_random_se3_transform = lambda device: (None, None)
    apply_se3_transform = lambda pos, rot, trans: pos
    compute_geodesic_distance = lambda *args: 0.0
    validate_edge_index = lambda edge_index, num_nodes: edge_index
    create_fallback_edges = lambda num_nodes, device: None
    safe_tensor_operation = lambda op, *tensors, **kwargs: None
    compute_attention_entropy = lambda weights: 0.0
    extract_geometric_features = lambda embeddings, method="pca": {}

# Export all public components
__all__ = [
    # Main classes
    "MathematicallyCorrectGASM",
    "EnhancedGASM", 
    "GASM",
    
    # Core components
    "SE3InvariantAttention",
    "UniversalInvariantAttention",
    "EfficientCurvatureComputation", 
    "ConstraintHandler",
    
    # Enhanced components
    "EnhancedBatchProcessor",
    "ErrorRecoveryWrapper",
    "robust_function",
    
    # Utility functions
    "check_se3_invariance",
    "generate_random_se3_transform",
    "apply_se3_transform",
    "compute_geodesic_distance",
    "validate_edge_index",
    "create_fallback_edges",
    "safe_tensor_operation",
    "compute_attention_entropy",
    "extract_geometric_features",
    
    # Factory functions
    "create_gasm_model",
    "get_config",
    
    # Package metadata
    "__version__",
    "__author__",
    "__description__",
]

# Configuration settings
DEFAULT_CONFIG = {
    "feature_dim": 768,
    "hidden_dim": 256,
    "output_dim": 3,
    "num_heads": 8,
    "max_iterations": 10,
    "dropout": 0.1,
    "target_curvature": 0.1,
    "enable_mixed_precision": True,
    "max_batch_size": 8,
    "error_recovery_retries": 2,
}

def get_config():
    """Get default GASM configuration"""
    return DEFAULT_CONFIG.copy()

def create_gasm_model(config=None, **kwargs):
    """
    Factory function to create a GASM model with default configuration
    
    Args:
        config: Configuration dictionary (optional)
        **kwargs: Additional configuration overrides
        
    Returns:
        EnhancedGASM model instance or None if not available
    """
    if not _core_available:
        logger.error("GASM core components not available - cannot create model")
        return None
        
    model_config = get_config()
    
    if config:
        model_config.update(config)
    
    if kwargs:
        model_config.update(kwargs)
    
    try:
        return EnhancedGASM(
            feature_dim=model_config["feature_dim"],
            hidden_dim=model_config["hidden_dim"], 
            output_dim=model_config["output_dim"],
            num_heads=model_config["num_heads"],
            max_iterations=model_config["max_iterations"],
            dropout=model_config["dropout"]
        )
    except Exception as e:
        logger.error(f"Failed to create GASM model: {e}")
        return None

# Add availability flags for external code
__all__.extend(["DEFAULT_CONFIG", "_core_available", "_utils_available"])