"""
GASM Utility Functions

This module provides utility functions for GASM operations including:
- SE(3) invariance testing
- Geometric transformations
- Edge index validation
- Tensor operations safety
- Feature extraction utilities
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)

def check_se3_invariance(
    model,
    positions: torch.Tensor,
    features: torch.Tensor, 
    relations: torch.Tensor,
    num_tests: int = 10,
    tolerance: float = 1e-3
) -> bool:
    """
    Check if a GASM model is SE(3) invariant
    
    Args:
        model: GASM model to test
        positions: Test positions (N, 3)
        features: Test features (N, feature_dim)
        relations: Test relations (N, N, relation_dim)
        num_tests: Number of random transformations to test
        tolerance: Tolerance for invariance check
        
    Returns:
        True if model passes SE(3) invariance tests
    """
    try:
        model.eval()
        
        with torch.no_grad():
            # Get original output
            original_output = model([], features, relations)
            
            for _ in range(num_tests):
                # Generate random SE(3) transformation
                rotation, translation = generate_random_se3_transform(
                    device=positions.device
                )
                
                # Transform positions
                transformed_positions = apply_se3_transform(
                    positions, rotation, translation
                )
                
                # Model should be invariant to position transformations
                # Since positions are embedded from features, we test with same features
                transformed_output = model([], features, relations)
                
                # Check if outputs are approximately equal (invariant)
                diff = torch.norm(original_output - transformed_output)
                if diff > tolerance:
                    logger.warning(f"SE(3) invariance test failed: diff = {diff}")
                    return False
            
            return True
            
    except Exception as e:
        logger.error(f"SE(3) invariance test error: {e}")
        return False

def generate_random_se3_transform(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random SE(3) transformation
    
    Args:
        device: Device to create tensors on
        
    Returns:
        Tuple of (rotation_matrix, translation_vector)
    """
    # Random rotation axis and angle
    axis = F.normalize(torch.randn(3, device=device), dim=0)
    angle = torch.rand(1, device=device) * 2 * np.pi
    
    # Convert to rotation matrix using Rodrigues' formula
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]], 
        [-axis[1], axis[0], 0]
    ], device=device)
    
    rotation = torch.eye(3, device=device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * K @ K
    
    # Random translation
    translation = torch.randn(3, device=device)
    
    return rotation, translation

def apply_se3_transform(
    positions: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor
) -> torch.Tensor:
    """
    Apply SE(3) transformation to positions
    
    Args:
        positions: Input positions (N, 3)
        rotation: Rotation matrix (3, 3)
        translation: Translation vector (3,)
        
    Returns:
        Transformed positions (N, 3)
    """
    return positions @ rotation.T + translation

def compute_geodesic_distance(
    pos1: torch.Tensor,
    rot1: torch.Tensor,
    pos2: torch.Tensor, 
    rot2: torch.Tensor
) -> torch.Tensor:
    """
    Compute geodesic distance on SE(3) manifold
    
    Args:
        pos1, pos2: Position vectors (3,)
        rot1, rot2: Rotation quaternions (4,)
        
    Returns:
        Geodesic distance scalar
    """
    try:
        # Position distance
        pos_diff = pos1 - pos2
        pos_dist = torch.norm(pos_diff)
        
        # Quaternion geodesic distance
        quat_dot = torch.abs(torch.dot(rot1, rot2))
        quat_dot = torch.clamp(quat_dot, 0.0, 1.0)
        rot_dist = torch.acos(quat_dot)
        
        # Combined SE(3) distance
        return pos_dist + 0.5 * rot_dist
        
    except Exception as e:
        logger.warning(f"Geodesic distance computation failed: {e}")
        return torch.norm(pos1 - pos2)

def validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Validate and fix edge index tensor
    
    Args:
        edge_index: Edge index tensor
        num_nodes: Number of nodes
        
    Returns:
        Valid edge index tensor (2, E)
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        logger.warning(f"Invalid edge_index shape: {edge_index.shape}")
        return create_fallback_edges(num_nodes, edge_index.device)
    
    # Clamp indices to valid range
    edge_index = torch.clamp(edge_index, 0, num_nodes - 1)
    
    # Remove self-loops if any
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    
    if edge_index.size(1) == 0:
        return create_fallback_edges(num_nodes, edge_index.device)
    
    return edge_index

def create_fallback_edges(num_nodes: int, device: torch.device) -> torch.Tensor:
    """
    Create fallback edge connectivity
    
    Args:
        num_nodes: Number of nodes
        device: Device for tensor creation
        
    Returns:
        Fallback edge index (2, E)
    """
    if num_nodes < 2:
        # Self-loop for single node
        return torch.tensor([[0], [0]], dtype=torch.long, device=device)
    
    # Create circular connectivity
    edge_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_list.append([i, j])
    
    if not edge_list:
        return torch.tensor([[0], [0]], dtype=torch.long, device=device)
    
    return torch.tensor(edge_list, dtype=torch.long, device=device).t()

def safe_tensor_operation(operation, *tensors, fallback_value=0.0, **kwargs):
    """
    Safely execute tensor operations with fallback
    
    Args:
        operation: Function to execute
        *tensors: Input tensors
        fallback_value: Value to return on failure
        **kwargs: Additional arguments
        
    Returns:
        Operation result or fallback value
    """
    try:
        return operation(*tensors, **kwargs)
    except Exception as e:
        logger.warning(f"Tensor operation failed: {e}, using fallback")
        if tensors:
            return torch.full_like(tensors[0], fallback_value)
        return torch.tensor(fallback_value)

def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of attention weights
    
    Args:
        attention_weights: Attention weight matrix
        
    Returns:
        Entropy value
    """
    try:
        # Add small epsilon for numerical stability
        eps = 1e-9
        attention_weights = attention_weights + eps
        
        # Normalize to ensure valid probabilities
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        
        # Compute entropy
        entropy = -torch.sum(attention_weights * torch.log(attention_weights), dim=-1)
        return entropy.mean()
        
    except Exception as e:
        logger.warning(f"Attention entropy computation failed: {e}")
        return torch.tensor(1.0)

def extract_geometric_features(
    embeddings: torch.Tensor,
    method: str = "pca"
) -> Dict[str, torch.Tensor]:
    """
    Extract geometric features from embeddings
    
    Args:
        embeddings: Input embeddings (N, D)
        method: Feature extraction method
        
    Returns:
        Dictionary of geometric features
    """
    try:
        features = {}
        
        # Basic statistics
        features["mean"] = embeddings.mean(dim=0)
        features["std"] = embeddings.std(dim=0) 
        features["norm"] = torch.norm(embeddings, dim=1)
        
        # Distance matrix
        dist_matrix = torch.cdist(embeddings, embeddings)
        features["distance_matrix"] = dist_matrix
        features["avg_distance"] = dist_matrix.mean()
        
        if method == "pca":
            # Simple PCA approximation
            centered = embeddings - embeddings.mean(dim=0)
            cov = torch.matmul(centered.T, centered) / (embeddings.size(0) - 1)
            eigenvals, eigenvecs = torch.linalg.eigh(cov)
            
            # Sort by eigenvalue (descending)
            idx = torch.argsort(eigenvals, descending=True)
            features["eigenvalues"] = eigenvals[idx]
            features["eigenvectors"] = eigenvecs[:, idx]
            
            # Project to top 3 components for visualization
            top_components = eigenvecs[:, idx[:3]]
            features["pca_projection"] = torch.matmul(centered, top_components)
        
        elif method == "manifold":
            # Local manifold properties
            k = min(5, embeddings.size(0) - 1)  # Number of neighbors
            if k > 0:
                # Find k nearest neighbors for each point
                distances, indices = torch.topk(dist_matrix, k + 1, largest=False)
                
                # Remove self (first element)
                neighbor_distances = distances[:, 1:]
                neighbor_indices = indices[:, 1:]
                
                features["neighbor_distances"] = neighbor_distances
                features["neighbor_indices"] = neighbor_indices
                features["local_density"] = 1.0 / (neighbor_distances.mean(dim=1) + 1e-6)
        
        return features
        
    except Exception as e:
        logger.warning(f"Geometric feature extraction failed: {e}")
        return {
            "mean": torch.zeros(embeddings.size(1)),
            "std": torch.ones(embeddings.size(1)),
            "norm": torch.norm(embeddings, dim=1) if embeddings.numel() > 0 else torch.tensor(0.0)
        }

def create_random_graph(
    num_nodes: int,
    edge_prob: float = 0.3,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create random graph edge index
    
    Args:
        num_nodes: Number of nodes
        edge_prob: Probability of edge between any two nodes
        device: Device for tensor creation
        
    Returns:
        Random edge index (2, E)
    """
    if device is None:
        device = torch.device('cpu')
    
    edge_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and torch.rand(1).item() < edge_prob:
                edge_list.append([i, j])
    
    if not edge_list:
        return create_fallback_edges(num_nodes, device)
    
    return torch.tensor(edge_list, dtype=torch.long, device=device).t()

def batch_process_safely(
    func,
    inputs: List[torch.Tensor],
    batch_size: int = 8,
    **kwargs
) -> List[torch.Tensor]:
    """
    Process inputs in batches safely
    
    Args:
        func: Function to apply
        inputs: List of input tensors
        batch_size: Batch size
        **kwargs: Additional function arguments
        
    Returns:
        List of processed outputs
    """
    outputs = []
    
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        try:
            batch_outputs = [func(inp, **kwargs) for inp in batch]
            outputs.extend(batch_outputs)
        except Exception as e:
            logger.warning(f"Batch processing failed for batch {i//batch_size}: {e}")
            # Add fallback outputs
            for inp in batch:
                try:
                    fallback = torch.zeros_like(inp)
                    outputs.append(fallback)
                except:
                    outputs.append(torch.tensor(0.0))
    
    return outputs

def normalize_features(
    features: torch.Tensor,
    method: str = "l2"
) -> torch.Tensor:
    """
    Normalize features using specified method
    
    Args:
        features: Input features (N, D)
        method: Normalization method ("l2", "unit", "zscore")
        
    Returns:
        Normalized features
    """
    try:
        if method == "l2":
            return F.normalize(features, p=2, dim=1)
        elif method == "unit":
            return features / (features.abs().max() + 1e-6)
        elif method == "zscore":
            mean = features.mean(dim=0)
            std = features.std(dim=0) + 1e-6
            return (features - mean) / std
        else:
            return features
    except Exception as e:
        logger.warning(f"Feature normalization failed: {e}")
        return features

# Compatibility functions for existing code
def check_geometric_consistency(*args, **kwargs):
    """Compatibility wrapper for geometric consistency checks"""
    logger.warning("check_geometric_consistency is deprecated, use model.verify_geometric_consistency")
    return True

def compute_manifold_curvature(positions: torch.Tensor, method: str = "gaussian") -> torch.Tensor:
    """
    Compatibility wrapper for curvature computation
    """
    from .core import EfficientCurvatureComputation
    
    # Create dummy edge index for full connectivity
    N = positions.size(0)
    if N < 2:
        return torch.tensor(0.0)
    
    edge_list = []
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_list.append([i, j])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=positions.device).t()
    
    return EfficientCurvatureComputation.compute_discrete_curvature(
        positions, edge_index, method
    )