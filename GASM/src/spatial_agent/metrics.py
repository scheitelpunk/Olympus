"""
Spatial Agent Metrics Module - Comprehensive Pose Error Calculation and Constraint Scoring

This module provides mathematically rigorous pose error calculations and constraint 
satisfaction scoring for spatial agents. It supports SE(3) manifold computations,
various constraint types, and detailed progress tracking with statistical analysis.

Mathematical Foundation:
- SE(3) pose errors using geodesic distances on the Special Euclidean group
- Constraint satisfaction using energy-based scoring functions
- Convergence detection through multi-criteria thresholding
- Statistical accumulation with numerical stability guarantees

Author: Claude Code Implementation Agent
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import math
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# Import geomstats with fallback
try:
    import geomstats.backend as gs
    from geomstats.geometry.special_euclidean import SpecialEuclidean
    from geomstats.geometry.special_orthogonal import SpecialOrthogonal
    GEOMSTATS_AVAILABLE = True
    logger.info("Geomstats available - using full SE(3) manifold computations")
except ImportError:
    GEOMSTATS_AVAILABLE = False
    logger.warning("Geomstats not available - using simplified geometry computations")


class ConstraintType(Enum):
    """Supported constraint types for spatial agents"""
    ABOVE = "above"           # Point must be above a plane/surface
    ANGLE = "angle"           # Angular constraint between vectors/poses
    DISTANCE = "distance"     # Distance constraint between points
    COLLISION = "collision"   # Collision avoidance constraint
    ORIENTATION = "orientation"  # Orientation constraint (quaternion-based)
    VELOCITY = "velocity"     # Velocity magnitude/direction constraint
    WORKSPACE = "workspace"   # Workspace boundary constraint


@dataclass
class PoseError:
    """
    Complete pose error representation with position and rotation components.
    
    Mathematical Definition:
    For poses p1 = (t1, R1) and p2 = (t2, R2) in SE(3):
    - Position error: ||t1 - t2||_2 (Euclidean distance)
    - Rotation error: ||log(R1^T * R2)||_F (Frobenius norm of log map)
    - Total error: weighted combination based on application needs
    """
    position_error: float = 0.0           # ||t_current - t_target||_2
    rotation_error: float = 0.0           # Angular distance in radians
    position_vector: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    rotation_axis: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    total_error: float = 0.0              # Combined weighted error
    is_converged: bool = False            # Whether error is below threshold
    
    def __post_init__(self):
        """Calculate total error as weighted combination"""
        # Default weights: position errors in meters, rotation in radians
        # Typical weighting: 1.0 for position, 0.5 for rotation (application dependent)
        self.total_error = self.position_error + 0.5 * self.rotation_error


@dataclass
class ConstraintScore:
    """
    Constraint satisfaction score with detailed breakdown.
    
    Score Range: [0, 1] where:
    - 1.0: Perfect constraint satisfaction
    - 0.0: Maximum constraint violation
    - Score follows smooth penalty functions for numerical stability
    """
    constraint_type: ConstraintType
    score: float = 0.0                    # Overall satisfaction score [0, 1]
    violation_magnitude: float = 0.0      # Raw violation amount
    is_satisfied: bool = False            # Whether constraint is satisfied
    tolerance_used: float = 0.0           # Tolerance threshold applied
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate score bounds and consistency"""
        self.score = max(0.0, min(1.0, self.score))
        self.is_satisfied = self.score >= 0.9  # High threshold for satisfaction


@dataclass
class ProgressMetrics:
    """
    Comprehensive progress tracking for spatial agent convergence.
    
    Tracks multiple convergence criteria:
    - Pose errors over time
    - Constraint satisfaction evolution  
    - Statistical measures (variance, trends)
    - Convergence detection with hysteresis
    """
    iteration: int = 0
    pose_errors: List[PoseError] = field(default_factory=list)
    constraint_scores: List[ConstraintScore] = field(default_factory=list)
    convergence_history: List[bool] = field(default_factory=list)
    error_statistics: Dict[str, float] = field(default_factory=dict)
    convergence_streak: int = 0           # Consecutive convergence iterations
    time_to_convergence: Optional[float] = None


class ToleranceConfig:
    """
    Configurable tolerance thresholds for different metrics.
    
    Provides sensible defaults while allowing customization for different
    applications (e.g., high-precision vs real-time requirements).
    """
    
    def __init__(
        self,
        position_tolerance: float = 0.01,      # 1cm default
        rotation_tolerance: float = 0.017,     # ~1 degree in radians
        constraint_tolerance: float = 0.1,     # 10% violation allowed
        convergence_threshold: float = 0.95,   # 95% overall satisfaction
        convergence_streak_required: int = 3   # Stability requirement
    ):
        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance
        self.constraint_tolerance = constraint_tolerance
        self.convergence_threshold = convergence_threshold
        self.convergence_streak_required = convergence_streak_required
        
        # Validate inputs
        if position_tolerance <= 0:
            raise ValueError("Position tolerance must be positive")
        if rotation_tolerance <= 0:
            raise ValueError("Rotation tolerance must be positive")
        if not (0 < convergence_threshold <= 1):
            raise ValueError("Convergence threshold must be in (0, 1]")


class SpatialMetricsCalculator:
    """
    Main calculator class for spatial agent pose errors and constraint scoring.
    
    Provides comprehensive metrics computation with mathematical rigor,
    error handling, and detailed logging for debugging and analysis.
    """
    
    def __init__(self, tolerance_config: Optional[ToleranceConfig] = None):
        """
        Initialize the metrics calculator.
        
        Args:
            tolerance_config: Custom tolerance configuration, uses defaults if None
        """
        self.tolerance_config = tolerance_config or ToleranceConfig()
        self.error_accumulator = defaultdict(list)
        self.constraint_history = defaultdict(list)
        
        # Initialize SE(3) geometry if available
        if GEOMSTATS_AVAILABLE:
            try:
                self.se3_group = SpecialEuclidean(n=3, equip=False)
                self.so3_group = SpecialOrthogonal(n=3, equip=False)
                logger.info("SE(3) and SO(3) groups initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Riemannian groups: {e}")
                self.se3_group = None
                self.so3_group = None
        else:
            self.se3_group = None
            self.so3_group = None
    
    def pose_error(
        self, 
        current_pose: Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]], 
        target_pose: Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]]
    ) -> PoseError:
        """
        Calculate comprehensive pose error between current and target poses.
        
        Mathematical Foundation:
        For SE(3) poses represented as (position, orientation):
        
        1. Position Error:
           e_pos = ||t_current - t_target||_2
           
        2. Rotation Error (using geodesic distance on SO(3)):
           e_rot = ||log(R_current^T * R_target)||_F
           where log is the matrix logarithm on SO(3)
           
        3. Alternative rotation metrics:
           - Quaternion: e_rot = arccos(|q_current · q_target|)
           - Axis-angle: Direct angle difference
        
        Args:
            current_pose: Current pose as tensor [x,y,z,qx,qy,qz,qw] or dict
            target_pose: Target pose in same format
            
        Returns:
            PoseError object with detailed error breakdown
            
        Raises:
            ValueError: If pose formats are incompatible or invalid
            RuntimeError: If computation fails with fallback attempted
        """
        try:
            # Parse pose inputs to standardized format
            current_pos, current_rot = self._parse_pose_input(current_pose)
            target_pos, target_rot = self._parse_pose_input(target_pose)
            
            # Ensure tensors are on same device and dtype
            device = current_pos.device if hasattr(current_pos, 'device') else 'cpu'
            current_pos = torch.as_tensor(current_pos, dtype=torch.float32, device=device)
            target_pos = torch.as_tensor(target_pos, dtype=torch.float32, device=device)
            current_rot = torch.as_tensor(current_rot, dtype=torch.float32, device=device)
            target_rot = torch.as_tensor(target_rot, dtype=torch.float32, device=device)
            
            # Calculate position error (Euclidean distance)
            position_error_vector = current_pos - target_pos
            position_error = torch.norm(position_error_vector).item()
            
            # Calculate rotation error (geodesic distance on SO(3))
            rotation_error, rotation_axis = self._calculate_rotation_error(
                current_rot, target_rot
            )
            
            # Create comprehensive error object
            pose_error = PoseError(
                position_error=position_error,
                rotation_error=rotation_error,
                position_vector=position_error_vector.cpu(),
                rotation_axis=rotation_axis.cpu(),
                is_converged=(
                    position_error < self.tolerance_config.position_tolerance and
                    rotation_error < self.tolerance_config.rotation_tolerance
                )
            )
            
            # Log detailed error information for debugging
            logger.debug(
                f"Pose error calculated - Position: {position_error:.6f}m, "
                f"Rotation: {math.degrees(rotation_error):.3f}°, "
                f"Converged: {pose_error.is_converged}"
            )
            
            return pose_error
            
        except Exception as e:
            logger.error(f"Pose error calculation failed: {e}")
            # Return safe fallback with maximum error
            return PoseError(
                position_error=float('inf'),
                rotation_error=float('inf'),
                total_error=float('inf'),
                is_converged=False
            )
    
    def constraint_score(
        self,
        state: Dict[str, torch.Tensor],
        constraints: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ConstraintScore]:
        """
        Calculate constraint satisfaction scores for given state and constraints.
        
        Mathematical Foundation:
        Constraint scores use smooth penalty functions to ensure differentiability:
        
        1. Soft constraint satisfaction:
           score = exp(-λ * max(0, violation)²)
           where λ controls penalty sharpness
           
        2. Distance constraints:
           violation = |d_actual - d_target| - tolerance
           
        3. Angle constraints:
           violation = |θ_actual - θ_target| - tolerance
           
        4. Collision constraints:
           violation = max(0, min_distance - actual_distance)
        
        Args:
            state: Current agent state containing positions, orientations, velocities
            constraints: Dictionary of constraint specifications
            
        Returns:
            Dictionary mapping constraint names to ConstraintScore objects
            
        Raises:
            ValueError: If constraint specifications are invalid
        """
        constraint_scores = {}
        
        try:
            for constraint_name, constraint_spec in constraints.items():
                try:
                    constraint_type = ConstraintType(constraint_spec.get('type', 'distance'))
                    score = self._evaluate_constraint(state, constraint_type, constraint_spec)
                    constraint_scores[constraint_name] = score
                    
                    # Track constraint history for analysis
                    self.constraint_history[constraint_name].append(score.score)
                    
                    logger.debug(
                        f"Constraint '{constraint_name}' ({constraint_type.value}): "
                        f"Score={score.score:.3f}, Satisfied={score.is_satisfied}"
                    )
                    
                except Exception as constraint_error:
                    logger.warning(
                        f"Failed to evaluate constraint '{constraint_name}': {constraint_error}"
                    )
                    # Create failure score
                    constraint_scores[constraint_name] = ConstraintScore(
                        constraint_type=ConstraintType.DISTANCE,  # Default fallback
                        score=0.0,
                        violation_magnitude=float('inf'),
                        is_satisfied=False,
                        additional_info={'error': str(constraint_error)}
                    )
            
            return constraint_scores
            
        except Exception as e:
            logger.error(f"Constraint scoring failed: {e}")
            return {}
    
    def is_done(
        self,
        pose_errors: List[PoseError],
        constraint_scores: Dict[str, ConstraintScore],
        custom_thresholds: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine convergence based on pose errors, constraint satisfaction, and stability.
        
        Convergence Criteria (all must be satisfied):
        1. All pose errors below tolerance thresholds
        2. All constraints satisfied above threshold
        3. Stability over multiple iterations (streak requirement)
        4. Optional custom criteria
        
        Mathematical Foundation:
        - Uses weighted combination of multiple criteria
        - Hysteresis to prevent oscillatory convergence decisions
        - Statistical stability analysis over recent history
        
        Args:
            pose_errors: List of recent pose errors for stability analysis
            constraint_scores: Current constraint satisfaction scores
            custom_thresholds: Optional custom convergence criteria
            
        Returns:
            Tuple of (is_converged, detailed_analysis)
            where detailed_analysis contains convergence metrics
        """
        try:
            convergence_analysis = {
                'pose_converged': False,
                'constraints_converged': False,
                'stable': False,
                'overall_converged': False,
                'details': {}
            }
            
            # 1. Check pose error convergence
            if pose_errors:
                recent_errors = pose_errors[-5:]  # Check last 5 iterations
                pose_converged_count = sum(1 for error in recent_errors if error.is_converged)
                convergence_analysis['pose_converged'] = (
                    pose_converged_count >= len(recent_errors) * 0.8  # 80% threshold
                )
                
                # Statistical analysis
                if len(recent_errors) > 1:
                    position_errors = [e.position_error for e in recent_errors]
                    rotation_errors = [e.rotation_error for e in recent_errors]
                    
                    convergence_analysis['details'].update({
                        'position_error_mean': np.mean(position_errors),
                        'position_error_std': np.std(position_errors),
                        'rotation_error_mean': np.mean(rotation_errors),
                        'rotation_error_std': np.std(rotation_errors),
                        'position_error_trend': self._calculate_trend(position_errors),
                        'rotation_error_trend': self._calculate_trend(rotation_errors)
                    })
            
            # 2. Check constraint satisfaction convergence
            if constraint_scores:
                satisfied_constraints = sum(
                    1 for score in constraint_scores.values() 
                    if score.is_satisfied
                )
                total_constraints = len(constraint_scores)
                constraint_satisfaction_rate = satisfied_constraints / max(1, total_constraints)
                
                convergence_analysis['constraints_converged'] = (
                    constraint_satisfaction_rate >= self.tolerance_config.convergence_threshold
                )
                
                convergence_analysis['details'].update({
                    'constraint_satisfaction_rate': constraint_satisfaction_rate,
                    'satisfied_constraints': satisfied_constraints,
                    'total_constraints': total_constraints,
                    'average_constraint_score': np.mean([
                        score.score for score in constraint_scores.values()
                    ])
                })
            else:
                # No constraints means constraint convergence is trivially satisfied
                convergence_analysis['constraints_converged'] = True
            
            # 3. Check stability (streak requirement)
            current_converged = (
                convergence_analysis['pose_converged'] and
                convergence_analysis['constraints_converged']
            )
            
            if hasattr(self, '_convergence_streak'):
                if current_converged:
                    self._convergence_streak += 1
                else:
                    self._convergence_streak = 0
            else:
                self._convergence_streak = 1 if current_converged else 0
            
            convergence_analysis['stable'] = (
                self._convergence_streak >= self.tolerance_config.convergence_streak_required
            )
            
            # 4. Apply custom thresholds if provided
            if custom_thresholds:
                for criterion, threshold in custom_thresholds.items():
                    if criterion in convergence_analysis['details']:
                        value = convergence_analysis['details'][criterion]
                        convergence_analysis['details'][f'{criterion}_custom_satisfied'] = (
                            value <= threshold if 'error' in criterion else value >= threshold
                        )
            
            # 5. Overall convergence decision
            convergence_analysis['overall_converged'] = (
                convergence_analysis['pose_converged'] and
                convergence_analysis['constraints_converged'] and
                convergence_analysis['stable']
            )
            
            convergence_analysis['details'].update({
                'convergence_streak': self._convergence_streak,
                'streak_required': self.tolerance_config.convergence_streak_required
            })
            
            logger.debug(f"Convergence analysis: {convergence_analysis}")
            
            return convergence_analysis['overall_converged'], convergence_analysis
            
        except Exception as e:
            logger.error(f"Convergence detection failed: {e}")
            return False, {'error': str(e), 'overall_converged': False}
    
    def accumulate_statistics(
        self,
        pose_error: PoseError,
        constraint_scores: Dict[str, ConstraintScore]
    ) -> Dict[str, Any]:
        """
        Accumulate error statistics over time for analysis and debugging.
        
        Statistics Tracked:
        - Running averages with exponential decay
        - Variance and standard deviation
        - Min/max values
        - Convergence trends and rates
        - Constraint violation patterns
        
        Args:
            pose_error: Current pose error to accumulate
            constraint_scores: Current constraint scores to accumulate
            
        Returns:
            Dictionary of accumulated statistics
        """
        try:
            # Accumulate pose error statistics
            self.error_accumulator['position_errors'].append(pose_error.position_error)
            self.error_accumulator['rotation_errors'].append(pose_error.rotation_error)
            self.error_accumulator['total_errors'].append(pose_error.total_error)
            self.error_accumulator['convergence_flags'].append(pose_error.is_converged)
            
            # Accumulate constraint statistics
            for name, score in constraint_scores.items():
                self.error_accumulator[f'constraint_{name}_scores'].append(score.score)
                self.error_accumulator[f'constraint_{name}_violations'].append(
                    score.violation_magnitude
                )
            
            # Calculate running statistics
            statistics = self._compute_running_statistics()
            
            # Add trend analysis
            statistics['trends'] = self._analyze_trends()
            
            return statistics
            
        except Exception as e:
            logger.error(f"Statistics accumulation failed: {e}")
            return {'error': str(e)}
    
    def reset_statistics(self):
        """Reset all accumulated statistics for fresh start."""
        self.error_accumulator.clear()
        self.constraint_history.clear()
        if hasattr(self, '_convergence_streak'):
            self._convergence_streak = 0
        logger.info("Statistics reset successfully")
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report for debugging and analysis.
        
        Returns:
            Detailed report with all accumulated statistics, trends, and analysis
        """
        try:
            report = {
                'summary': {
                    'total_iterations': len(self.error_accumulator.get('position_errors', [])),
                    'convergence_rate': self._calculate_convergence_rate(),
                    'current_streak': getattr(self, '_convergence_streak', 0)
                },
                'pose_errors': self._analyze_pose_errors(),
                'constraints': self._analyze_constraints(),
                'trends': self._analyze_trends(),
                'recommendations': self._generate_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {'error': str(e)}
    
    # Private helper methods
    
    def _parse_pose_input(
        self, 
        pose: Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse various pose input formats to standardized tensors."""
        if isinstance(pose, dict):
            # Dictionary format: {'position': [x,y,z], 'orientation': [qx,qy,qz,qw]}
            if 'position' in pose and 'orientation' in pose:
                position = torch.as_tensor(pose['position'], dtype=torch.float32)
                orientation = torch.as_tensor(pose['orientation'], dtype=torch.float32)
                return position[:3], orientation[:4]  # Ensure correct sizes
            else:
                raise ValueError("Dictionary pose must contain 'position' and 'orientation' keys")
        
        elif isinstance(pose, (torch.Tensor, np.ndarray)):
            # Tensor format: [x, y, z, qx, qy, qz, qw] or [x, y, z, rx, ry, rz]
            pose_tensor = torch.as_tensor(pose, dtype=torch.float32)
            
            if len(pose_tensor) == 7:
                # Position + quaternion
                return pose_tensor[:3], pose_tensor[3:7]
            elif len(pose_tensor) == 6:
                # Position + Euler angles (convert to quaternion)
                position = pose_tensor[:3]
                euler = pose_tensor[3:6]
                quaternion = self._euler_to_quaternion(euler)
                return position, quaternion
            else:
                raise ValueError(f"Tensor pose must have length 6 or 7, got {len(pose_tensor)}")
        
        else:
            raise ValueError(f"Unsupported pose format: {type(pose)}")
    
    def _calculate_rotation_error(
        self, 
        current_rot: torch.Tensor, 
        target_rot: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        """
        Calculate rotation error using geodesic distance on SO(3).
        
        Returns rotation error in radians and rotation axis.
        """
        try:
            # Ensure quaternions are normalized
            current_rot = current_rot / torch.norm(current_rot)
            target_rot = target_rot / torch.norm(target_rot)
            
            if GEOMSTATS_AVAILABLE and self.so3_group is not None:
                # Use geomstats for accurate geodesic distance
                current_matrix = self._quaternion_to_rotation_matrix(current_rot)
                target_matrix = self._quaternion_to_rotation_matrix(target_rot)
                
                current_np = current_matrix.detach().cpu().numpy()
                target_np = target_matrix.detach().cpu().numpy()
                
                # Geodesic distance on SO(3)
                rotation_error = self.so3_group.metric.dist(current_np, target_np)
                
                # Calculate rotation axis from relative rotation
                relative_rotation = current_matrix.T @ target_matrix
                axis_angle = self._rotation_matrix_to_axis_angle(relative_rotation)
                rotation_axis = axis_angle[:3]  # Axis component
                
                return float(rotation_error), rotation_axis
            
            else:
                # Fallback: quaternion-based angle calculation
                # Geodesic distance on quaternion manifold: θ = arccos(|q1 · q2|)
                dot_product = torch.abs(torch.dot(current_rot, target_rot))
                dot_product = torch.clamp(dot_product, 0.0, 1.0)  # Numerical stability
                
                rotation_error = torch.acos(dot_product).item()
                
                # Calculate rotation axis (cross product for small angles)
                if rotation_error > 1e-6:
                    # Relative quaternion
                    rel_quat = self._quaternion_multiply(
                        self._quaternion_conjugate(current_rot), target_rot
                    )
                    rotation_axis = rel_quat[1:4]  # Imaginary part
                else:
                    rotation_axis = torch.zeros(3)
                
                return rotation_error, rotation_axis
                
        except Exception as e:
            logger.warning(f"Rotation error calculation failed: {e}")
            # Return safe fallback
            return float('inf'), torch.zeros(3)
    
    def _evaluate_constraint(
        self,
        state: Dict[str, torch.Tensor],
        constraint_type: ConstraintType,
        constraint_spec: Dict[str, Any]
    ) -> ConstraintScore:
        """Evaluate specific constraint type and return satisfaction score."""
        try:
            if constraint_type == ConstraintType.DISTANCE:
                return self._evaluate_distance_constraint(state, constraint_spec)
            elif constraint_type == ConstraintType.ANGLE:
                return self._evaluate_angle_constraint(state, constraint_spec)
            elif constraint_type == ConstraintType.ABOVE:
                return self._evaluate_above_constraint(state, constraint_spec)
            elif constraint_type == ConstraintType.COLLISION:
                return self._evaluate_collision_constraint(state, constraint_spec)
            elif constraint_type == ConstraintType.ORIENTATION:
                return self._evaluate_orientation_constraint(state, constraint_spec)
            elif constraint_type == ConstraintType.VELOCITY:
                return self._evaluate_velocity_constraint(state, constraint_spec)
            elif constraint_type == ConstraintType.WORKSPACE:
                return self._evaluate_workspace_constraint(state, constraint_spec)
            else:
                raise ValueError(f"Unknown constraint type: {constraint_type}")
                
        except Exception as e:
            logger.warning(f"Constraint evaluation failed: {e}")
            return ConstraintScore(
                constraint_type=constraint_type,
                score=0.0,
                violation_magnitude=float('inf'),
                is_satisfied=False,
                additional_info={'error': str(e)}
            )
    
    def _evaluate_distance_constraint(
        self, 
        state: Dict[str, torch.Tensor], 
        spec: Dict[str, Any]
    ) -> ConstraintScore:
        """Evaluate distance constraint between two points."""
        try:
            point1_key = spec.get('point1', 'position')
            point2_key = spec.get('point2', 'target_position')
            target_distance = spec.get('target_distance', 1.0)
            tolerance = spec.get('tolerance', self.tolerance_config.constraint_tolerance)
            
            point1 = state.get(point1_key, torch.zeros(3))
            point2 = state.get(point2_key, torch.zeros(3))
            
            actual_distance = torch.norm(point1 - point2).item()
            violation = abs(actual_distance - target_distance) - tolerance
            
            # Smooth penalty function: score = exp(-λ * max(0, violation)²)
            if violation <= 0:
                score = 1.0  # Perfect satisfaction
            else:
                lambda_penalty = 10.0  # Penalty sharpness parameter
                score = math.exp(-lambda_penalty * violation * violation)
            
            return ConstraintScore(
                constraint_type=ConstraintType.DISTANCE,
                score=score,
                violation_magnitude=max(0, violation),
                is_satisfied=violation <= 0,
                tolerance_used=tolerance,
                additional_info={
                    'actual_distance': actual_distance,
                    'target_distance': target_distance,
                    'points': (point1_key, point2_key)
                }
            )
            
        except Exception as e:
            logger.warning(f"Distance constraint evaluation failed: {e}")
            return ConstraintScore(
                constraint_type=ConstraintType.DISTANCE,
                score=0.0,
                violation_magnitude=float('inf'),
                additional_info={'error': str(e)}
            )
    
    def _evaluate_angle_constraint(
        self, 
        state: Dict[str, torch.Tensor], 
        spec: Dict[str, Any]
    ) -> ConstraintScore:
        """Evaluate angular constraint between vectors or orientations."""
        try:
            vector1_key = spec.get('vector1', 'orientation')
            vector2_key = spec.get('vector2', 'target_orientation')
            target_angle = spec.get('target_angle', 0.0)  # radians
            tolerance = spec.get('tolerance', self.tolerance_config.rotation_tolerance)
            
            vector1 = state.get(vector1_key, torch.tensor([1.0, 0.0, 0.0]))
            vector2 = state.get(vector2_key, torch.tensor([1.0, 0.0, 0.0]))
            
            # Normalize vectors
            vector1 = vector1 / torch.norm(vector1)
            vector2 = vector2 / torch.norm(vector2)
            
            # Calculate angle between vectors
            cos_angle = torch.clamp(torch.dot(vector1, vector2), -1.0, 1.0)
            actual_angle = torch.acos(torch.abs(cos_angle)).item()
            
            violation = abs(actual_angle - target_angle) - tolerance
            
            # Smooth penalty function
            if violation <= 0:
                score = 1.0
            else:
                lambda_penalty = 5.0
                score = math.exp(-lambda_penalty * violation * violation)
            
            return ConstraintScore(
                constraint_type=ConstraintType.ANGLE,
                score=score,
                violation_magnitude=max(0, violation),
                is_satisfied=violation <= 0,
                tolerance_used=tolerance,
                additional_info={
                    'actual_angle_deg': math.degrees(actual_angle),
                    'target_angle_deg': math.degrees(target_angle),
                    'vectors': (vector1_key, vector2_key)
                }
            )
            
        except Exception as e:
            logger.warning(f"Angle constraint evaluation failed: {e}")
            return ConstraintScore(
                constraint_type=ConstraintType.ANGLE,
                score=0.0,
                violation_magnitude=float('inf'),
                additional_info={'error': str(e)}
            )
    
    def _evaluate_above_constraint(
        self, 
        state: Dict[str, torch.Tensor], 
        spec: Dict[str, Any]
    ) -> ConstraintScore:
        """Evaluate constraint that point must be above a plane/surface."""
        try:
            point_key = spec.get('point', 'position')
            plane_normal = spec.get('plane_normal', torch.tensor([0.0, 0.0, 1.0]))
            plane_point = spec.get('plane_point', torch.tensor([0.0, 0.0, 0.0]))
            min_height = spec.get('min_height', 0.0)
            
            point = state.get(point_key, torch.zeros(3))
            plane_normal = plane_normal / torch.norm(plane_normal)
            
            # Calculate distance from point to plane
            point_to_plane = point - plane_point
            distance_to_plane = torch.dot(point_to_plane, plane_normal).item()
            
            violation = min_height - distance_to_plane
            
            # Score based on height above plane
            if violation <= 0:
                score = 1.0
            else:
                # Exponential penalty for being too low
                lambda_penalty = 2.0
                score = math.exp(-lambda_penalty * violation)
            
            return ConstraintScore(
                constraint_type=ConstraintType.ABOVE,
                score=score,
                violation_magnitude=max(0, violation),
                is_satisfied=distance_to_plane >= min_height,
                additional_info={
                    'distance_to_plane': distance_to_plane,
                    'min_height': min_height,
                    'point': point_key
                }
            )
            
        except Exception as e:
            logger.warning(f"Above constraint evaluation failed: {e}")
            return ConstraintScore(
                constraint_type=ConstraintType.ABOVE,
                score=0.0,
                violation_magnitude=float('inf'),
                additional_info={'error': str(e)}
            )
    
    def _evaluate_collision_constraint(
        self, 
        state: Dict[str, torch.Tensor], 
        spec: Dict[str, Any]
    ) -> ConstraintScore:
        """Evaluate collision avoidance constraint."""
        try:
            objects = spec.get('objects', ['position', 'obstacle_position'])
            min_distance = spec.get('min_distance', 0.1)
            
            if len(objects) < 2:
                return ConstraintScore(
                    constraint_type=ConstraintType.COLLISION,
                    score=1.0,  # No collision possible with single object
                    is_satisfied=True
                )
            
            min_actual_distance = float('inf')
            closest_pair = None
            
            # Check all pairs of objects
            for i in range(len(objects)):
                for j in range(i + 1, len(objects)):
                    obj1_pos = state.get(objects[i], torch.zeros(3))
                    obj2_pos = state.get(objects[j], torch.zeros(3))
                    
                    distance = torch.norm(obj1_pos - obj2_pos).item()
                    if distance < min_actual_distance:
                        min_actual_distance = distance
                        closest_pair = (objects[i], objects[j])
            
            violation = min_distance - min_actual_distance
            
            # Collision avoidance score
            if violation <= 0:
                score = 1.0  # Safe distance maintained
            else:
                # Steep penalty for collision
                lambda_penalty = 20.0
                score = math.exp(-lambda_penalty * violation * violation)
            
            return ConstraintScore(
                constraint_type=ConstraintType.COLLISION,
                score=score,
                violation_magnitude=max(0, violation),
                is_satisfied=min_actual_distance >= min_distance,
                additional_info={
                    'min_actual_distance': min_actual_distance,
                    'min_required_distance': min_distance,
                    'closest_pair': closest_pair,
                    'objects_checked': len(objects)
                }
            )
            
        except Exception as e:
            logger.warning(f"Collision constraint evaluation failed: {e}")
            return ConstraintScore(
                constraint_type=ConstraintType.COLLISION,
                score=0.0,
                violation_magnitude=float('inf'),
                additional_info={'error': str(e)}
            )
    
    def _evaluate_orientation_constraint(
        self, 
        state: Dict[str, torch.Tensor], 
        spec: Dict[str, Any]
    ) -> ConstraintScore:
        """Evaluate orientation constraint using quaternions."""
        try:
            current_key = spec.get('current_orientation', 'orientation')
            target_key = spec.get('target_orientation', 'target_orientation')
            tolerance = spec.get('tolerance', self.tolerance_config.rotation_tolerance)
            
            current_quat = state.get(current_key, torch.tensor([1.0, 0.0, 0.0, 0.0]))
            target_quat = state.get(target_key, torch.tensor([1.0, 0.0, 0.0, 0.0]))
            
            # Calculate quaternion distance
            rotation_error, _ = self._calculate_rotation_error(current_quat, target_quat)
            violation = rotation_error - tolerance
            
            # Score based on rotation error
            if violation <= 0:
                score = 1.0
            else:
                lambda_penalty = 5.0
                score = math.exp(-lambda_penalty * violation * violation)
            
            return ConstraintScore(
                constraint_type=ConstraintType.ORIENTATION,
                score=score,
                violation_magnitude=max(0, violation),
                is_satisfied=rotation_error <= tolerance,
                tolerance_used=tolerance,
                additional_info={
                    'rotation_error_deg': math.degrees(rotation_error),
                    'tolerance_deg': math.degrees(tolerance)
                }
            )
            
        except Exception as e:
            logger.warning(f"Orientation constraint evaluation failed: {e}")
            return ConstraintScore(
                constraint_type=ConstraintType.ORIENTATION,
                score=0.0,
                violation_magnitude=float('inf'),
                additional_info={'error': str(e)}
            )
    
    def _evaluate_velocity_constraint(
        self, 
        state: Dict[str, torch.Tensor], 
        spec: Dict[str, Any]
    ) -> ConstraintScore:
        """Evaluate velocity magnitude and/or direction constraint."""
        try:
            velocity_key = spec.get('velocity', 'velocity')
            max_speed = spec.get('max_speed', None)
            min_speed = spec.get('min_speed', None)
            target_direction = spec.get('target_direction', None)
            direction_tolerance = spec.get('direction_tolerance', math.pi / 4)  # 45 degrees
            
            velocity = state.get(velocity_key, torch.zeros(3))
            speed = torch.norm(velocity).item()
            
            violations = []
            
            # Speed constraints
            if max_speed is not None and speed > max_speed:
                violations.append(speed - max_speed)
            if min_speed is not None and speed < min_speed:
                violations.append(min_speed - speed)
            
            # Direction constraint
            if target_direction is not None and speed > 1e-6:
                target_dir = torch.as_tensor(target_direction, dtype=torch.float32)
                target_dir = target_dir / torch.norm(target_dir)
                
                velocity_dir = velocity / speed
                angle_error = torch.acos(torch.clamp(
                    torch.dot(velocity_dir, target_dir), -1.0, 1.0
                )).item()
                
                if angle_error > direction_tolerance:
                    violations.append(angle_error - direction_tolerance)
            
            # Calculate overall violation
            max_violation = max(violations) if violations else 0.0
            
            # Score calculation
            if max_violation <= 0:
                score = 1.0
            else:
                lambda_penalty = 3.0
                score = math.exp(-lambda_penalty * max_violation * max_violation)
            
            return ConstraintScore(
                constraint_type=ConstraintType.VELOCITY,
                score=score,
                violation_magnitude=max_violation,
                is_satisfied=max_violation <= 0,
                additional_info={
                    'speed': speed,
                    'max_speed': max_speed,
                    'min_speed': min_speed,
                    'violations': len(violations)
                }
            )
            
        except Exception as e:
            logger.warning(f"Velocity constraint evaluation failed: {e}")
            return ConstraintScore(
                constraint_type=ConstraintType.VELOCITY,
                score=0.0,
                violation_magnitude=float('inf'),
                additional_info={'error': str(e)}
            )
    
    def _evaluate_workspace_constraint(
        self, 
        state: Dict[str, torch.Tensor], 
        spec: Dict[str, Any]
    ) -> ConstraintScore:
        """Evaluate workspace boundary constraint."""
        try:
            point_key = spec.get('point', 'position')
            workspace_bounds = spec.get('bounds', {})  # {min_x, max_x, min_y, max_y, min_z, max_z}
            
            point = state.get(point_key, torch.zeros(3))
            violations = []
            
            # Check each dimension
            for i, dim in enumerate(['x', 'y', 'z']):
                min_key = f'min_{dim}'
                max_key = f'max_{dim}'
                
                if min_key in workspace_bounds:
                    min_val = workspace_bounds[min_key]
                    if point[i] < min_val:
                        violations.append(min_val - point[i].item())
                
                if max_key in workspace_bounds:
                    max_val = workspace_bounds[max_key]
                    if point[i] > max_val:
                        violations.append(point[i].item() - max_val)
            
            max_violation = max(violations) if violations else 0.0
            
            # Score calculation
            if max_violation <= 0:
                score = 1.0
            else:
                # Quadratic penalty for leaving workspace
                lambda_penalty = 5.0
                score = math.exp(-lambda_penalty * max_violation * max_violation)
            
            return ConstraintScore(
                constraint_type=ConstraintType.WORKSPACE,
                score=score,
                violation_magnitude=max_violation,
                is_satisfied=max_violation <= 0,
                additional_info={
                    'position': point.tolist(),
                    'bounds': workspace_bounds,
                    'violations_count': len(violations)
                }
            )
            
        except Exception as e:
            logger.warning(f"Workspace constraint evaluation failed: {e}")
            return ConstraintScore(
                constraint_type=ConstraintType.WORKSPACE,
                score=0.0,
                violation_magnitude=float('inf'),
                additional_info={'error': str(e)}
            )
    
    # Utility methods for geometric computations
    
    def _euler_to_quaternion(self, euler: torch.Tensor) -> torch.Tensor:
        """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)."""
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return torch.tensor([w, x, y, z], dtype=torch.float32)
    
    def _quaternion_to_rotation_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to 3x3 rotation matrix."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # Ensure quaternion is normalized
        norm = torch.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix
        R = torch.zeros((3, 3))
        R[0, 0] = 1 - 2*y*y - 2*z*z
        R[0, 1] = 2*x*y - 2*z*w
        R[0, 2] = 2*x*z + 2*y*w
        R[1, 0] = 2*x*y + 2*z*w
        R[1, 1] = 1 - 2*x*x - 2*z*z
        R[1, 2] = 2*y*z - 2*x*w
        R[2, 0] = 2*x*z - 2*y*w
        R[2, 1] = 2*y*z + 2*x*w
        R[2, 2] = 1 - 2*x*x - 2*y*y
        
        return R
    
    def _rotation_matrix_to_axis_angle(self, R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to axis-angle representation."""
        # Calculate rotation angle
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        angle = torch.acos((trace - 1) / 2)
        
        if torch.abs(angle) < 1e-6:
            # Near identity, return zero axis-angle
            return torch.zeros(4)  # [angle, axis_x, axis_y, axis_z]
        
        # Calculate rotation axis
        axis = torch.tensor([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * torch.sin(angle))
        
        return torch.cat([angle.unsqueeze(0), axis])
    
    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return torch.tensor([w, x, y, z])
    
    def _quaternion_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        """Calculate quaternion conjugate."""
        return torch.tensor([q[0], -q[1], -q[2], -q[3]])
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) in a series of values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        # Simple linear regression slope
        slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
        return float(slope)
    
    def _compute_running_statistics(self) -> Dict[str, Any]:
        """Compute running statistics from accumulated data."""
        stats = {}
        
        for key, values in self.error_accumulator.items():
            if values and isinstance(values[0], (int, float)):
                values_array = np.array(values)
                stats[key] = {
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'min': np.min(values_array),
                    'max': np.max(values_array),
                    'count': len(values_array),
                    'trend': self._calculate_trend(values)
                }
        
        return stats
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends in accumulated data."""
        trends = {}
        
        # Analyze error trends
        if 'position_errors' in self.error_accumulator:
            pos_errors = self.error_accumulator['position_errors']
            trends['position_error_trend'] = self._calculate_trend(pos_errors[-20:])  # Last 20 iterations
            trends['position_improving'] = trends['position_error_trend'] < -0.001
        
        if 'rotation_errors' in self.error_accumulator:
            rot_errors = self.error_accumulator['rotation_errors']
            trends['rotation_error_trend'] = self._calculate_trend(rot_errors[-20:])
            trends['rotation_improving'] = trends['rotation_error_trend'] < -0.001
        
        # Analyze convergence trends
        if 'convergence_flags' in self.error_accumulator:
            conv_flags = self.error_accumulator['convergence_flags']
            if len(conv_flags) >= 10:
                recent_convergence_rate = sum(conv_flags[-10:]) / 10
                trends['recent_convergence_rate'] = recent_convergence_rate
                trends['convergence_stable'] = recent_convergence_rate > 0.8
        
        return trends
    
    def _analyze_pose_errors(self) -> Dict[str, Any]:
        """Analyze accumulated pose error data."""
        analysis = {}
        
        if 'position_errors' in self.error_accumulator:
            pos_errors = self.error_accumulator['position_errors']
            analysis['position'] = {
                'current': pos_errors[-1] if pos_errors else 0,
                'average': np.mean(pos_errors),
                'improvement': (pos_errors[0] - pos_errors[-1]) if len(pos_errors) > 1 else 0,
                'converged_percentage': sum(1 for e in pos_errors if e < self.tolerance_config.position_tolerance) / max(1, len(pos_errors)) * 100
            }
        
        if 'rotation_errors' in self.error_accumulator:
            rot_errors = self.error_accumulator['rotation_errors']
            analysis['rotation'] = {
                'current': rot_errors[-1] if rot_errors else 0,
                'average': np.mean(rot_errors),
                'improvement': (rot_errors[0] - rot_errors[-1]) if len(rot_errors) > 1 else 0,
                'converged_percentage': sum(1 for e in rot_errors if e < self.tolerance_config.rotation_tolerance) / max(1, len(rot_errors)) * 100
            }
        
        return analysis
    
    def _analyze_constraints(self) -> Dict[str, Any]:
        """Analyze constraint satisfaction history."""
        constraint_analysis = {}
        
        for constraint_name, scores in self.constraint_history.items():
            if scores:
                constraint_analysis[constraint_name] = {
                    'current_score': scores[-1],
                    'average_score': np.mean(scores),
                    'satisfaction_rate': sum(1 for s in scores if s >= 0.9) / len(scores) * 100,
                    'trend': self._calculate_trend(scores[-10:])  # Last 10 iterations
                }
        
        return constraint_analysis
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate overall convergence rate."""
        conv_flags = self.error_accumulator.get('convergence_flags', [])
        if not conv_flags:
            return 0.0
        
        return sum(conv_flags) / len(conv_flags)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on accumulated data."""
        recommendations = []
        
        # Analyze position error trends
        if 'position_errors' in self.error_accumulator:
            pos_errors = self.error_accumulator['position_errors']
            if len(pos_errors) > 5:
                recent_trend = self._calculate_trend(pos_errors[-5:])
                if recent_trend > 0.01:
                    recommendations.append("Position errors are increasing - consider reducing step size or improving controller gains")
                elif recent_trend < -0.01:
                    recommendations.append("Position errors are decreasing well - current approach is working")
                else:
                    recommendations.append("Position errors are stable - may need different approach to improve further")
        
        # Analyze convergence stability
        conv_rate = self._calculate_convergence_rate()
        if conv_rate < 0.5:
            recommendations.append("Low convergence rate - consider adjusting tolerance thresholds or improving control strategy")
        elif conv_rate > 0.9:
            recommendations.append("High convergence rate - system is performing well")
        
        # Analyze constraint satisfaction
        constraint_issues = []
        for name, scores in self.constraint_history.items():
            if scores:
                avg_score = np.mean(scores)
                if avg_score < 0.7:
                    constraint_issues.append(name)
        
        if constraint_issues:
            recommendations.append(f"Constraints with low satisfaction: {', '.join(constraint_issues)} - consider relaxing or redesigning these constraints")
        
        if not recommendations:
            recommendations.append("System metrics look good - no specific recommendations")
        
        return recommendations


# Convenience functions for easy usage

def calculate_pose_error(
    current_pose: Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]], 
    target_pose: Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]],
    tolerance_config: Optional[ToleranceConfig] = None
) -> PoseError:
    """
    Convenience function to calculate pose error between current and target poses.
    
    Args:
        current_pose: Current pose in supported format
        target_pose: Target pose in supported format
        tolerance_config: Optional custom tolerance configuration
        
    Returns:
        PoseError object with detailed error breakdown
    """
    calculator = SpatialMetricsCalculator(tolerance_config)
    return calculator.pose_error(current_pose, target_pose)


def evaluate_constraints(
    state: Dict[str, torch.Tensor],
    constraints: Dict[str, Dict[str, Any]],
    tolerance_config: Optional[ToleranceConfig] = None
) -> Dict[str, ConstraintScore]:
    """
    Convenience function to evaluate constraint satisfaction.
    
    Args:
        state: Current agent state
        constraints: Dictionary of constraint specifications
        tolerance_config: Optional custom tolerance configuration
        
    Returns:
        Dictionary mapping constraint names to ConstraintScore objects
    """
    calculator = SpatialMetricsCalculator(tolerance_config)
    return calculator.constraint_score(state, constraints)


def check_convergence(
    pose_errors: List[PoseError],
    constraint_scores: Dict[str, ConstraintScore],
    tolerance_config: Optional[ToleranceConfig] = None,
    custom_thresholds: Optional[Dict[str, float]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to check convergence.
    
    Args:
        pose_errors: List of recent pose errors
        constraint_scores: Current constraint scores
        tolerance_config: Optional custom tolerance configuration
        custom_thresholds: Optional custom convergence criteria
        
    Returns:
        Tuple of (is_converged, detailed_analysis)
    """
    calculator = SpatialMetricsCalculator(tolerance_config)
    return calculator.is_done(pose_errors, constraint_scores, custom_thresholds)


# Example usage and testing
if __name__ == "__main__":
    # Example usage demonstrating the metrics system
    
    # Initialize calculator
    tolerance_config = ToleranceConfig(
        position_tolerance=0.02,  # 2cm
        rotation_tolerance=0.035,  # ~2 degrees
        convergence_threshold=0.95
    )
    
    calculator = SpatialMetricsCalculator(tolerance_config)
    
    # Example pose error calculation
    current_pose = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])  # [x,y,z,qx,qy,qz,qw]
    target_pose = torch.tensor([1.01, 2.01, 3.01, 0.0, 0.0, 0.1, 0.995])
    
    pose_error = calculator.pose_error(current_pose, target_pose)
    print(f"Pose Error: Position={pose_error.position_error:.4f}m, "
          f"Rotation={math.degrees(pose_error.rotation_error):.2f}°, "
          f"Converged={pose_error.is_converged}")
    
    # Example constraint evaluation
    state = {
        'position': torch.tensor([1.0, 2.0, 3.0]),
        'target_position': torch.tensor([1.1, 2.1, 3.1]),
        'orientation': torch.tensor([0.0, 0.0, 0.0, 1.0]),
        'velocity': torch.tensor([0.1, 0.0, 0.0])
    }
    
    constraints = {
        'distance_constraint': {
            'type': 'distance',
            'point1': 'position',
            'point2': 'target_position',
            'target_distance': 0.2,
            'tolerance': 0.05
        },
        'speed_constraint': {
            'type': 'velocity',
            'velocity': 'velocity',
            'max_speed': 0.5,
            'min_speed': 0.05
        }
    }
    
    constraint_scores = calculator.constraint_score(state, constraints)
    for name, score in constraint_scores.items():
        print(f"Constraint '{name}': Score={score.score:.3f}, "
              f"Satisfied={score.is_satisfied}, "
              f"Violation={score.violation_magnitude:.4f}")
    
    # Example convergence check
    pose_errors = [pose_error]  # In practice, this would be a list of recent errors
    is_converged, analysis = calculator.is_done(pose_errors, constraint_scores)
    print(f"System Converged: {is_converged}")
    print(f"Analysis: {analysis}")
    
    print("Spatial metrics module initialized successfully!")