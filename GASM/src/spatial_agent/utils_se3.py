"""
SE(3) Utility Functions for Spatial Transformations

This module provides comprehensive utilities for working with SE(3) group transformations,
including homogeneous matrices, quaternions, Euler angles, and various spatial operations.

Mathematical Background:
- SE(3): Special Euclidean Group in 3D (rigid body transformations)
- SO(3): Special Orthogonal Group in 3D (rotations)
- Lie algebra se(3) and so(3) for tangent space operations

Author: Generated for GASM-Roboting Project
"""

import numpy as np
from typing import Tuple, Union, Optional, List
import warnings
from scipy.spatial.transform import Rotation as ScipyRotation


class SE3ValidationError(Exception):
    """Custom exception for SE(3) validation errors."""
    pass


class SE3Utils:
    """Comprehensive SE(3) utility class for spatial transformations."""
    
    # Constants for numerical stability
    EPS = 1e-12  # Increased precision
    PI = np.pi
    TWO_PI = 2 * np.pi
    SQRT_EPS = np.sqrt(np.finfo(np.float64).eps)  # ~1e-8
    
    # Rotation representation limits
    MAX_ANGLE = np.pi - 1e-12  # Avoid singularities at ±π
    MIN_ANGLE = 1e-12
    
    # Optimization constants
    MAX_ITERATIONS = 100
    CONVERGENCE_TOL = 1e-12
    
    @staticmethod
    def validate_rotation_matrix(R_matrix: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Validate if a matrix is a proper rotation matrix.
        
        A proper rotation matrix R must satisfy:
        1. R^T * R = I (orthogonality)
        2. det(R) = 1 (proper rotation, not reflection)
        3. Shape is (3, 3)
        
        Args:
            R_matrix: 3x3 matrix to validate
            tolerance: Numerical tolerance for validation
            
        Returns:
            bool: True if valid rotation matrix
            
        Raises:
            SE3ValidationError: If matrix is invalid
            
        Examples:
            >>> import numpy as np
            >>> R_identity = np.eye(3)
            >>> SE3Utils.validate_rotation_matrix(R_identity)
            True
            
            >>> # 90-degree rotation about Z-axis
            >>> R_z90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            >>> SE3Utils.validate_rotation_matrix(R_z90)
            True
            
            >>> # Invalid matrix (not orthogonal)
            >>> R_invalid = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
            >>> try:
            ...     SE3Utils.validate_rotation_matrix(R_invalid)
            ... except SE3ValidationError:
            ...     print("Invalid rotation matrix detected")
            Invalid rotation matrix detected
        """
        if R_matrix.shape != (3, 3):
            raise SE3ValidationError(f"Rotation matrix must be 3x3, got {R_matrix.shape}")
        
        # Check orthogonality: R^T * R = I
        should_be_identity = R_matrix.T @ R_matrix
        identity_error = np.abs(should_be_identity - np.eye(3)).max()
        
        # Check determinant = 1 (proper rotation, not reflection)
        det = np.linalg.det(R_matrix)
        det_error = abs(det - 1.0)
        
        if identity_error > tolerance:
            raise SE3ValidationError(f"Matrix not orthogonal. Max error: {identity_error}")
        
        if det_error > tolerance:
            raise SE3ValidationError(f"Determinant not 1. Det: {det}, Error: {det_error}")
        
        return True
    
    @staticmethod
    def validate_homogeneous_matrix(T: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Validate if a matrix is a proper homogeneous transformation matrix.
        
        Args:
            T: 4x4 homogeneous transformation matrix
            tolerance: Numerical tolerance for validation
            
        Returns:
            bool: True if valid transformation matrix
        """
        if T.shape != (4, 4):
            raise SE3ValidationError(f"Homogeneous matrix must be 4x4, got {T.shape}")
        
        # Validate rotation part (top-left 3x3)
        R_part = T[:3, :3]
        SE3Utils.validate_rotation_matrix(R_part, tolerance)
        
        # Check bottom row is [0, 0, 0, 1]
        bottom_row = T[3, :]
        expected_bottom = np.array([0, 0, 0, 1])
        bottom_error = np.abs(bottom_row - expected_bottom).max()
        
        if bottom_error > tolerance:
            raise SE3ValidationError(f"Bottom row invalid. Expected [0,0,0,1], got {bottom_row}")
        
        return True
    
    @staticmethod
    def homogeneous_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """
        Create homogeneous transformation matrix from rotation and translation.
        
        The homogeneous transformation matrix has the form:
        T = [R  t]
            [0  1]
        where R is 3x3 rotation matrix and t is 3x1 translation vector.
        
        Args:
            rotation: 3x3 rotation matrix or 4-element quaternion [x,y,z,w]
            translation: 3-element translation vector
            
        Returns:
            4x4 homogeneous transformation matrix
            
        Examples:
            >>> import numpy as np
            >>> R = np.eye(3)
            >>> t = np.array([1, 2, 3])
            >>> T = SE3Utils.homogeneous_matrix(R, t)
            >>> T.shape
            (4, 4)
            >>> np.allclose(T[:3, :3], R)
            True
            >>> np.allclose(T[:3, 3], t)
            True
            >>> np.allclose(T[3, :], [0, 0, 0, 1])
            True
            
            >>> # With quaternion input
            >>> quat = np.array([0, 0, 0, 1])  # Identity quaternion
            >>> T_quat = SE3Utils.homogeneous_matrix(quat, t)
            >>> np.allclose(T_quat[:3, :3], np.eye(3))
            True
        """
        translation = np.asarray(translation, dtype=np.float64)
        if translation.shape != (3,):
            raise SE3ValidationError(f"Translation must be 3-element vector, got shape {translation.shape}")
        
        T = np.eye(4, dtype=np.float64)
        
        # Handle rotation input
        if rotation.shape == (4,):
            # Quaternion input [x, y, z, w]
            T[:3, :3] = SE3Utils.quaternion_to_rotation_matrix(rotation)
        elif rotation.shape == (3, 3):
            # Rotation matrix input
            SE3Utils.validate_rotation_matrix(rotation)
            T[:3, :3] = rotation
        else:
            raise SE3ValidationError(f"Rotation must be 3x3 matrix or 4-element quaternion, got shape {rotation.shape}")
        
        T[:3, 3] = translation
        return T
    
    @staticmethod
    def extract_rotation_translation(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract rotation matrix and translation vector from homogeneous matrix.
        
        Args:
            T: 4x4 homogeneous transformation matrix
            
        Returns:
            Tuple of (3x3 rotation matrix, 3-element translation vector)
        """
        SE3Utils.validate_homogeneous_matrix(T)
        return T[:3, :3].copy(), T[:3, 3].copy()
    
    @staticmethod
    def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix using quaternion formula.
        
        Uses the standard quaternion-to-rotation-matrix formula:
        R = I + 2s[q]× + 2[q]×²
        where q = [x,y,z], s = w, and [v]× is the skew-symmetric matrix of v.
        
        Args:
            quat: Quaternion as [x, y, z, w] (scalar-last convention)
            
        Returns:
            3x3 rotation matrix
            
        Examples:
            >>> import numpy as np
            >>> # Identity quaternion
            >>> quat_identity = np.array([0, 0, 0, 1])
            >>> R = SE3Utils.quaternion_to_rotation_matrix(quat_identity)
            >>> np.allclose(R, np.eye(3))
            True
            
            >>> # 90-degree rotation about Z-axis
            >>> quat_z90 = np.array([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
            >>> R_z90 = SE3Utils.quaternion_to_rotation_matrix(quat_z90)
            >>> expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            >>> np.allclose(R_z90, expected, atol=1e-10)
            True
        """
        quat = np.asarray(quat, dtype=np.float64)
        if quat.shape != (4,):
            raise SE3ValidationError(f"Quaternion must be 4-element vector, got shape {quat.shape}")
        
        # Normalize quaternion
        quat = quat / np.linalg.norm(quat)
        
        x, y, z, w = quat
        
        # Compute rotation matrix using quaternion formula
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ], dtype=np.float64)
        
        return R
    
    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion using Shepperd's method.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Quaternion as [x, y, z, w] (scalar-last convention)
        """
        SE3Utils.validate_rotation_matrix(R)
        
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w], dtype=np.float64)
    
    @staticmethod
    def euler_to_rotation_matrix(euler: np.ndarray, order: str = 'xyz') -> np.ndarray:
        """
        Convert Euler angles to rotation matrix.
        
        Args:
            euler: 3-element Euler angles in radians
            order: Rotation order (e.g., 'xyz', 'zyx', 'zxy')
            
        Returns:
            3x3 rotation matrix
        """
        euler = np.asarray(euler, dtype=np.float64)
        if euler.shape != (3,):
            raise SE3ValidationError(f"Euler angles must be 3-element vector, got shape {euler.shape}")
        
        # Use scipy for robust conversion
        scipy_rot = ScipyRotation.from_euler(order, euler)
        return scipy_rot.as_matrix().astype(np.float64)
    
    @staticmethod
    def rotation_matrix_to_euler(R: np.ndarray, order: str = 'xyz') -> np.ndarray:
        """
        Convert rotation matrix to Euler angles.
        
        Args:
            R: 3x3 rotation matrix
            order: Rotation order (e.g., 'xyz', 'zyx', 'zxy')
            
        Returns:
            3-element Euler angles in radians
        """
        SE3Utils.validate_rotation_matrix(R)
        scipy_rot = ScipyRotation.from_matrix(R)
        return scipy_rot.as_euler(order).astype(np.float64)
    
    @staticmethod
    def skew_symmetric(v: np.ndarray) -> np.ndarray:
        """
        Create skew-symmetric matrix from 3D vector.
        
        Args:
            v: 3-element vector
            
        Returns:
            3x3 skew-symmetric matrix
        """
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (3,):
            raise SE3ValidationError(f"Vector must be 3-element, got shape {v.shape}")
        
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ], dtype=np.float64)
    
    @staticmethod
    def unskew_symmetric(S: np.ndarray) -> np.ndarray:
        """
        Extract vector from skew-symmetric matrix.
        
        Args:
            S: 3x3 skew-symmetric matrix
            
        Returns:
            3-element vector
        """
        if S.shape != (3, 3):
            raise SE3ValidationError(f"Matrix must be 3x3, got shape {S.shape}")
        
        # Check skew-symmetry
        if not np.allclose(S, -S.T, atol=SE3Utils.EPS):
            warnings.warn("Matrix is not skew-symmetric")
        
        return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=np.float64)
    
    @staticmethod
    def rodrigues_formula(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Compute rotation matrix using Rodrigues' rotation formula.
        
        Rodrigues' formula: R = I + sin(θ)[k]× + (1-cos(θ))[k]×²
        where k is the unit rotation axis, θ is the angle, and [k]× is
        the skew-symmetric matrix of k.
        
        Args:
            axis: 3-element unit rotation axis
            angle: Rotation angle in radians
            
        Returns:
            3x3 rotation matrix
            
        Examples:
            >>> import numpy as np
            >>> # Zero rotation
            >>> axis = np.array([0, 0, 1])
            >>> R_zero = SE3Utils.rodrigues_formula(axis, 0.0)
            >>> np.allclose(R_zero, np.eye(3))
            True
            
            >>> # 90-degree rotation about Z-axis
            >>> R_z90 = SE3Utils.rodrigues_formula(axis, np.pi/2)
            >>> expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            >>> np.allclose(R_z90, expected)
            True
            
            >>> # 180-degree rotation about X-axis
            >>> axis_x = np.array([1, 0, 0])
            >>> R_x180 = SE3Utils.rodrigues_formula(axis_x, np.pi)
            >>> expected_x180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            >>> np.allclose(R_x180, expected_x180)
            True
        """
        axis = np.asarray(axis, dtype=np.float64)
        if axis.shape != (3,):
            raise SE3ValidationError(f"Axis must be 3-element vector, got shape {axis.shape}")
        
        # Normalize axis
        axis_norm = np.linalg.norm(axis)
        if axis_norm < SE3Utils.EPS:
            return np.eye(3)
        
        axis = axis / axis_norm
        angle = float(angle)
        
        # Rodrigues' formula: R = I + sin(θ)[k]× + (1-cos(θ))[k]×²
        K = SE3Utils.skew_symmetric(axis)
        R = (np.eye(3) + 
             np.sin(angle) * K + 
             (1 - np.cos(angle)) * np.dot(K, K))
        
        return R
    
    @staticmethod
    def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
        """
        Convert axis-angle representation to rotation matrix.
        
        Args:
            axis_angle: 3-element vector where magnitude is angle and direction is axis
            
        Returns:
            3x3 rotation matrix
        """
        axis_angle = np.asarray(axis_angle, dtype=np.float64)
        if axis_angle.shape != (3,):
            raise SE3ValidationError(f"Axis-angle must be 3-element vector, got shape {axis_angle.shape}")
        
        angle = np.linalg.norm(axis_angle)
        if angle < SE3Utils.EPS:
            return np.eye(3)
        
        axis = axis_angle / angle
        return SE3Utils.rodrigues_formula(axis, angle)
    
    @staticmethod
    def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to axis-angle representation.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            3-element axis-angle vector
        """
        SE3Utils.validate_rotation_matrix(R)
        
        # Extract angle from trace
        trace = np.trace(R)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if angle < SE3Utils.EPS:
            return np.zeros(3)
        
        if abs(angle - np.pi) < SE3Utils.EPS:
            # Special case: angle ≈ π
            # Find eigenvector corresponding to eigenvalue 1
            eigenvals, eigenvecs = np.linalg.eig(R)
            idx = np.argmin(np.abs(eigenvals - 1))
            axis = np.real(eigenvecs[:, idx])
            axis = axis / np.linalg.norm(axis)
        else:
            # General case
            axis = np.array([R[2, 1] - R[1, 2],
                           R[0, 2] - R[2, 0],
                           R[1, 0] - R[0, 1]])
            axis = axis / (2 * np.sin(angle))
        
        return axis * angle
    
    @staticmethod
    def adjoint_matrix(T: np.ndarray) -> np.ndarray:
        """
        Compute the adjoint matrix of an SE(3) transformation.
        
        The adjoint matrix Ad_T maps se(3) vectors from one coordinate
        frame to another: Ad_T * xi = T * xi * T^(-1) (in matrix form).
        
        Args:
            T: 4x4 transformation matrix
            
        Returns:
            6x6 adjoint matrix
            
        Examples:
            >>> import numpy as np
            >>> T = create_pose([1, 2, 3], [0, 0, np.pi/4], 'euler')
            >>> Ad = SE3Utils.adjoint_matrix(T)
            >>> Ad.shape
            (6, 6)
            >>> # Adjoint of identity should be identity
            >>> Ad_identity = SE3Utils.adjoint_matrix(np.eye(4))
            >>> np.allclose(Ad_identity, np.eye(6))
            True
        """
        SE3Utils.validate_homogeneous_matrix(T)
        R, t = SE3Utils.extract_rotation_translation(T)
        
        # Adjoint matrix has block structure:
        # Ad = [R    [t]×R]
        #      [0    R   ]
        Ad = np.zeros((6, 6))
        Ad[:3, :3] = R
        Ad[3:, 3:] = R
        Ad[:3, 3:] = SE3Utils.skew_symmetric(t) @ R
        
        return Ad
    
    @staticmethod
    def se3_exp_map(xi: np.ndarray) -> np.ndarray:
        """
        Exponential map from se(3) to SE(3).
        
        Maps a 6-element se(3) algebra vector to SE(3) group element.
        For xi = [rho; phi], the exponential map is:
        exp(xi) = [R    V*rho]
                  [0    1    ]
        where R = exp([phi]×) and V is the left Jacobian.
        
        Args:
            xi: 6-element se(3) vector [rho; phi] where rho is translation, phi is rotation
            
        Returns:
            4x4 homogeneous transformation matrix
            
        Examples:
            >>> import numpy as np
            >>> # Zero motion
            >>> xi_zero = np.zeros(6)
            >>> T_zero = SE3Utils.se3_exp_map(xi_zero)
            >>> np.allclose(T_zero, np.eye(4))
            True
            
            >>> # Pure translation
            >>> xi_trans = np.array([1, 2, 3, 0, 0, 0])
            >>> T_trans = SE3Utils.se3_exp_map(xi_trans)
            >>> np.allclose(T_trans[:3, :3], np.eye(3))
            True
            >>> np.allclose(T_trans[:3, 3], [1, 2, 3])
            True
            
            >>> # Pure rotation (90 degrees about Z)
            >>> xi_rot = np.array([0, 0, 0, 0, 0, np.pi/2])
            >>> T_rot = SE3Utils.se3_exp_map(xi_rot)
            >>> expected_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            >>> np.allclose(T_rot[:3, :3], expected_R)
            True
        """
        xi = np.asarray(xi, dtype=np.float64)
        if xi.shape != (6,):
            raise SE3ValidationError(f"se(3) vector must be 6-element, got shape {xi.shape}")
        
        rho = xi[:3]  # Translation part
        phi = xi[3:]  # Rotation part
        
        # Rotation matrix from axis-angle
        R = SE3Utils.axis_angle_to_rotation_matrix(phi)
        
        # Compute V matrix for translation
        angle = np.linalg.norm(phi)
        if angle < SE3Utils.EPS:
            V = np.eye(3)
        else:
            axis = phi / angle
            K = SE3Utils.skew_symmetric(axis)
            V = (np.eye(3) + 
                 (1 - np.cos(angle)) / (angle**2) * SE3Utils.skew_symmetric(phi) +
                 (angle - np.sin(angle)) / (angle**3) * np.dot(SE3Utils.skew_symmetric(phi), SE3Utils.skew_symmetric(phi)))
        
        t = V @ rho
        
        return SE3Utils.homogeneous_matrix(R, t)
    
    @staticmethod
    def se3_log_map(T: np.ndarray) -> np.ndarray:
        """
        Logarithm map from SE(3) to se(3).
        
        Maps SE(3) group element to se(3) algebra vector.
        For T = [R t; 0 1], the logarithm is xi = [rho; phi] where
        phi = log(R) and rho = V^(-1) * t, with V^(-1) being the
        inverse of the left Jacobian.
        
        Args:
            T: 4x4 homogeneous transformation matrix
            
        Returns:
            6-element se(3) vector [rho; phi]
            
        Examples:
            >>> import numpy as np
            >>> # Identity transformation
            >>> T_identity = np.eye(4)
            >>> xi_identity = SE3Utils.se3_log_map(T_identity)
            >>> np.allclose(xi_identity, np.zeros(6))
            True
            
            >>> # Round trip test
            >>> xi_original = np.array([1, 2, 3, 0.1, 0.2, 0.3])
            >>> T_exp = SE3Utils.se3_exp_map(xi_original)
            >>> xi_back = SE3Utils.se3_log_map(T_exp)
            >>> np.allclose(xi_back, xi_original, atol=1e-10)
            True
        """
        SE3Utils.validate_homogeneous_matrix(T)
        
        R, t = SE3Utils.extract_rotation_translation(T)
        phi = SE3Utils.rotation_matrix_to_axis_angle(R)
        
        angle = np.linalg.norm(phi)
        if angle < SE3Utils.EPS:
            V_inv = np.eye(3)
        else:
            axis = phi / angle
            K = SE3Utils.skew_symmetric(axis)
            half_angle = angle / 2
            cot_half = 1 / np.tan(half_angle)
            V_inv = (half_angle * cot_half * np.eye(3) + 
                     (1 - half_angle * cot_half) * np.outer(axis, axis) -
                     half_angle * K)
        
        rho = V_inv @ t
        
        return np.concatenate([rho, phi])
    
    @staticmethod
    def pose_composition(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """
        Compose two SE(3) transformations: T1 ∘ T2.
        
        Args:
            T1: First 4x4 transformation matrix
            T2: Second 4x4 transformation matrix
            
        Returns:
            Composed 4x4 transformation matrix
        """
        SE3Utils.validate_homogeneous_matrix(T1)
        SE3Utils.validate_homogeneous_matrix(T2)
        
        return T1 @ T2
    
    @staticmethod
    def pose_inverse(T: np.ndarray) -> np.ndarray:
        """
        Compute inverse of SE(3) transformation.
        
        Args:
            T: 4x4 transformation matrix
            
        Returns:
            Inverse 4x4 transformation matrix
        """
        SE3Utils.validate_homogeneous_matrix(T)
        
        R, t = SE3Utils.extract_rotation_translation(T)
        R_inv = R.T
        t_inv = -R_inv @ t
        
        return SE3Utils.homogeneous_matrix(R_inv, t_inv)
    
    @staticmethod
    def geodesic_distance_SO3(R1: np.ndarray, R2: np.ndarray) -> float:
        """
        Compute geodesic distance between two rotations in SO(3).
        
        Args:
            R1: First 3x3 rotation matrix
            R2: Second 3x3 rotation matrix
            
        Returns:
            Geodesic distance (angle in radians)
        """
        SE3Utils.validate_rotation_matrix(R1)
        SE3Utils.validate_rotation_matrix(R2)
        
        # Relative rotation
        R_rel = R1.T @ R2
        
        # Extract angle
        trace = np.trace(R_rel)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        return angle
    
    @staticmethod
    def geodesic_distance_SE3(T1: np.ndarray, T2: np.ndarray, 
                            translation_weight: float = 1.0) -> float:
        """
        Compute geodesic distance between two SE(3) transformations.
        
        Computes the weighted sum of rotation and translation distances:
        d(T1, T2) = d_rot(R1, R2) + w * ||t2 - t1||₂
        where d_rot is the geodesic distance on SO(3).
        
        Args:
            T1: First 4x4 transformation matrix
            T2: Second 4x4 transformation matrix
            translation_weight: Weight for translation vs rotation distance
            
        Returns:
            Combined geodesic distance
            
        Examples:
            >>> import numpy as np
            >>> T1 = np.eye(4)
            >>> T2 = SE3Utils.homogeneous_matrix(np.eye(3), np.array([1, 0, 0]))
            >>> 
            >>> # Pure translation distance
            >>> dist = SE3Utils.geodesic_distance_SE3(T1, T2)
            >>> np.isclose(dist, 1.0)
            True
            >>> 
            >>> # With rotation
            >>> R_z90 = SE3Utils.axis_angle_to_rotation_matrix(np.array([0, 0, np.pi/2]))
            >>> T3 = SE3Utils.homogeneous_matrix(R_z90, np.zeros(3))
            >>> dist_rot = SE3Utils.geodesic_distance_SE3(T1, T3)
            >>> np.isclose(dist_rot, np.pi/2)
            True
        """
        SE3Utils.validate_homogeneous_matrix(T1)
        SE3Utils.validate_homogeneous_matrix(T2)
        
        R1, t1 = SE3Utils.extract_rotation_translation(T1)
        R2, t2 = SE3Utils.extract_rotation_translation(T2)
        
        # Rotation distance
        rotation_dist = SE3Utils.geodesic_distance_SO3(R1, R2)
        
        # Translation distance
        translation_dist = np.linalg.norm(t2 - t1)
        
        return rotation_dist + translation_weight * translation_dist
    
    @staticmethod
    def bounded_pose_step(T_current: np.ndarray, xi_step: np.ndarray,
                         max_translation: float = 1.0, 
                         max_rotation: float = np.pi/4) -> np.ndarray:
        """
        Apply bounded step in se(3) to current pose.
        
        Clips the translation and rotation components of the step to stay
        within specified bounds, then applies the step to the current pose.
        This is useful for gradient-based optimization with constraints.
        
        Args:
            T_current: Current 4x4 transformation matrix
            xi_step: 6-element se(3) step vector
            max_translation: Maximum translation step magnitude
            max_rotation: Maximum rotation step magnitude (radians)
            
        Returns:
            New bounded 4x4 transformation matrix
            
        Examples:
            >>> import numpy as np
            >>> T_start = np.eye(4)
            >>> # Large step that should be clipped
            >>> xi_large = np.array([10, 10, 10, np.pi, np.pi, np.pi])
            >>> T_bounded = SE3Utils.bounded_pose_step(T_start, xi_large, 
            ...                                      max_translation=1.0, 
            ...                                      max_rotation=np.pi/4)
            >>> 
            >>> # Check translation bound
            >>> translation_norm = np.linalg.norm(T_bounded[:3, 3])
            >>> translation_norm <= 1.0 + 1e-10
            True
        """
        SE3Utils.validate_homogeneous_matrix(T_current)
        
        xi_step = np.asarray(xi_step, dtype=np.float64)
        if xi_step.shape != (6,):
            raise SE3ValidationError(f"Step vector must be 6-element, got shape {xi_step.shape}")
        
        rho = xi_step[:3]
        phi = xi_step[3:]
        
        # Clip translation
        rho_norm = np.linalg.norm(rho)
        if rho_norm > max_translation:
            rho = rho * (max_translation / rho_norm)
        
        # Clip rotation
        phi_norm = np.linalg.norm(phi)
        if phi_norm > max_rotation:
            phi = phi * (max_rotation / phi_norm)
        
        xi_clipped = np.concatenate([rho, phi])
        T_step = SE3Utils.se3_exp_map(xi_clipped)
        
        return SE3Utils.pose_composition(T_current, T_step)
    
    @staticmethod
    def interpolate_poses(T1: np.ndarray, T2: np.ndarray, t: float) -> np.ndarray:
        """
        Interpolate between two SE(3) poses using geodesic interpolation.
        
        Performs geodesic interpolation in SE(3) by computing:
        T(t) = T1 * exp(t * log(T1^(-1) * T2))
        This gives the shortest path between T1 and T2 on the SE(3) manifold.
        
        Args:
            T1: Start 4x4 transformation matrix
            T2: End 4x4 transformation matrix
            t: Interpolation parameter [0, 1]
            
        Returns:
            Interpolated 4x4 transformation matrix
            
        Examples:
            >>> import numpy as np
            >>> T1 = np.eye(4)
            >>> T2 = SE3Utils.homogeneous_matrix(np.eye(3), np.array([2, 0, 0]))
            >>> 
            >>> # At t=0, should get T1
            >>> T_start = SE3Utils.interpolate_poses(T1, T2, 0.0)
            >>> np.allclose(T_start, T1)
            True
            >>> 
            >>> # At t=1, should get T2
            >>> T_end = SE3Utils.interpolate_poses(T1, T2, 1.0)
            >>> np.allclose(T_end, T2)
            True
            >>> 
            >>> # At t=0.5, should be halfway
            >>> T_mid = SE3Utils.interpolate_poses(T1, T2, 0.5)
            >>> np.allclose(T_mid[:3, 3], [1, 0, 0])
            True
        """
        SE3Utils.validate_homogeneous_matrix(T1)
        SE3Utils.validate_homogeneous_matrix(T2)
        
        if not (0 <= t <= 1):
            raise ValueError(f"Interpolation parameter t must be in [0,1], got {t}")
        
        # Compute relative transformation
        T_rel = SE3Utils.pose_composition(SE3Utils.pose_inverse(T1), T2)
        
        # Get se(3) representation
        xi_rel = SE3Utils.se3_log_map(T_rel)
        
        # Scale by interpolation parameter
        xi_interp = t * xi_rel
        
        # Apply to start pose
        T_step = SE3Utils.se3_exp_map(xi_interp)
        return SE3Utils.pose_composition(T1, T_step)
    
    @staticmethod
    def pose_difference(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """
        Compute pose difference T1^(-1) * T2 in se(3).
        
        Computes the se(3) vector that represents the transformation
        needed to go from T1 to T2. This is the logarithm of T1^(-1) * T2.
        
        Args:
            T1: Reference 4x4 transformation matrix
            T2: Target 4x4 transformation matrix
            
        Returns:
            6-element se(3) difference vector
            
        Examples:
            >>> import numpy as np
            >>> T1 = np.eye(4)
            >>> T2 = create_pose([1, 2, 3])
            >>> 
            >>> # Compute difference
            >>> xi_diff = SE3Utils.pose_difference(T1, T2)
            >>> 
            >>> # Apply difference should reconstruct T2
            >>> T_diff = SE3Utils.se3_exp_map(xi_diff)
            >>> T_result = SE3Utils.pose_composition(T1, T_diff)
            >>> np.allclose(T_result, T2, atol=1e-10)
            True
            >>> 
            >>> # For identity poses, difference should be zero
            >>> xi_zero = SE3Utils.pose_difference(T1, T1)
            >>> np.allclose(xi_zero, np.zeros(6))
            True
        """
        SE3Utils.validate_homogeneous_matrix(T1)
        SE3Utils.validate_homogeneous_matrix(T2)
        
        T_diff = SE3Utils.pose_composition(SE3Utils.pose_inverse(T1), T2)
        return SE3Utils.se3_log_map(T_diff)
    
    @staticmethod
    def random_se3_pose(translation_range: float = 1.0, 
                       rotation_range: float = np.pi) -> np.ndarray:
        """
        Generate random SE(3) pose.
        
        Creates a random pose by sampling translation uniformly and
        rotation uniformly on SO(3) using axis-angle representation.
        
        Args:
            translation_range: Range for random translation
            rotation_range: Range for random rotation (radians)
            
        Returns:
            Random 4x4 transformation matrix
            
        Examples:
            >>> import numpy as np
            >>> # Generate random pose
            >>> T_random = SE3Utils.random_se3_pose(translation_range=2.0, 
            ...                                    rotation_range=np.pi/2)
            >>> 
            >>> # Validate it's a proper SE(3) matrix
            >>> SE3Utils.validate_homogeneous_matrix(T_random)
            True
            >>> 
            >>> # Check translation is within bounds
            >>> translation_norm = np.linalg.norm(T_random[:3, 3])
            >>> translation_norm <= 2.0 * np.sqrt(3) + 1e-10  # max possible norm
            True
        """
        # Random translation
        t = np.random.uniform(-translation_range, translation_range, 3)
        
        # Random rotation (axis-angle)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(-rotation_range, rotation_range)
        R = SE3Utils.rodrigues_formula(axis, angle)
        
        return SE3Utils.homogeneous_matrix(R, t)


# Convenience functions for common operations
def create_pose(position: np.ndarray, orientation: Union[np.ndarray, str] = None, 
               orientation_type: str = 'quaternion') -> np.ndarray:
    """
    Convenience function to create SE(3) pose from position and orientation.
    
    This is a user-friendly wrapper around SE3Utils.homogeneous_matrix
    that accepts different orientation representations.
    
    Args:
        position: 3-element position vector
        orientation: Orientation as quaternion, euler angles, or rotation matrix
        orientation_type: Type of orientation ('quaternion', 'euler', 'matrix')
        
    Returns:
        4x4 homogeneous transformation matrix
        
    Examples:
        >>> import numpy as np
        >>> # Position only (identity orientation)
        >>> T1 = create_pose([1, 2, 3])
        >>> np.allclose(T1[:3, 3], [1, 2, 3])
        True
        >>> np.allclose(T1[:3, :3], np.eye(3))
        True
        
        >>> # With quaternion orientation
        >>> quat = [0, 0, 0, 1]  # Identity quaternion
        >>> T2 = create_pose([0, 0, 0], quat, 'quaternion')
        >>> np.allclose(T2[:3, :3], np.eye(3))
        True
        
        >>> # With Euler angles (90 deg about Z)
        >>> euler = [0, 0, np.pi/2]
        >>> T3 = create_pose([0, 0, 0], euler, 'euler')
        >>> expected_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        >>> np.allclose(T3[:3, :3], expected_R, atol=1e-10)
        True
    """
    position = np.asarray(position)
    
    if orientation is None:
        R = np.eye(3)
    elif orientation_type == 'quaternion':
        R = SE3Utils.quaternion_to_rotation_matrix(orientation)
    elif orientation_type == 'euler':
        R = SE3Utils.euler_to_rotation_matrix(orientation)
    elif orientation_type == 'matrix':
        R = np.asarray(orientation)
    else:
        raise ValueError(f"Unknown orientation type: {orientation_type}")
    
    return SE3Utils.homogeneous_matrix(R, position)


def pose_to_dict(T: np.ndarray) -> dict:
    """
    Convert SE(3) pose to dictionary representation.
    
    Provides multiple representations of the same pose for flexibility
    in different applications (visualization, serialization, etc.).
    
    Args:
        T: 4x4 transformation matrix
        
    Returns:
        Dictionary with position, quaternion, euler angles
        
    Examples:
        >>> import numpy as np
        >>> # Test with simple translation
        >>> T = create_pose([1, 2, 3])
        >>> pose_dict = pose_to_dict(T)
        >>> pose_dict['position']
        [1.0, 2.0, 3.0]
        >>> len(pose_dict['quaternion'])
        4
        >>> len(pose_dict['euler_xyz'])
        3
        
        >>> # Round trip test
        >>> T_original = create_pose([1, 2, 3], [0.1, 0.2, 0.3], 'euler')
        >>> pose_dict = pose_to_dict(T_original)
        >>> T_restored = dict_to_pose(pose_dict)
        >>> np.allclose(T_restored, T_original, atol=1e-10)
        True
    """
    SE3Utils.validate_homogeneous_matrix(T)
    R, t = SE3Utils.extract_rotation_translation(T)
    
    quat = SE3Utils.rotation_matrix_to_quaternion(R)
    euler = SE3Utils.rotation_matrix_to_euler(R)
    
    return {
        'position': t.tolist(),
        'quaternion': quat.tolist(),  # [x, y, z, w]
        'euler_xyz': euler.tolist(),
        'matrix': T.tolist()
    }


def dict_to_pose(pose_dict: dict) -> np.ndarray:
    """
    Convert dictionary representation to SE(3) pose.
    
    Args:
        pose_dict: Dictionary with 'position' and one of 'quaternion'/'euler_xyz'/'matrix'
        
    Returns:
        4x4 transformation matrix
    """
    position = np.array(pose_dict['position'])
    
    if 'matrix' in pose_dict:
        return np.array(pose_dict['matrix'])
    elif 'quaternion' in pose_dict:
        return create_pose(position, pose_dict['quaternion'], 'quaternion')
    elif 'euler_xyz' in pose_dict:
        return create_pose(position, pose_dict['euler_xyz'], 'euler')
    else:
        raise ValueError("Dictionary must contain 'quaternion', 'euler_xyz', or 'matrix'")


# Additional utility functions for advanced operations
def compute_pose_jacobian(T: np.ndarray, perturbation: float = 1e-8) -> np.ndarray:
    """
    Compute numerical Jacobian of pose with respect to se(3) perturbations.
    
    Useful for optimization and control applications where gradients
    with respect to pose parameters are needed.
    
    Args:
        T: 4x4 transformation matrix
        perturbation: Small perturbation for numerical differentiation
        
    Returns:
        6x6 Jacobian matrix dT/dxi evaluated at T
        
    Examples:
        >>> import numpy as np
        >>> T = create_pose([1, 2, 3], [0.1, 0.2, 0.3], 'euler')
        >>> J = compute_pose_jacobian(T)
        >>> J.shape
        (6, 6)
        >>> # Jacobian should be close to identity for small perturbations
        >>> np.allclose(np.diag(J), np.ones(6), rtol=0.1)
        True
    """
    # Convert current pose to se(3)
    xi_current = SE3Utils.se3_log_map(T)
    
    # Initialize Jacobian matrix
    J = np.zeros((6, 6))
    
    # Compute numerical derivatives
    for i in range(6):
        # Forward perturbation
        xi_plus = xi_current.copy()
        xi_plus[i] += perturbation
        T_plus = SE3Utils.se3_exp_map(xi_plus)
        
        # Backward perturbation  
        xi_minus = xi_current.copy()
        xi_minus[i] -= perturbation
        T_minus = SE3Utils.se3_exp_map(xi_minus)
        
        # Central difference
        xi_plus_log = SE3Utils.se3_log_map(T_plus)
        xi_minus_log = SE3Utils.se3_log_map(T_minus)
        J[:, i] = (xi_plus_log - xi_minus_log) / (2 * perturbation)
    
    return J


def pose_error_metrics(T_desired: np.ndarray, T_actual: np.ndarray) -> dict:
    """
    Compute comprehensive error metrics between desired and actual poses.
    
    Provides multiple error measures commonly used in robotics and
    computer vision applications.
    
    Args:
        T_desired: Desired 4x4 transformation matrix
        T_actual: Actual 4x4 transformation matrix
        
    Returns:
        Dictionary containing various error metrics
        
    Examples:
        >>> import numpy as np
        >>> T_desired = create_pose([1, 2, 3], [0, 0, 0], 'euler')
        >>> T_actual = create_pose([1.1, 2.05, 2.95], [0.01, 0, 0], 'euler')
        >>> errors = pose_error_metrics(T_desired, T_actual)
        >>> 
        >>> # Check that all expected metrics are present
        >>> required_keys = ['translation_error', 'rotation_error', 
        ...                  'geodesic_distance', 'frobenius_norm']
        >>> all(key in errors for key in required_keys)
        True
        >>> 
        >>> # Translation error should be reasonable
        >>> errors['translation_error'] < 0.2
        True
    """
    SE3Utils.validate_homogeneous_matrix(T_desired)
    SE3Utils.validate_homogeneous_matrix(T_actual)
    
    R_desired, t_desired = SE3Utils.extract_rotation_translation(T_desired)
    R_actual, t_actual = SE3Utils.extract_rotation_translation(T_actual)
    
    # Translation error (Euclidean distance)
    translation_error = np.linalg.norm(t_actual - t_desired)
    
    # Rotation error (geodesic distance on SO(3))
    rotation_error = SE3Utils.geodesic_distance_SO3(R_desired, R_actual)
    
    # Combined geodesic distance
    geodesic_distance = SE3Utils.geodesic_distance_SE3(T_desired, T_actual)
    
    # Frobenius norm of difference
    frobenius_norm = np.linalg.norm(T_actual - T_desired, 'fro')
    
    # se(3) difference norm
    xi_diff = SE3Utils.pose_difference(T_desired, T_actual)
    se3_norm = np.linalg.norm(xi_diff)
    
    # Angular error in degrees for convenience
    rotation_error_deg = np.degrees(rotation_error)
    
    return {
        'translation_error': float(translation_error),
        'rotation_error': float(rotation_error),
        'rotation_error_deg': float(rotation_error_deg),
        'geodesic_distance': float(geodesic_distance),
        'frobenius_norm': float(frobenius_norm),
        'se3_norm': float(se3_norm),
        'xi_difference': xi_diff
    }


# Module-level constants for common poses
IDENTITY_POSE = np.eye(4)
ORIGIN_POSE = IDENTITY_POSE.copy()

# Common rotation matrices
ROT_X_90 = SE3Utils.axis_angle_to_rotation_matrix(np.array([np.pi/2, 0, 0]))
ROT_Y_90 = SE3Utils.axis_angle_to_rotation_matrix(np.array([0, np.pi/2, 0]))
ROT_Z_90 = SE3Utils.axis_angle_to_rotation_matrix(np.array([0, 0, np.pi/2]))

# Common poses
POSE_X_90 = SE3Utils.homogeneous_matrix(ROT_X_90, np.zeros(3))
POSE_Y_90 = SE3Utils.homogeneous_matrix(ROT_Y_90, np.zeros(3))
POSE_Z_90 = SE3Utils.homogeneous_matrix(ROT_Z_90, np.zeros(3))
