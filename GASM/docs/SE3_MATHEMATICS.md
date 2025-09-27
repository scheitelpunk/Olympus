# SE(3) Mathematics and Constraint System Documentation

## Overview

This document provides a comprehensive mathematical foundation for the SE(3) (Special Euclidean Group in 3D) operations and constraint systems used throughout the GASM architecture. It covers theoretical background, practical implementation, and computational aspects of geometric transformations and spatial constraints.

## Table of Contents

1. [Mathematical Foundations](#mathematical-foundations)
2. [SE(3) Group Theory](#se3-group-theory)
3. [Lie Algebra and Exponential Maps](#lie-algebra-and-exponential-maps)
4. [Constraint System Mathematics](#constraint-system-mathematics)
5. [Optimization on Manifolds](#optimization-on-manifolds)
6. [Implementation Details](#implementation-details)
7. [Numerical Considerations](#numerical-considerations)
8. [Applications and Examples](#applications-and-examples)

## Mathematical Foundations

### Coordinate Systems and Transformations

The GASM system operates with 6-DOF (degree of freedom) spatial transformations representing both position and orientation in 3D space.

#### Homogeneous Transformations

Every rigid body transformation in SE(3) can be represented as a 4×4 homogeneous matrix:

```
T = [R  t]  ∈ SE(3)
    [0  1]

where:
- R ∈ SO(3) is a 3×3 rotation matrix
- t ∈ ℝ³ is a 3×1 translation vector
- 0 is a 1×3 zero vector
```

**Properties of SE(3):**
- **Group closure:** T₁, T₂ ∈ SE(3) ⟹ T₁T₂ ∈ SE(3)
- **Identity element:** I₄ = diag(1,1,1,1)
- **Inverse:** T⁻¹ = [Rᵀ  -Rᵀt; 0  1]
- **Associativity:** (T₁T₂)T₃ = T₁(T₂T₃)

#### Rotation Representations

The system supports multiple rotation representations:

**1. Rotation Matrices (SO(3))**
```
R ∈ SO(3) = {R ∈ ℝ³ˣ³ | RRᵀ = I, det(R) = 1}
```

**2. Quaternions**
```
q = [qₓ, qᵧ, qᵧ, qw] ∈ ℍ, ||q|| = 1

Rotation matrix from quaternion:
R = I + 2qw[q×] + 2[q×]²
where [q×] is the skew-symmetric matrix of q = [qₓ, qᵧ, qᵧ]
```

**3. Axis-Angle Representation**
```
ω = θu ∈ ℝ³
where θ is rotation angle and u is unit axis vector

Rodrigues' formula:
R = I + sin(θ)[u×] + (1-cos(θ))[u×]²
```

**4. Euler Angles**
```
(α, β, γ) representing rotations about fixed or moving axes
Multiple conventions: XYZ, ZYX, ZXZ, etc.
```

### Vector Spaces and Manifolds

**Euclidean Space ℝ³:**
- Translation vectors: t = [x, y, z]ᵀ
- Standard inner product: ⟨u,v⟩ = uᵀv
- Euclidean norm: ||v|| = √(vᵀv)

**Special Orthogonal Group SO(3):**
- 3D rotation group
- Lie group with manifold structure
- Tangent space at identity: so(3)

**Special Euclidean Group SE(3):**
- Semi-direct product: SE(3) = ℝ³ ⋊ SO(3)
- 6-dimensional Lie group
- Tangent space at identity: se(3)

## SE(3) Group Theory

### Group Operations

**Composition (Group Multiplication):**
```python
def compose_transforms(T1, T2):
    """
    Compose two SE(3) transformations: T1 ∘ T2
    
    Mathematical definition:
    T_result = T1 @ T2
    
    Physical interpretation:
    Apply T2 first, then T1
    """
    return T1 @ T2
```

**Inverse:**
```python
def inverse_transform(T):
    """
    Compute SE(3) inverse: T⁻¹
    
    Mathematical formula:
    T⁻¹ = [Rᵀ  -Rᵀt]
          [0   1    ]
    
    Property: T @ T⁻¹ = T⁻¹ @ T = I
    """
    R, t = T[:3, :3], T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv
```

### Adjoint Representation

The adjoint representation maps SE(3) elements to linear operators on se(3):

```
Ad_T : se(3) → se(3)

For T = [R t; 0 1] and ξ = [ρ; φ] ∈ se(3):

Ad_T = [R    [t×]R]  ∈ ℝ⁶ˣ⁶
       [0    R   ]

where [t×] is the skew-symmetric matrix of t
```

**Implementation:**
```python
def adjoint_matrix(T):
    """
    Compute adjoint matrix for SE(3) transformation
    
    Mathematics:
    Ad_T maps se(3) algebra elements according to:
    Ad_T · ξ = T · ξ · T⁻¹ (in matrix Lie algebra)
    """
    R, t = T[:3, :3], T[:3, 3]
    
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[:3, 3:] = skew_symmetric(t) @ R
    
    return Ad
```

## Lie Algebra and Exponential Maps

### se(3) Lie Algebra

The tangent space se(3) at the identity consists of 6×6 matrices:

```
ξ̂ = [ω̂  ρ] ∈ se(3)
    [0  0]

where:
- ω̂ ∈ so(3) is a 3×3 skew-symmetric matrix (rotation)
- ρ ∈ ℝ³ is translation
- Coordinate vector: ξ = [ρ; ω] ∈ ℝ⁶
```

### Exponential Map: se(3) → SE(3)

The exponential map connects the Lie algebra to the Lie group:

```
exp: se(3) → SE(3)
exp(ξ̂) = exp([ω̂  ρ]) = [R  V·ρ]
              [0  0]     [0  1 ]

where:
- R = exp(ω̂) (rotation matrix from so(3))
- V is the "left Jacobian" matrix
```

**Left Jacobian for Translation:**
```
For ω ≠ 0:
V = I + (1-cos(θ))/θ² [ω×] + (θ-sin(θ))/θ³ [ω×]²

For ω ≈ 0:
V ≈ I + ½[ω×] + 1/12[ω×]² + ...
```

**Implementation:**
```python
def se3_exp_map(xi):
    """
    Exponential map from se(3) to SE(3)
    
    Input: xi ∈ ℝ⁶ = [ρ; ω] where ρ is translation, ω is rotation
    Output: T ∈ SE(3) homogeneous transformation matrix
    
    Mathematical formula:
    T = exp(ξ̂) = [exp(ω̂)  V·ρ]
                  [0       1  ]
    """
    rho = xi[:3]  # Translation part
    omega = xi[3:]  # Rotation part
    
    # Rotation matrix from axis-angle
    theta = np.linalg.norm(omega)
    
    if theta < 1e-8:
        # Small angle approximation
        R = np.eye(3) + skew_symmetric(omega)
        V = np.eye(3) + 0.5 * skew_symmetric(omega)
    else:
        # Full formula
        u = omega / theta
        R = rodrigues_formula(u, theta)
        
        # Left Jacobian
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        V = (sin_theta / theta) * np.eye(3) + \
            (1 - cos_theta) / theta * skew_symmetric(u) + \
            (theta - sin_theta) / theta * np.outer(u, u)
    
    # Construct homogeneous matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    
    return T
```

### Logarithm Map: SE(3) → se(3)

The inverse of the exponential map:

```python
def se3_log_map(T):
    """
    Logarithm map from SE(3) to se(3)
    
    Input: T ∈ SE(3) homogeneous transformation matrix  
    Output: xi ∈ ℝ⁶ = [ρ; ω] coordinates in se(3)
    
    Mathematical process:
    1. Extract rotation matrix R and translation t
    2. Compute ω from log(R) 
    3. Compute ρ from V⁻¹ · t where V is left Jacobian
    """
    R, t = T[:3, :3], T[:3, 3]
    
    # Rotation part: R → ω
    omega = rotation_matrix_to_axis_angle(R)
    theta = np.linalg.norm(omega)
    
    if theta < 1e-8:
        # Small angle approximation
        V_inv = np.eye(3) - 0.5 * skew_symmetric(omega)
    else:
        # Full inverse left Jacobian
        u = omega / theta
        half_theta = 0.5 * theta
        cot_half = 1.0 / np.tan(half_theta)
        
        V_inv = half_theta * cot_half * np.eye(3) + \
                0.5 * skew_symmetric(omega) + \
                (1.0 / (2 * np.sin(half_theta)**2) - cot_half / (2 * half_theta)) * \
                np.outer(u, u)
    
    # Translation part: t → ρ
    rho = V_inv @ t
    
    return np.concatenate([rho, omega])
```

## Constraint System Mathematics

### Energy-Based Constraint Formulation

The GASM system formulates spatial constraints as energy functions to be minimized:

```
E_total(x) = E_data(x) + λ₁E_constraint₁(x) + λ₂E_constraint₂(x) + ...

where:
- x represents system state (poses, velocities, etc.)
- E_data(x) is data fitting term
- E_constraintᵢ(x) are constraint violation penalties
- λᵢ are constraint weights
```

### Geometric Constraints

**1. Distance Constraints:**
```
E_distance(x) = ½(||p₁ - p₂|| - d_target)²

Gradient:
∇E_distance = (||p₁ - p₂|| - d_target) · (p₁ - p₂)/||p₁ - p₂||
```

**2. Above/Below Constraints:**
```
E_above(x) = max(0, z_min - z_object)²

where z_object is object height and z_min is minimum allowed height
```

**3. Orientation Constraints:**
```
E_orientation(x) = ½||Log(R_target^T · R_current)||²

where Log: SO(3) → so(3) is the matrix logarithm
```

**4. Angular Constraints:**
```
E_angle(x) = ½(angle(v₁, v₂) - θ_target)²

where angle(v₁, v₂) = arccos(v₁^T v₂ / (||v₁|| ||v₂||))
```

### Constraint Jacobians

For optimization, we need constraint gradients:

```python
class ConstraintJacobian:
    """
    Compute constraint Jacobians for SE(3) optimization
    """
    
    @staticmethod
    def distance_jacobian(pose1, pose2, target_distance):
        """
        Jacobian of distance constraint w.r.t. poses
        
        Constraint: ||pos1 - pos2|| = target_distance
        Energy: E = ½(||pos1 - pos2|| - target_distance)²
        """
        pos1, pos2 = pose1[:3], pose2[:3]
        diff = pos1 - pos2
        current_distance = np.linalg.norm(diff)
        
        if current_distance < 1e-8:
            # Handle degenerate case
            direction = np.array([1, 0, 0])
        else:
            direction = diff / current_distance
        
        error = current_distance - target_distance
        
        # Jacobian w.r.t. position only (6 DOF poses)
        jacobian = np.zeros((1, 12))  # 1 constraint, 2 poses × 6 DOF
        jacobian[0, :3] = error * direction     # ∂E/∂pos1
        jacobian[0, 6:9] = -error * direction   # ∂E/∂pos2
        
        return jacobian
    
    @staticmethod  
    def orientation_jacobian(R_current, R_target):
        """
        Jacobian of orientation constraint
        
        Constraint: R_current = R_target
        Energy: E = ½||Log(R_target^T · R_current)||²
        """
        R_error = R_target.T @ R_current
        omega_error = rotation_matrix_to_axis_angle(R_error)
        
        # Jacobian computation involves adjoint representation
        # This is a simplified version
        jacobian = np.zeros((3, 6))
        jacobian[:, 3:] = np.eye(3)  # ∂E/∂rotation coordinates
        
        return jacobian
```

### Constraint Satisfaction Methods

**1. Penalty Methods:**
```
Minimize: f(x) + Σᵢ λᵢ max(0, gᵢ(x))²

Advantages: Unconstrained optimization
Disadvantages: Requires tuning penalty weights
```

**2. Lagrange Multipliers:**
```
L(x,μ) = f(x) + Σᵢ μᵢ gᵢ(x)

KKT conditions:
∇ₓL = 0
gᵢ(x) = 0  (equality constraints)
```

**3. Augmented Lagrangian:**
```
L_A(x,μ,ρ) = f(x) + Σᵢ μᵢ gᵢ(x) + ½ρ Σᵢ gᵢ(x)²

Combines benefits of penalty and Lagrangian methods
```

## Optimization on Manifolds

### Riemannian Optimization

Since SE(3) is a smooth manifold, we can use Riemannian optimization methods:

**1. Riemannian Gradient:**
```
grad_M f(x) = P_x(∇f(x))

where P_x projects Euclidean gradient onto tangent space T_x M
```

**2. Riemannian Exponential Map:**
```
Exp_x: T_x M → M
Exp_x(v) moves from point x in direction v on manifold
```

**3. Retraction:**
```
R_x: T_x M → M
R_x(v) ≈ Exp_x(v) but computationally cheaper
```

### SE(3) Optimization Implementation

```python
class SE3Optimizer:
    """
    Riemannian optimization on SE(3) manifold
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.lr = learning_rate
        self.max_iter = max_iterations
        
    def optimize(self, initial_poses, constraint_functions, constraint_weights):
        """
        Optimize poses to satisfy constraints
        
        Args:
            initial_poses: List of SE(3) matrices
            constraint_functions: List of constraint energy functions
            constraint_weights: Weights for each constraint
        """
        current_poses = [T.copy() for T in initial_poses]
        
        for iteration in range(self.max_iter):
            # Compute total gradient in se(3) coordinates
            total_gradient = self._compute_se3_gradient(
                current_poses, constraint_functions, constraint_weights
            )
            
            # Update poses using exponential map
            for i, pose in enumerate(current_poses):
                # Extract gradient for this pose
                grad_i = total_gradient[i * 6:(i + 1) * 6]
                
                # Take step in se(3) algebra
                step = -self.lr * grad_i
                
                # Map back to SE(3) using exponential map
                T_step = se3_exp_map(step)
                current_poses[i] = current_poses[i] @ T_step
            
            # Check convergence
            if self._check_convergence(total_gradient):
                break
                
        return current_poses
    
    def _compute_se3_gradient(self, poses, constraint_funcs, weights):
        """Compute gradient in se(3) algebra coordinates"""
        total_grad = np.zeros(len(poses) * 6)
        
        for constraint_func, weight in zip(constraint_funcs, weights):
            # Compute constraint gradient
            constraint_grad = constraint_func.gradient(poses)
            total_grad += weight * constraint_grad
            
        return total_grad
```

### Geodesic Interpolation

For smooth motion planning, we use geodesic interpolation on SE(3):

```python
def se3_interpolate(T1, T2, t):
    """
    Geodesic interpolation between SE(3) poses
    
    Mathematical formula:
    T(t) = T1 · exp(t · log(T1⁻¹ · T2))
    
    Args:
        T1, T2: SE(3) transformation matrices
        t: interpolation parameter ∈ [0,1]
        
    Returns:
        Interpolated SE(3) transformation
    """
    # Compute relative transformation
    T_rel = np.linalg.inv(T1) @ T2
    
    # Map to se(3) algebra
    xi_rel = se3_log_map(T_rel)
    
    # Scale by interpolation parameter
    xi_interp = t * xi_rel
    
    # Map back to SE(3) and compose with T1
    T_step = se3_exp_map(xi_interp)
    return T1 @ T_step
```

## Implementation Details

### Numerical Stability

**1. Small Angle Approximations:**
```python
def safe_angle_computation(omega, threshold=1e-8):
    """
    Numerically stable angle computations
    """
    theta = np.linalg.norm(omega)
    
    if theta < threshold:
        # Use Taylor series expansion
        # sin(θ)/θ ≈ 1 - θ²/6 + θ⁴/120 - ...
        sin_over_theta = 1.0 - theta**2/6.0 + theta**4/120.0
        cos_minus_one_over_theta_sq = -0.5 + theta**2/24.0 - theta**4/720.0
    else:
        sin_over_theta = np.sin(theta) / theta
        cos_minus_one_over_theta_sq = (np.cos(theta) - 1.0) / (theta**2)
    
    return sin_over_theta, cos_minus_one_over_theta_sq
```

**2. Orthogonalization:**
```python
def orthogonalize_rotation_matrix(R, method='gram_schmidt'):
    """
    Ensure rotation matrix maintains SO(3) properties
    """
    if method == 'gram_schmidt':
        # Gram-Schmidt orthogonalization
        u1 = R[:, 0] / np.linalg.norm(R[:, 0])
        u2 = R[:, 1] - np.dot(R[:, 1], u1) * u1
        u2 = u2 / np.linalg.norm(u2)
        u3 = np.cross(u1, u2)
        
        R_ortho = np.column_stack([u1, u2, u3])
        
    elif method == 'svd':
        # SVD-based orthogonalization
        U, S, Vh = np.linalg.svd(R)
        R_ortho = U @ Vh
        
        # Ensure proper rotation (det = +1)
        if np.linalg.det(R_ortho) < 0:
            R_ortho[:, -1] *= -1
    
    return R_ortho
```

### Computational Optimization

**1. Vectorized Operations:**
```python
def batch_se3_operations(transforms):
    """
    Vectorized SE(3) operations for better performance
    """
    batch_size = len(transforms)
    
    # Stack all matrices for batch processing
    T_batch = np.stack(transforms, axis=0)  # Shape: [batch, 4, 4]
    
    # Batch extract rotations and translations
    R_batch = T_batch[:, :3, :3]  # Shape: [batch, 3, 3]
    t_batch = T_batch[:, :3, 3]   # Shape: [batch, 3]
    
    # Batch operations...
    return results
```

**2. Caching and Memoization:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_rodrigues_matrix(axis_tuple, angle):
    """
    Cache frequently computed rotation matrices
    """
    axis = np.array(axis_tuple)
    return rodrigues_formula(axis, angle)
```

### Error Handling and Validation

```python
class SE3ValidationError(Exception):
    """Custom exception for SE(3) validation errors"""
    pass

def validate_se3_matrix(T, tolerance=1e-6):
    """
    Comprehensive SE(3) matrix validation
    """
    if T.shape != (4, 4):
        raise SE3ValidationError(f"Matrix must be 4x4, got {T.shape}")
    
    # Check rotation part
    R = T[:3, :3]
    
    # Orthogonality check
    should_be_identity = R.T @ R
    identity_error = np.abs(should_be_identity - np.eye(3)).max()
    if identity_error > tolerance:
        raise SE3ValidationError(f"Rotation not orthogonal. Error: {identity_error}")
    
    # Determinant check  
    det = np.linalg.det(R)
    if abs(det - 1.0) > tolerance:
        raise SE3ValidationError(f"Determinant not 1. Det: {det}")
    
    # Bottom row check
    bottom_row = T[3, :]
    expected = np.array([0, 0, 0, 1])
    bottom_error = np.abs(bottom_row - expected).max()
    if bottom_error > tolerance:
        raise SE3ValidationError(f"Bottom row invalid. Expected [0,0,0,1], got {bottom_row}")
    
    return True
```

## Numerical Considerations

### Precision and Accuracy

**1. Floating Point Precision:**
```python
# Use appropriate precision for different operations
EPS = np.finfo(np.float64).eps  # ~2.22e-16
SQRT_EPS = np.sqrt(EPS)         # ~1.49e-8

# For angular calculations
ANGLE_EPS = 1e-12
MIN_ANGLE = 1e-12
MAX_ANGLE = np.pi - 1e-12
```

**2. Condition Number Monitoring:**
```python
def check_matrix_conditioning(A, threshold=1e12):
    """
    Monitor numerical conditioning of matrices
    """
    cond_num = np.linalg.cond(A)
    if cond_num > threshold:
        logger.warning(f"Matrix is ill-conditioned. Condition number: {cond_num:.2e}")
    return cond_num
```

### Singularity Handling

**1. Rotation Singularities:**
```python
def handle_rotation_singularities(omega, threshold=1e-8):
    """
    Handle singularities in rotation computations
    """
    theta = np.linalg.norm(omega)
    
    if theta < threshold:
        # Near identity: use series expansion
        return small_angle_rodrigues(omega)
    elif abs(theta - np.pi) < threshold:
        # Near π rotation: special handling required
        return near_pi_rotation(omega)
    else:
        # Normal case
        return rodrigues_formula(omega / theta, theta)
```

**2. Translation Singularities:**
```python
def handle_translation_singularities(t, R, threshold=1e-8):
    """
    Handle singularities in se(3) computations
    """
    # Check for degenerate translations
    if np.linalg.norm(t) < threshold:
        return np.eye(3)  # Identity left Jacobian
    
    # Check for rotation singularities affecting translation
    axis_angle = rotation_matrix_to_axis_angle(R)
    theta = np.linalg.norm(axis_angle)
    
    if theta < threshold:
        # Small rotation case
        return compute_left_jacobian_small_angle(axis_angle)
    else:
        return compute_left_jacobian_full(axis_angle)
```

## Applications and Examples

### Robotic Assembly Example

```python
def assembly_constraint_optimization():
    """
    Example: Optimize poses for robotic assembly task
    """
    # Initial poses
    peg_pose = create_pose([0.1, 0.0, 0.2], [0, 0, 0])  # Above hole
    hole_pose = create_pose([0.0, 0.0, 0.0], [0, 0, 0])  # On surface
    
    # Define constraints
    constraints = [
        # Peg should be directly above hole
        DistanceConstraint(
            peg_pose[:3, 3], hole_pose[:3, 3], 
            target_distance=0.05,  # 5cm apart horizontally
            axis='xy'  # Only x-y distance
        ),
        
        # Peg should be above hole vertically
        HeightConstraint(
            peg_pose[:3, 3], hole_pose[:3, 3],
            min_height=0.1  # At least 10cm above
        ),
        
        # Peg orientation should align with hole
        OrientationConstraint(
            peg_pose[:3, :3], hole_pose[:3, :3],
            tolerance=0.1  # 0.1 radian tolerance
        )
    ]
    
    # Optimize
    optimizer = SE3ConstraintOptimizer()
    optimized_poses = optimizer.optimize([peg_pose, hole_pose], constraints)
    
    return optimized_poses

def multi_object_arrangement():
    """
    Example: Arrange multiple objects with spatial relationships
    """
    # Objects: table, box1, box2, robot_arm
    initial_poses = {
        'table': create_pose([0, 0, 0], [0, 0, 0]),
        'box1': create_pose([0.2, 0.2, 0.1], [0, 0, 0.1]),  
        'box2': create_pose([-0.2, 0.2, 0.1], [0, 0, -0.1]),
        'robot': create_pose([0, -0.5, 0.3], [0, 0, np.pi])
    }
    
    # Constraints from natural language: 
    # "place box1 above box2, both on table, robot facing boxes"
    constraints = [
        # Box1 above box2
        RelativePositionConstraint('box1', 'box2', 'above', distance=0.1),
        
        # Both boxes on table
        RelativePositionConstraint('box1', 'table', 'above', distance=0.05),
        RelativePositionConstraint('box2', 'table', 'above', distance=0.05),
        
        # Robot facing boxes
        OrientationConstraint('robot', target_direction=[0, 1, 0])  # Y-axis
    ]
    
    # Solve constraint system
    solver = ConstraintSolver()
    final_poses = solver.solve(initial_poses, constraints)
    
    return final_poses
```

### Neural Network Integration

```python
class SE3EquivariantLayer(nn.Module):
    """
    Neural network layer that respects SE(3) equivariance
    """
    
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Equivariant operations
        self.linear = nn.Linear(feature_dim, feature_dim)
        self.se3_conv = SE3ConvolutionLayer(feature_dim)
        
    def forward(self, features, poses):
        """
        Forward pass maintaining SE(3) equivariance
        
        Property: If poses are transformed by T ∈ SE(3), 
        then output features transform consistently
        """
        # Standard feature processing
        processed_features = self.linear(features)
        
        # SE(3) geometric processing
        geometric_features = self.se3_conv(processed_features, poses)
        
        return geometric_features

def test_equivariance():
    """
    Test SE(3) equivariance property
    """
    layer = SE3EquivariantLayer(64)
    
    # Random input
    features = torch.randn(10, 64)
    poses = torch.randn(10, 6)  # se(3) coordinates
    
    # Compute output
    output1 = layer(features, poses)
    
    # Apply random SE(3) transformation
    T_random = se3_exp_map(torch.randn(6) * 0.1)
    T_matrix = torch.tensor(T_random, dtype=torch.float32)
    
    # Transform poses
    transformed_poses = []
    for pose in poses:
        T_pose = se3_exp_map(pose)
        T_transformed = T_matrix @ torch.tensor(T_pose)
        transformed_pose = se3_log_map(T_transformed.numpy())
        transformed_poses.append(torch.tensor(transformed_pose))
    
    transformed_poses = torch.stack(transformed_poses)
    
    # Compute output with transformed poses
    output2 = layer(features, transformed_poses)
    
    # Check equivariance: outputs should be consistently transformed
    # This is a simplified check - full equivariance testing is more complex
    print(f"Original output norm: {output1.norm():.4f}")
    print(f"Transformed output norm: {output2.norm():.4f}")
    print(f"Relative difference: {(output2 - output1).norm() / output1.norm():.4f}")
```

### Performance Benchmarking

```python
def benchmark_se3_operations():
    """
    Benchmark SE(3) operations for performance analysis
    """
    import time
    
    # Test data
    num_poses = 1000
    poses = [se3_random_pose() for _ in range(num_poses)]
    
    # Benchmark exponential map
    start_time = time.time()
    for _ in range(1000):
        xi = np.random.randn(6) * 0.1
        T = se3_exp_map(xi)
    exp_time = time.time() - start_time
    
    # Benchmark logarithm map  
    start_time = time.time()
    for pose in poses:
        xi = se3_log_map(pose)
    log_time = time.time() - start_time
    
    # Benchmark composition
    start_time = time.time()
    for i in range(len(poses)-1):
        T_comp = poses[i] @ poses[i+1]
    comp_time = time.time() - start_time
    
    print(f"SE(3) Operation Benchmarks:")
    print(f"Exponential map: {exp_time*1000:.2f}ms for 1000 ops")
    print(f"Logarithm map: {log_time*1000:.2f}ms for {num_poses} ops") 
    print(f"Composition: {comp_time*1000:.2f}ms for {num_poses-1} ops")
    
    return {
        'exp_map_time': exp_time,
        'log_map_time': log_time, 
        'composition_time': comp_time
    }
```

This comprehensive documentation provides the mathematical foundation needed to understand and extend the SE(3) operations and constraint systems in GASM. It serves as both theoretical reference and practical implementation guide for developers working with spatial transformations and geometric constraints.