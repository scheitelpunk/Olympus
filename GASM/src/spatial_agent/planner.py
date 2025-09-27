"""
Intelligent Rule-Based Motion Planner for Spatial Agents

This module provides a configurable motion planning system with multiple strategies:
- Direct gradient descent toward targets
- Constraint-aware motion planning
- Obstacle avoidance and collision checking
- Safety-bounded step limits
- Extensible interface for future LLM policy integration

The planner is designed to be easily replaceable while maintaining consistent interfaces.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import math

# Configure logging
logger = logging.getLogger(__name__)


class PlanningStrategy(Enum):
    """Available planning strategies"""
    DIRECT = "direct"           # Direct path to target
    CONSTRAINED = "constrained" # Respect all constraints
    SAFE = "safe"              # Conservative with extra safety margins
    ADAPTIVE = "adaptive"       # Adapt strategy based on situation


@dataclass
class Pose:
    """6-DOF pose representation (position + orientation)"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0   # Rotation around x-axis (radians)
    pitch: float = 0.0  # Rotation around y-axis (radians)
    yaw: float = 0.0    # Rotation around z-axis (radians)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z, roll, pitch, yaw]"""
        return np.array([self.x, self.y, self.z, self.roll, self.pitch, self.yaw])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Pose':
        """Create pose from numpy array"""
        return cls(x=arr[0], y=arr[1], z=arr[2], 
                  roll=arr[3], pitch=arr[4], yaw=arr[5])
    
    def distance_to(self, other: 'Pose') -> float:
        """Calculate Euclidean distance to another pose (position only)"""
        return math.sqrt((self.x - other.x)**2 + 
                        (self.y - other.y)**2 + 
                        (self.z - other.z)**2)
    
    def angular_distance_to(self, other: 'Pose') -> float:
        """Calculate angular distance to another pose (orientation only)"""
        dr = abs(self.roll - other.roll)
        dp = abs(self.pitch - other.pitch)
        dy = abs(self.yaw - other.yaw)
        
        # Normalize angles to [-π, π]
        dr = min(dr, 2*math.pi - dr)
        dp = min(dp, 2*math.pi - dp)
        dy = min(dy, 2*math.pi - dy)
        
        return math.sqrt(dr**2 + dp**2 + dy**2)


@dataclass
class Constraint:
    """Motion constraint definition"""
    type: str  # "above", "below", "angle", "speed", "collision"
    params: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # Higher = more important
    active: bool = True


@dataclass
class Obstacle:
    """Obstacle representation for collision checking"""
    center: Pose
    radius: float  # Simplified as sphere for now
    type: str = "sphere"  # Future: "box", "cylinder", etc.


@dataclass
class PlanningConfig:
    """Configuration parameters for motion planning"""
    # Step size limits (safety bounds)
    max_position_step: float = 0.1  # meters
    max_rotation_step: float = 0.2  # radians (~11 degrees)
    
    # Gradient descent parameters
    position_gain: float = 1.0
    rotation_gain: float = 0.5
    
    # Constraint parameters
    constraint_gain: float = 2.0
    safety_margin: float = 0.05  # meters
    
    # Convergence criteria
    position_tolerance: float = 0.01  # meters
    rotation_tolerance: float = 0.05  # radians (~3 degrees)
    
    # Planning behavior
    max_iterations: int = 1000
    enable_obstacle_avoidance: bool = True
    enable_constraint_checking: bool = True


@dataclass
class PlanningResult:
    """Result of a planning step"""
    success: bool
    next_pose: Optional[Pose] = None
    step_size: float = 0.0
    constraints_violated: List[str] = field(default_factory=list)
    obstacles_detected: List[int] = field(default_factory=list)  # Obstacle indices
    reasoning: str = ""
    debug_info: Dict[str, Any] = field(default_factory=dict)


class MotionPlanner:
    """
    Intelligent rule-based motion planner with multiple strategies.
    
    This planner provides the core functionality for spatial agent navigation
    while maintaining a clean interface that can be easily replaced by more
    advanced planners (including LLM-based policies) in the future.
    """
    
    def __init__(self, config: Optional[PlanningConfig] = None):
        """Initialize the motion planner with configuration"""
        self.config = config or PlanningConfig()
        self.obstacles: List[Obstacle] = []
        self.constraints: List[Constraint] = []
        self.strategy = PlanningStrategy.CONSTRAINED
        
        # Planning state
        self.planning_history: List[Dict[str, Any]] = []
        self.iteration_count = 0
        
        logger.info(f"MotionPlanner initialized with strategy: {self.strategy}")
    
    def set_strategy(self, strategy: PlanningStrategy):
        """Change the planning strategy"""
        self.strategy = strategy
        logger.info(f"Planning strategy changed to: {strategy}")
    
    def add_obstacle(self, obstacle: Obstacle) -> int:
        """Add an obstacle and return its index"""
        self.obstacles.append(obstacle)
        index = len(self.obstacles) - 1
        logger.debug(f"Added obstacle {index} at ({obstacle.center.x}, {obstacle.center.y}, {obstacle.center.z})")
        return index
    
    def add_constraint(self, constraint: Constraint) -> int:
        """Add a constraint and return its index"""
        self.constraints.append(constraint)
        index = len(self.constraints) - 1
        logger.debug(f"Added constraint {index}: {constraint.type}")
        return index
    
    def clear_obstacles(self):
        """Remove all obstacles"""
        self.obstacles.clear()
        logger.debug("All obstacles cleared")
    
    def clear_constraints(self):
        """Remove all constraints"""
        self.constraints.clear()
        logger.debug("All constraints cleared")
    
    def plan_step(self, current_pose: Pose, target_pose: Pose, 
                  constraints: Optional[List[Constraint]] = None) -> PlanningResult:
        """
        Core planning function - computes the next step toward the target.
        
        Args:
            current_pose: Current robot pose
            target_pose: Desired target pose
            constraints: Additional constraints for this step (optional)
            
        Returns:
            PlanningResult with next pose and planning information
        """
        self.iteration_count += 1
        
        # Combine global and local constraints
        all_constraints = self.constraints.copy()
        if constraints:
            all_constraints.extend(constraints)
        
        # Log planning attempt
        logger.debug(f"Planning step {self.iteration_count}: "
                    f"current=({current_pose.x:.3f}, {current_pose.y:.3f}, {current_pose.z:.3f}) "
                    f"target=({target_pose.x:.3f}, {target_pose.y:.3f}, {target_pose.z:.3f})")
        
        # Check if already at target
        if self._at_target(current_pose, target_pose):
            return PlanningResult(
                success=True,
                next_pose=current_pose,
                step_size=0.0,
                reasoning="Already at target within tolerance"
            )
        
        # Execute planning strategy
        if self.strategy == PlanningStrategy.DIRECT:
            return self._plan_direct(current_pose, target_pose, all_constraints)
        elif self.strategy == PlanningStrategy.CONSTRAINED:
            return self._plan_constrained(current_pose, target_pose, all_constraints)
        elif self.strategy == PlanningStrategy.SAFE:
            return self._plan_safe(current_pose, target_pose, all_constraints)
        elif self.strategy == PlanningStrategy.ADAPTIVE:
            return self._plan_adaptive(current_pose, target_pose, all_constraints)
        else:
            raise ValueError(f"Unknown planning strategy: {self.strategy}")
    
    def _plan_direct(self, current: Pose, target: Pose, 
                    constraints: List[Constraint]) -> PlanningResult:
        """Direct gradient descent toward target"""
        # Calculate direction vector
        direction = self._calculate_direction(current, target)
        
        # Apply gains and compute step
        step = direction * np.array([
            self.config.position_gain, self.config.position_gain, self.config.position_gain,
            self.config.rotation_gain, self.config.rotation_gain, self.config.rotation_gain
        ])
        
        # Apply step size limits
        step = self._apply_step_limits(step)
        
        # Compute next pose
        next_array = current.to_array() + step
        next_pose = Pose.from_array(next_array)
        
        # Basic collision checking
        obstacles_detected = []
        if self.config.enable_obstacle_avoidance:
            obstacles_detected = self._check_collisions(next_pose)
            if obstacles_detected:
                return PlanningResult(
                    success=False,
                    obstacles_detected=obstacles_detected,
                    reasoning="Direct path blocked by obstacles"
                )
        
        step_size = np.linalg.norm(step[:3])  # Position step size
        
        return PlanningResult(
            success=True,
            next_pose=next_pose,
            step_size=step_size,
            reasoning=f"Direct step toward target (size: {step_size:.4f}m)",
            debug_info={"direction": direction.tolist(), "step": step.tolist()}
        )
    
    def _plan_constrained(self, current: Pose, target: Pose, 
                         constraints: List[Constraint]) -> PlanningResult:
        """Constraint-aware motion planning"""
        # Start with direct path
        direction = self._calculate_direction(current, target)
        
        # Apply constraint forces
        constraint_forces = self._calculate_constraint_forces(current, target, constraints)
        
        # Combine forces
        total_force = direction + constraint_forces
        
        # Apply gains
        step = total_force * np.array([
            self.config.position_gain, self.config.position_gain, self.config.position_gain,
            self.config.rotation_gain, self.config.rotation_gain, self.config.rotation_gain
        ])
        
        # Apply step size limits
        step = self._apply_step_limits(step)
        
        # Compute next pose
        next_array = current.to_array() + step
        next_pose = Pose.from_array(next_array)
        
        # Check constraints and collisions
        violated_constraints = []
        obstacles_detected = []
        
        if self.config.enable_constraint_checking:
            violated_constraints = self._check_constraints(next_pose, constraints)
        
        if self.config.enable_obstacle_avoidance:
            obstacles_detected = self._check_collisions(next_pose)
        
        success = len(violated_constraints) == 0 and len(obstacles_detected) == 0
        step_size = np.linalg.norm(step[:3])
        
        reasoning = f"Constrained step (size: {step_size:.4f}m)"
        if violated_constraints:
            reasoning += f", violated: {violated_constraints}"
        if obstacles_detected:
            reasoning += f", obstacles: {obstacles_detected}"
        
        return PlanningResult(
            success=success,
            next_pose=next_pose if success else None,
            step_size=step_size,
            constraints_violated=violated_constraints,
            obstacles_detected=obstacles_detected,
            reasoning=reasoning,
            debug_info={
                "direction": direction.tolist(),
                "constraint_forces": constraint_forces.tolist(),
                "total_force": total_force.tolist(),
                "step": step.tolist()
            }
        )
    
    def _plan_safe(self, current: Pose, target: Pose, 
                   constraints: List[Constraint]) -> PlanningResult:
        """Conservative planning with extra safety margins"""
        # Use smaller step sizes for safety
        safe_config = PlanningConfig(
            max_position_step=self.config.max_position_step * 0.5,
            max_rotation_step=self.config.max_rotation_step * 0.5,
            position_gain=self.config.position_gain * 0.7,
            rotation_gain=self.config.rotation_gain * 0.7,
            safety_margin=self.config.safety_margin * 2.0
        )
        
        # Temporarily use safe config
        original_config = self.config
        self.config = safe_config
        
        try:
            result = self._plan_constrained(current, target, constraints)
            result.reasoning = f"Safe mode: {result.reasoning}"
            return result
        finally:
            self.config = original_config
    
    def _plan_adaptive(self, current: Pose, target: Pose, 
                      constraints: List[Constraint]) -> PlanningResult:
        """Adaptive planning that chooses strategy based on situation"""
        # Analyze situation
        distance_to_target = current.distance_to(target)
        nearby_obstacles = len([obs for obs in self.obstacles 
                              if current.distance_to(obs.center) < obs.radius + 0.2])
        high_priority_constraints = len([c for c in constraints if c.priority > 2])
        
        # Choose strategy based on analysis
        if nearby_obstacles > 0 or high_priority_constraints > 0:
            chosen_strategy = PlanningStrategy.SAFE
        elif distance_to_target > 0.5:
            chosen_strategy = PlanningStrategy.DIRECT
        else:
            chosen_strategy = PlanningStrategy.CONSTRAINED
        
        # Execute chosen strategy
        original_strategy = self.strategy
        self.strategy = chosen_strategy
        
        try:
            if chosen_strategy == PlanningStrategy.DIRECT:
                result = self._plan_direct(current, target, constraints)
            elif chosen_strategy == PlanningStrategy.SAFE:
                result = self._plan_safe(current, target, constraints)
            else:
                result = self._plan_constrained(current, target, constraints)
            
            result.reasoning = f"Adaptive ({chosen_strategy.value}): {result.reasoning}"
            return result
        finally:
            self.strategy = original_strategy
    
    def _calculate_direction(self, current: Pose, target: Pose) -> np.ndarray:
        """Calculate normalized direction vector from current to target"""
        current_arr = current.to_array()
        target_arr = target.to_array()
        
        direction = target_arr - current_arr
        
        # Normalize position and rotation components separately
        pos_norm = np.linalg.norm(direction[:3])
        rot_norm = np.linalg.norm(direction[3:])
        
        if pos_norm > 0:
            direction[:3] = direction[:3] / pos_norm
        if rot_norm > 0:
            direction[3:] = direction[3:] / rot_norm
            
        return direction
    
    def _calculate_constraint_forces(self, current: Pose, target: Pose, 
                                   constraints: List[Constraint]) -> np.ndarray:
        """Calculate force vectors from active constraints"""
        total_force = np.zeros(6)
        
        for constraint in constraints:
            if not constraint.active:
                continue
                
            force = self._calculate_single_constraint_force(current, target, constraint)
            total_force += force * constraint.priority * self.config.constraint_gain
        
        return total_force
    
    def _calculate_single_constraint_force(self, current: Pose, target: Pose, 
                                         constraint: Constraint) -> np.ndarray:
        """Calculate force from a single constraint"""
        force = np.zeros(6)
        
        if constraint.type == "above":
            # Stay above specified z-level
            min_z = constraint.params.get("z", 0.0)
            if current.z < min_z + self.config.safety_margin:
                force[2] = 1.0  # Upward force
                
        elif constraint.type == "below":
            # Stay below specified z-level
            max_z = constraint.params.get("z", 1.0)
            if current.z > max_z - self.config.safety_margin:
                force[2] = -1.0  # Downward force
                
        elif constraint.type == "angle":
            # Maintain specific orientation constraints
            target_angle = constraint.params.get("target", 0.0)
            axis = constraint.params.get("axis", "yaw")  # roll, pitch, or yaw
            
            if axis == "roll":
                error = target_angle - current.roll
                force[3] = np.sign(error) * min(abs(error), 1.0)
            elif axis == "pitch":
                error = target_angle - current.pitch
                force[4] = np.sign(error) * min(abs(error), 1.0)
            elif axis == "yaw":
                error = target_angle - current.yaw
                force[5] = np.sign(error) * min(abs(error), 1.0)
                
        elif constraint.type == "speed":
            # Speed limiting constraint (affects step size, not direction)
            pass  # Handled in step size limiting
            
        return force
    
    def _apply_step_limits(self, step: np.ndarray) -> np.ndarray:
        """Apply safety step size limits"""
        # Limit position steps
        pos_step = step[:3]
        pos_magnitude = np.linalg.norm(pos_step)
        if pos_magnitude > self.config.max_position_step:
            step[:3] = pos_step * (self.config.max_position_step / pos_magnitude)
        
        # Limit rotation steps
        rot_step = step[3:]
        rot_magnitude = np.linalg.norm(rot_step)
        if rot_magnitude > self.config.max_rotation_step:
            step[3:] = rot_step * (self.config.max_rotation_step / rot_magnitude)
            
        return step
    
    def _check_collisions(self, pose: Pose) -> List[int]:
        """Check for collisions with obstacles"""
        colliding_obstacles = []
        
        for i, obstacle in enumerate(self.obstacles):
            distance = pose.distance_to(obstacle.center)
            if distance < obstacle.radius + self.config.safety_margin:
                colliding_obstacles.append(i)
                
        return colliding_obstacles
    
    def _check_constraints(self, pose: Pose, constraints: List[Constraint]) -> List[str]:
        """Check constraint violations"""
        violations = []
        
        for constraint in constraints:
            if not constraint.active:
                continue
                
            if constraint.type == "above":
                min_z = constraint.params.get("z", 0.0)
                if pose.z < min_z:
                    violations.append(f"above_z_{min_z}")
                    
            elif constraint.type == "below":
                max_z = constraint.params.get("z", 1.0)
                if pose.z > max_z:
                    violations.append(f"below_z_{max_z}")
                    
            elif constraint.type == "angle":
                # Check angle constraints
                target_angle = constraint.params.get("target", 0.0)
                tolerance = constraint.params.get("tolerance", 0.1)
                axis = constraint.params.get("axis", "yaw")
                
                current_angle = getattr(pose, axis)
                if abs(current_angle - target_angle) > tolerance:
                    violations.append(f"angle_{axis}_{target_angle}")
                    
        return violations
    
    def _at_target(self, current: Pose, target: Pose) -> bool:
        """Check if current pose is close enough to target"""
        pos_dist = current.distance_to(target)
        rot_dist = current.angular_distance_to(target)
        
        return (pos_dist < self.config.position_tolerance and 
                rot_dist < self.config.rotation_tolerance)
    
    def get_planning_stats(self) -> Dict[str, Any]:
        """Get planning statistics and performance metrics"""
        return {
            "iteration_count": self.iteration_count,
            "obstacles_count": len(self.obstacles),
            "constraints_count": len(self.constraints),
            "strategy": self.strategy.value,
            "config": {
                "max_position_step": self.config.max_position_step,
                "max_rotation_step": self.config.max_rotation_step,
                "position_tolerance": self.config.position_tolerance,
                "rotation_tolerance": self.config.rotation_tolerance
            }
        }
    
    def reset_planning_state(self):
        """Reset planning state for new planning session"""
        self.planning_history.clear()
        self.iteration_count = 0
        logger.debug("Planning state reset")


# Utility functions for easy planner creation and configuration

def create_default_planner() -> MotionPlanner:
    """Create a motion planner with default configuration"""
    return MotionPlanner()

def create_safe_planner() -> MotionPlanner:
    """Create a conservative motion planner for safety-critical applications"""
    config = PlanningConfig(
        max_position_step=0.05,  # Smaller steps
        max_rotation_step=0.1,   # Smaller rotations
        position_gain=0.5,       # Lower gains
        rotation_gain=0.3,
        safety_margin=0.1        # Larger margins
    )
    planner = MotionPlanner(config)
    planner.set_strategy(PlanningStrategy.SAFE)
    return planner

def create_fast_planner() -> MotionPlanner:
    """Create a fast motion planner for open environments"""
    config = PlanningConfig(
        max_position_step=0.2,   # Larger steps
        max_rotation_step=0.5,   # Larger rotations
        position_gain=1.5,       # Higher gains
        rotation_gain=1.0,
        safety_margin=0.02       # Smaller margins
    )
    planner = MotionPlanner(config)
    planner.set_strategy(PlanningStrategy.DIRECT)
    return planner


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.DEBUG)
    
    # Create planner
    planner = create_default_planner()
    
    # Define poses
    start = Pose(x=0, y=0, z=1, yaw=0)
    target = Pose(x=1, y=1, z=1, yaw=math.pi/2)
    
    # Add some constraints
    planner.add_constraint(Constraint("above", {"z": 0.5}))
    planner.add_obstacle(Obstacle(Pose(x=0.5, y=0.5, z=1), radius=0.2))
    
    # Plan a step
    result = planner.plan_step(start, target)
    
    print(f"Planning result: {result.success}")
    if result.next_pose:
        print(f"Next pose: ({result.next_pose.x:.3f}, {result.next_pose.y:.3f}, {result.next_pose.z:.3f})")
    print(f"Reasoning: {result.reasoning}")