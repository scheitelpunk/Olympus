"""
Multi-Layer Action Filtering System

Provides comprehensive action filtering through multiple validation layers:
1. Physics validation (force, speed, acceleration limits)
2. Spatial validation (workspace boundaries, collision avoidance)
3. Intention validation (analyzing action purpose and safety)
4. Context validation (environmental conditions)
5. Human safety validation (proximity and interaction safety)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FilterLayer(Enum):
    """Action filter layers in order of execution"""
    PHYSICS = "physics"
    SPATIAL = "spatial"
    INTENTION = "intention"
    CONTEXT = "context"
    HUMAN_SAFETY = "human_safety"


class FilterStatus(Enum):
    """Filter result status"""
    APPROVED = "approved"
    BLOCKED = "blocked"
    MODIFIED = "modified"
    REQUIRES_CONFIRMATION = "requires_confirmation"


@dataclass
class FilterResult:
    """Result of action filtering"""
    status: FilterStatus
    layer: FilterLayer
    original_action: Dict[str, Any]
    filtered_action: Optional[Dict[str, Any]] = None
    reason: str = ""
    risk_score: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class PhysicsLimits:
    """Physics-based safety limits"""
    max_force: float = 20.0  # Newtons
    max_speed: float = 1.0   # m/s
    max_acceleration: float = 2.0  # m/s²
    max_jerk: float = 10.0   # m/s³
    max_torque: float = 5.0  # N⋅m


@dataclass
class SpatialLimits:
    """Spatial safety limits"""
    workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]  # ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    min_obstacle_distance: float = 0.1  # meters
    max_reach_distance: float = 1.5  # meters


class ActionFilter:
    """Multi-layer action filtering system"""
    
    def __init__(self, 
                 physics_limits: Optional[PhysicsLimits] = None,
                 spatial_limits: Optional[SpatialLimits] = None,
                 strict_mode: bool = True):
        """
        Initialize action filter
        
        Args:
            physics_limits: Physical safety limits
            spatial_limits: Spatial safety limits
            strict_mode: Enable strict safety mode (more conservative)
        """
        self.physics_limits = physics_limits or PhysicsLimits()
        self.spatial_limits = spatial_limits or SpatialLimits(
            workspace_bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0))
        )
        self.strict_mode = strict_mode
        self.filter_layers = [
            self._physics_filter,
            self._spatial_filter,
            self._intention_filter,
            self._context_filter,
            self._human_safety_filter
        ]
        
        logger.info(f"ActionFilter initialized with strict_mode={strict_mode}")
    
    def filter_action(self, action: Dict[str, Any]) -> FilterResult:
        """
        Filter action through all safety layers
        
        Args:
            action: Action to be filtered
            
        Returns:
            FilterResult with filtering outcome
        """
        current_action = action.copy()
        
        for i, filter_func in enumerate(self.filter_layers):
            layer = FilterLayer(list(FilterLayer)[i].value)
            
            try:
                result = filter_func(current_action)
                
                if result.status == FilterStatus.BLOCKED:
                    logger.warning(f"Action blocked at {layer.value} layer: {result.reason}")
                    return result
                
                elif result.status == FilterStatus.REQUIRES_CONFIRMATION:
                    logger.info(f"Action requires confirmation at {layer.value} layer: {result.reason}")
                    return result
                
                elif result.status == FilterStatus.MODIFIED:
                    logger.info(f"Action modified at {layer.value} layer: {result.reason}")
                    current_action = result.filtered_action
                
            except Exception as e:
                logger.error(f"Error in {layer.value} filter: {str(e)}")
                return FilterResult(
                    status=FilterStatus.BLOCKED,
                    layer=layer,
                    original_action=action,
                    reason=f"Filter error: {str(e)}",
                    risk_score=1.0
                )
        
        # All filters passed
        final_result = FilterResult(
            status=FilterStatus.APPROVED if current_action == action else FilterStatus.MODIFIED,
            layer=FilterLayer.HUMAN_SAFETY,  # Last layer
            original_action=action,
            filtered_action=current_action if current_action != action else None,
            reason="Action passed all safety filters",
            risk_score=self._calculate_overall_risk(current_action)
        )
        
        logger.info(f"Action filtering complete: {final_result.status.value}")
        return final_result
    
    def _physics_filter(self, action: Dict[str, Any]) -> FilterResult:
        """Physics-based filtering (Layer 1)"""
        violations = []
        modified_action = action.copy()
        
        # Check force limits
        if 'force' in action:
            force_magnitude = np.linalg.norm(action['force'])
            if force_magnitude > self.physics_limits.max_force:
                if self.strict_mode:
                    return FilterResult(
                        status=FilterStatus.BLOCKED,
                        layer=FilterLayer.PHYSICS,
                        original_action=action,
                        reason=f"Force {force_magnitude:.2f}N exceeds limit {self.physics_limits.max_force}N",
                        risk_score=min(1.0, force_magnitude / self.physics_limits.max_force)
                    )
                else:
                    # Scale down force
                    scale_factor = self.physics_limits.max_force / force_magnitude
                    modified_action['force'] = np.array(action['force']) * scale_factor
                    violations.append(f"Force scaled from {force_magnitude:.2f}N to {self.physics_limits.max_force}N")
        
        # Check speed limits
        if 'velocity' in action:
            speed = np.linalg.norm(action['velocity'])
            if speed > self.physics_limits.max_speed:
                if self.strict_mode:
                    return FilterResult(
                        status=FilterStatus.BLOCKED,
                        layer=FilterLayer.PHYSICS,
                        original_action=action,
                        reason=f"Speed {speed:.2f}m/s exceeds limit {self.physics_limits.max_speed}m/s",
                        risk_score=min(1.0, speed / self.physics_limits.max_speed)
                    )
                else:
                    # Scale down velocity
                    scale_factor = self.physics_limits.max_speed / speed
                    modified_action['velocity'] = np.array(action['velocity']) * scale_factor
                    violations.append(f"Speed scaled from {speed:.2f}m/s to {self.physics_limits.max_speed}m/s")
        
        # Check acceleration limits
        if 'acceleration' in action:
            accel = np.linalg.norm(action['acceleration'])
            if accel > self.physics_limits.max_acceleration:
                if self.strict_mode:
                    return FilterResult(
                        status=FilterStatus.BLOCKED,
                        layer=FilterLayer.PHYSICS,
                        original_action=action,
                        reason=f"Acceleration {accel:.2f}m/s² exceeds limit {self.physics_limits.max_acceleration}m/s²",
                        risk_score=min(1.0, accel / self.physics_limits.max_acceleration)
                    )
                else:
                    scale_factor = self.physics_limits.max_acceleration / accel
                    modified_action['acceleration'] = np.array(action['acceleration']) * scale_factor
                    violations.append(f"Acceleration scaled from {accel:.2f}m/s² to {self.physics_limits.max_acceleration}m/s²")
        
        if violations:
            return FilterResult(
                status=FilterStatus.MODIFIED,
                layer=FilterLayer.PHYSICS,
                original_action=action,
                filtered_action=modified_action,
                reason="; ".join(violations),
                risk_score=0.3  # Modified actions have moderate risk
            )
        
        return FilterResult(
            status=FilterStatus.APPROVED,
            layer=FilterLayer.PHYSICS,
            original_action=action,
            reason="Physics constraints satisfied",
            risk_score=0.1
        )
    
    def _spatial_filter(self, action: Dict[str, Any]) -> FilterResult:
        """Spatial boundary and collision filtering (Layer 2)"""
        violations = []
        
        # Check workspace boundaries
        if 'target_position' in action:
            pos = action['target_position']
            bounds = self.spatial_limits.workspace_bounds
            
            for i, (coord, (min_val, max_val)) in enumerate(zip(pos, bounds)):
                if coord < min_val or coord > max_val:
                    axis = ['x', 'y', 'z'][i]
                    return FilterResult(
                        status=FilterStatus.BLOCKED,
                        layer=FilterLayer.SPATIAL,
                        original_action=action,
                        reason=f"Target position {axis}={coord:.2f} outside workspace bounds [{min_val}, {max_val}]",
                        risk_score=0.8
                    )
        
        # Check reach distance
        if 'target_position' in action and 'current_position' in action:
            distance = np.linalg.norm(np.array(action['target_position']) - np.array(action['current_position']))
            if distance > self.spatial_limits.max_reach_distance:
                return FilterResult(
                    status=FilterStatus.BLOCKED,
                    layer=FilterLayer.SPATIAL,
                    original_action=action,
                    reason=f"Target distance {distance:.2f}m exceeds max reach {self.spatial_limits.max_reach_distance}m",
                    risk_score=0.7
                )
        
        # Check obstacle clearance
        if 'obstacles' in action and 'trajectory' in action:
            min_distance = float('inf')
            for point in action['trajectory']:
                for obstacle in action['obstacles']:
                    dist = np.linalg.norm(np.array(point) - np.array(obstacle['position']))
                    min_distance = min(min_distance, dist - obstacle.get('radius', 0.05))
            
            if min_distance < self.spatial_limits.min_obstacle_distance:
                return FilterResult(
                    status=FilterStatus.BLOCKED,
                    layer=FilterLayer.SPATIAL,
                    original_action=action,
                    reason=f"Trajectory clearance {min_distance:.3f}m below minimum {self.spatial_limits.min_obstacle_distance}m",
                    risk_score=0.9
                )
        
        return FilterResult(
            status=FilterStatus.APPROVED,
            layer=FilterLayer.SPATIAL,
            original_action=action,
            reason="Spatial constraints satisfied",
            risk_score=0.1
        )
    
    def _intention_filter(self, action: Dict[str, Any]) -> FilterResult:
        """Intention analysis filtering (Layer 3)"""
        # Analyze action intention for potential risks
        risk_indicators = []
        risk_score = 0.0
        
        # Check for rapid movements
        if 'velocity' in action:
            speed = np.linalg.norm(action['velocity'])
            if speed > 0.5:  # Half of max speed
                risk_indicators.append("High speed movement")
                risk_score += 0.2
        
        # Check for high force operations
        if 'force' in action:
            force = np.linalg.norm(action['force'])
            if force > 10.0:  # Half of max force
                risk_indicators.append("High force operation")
                risk_score += 0.3
        
        # Check for potentially dangerous tool usage
        if 'tool' in action:
            dangerous_tools = ['cutter', 'drill', 'grinder', 'plasma', 'laser']
            if action['tool'].lower() in dangerous_tools:
                risk_indicators.append(f"Dangerous tool: {action['tool']}")
                risk_score += 0.4
                
                # Require confirmation for dangerous tools
                return FilterResult(
                    status=FilterStatus.REQUIRES_CONFIRMATION,
                    layer=FilterLayer.INTENTION,
                    original_action=action,
                    reason=f"Dangerous tool operation requires confirmation: {action['tool']}",
                    risk_score=risk_score
                )
        
        # Check for repetitive motions (potential automation risk)
        if 'repetitions' in action and action['repetitions'] > 100:
            risk_indicators.append("High repetition count")
            risk_score += 0.2
        
        if risk_score > 0.6:
            return FilterResult(
                status=FilterStatus.REQUIRES_CONFIRMATION,
                layer=FilterLayer.INTENTION,
                original_action=action,
                reason=f"High risk intention detected: {'; '.join(risk_indicators)}",
                risk_score=risk_score
            )
        
        return FilterResult(
            status=FilterStatus.APPROVED,
            layer=FilterLayer.INTENTION,
            original_action=action,
            reason="Intention analysis passed",
            risk_score=max(0.1, risk_score)
        )
    
    def _context_filter(self, action: Dict[str, Any]) -> FilterResult:
        """Environmental context filtering (Layer 4)"""
        risk_factors = []
        risk_score = 0.0
        
        # Check environmental conditions
        if 'environment' in action:
            env = action['environment']
            
            # Check lighting conditions
            if env.get('lighting', 100) < 20:  # Too dark
                risk_factors.append("Poor lighting conditions")
                risk_score += 0.3
            
            # Check temperature
            temp = env.get('temperature', 20)
            if temp < 0 or temp > 50:
                risk_factors.append(f"Extreme temperature: {temp}°C")
                risk_score += 0.2
            
            # Check vibration levels
            if env.get('vibration_level', 0) > 5:
                risk_factors.append("High vibration environment")
                risk_score += 0.2
            
            # Check for hazardous materials
            if env.get('hazardous_materials', False):
                risk_factors.append("Hazardous materials present")
                risk_score += 0.4
        
        # Check system status
        if 'system_status' in action:
            status = action['system_status']
            
            if status.get('battery_level', 100) < 20:
                risk_factors.append("Low battery level")
                risk_score += 0.3
            
            if status.get('error_count', 0) > 5:
                risk_factors.append("Multiple system errors")
                risk_score += 0.4
            
            if not status.get('sensors_operational', True):
                return FilterResult(
                    status=FilterStatus.BLOCKED,
                    layer=FilterLayer.CONTEXT,
                    original_action=action,
                    reason="Critical sensors not operational",
                    risk_score=1.0
                )
        
        if risk_score > 0.7:
            return FilterResult(
                status=FilterStatus.REQUIRES_CONFIRMATION,
                layer=FilterLayer.CONTEXT,
                original_action=action,
                reason=f"High environmental risk: {'; '.join(risk_factors)}",
                risk_score=risk_score
            )
        
        return FilterResult(
            status=FilterStatus.APPROVED,
            layer=FilterLayer.CONTEXT,
            original_action=action,
            reason="Environmental context acceptable",
            risk_score=max(0.1, risk_score)
        )
    
    def _human_safety_filter(self, action: Dict[str, Any]) -> FilterResult:
        """Human safety filtering (Layer 5 - Final)"""
        # Check for human presence and proximity
        if 'humans_detected' in action and action['humans_detected']:
            humans = action['humans_detected']
            
            for human in humans:
                distance = human.get('distance', float('inf'))
                
                # Minimum safe distance check
                min_distance = human.get('min_safe_distance', 1.0)
                if distance < min_distance:
                    return FilterResult(
                        status=FilterStatus.BLOCKED,
                        layer=FilterLayer.HUMAN_SAFETY,
                        original_action=action,
                        reason=f"Human detected at {distance:.2f}m, below minimum safe distance {min_distance:.2f}m",
                        risk_score=1.0
                    )
                
                # Warning zone check
                warning_distance = min_distance * 1.5
                if distance < warning_distance:
                    return FilterResult(
                        status=FilterStatus.REQUIRES_CONFIRMATION,
                        layer=FilterLayer.HUMAN_SAFETY,
                        original_action=action,
                        reason=f"Human in warning zone at {distance:.2f}m",
                        risk_score=0.8
                    )
        
        # Check for direct human interaction
        if action.get('interaction_type') == 'direct_human':
            # Direct interaction requires extra safety measures
            return FilterResult(
                status=FilterStatus.REQUIRES_CONFIRMATION,
                layer=FilterLayer.HUMAN_SAFETY,
                original_action=action,
                reason="Direct human interaction requires confirmation",
                risk_score=0.6
            )
        
        return FilterResult(
            status=FilterStatus.APPROVED,
            layer=FilterLayer.HUMAN_SAFETY,
            original_action=action,
            reason="Human safety requirements satisfied",
            risk_score=0.1
        )
    
    def _calculate_overall_risk(self, action: Dict[str, Any]) -> float:
        """Calculate overall risk score for an action"""
        base_risk = 0.1
        
        # Add risk based on action parameters
        if 'force' in action:
            force_risk = np.linalg.norm(action['force']) / self.physics_limits.max_force * 0.3
            base_risk += force_risk
        
        if 'velocity' in action:
            speed_risk = np.linalg.norm(action['velocity']) / self.physics_limits.max_speed * 0.2
            base_risk += speed_risk
        
        if 'tool' in action:
            dangerous_tools = ['cutter', 'drill', 'grinder', 'plasma', 'laser']
            if action['tool'].lower() in dangerous_tools:
                base_risk += 0.3
        
        if 'humans_detected' in action and action['humans_detected']:
            base_risk += 0.2
        
        return min(1.0, base_risk)
    
    def update_limits(self, 
                     physics_limits: Optional[PhysicsLimits] = None,
                     spatial_limits: Optional[SpatialLimits] = None):
        """Update safety limits"""
        if physics_limits:
            self.physics_limits = physics_limits
            logger.info("Physics limits updated")
        
        if spatial_limits:
            self.spatial_limits = spatial_limits
            logger.info("Spatial limits updated")
    
    def get_filter_status(self) -> Dict[str, Any]:
        """Get current filter configuration status"""
        return {
            'strict_mode': self.strict_mode,
            'physics_limits': {
                'max_force': self.physics_limits.max_force,
                'max_speed': self.physics_limits.max_speed,
                'max_acceleration': self.physics_limits.max_acceleration,
                'max_jerk': self.physics_limits.max_jerk,
                'max_torque': self.physics_limits.max_torque
            },
            'spatial_limits': {
                'workspace_bounds': self.spatial_limits.workspace_bounds,
                'min_obstacle_distance': self.spatial_limits.min_obstacle_distance,
                'max_reach_distance': self.spatial_limits.max_reach_distance
            },
            'active_layers': [layer.value for layer in FilterLayer]
        }