"""
Human Protection System

Active protection system for human-robot interaction safety:
- Real-time human detection and tracking
- Dynamic safety zone management
- Proximity-based speed and force limiting
- Collision prediction and avoidance
- Emergency human override mechanisms
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
from datetime import datetime, timedelta
import logging
import math

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Human proximity alert levels"""
    SAFE = ("safe", 0)
    CAUTION = ("caution", 1)
    WARNING = ("warning", 2)
    DANGER = ("danger", 3)
    EMERGENCY = ("emergency", 4)


class HumanZone(Enum):
    """Human safety zones"""
    OUTSIDE = "outside"      # Beyond monitoring range
    MONITORING = "monitoring"  # Being monitored
    WARNING = "warning"      # Approaching safety boundary
    SAFETY = "safety"        # Within minimum safe distance
    CRITICAL = "critical"    # Immediate danger zone


@dataclass
class Human:
    """Human detection data"""
    id: str
    position: Tuple[float, float, float]  # (x, y, z)
    velocity: Optional[Tuple[float, float, float]] = None
    size_estimate: float = 0.6  # meters (approximate human width)
    confidence: float = 1.0
    last_seen: datetime = None
    
    def __post_init__(self):
        if self.last_seen is None:
            self.last_seen = datetime.utcnow()
    
    @property
    def distance_from_robot(self) -> float:
        """Calculate distance from robot (assuming robot at origin)"""
        return np.linalg.norm(self.position)
    
    def predicted_position(self, time_delta: float) -> Tuple[float, float, float]:
        """Predict position after time_delta seconds"""
        if self.velocity is None:
            return self.position
        
        return (
            self.position[0] + self.velocity[0] * time_delta,
            self.position[1] + self.velocity[1] * time_delta,
            self.position[2] + self.velocity[2] * time_delta
        )


@dataclass
class ProximityAlert:
    """Human proximity alert"""
    human_id: str
    alert_level: AlertLevel
    distance: float
    predicted_distance: Optional[float]
    zone: HumanZone
    recommended_action: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class SafetyZoneConfig:
    """Configuration for human safety zones"""
    critical_distance: float = 0.3    # meters - immediate stop required
    safety_distance: float = 1.0      # meters - minimum safe distance
    warning_distance: float = 1.5     # meters - warning zone
    monitoring_distance: float = 3.0   # meters - monitoring range
    
    # Speed limiting factors for each zone
    critical_speed_factor: float = 0.0    # Complete stop
    safety_speed_factor: float = 0.1      # 10% normal speed
    warning_speed_factor: float = 0.3     # 30% normal speed
    monitoring_speed_factor: float = 0.7   # 70% normal speed
    
    # Force limiting factors for each zone
    critical_force_factor: float = 0.0    # No force allowed
    safety_force_factor: float = 0.1      # 10% normal force
    warning_force_factor: float = 0.3     # 30% normal force
    monitoring_force_factor: float = 0.7   # 70% normal force


class HumanProtection:
    """Active human protection system"""
    
    def __init__(self, 
                 safety_config: Optional[SafetyZoneConfig] = None,
                 prediction_horizon: float = 2.0,  # seconds
                 human_timeout: float = 5.0):      # seconds
        """
        Initialize human protection system
        
        Args:
            safety_config: Safety zone configuration
            prediction_horizon: Time ahead to predict human positions
            human_timeout: Time before removing undetected humans
        """
        self.safety_config = safety_config or SafetyZoneConfig()
        self.prediction_horizon = prediction_horizon
        self.human_timeout = human_timeout
        
        self.detected_humans: Dict[str, Human] = {}
        self.alert_history: List[ProximityAlert] = []
        self.emergency_stops_triggered = 0
        self.protection_callbacks: Dict[str, Callable] = {}
        
        # Runtime statistics
        self.stats = {
            'total_detections': 0,
            'safety_violations': 0,
            'emergency_stops': 0,
            'protection_activations': 0
        }
        
        logger.info(f"HumanProtection initialized with {self.safety_config.safety_distance}m safety distance")
    
    def update_human_detections(self, detections: List[Dict[str, Any]]) -> List[ProximityAlert]:
        """
        Update human detections and generate proximity alerts
        
        Args:
            detections: List of human detection data
            
        Returns:
            List of proximity alerts
        """
        current_time = datetime.utcnow()
        alerts = []
        
        # Process new detections
        for detection in detections:
            human_id = detection.get('id', f"human_{len(self.detected_humans)}")
            
            human = Human(
                id=human_id,
                position=tuple(detection['position']),
                velocity=tuple(detection.get('velocity', [0, 0, 0])),
                size_estimate=detection.get('size_estimate', 0.6),
                confidence=detection.get('confidence', 1.0),
                last_seen=current_time
            )
            
            self.detected_humans[human_id] = human
            self.stats['total_detections'] += 1
            
            # Generate proximity alert
            alert = self._assess_human_proximity(human)
            if alert:
                alerts.append(alert)
                self.alert_history.append(alert)
        
        # Remove timed-out humans
        self._cleanup_old_detections(current_time)
        
        # Limit alert history size
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-50:]
        
        return alerts
    
    def get_safety_constraints(self, robot_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get safety constraints based on current human positions
        
        Args:
            robot_action: Planned robot action
            
        Returns:
            Dictionary of safety constraints
        """
        if not self.detected_humans:
            return {
                'speed_limit_factor': 1.0,
                'force_limit_factor': 1.0,
                'emergency_stop_required': False,
                'recommended_action': 'proceed',
                'active_constraints': []
            }
        
        # Find most restrictive constraints
        min_speed_factor = 1.0
        min_force_factor = 1.0
        emergency_stop = False
        active_constraints = []
        most_critical_zone = HumanZone.OUTSIDE
        
        for human in self.detected_humans.values():
            zone = self._determine_human_zone(human)
            constraints = self._get_zone_constraints(zone)
            
            if constraints['speed_factor'] < min_speed_factor:
                min_speed_factor = constraints['speed_factor']
            
            if constraints['force_factor'] < min_force_factor:
                min_force_factor = constraints['force_factor']
            
            if constraints['emergency_stop']:
                emergency_stop = True
            
            active_constraints.append({
                'human_id': human.id,
                'zone': zone.value,
                'distance': human.distance_from_robot,
                'constraints': constraints
            })
            
            # Track most critical zone
            if zone.value == 'critical' or (zone.value == 'safety' and most_critical_zone.value != 'critical'):
                most_critical_zone = zone
        
        # Determine recommended action
        if emergency_stop:
            recommended_action = 'emergency_stop'
        elif min_speed_factor < 0.3:
            recommended_action = 'slow_motion'
        elif min_speed_factor < 0.7:
            recommended_action = 'reduced_speed'
        else:
            recommended_action = 'proceed_with_caution'
        
        return {
            'speed_limit_factor': min_speed_factor,
            'force_limit_factor': min_force_factor,
            'emergency_stop_required': emergency_stop,
            'recommended_action': recommended_action,
            'active_constraints': active_constraints,
            'most_critical_zone': most_critical_zone.value,
            'human_count': len(self.detected_humans)
        }
    
    def predict_collision_risk(self, robot_trajectory: List[Tuple[float, float, float]], 
                              time_steps: List[float]) -> Dict[str, Any]:
        """
        Predict collision risk along robot trajectory
        
        Args:
            robot_trajectory: List of robot positions over time
            time_steps: Corresponding time steps
            
        Returns:
            Collision risk assessment
        """
        if not self.detected_humans or not robot_trajectory:
            return {
                'collision_risk': 0.0,
                'risk_points': [],
                'recommendations': ['No humans detected or no trajectory provided']
            }
        
        risk_points = []
        max_risk = 0.0
        
        for i, (robot_pos, time_step) in enumerate(zip(robot_trajectory, time_steps)):
            time_from_now = time_step
            
            for human in self.detected_humans.values():
                # Predict human position at this time
                predicted_human_pos = human.predicted_position(time_from_now)
                
                # Calculate distance between predicted positions
                distance = np.linalg.norm(np.array(robot_pos) - np.array(predicted_human_pos))
                
                # Add safety margin based on uncertainties
                uncertainty_margin = 0.2 + (time_from_now * 0.1)  # Increases with prediction time
                effective_distance = distance - uncertainty_margin - human.size_estimate
                
                # Calculate risk (inverse of distance with safety considerations)
                if effective_distance <= 0:
                    risk = 1.0  # Certain collision
                elif effective_distance < self.safety_config.critical_distance:
                    risk = 0.9
                elif effective_distance < self.safety_config.safety_distance:
                    risk = 0.7 * (self.safety_config.safety_distance - effective_distance) / self.safety_config.safety_distance
                elif effective_distance < self.safety_config.warning_distance:
                    risk = 0.4 * (self.safety_config.warning_distance - effective_distance) / self.safety_config.warning_distance
                else:
                    risk = 0.0
                
                if risk > 0:
                    risk_points.append({
                        'trajectory_index': i,
                        'time_step': time_step,
                        'robot_position': robot_pos,
                        'human_id': human.id,
                        'human_position': predicted_human_pos,
                        'distance': distance,
                        'effective_distance': effective_distance,
                        'risk': risk
                    })
                
                max_risk = max(max_risk, risk)
        
        # Generate recommendations
        recommendations = []
        if max_risk >= 0.9:
            recommendations.append("CRITICAL: Trajectory leads to likely collision - abort motion")
        elif max_risk >= 0.7:
            recommendations.append("HIGH RISK: Modify trajectory to increase clearance")
        elif max_risk >= 0.4:
            recommendations.append("MODERATE RISK: Reduce speed and monitor closely")
        elif max_risk > 0:
            recommendations.append("LOW RISK: Proceed with enhanced monitoring")
        else:
            recommendations.append("No significant collision risk detected")
        
        return {
            'collision_risk': max_risk,
            'risk_points': sorted(risk_points, key=lambda x: x['risk'], reverse=True)[:10],  # Top 10 highest risks
            'recommendations': recommendations,
            'total_risk_points': len(risk_points)
        }
    
    def check_emergency_conditions(self) -> Dict[str, Any]:
        """
        Check for emergency conditions requiring immediate action
        
        Returns:
            Emergency status and required actions
        """
        emergency_conditions = []
        immediate_actions = []
        risk_level = 0
        
        for human in self.detected_humans.values():
            distance = human.distance_from_robot
            
            # Critical distance breach
            if distance < self.safety_config.critical_distance:
                emergency_conditions.append(f"Human {human.id} in critical zone at {distance:.2f}m")
                immediate_actions.append("EMERGENCY STOP")
                risk_level = max(risk_level, 4)
            
            # High-speed approach detection
            elif human.velocity is not None:
                approach_speed = -np.dot(
                    np.array(human.velocity),
                    np.array(human.position) / np.linalg.norm(human.position)
                )
                
                if approach_speed > 1.0 and distance < self.safety_config.warning_distance:
                    emergency_conditions.append(f"Human {human.id} approaching rapidly at {approach_speed:.2f}m/s")
                    immediate_actions.append("REDUCE SPEED AND MONITOR")
                    risk_level = max(risk_level, 3)
            
            # Loss of tracking
            time_since_seen = (datetime.utcnow() - human.last_seen).total_seconds()
            if time_since_seen > 1.0 and distance < self.safety_config.safety_distance:
                emergency_conditions.append(f"Lost tracking of human {human.id} in safety zone")
                immediate_actions.append("CAUTIOUS OPERATION")
                risk_level = max(risk_level, 2)
        
        return {
            'emergency_required': risk_level >= 4,
            'risk_level': risk_level,
            'conditions': emergency_conditions,
            'immediate_actions': list(set(immediate_actions)),  # Remove duplicates
            'human_count_in_danger': sum(1 for h in self.detected_humans.values() 
                                       if h.distance_from_robot < self.safety_config.safety_distance)
        }
    
    def register_protection_callback(self, name: str, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for protection events"""
        self.protection_callbacks[name] = callback
        logger.info(f"Registered protection callback: {name}")
    
    def _assess_human_proximity(self, human: Human) -> Optional[ProximityAlert]:
        """Assess proximity of a human and generate alert if needed"""
        distance = human.distance_from_robot
        zone = self._determine_human_zone(human)
        
        # Predict future position
        predicted_distance = None
        if human.velocity is not None:
            future_pos = human.predicted_position(self.prediction_horizon)
            predicted_distance = np.linalg.norm(future_pos)
        
        # Determine alert level
        if zone == HumanZone.CRITICAL:
            alert_level = AlertLevel.EMERGENCY
            recommended_action = "IMMEDIATE EMERGENCY STOP"
        elif zone == HumanZone.SAFETY:
            alert_level = AlertLevel.DANGER
            recommended_action = "Stop motion and withdraw to safe distance"
        elif zone == HumanZone.WARNING:
            alert_level = AlertLevel.WARNING
            recommended_action = "Reduce speed and force, monitor closely"
        elif zone == HumanZone.MONITORING:
            alert_level = AlertLevel.CAUTION
            recommended_action = "Continue monitoring, be prepared to slow"
        else:
            return None  # No alert needed
        
        # Check if this warrants triggering callbacks
        if alert_level.value[1] >= 3:  # Danger or Emergency
            self._trigger_protection_callbacks({
                'alert_level': alert_level.value[0],
                'human_id': human.id,
                'distance': distance,
                'zone': zone.value,
                'recommended_action': recommended_action
            })
        
        return ProximityAlert(
            human_id=human.id,
            alert_level=alert_level,
            distance=distance,
            predicted_distance=predicted_distance,
            zone=zone,
            recommended_action=recommended_action
        )
    
    def _determine_human_zone(self, human: Human) -> HumanZone:
        """Determine which safety zone the human is in"""
        distance = human.distance_from_robot
        
        if distance < self.safety_config.critical_distance:
            return HumanZone.CRITICAL
        elif distance < self.safety_config.safety_distance:
            return HumanZone.SAFETY
        elif distance < self.safety_config.warning_distance:
            return HumanZone.WARNING
        elif distance < self.safety_config.monitoring_distance:
            return HumanZone.MONITORING
        else:
            return HumanZone.OUTSIDE
    
    def _get_zone_constraints(self, zone: HumanZone) -> Dict[str, Any]:
        """Get safety constraints for a given zone"""
        config = self.safety_config
        
        if zone == HumanZone.CRITICAL:
            return {
                'speed_factor': config.critical_speed_factor,
                'force_factor': config.critical_force_factor,
                'emergency_stop': True,
                'description': 'Human in critical danger zone'
            }
        elif zone == HumanZone.SAFETY:
            return {
                'speed_factor': config.safety_speed_factor,
                'force_factor': config.safety_force_factor,
                'emergency_stop': False,
                'description': 'Human in minimum safe distance zone'
            }
        elif zone == HumanZone.WARNING:
            return {
                'speed_factor': config.warning_speed_factor,
                'force_factor': config.warning_force_factor,
                'emergency_stop': False,
                'description': 'Human in warning zone'
            }
        elif zone == HumanZone.MONITORING:
            return {
                'speed_factor': config.monitoring_speed_factor,
                'force_factor': config.monitoring_force_factor,
                'emergency_stop': False,
                'description': 'Human in monitoring zone'
            }
        else:
            return {
                'speed_factor': 1.0,
                'force_factor': 1.0,
                'emergency_stop': False,
                'description': 'No humans in safety zones'
            }
    
    def _cleanup_old_detections(self, current_time: datetime):
        """Remove old human detections that have timed out"""
        timeout_threshold = current_time - timedelta(seconds=self.human_timeout)
        
        humans_to_remove = [
            human_id for human_id, human in self.detected_humans.items()
            if human.last_seen < timeout_threshold
        ]
        
        for human_id in humans_to_remove:
            logger.info(f"Removing timed-out human detection: {human_id}")
            del self.detected_humans[human_id]
    
    def _trigger_protection_callbacks(self, event_data: Dict[str, Any]):
        """Trigger registered protection callbacks"""
        self.stats['protection_activations'] += 1
        
        for name, callback in self.protection_callbacks.items():
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in protection callback {name}: {str(e)}")
    
    def get_human_status(self) -> Dict[str, Any]:
        """Get current status of all detected humans"""
        humans_status = []
        
        for human in self.detected_humans.values():
            zone = self._determine_human_zone(human)
            constraints = self._get_zone_constraints(zone)
            
            humans_status.append({
                'id': human.id,
                'position': human.position,
                'distance': human.distance_from_robot,
                'velocity': human.velocity,
                'zone': zone.value,
                'last_seen': human.last_seen.isoformat(),
                'confidence': human.confidence,
                'constraints': constraints
            })
        
        return {
            'humans_detected': len(self.detected_humans),
            'humans': humans_status,
            'active_alerts': len([a for a in self.alert_history[-10:] 
                                 if (datetime.utcnow() - a.timestamp).total_seconds() < 30]),
            'emergency_conditions': self.check_emergency_conditions()
        }
    
    def get_protection_statistics(self) -> Dict[str, Any]:
        """Get protection system statistics"""
        current_time = datetime.utcnow()
        recent_alerts = [
            a for a in self.alert_history 
            if (current_time - a.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        return {
            'runtime_stats': self.stats.copy(),
            'recent_alerts_count': len(recent_alerts),
            'alert_distribution': {
                level.value[0]: sum(1 for a in recent_alerts if a.alert_level == level)
                for level in AlertLevel
            },
            'active_callbacks': list(self.protection_callbacks.keys()),
            'configuration': {
                'safety_distance': self.safety_config.safety_distance,
                'warning_distance': self.safety_config.warning_distance,
                'monitoring_distance': self.safety_config.monitoring_distance,
                'prediction_horizon': self.prediction_horizon,
                'human_timeout': self.human_timeout
            }
        }
    
    def update_safety_config(self, new_config: SafetyZoneConfig):
        """Update safety zone configuration"""
        old_config = self.safety_config
        self.safety_config = new_config
        
        logger.info(f"Safety configuration updated: "
                   f"safety_distance {old_config.safety_distance}m -> {new_config.safety_distance}m, "
                   f"warning_distance {old_config.warning_distance}m -> {new_config.warning_distance}m")
    
    def force_emergency_mode(self, reason: str = "Manual activation"):
        """Force the system into emergency protection mode"""
        self.stats['emergency_stops'] += 1
        
        emergency_event = {
            'alert_level': 'emergency',
            'human_id': 'system',
            'distance': 0.0,
            'zone': 'critical',
            'recommended_action': 'EMERGENCY STOP - ' + reason
        }
        
        self._trigger_protection_callbacks(emergency_event)
        
        logger.warning(f"Emergency protection mode activated: {reason}")
    
    def reset_protection_system(self):
        """Reset the protection system (clear all detections and history)"""
        self.detected_humans.clear()
        self.alert_history.clear()
        self.stats = {
            'total_detections': 0,
            'safety_violations': 0,
            'emergency_stops': 0,
            'protection_activations': 0
        }
        
        logger.info("Human protection system reset")