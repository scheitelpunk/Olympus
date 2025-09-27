"""
Comprehensive Risk Assessment System

Provides multi-dimensional risk analysis for robotic actions:
- Real-time risk calculation and monitoring
- Environmental hazard assessment
- Cumulative risk tracking
- Predictive risk modeling
- Risk mitigation strategies
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
from datetime import datetime, timedelta
import logging
import math

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels with numerical values"""
    MINIMAL = ("minimal", 0.0, 0.2)
    LOW = ("low", 0.2, 0.4)
    MODERATE = ("moderate", 0.4, 0.6)
    HIGH = ("high", 0.6, 0.8)
    CRITICAL = ("critical", 0.8, 1.0)
    
    def __init__(self, label: str, min_score: float, max_score: float):
        self.label = label
        self.min_score = min_score
        self.max_score = max_score
    
    @classmethod
    def from_score(cls, score: float) -> 'RiskLevel':
        """Get risk level from numerical score"""
        for level in cls:
            if level.min_score <= score < level.max_score:
                return level
        return cls.CRITICAL  # For scores >= 1.0


class RiskDimension(Enum):
    """Different dimensions of risk assessment"""
    PHYSICAL = "physical"
    ENVIRONMENTAL = "environmental"
    OPERATIONAL = "operational"
    HUMAN_SAFETY = "human_safety"
    EQUIPMENT = "equipment"
    MISSION = "mission"


@dataclass
class RiskFactor:
    """Individual risk factor"""
    name: str
    category: RiskDimension
    severity: float  # 0.0 to 1.0
    probability: float  # 0.0 to 1.0
    description: str
    mitigation_strategies: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def risk_score(self) -> float:
        """Calculate risk score as severity × probability"""
        return self.severity * self.probability


@dataclass
class RiskAssessmentResult:
    """Complete risk assessment result"""
    overall_risk_score: float
    risk_level: RiskLevel
    risk_factors: List[RiskFactor]
    cumulative_risk: float
    predicted_risk: float
    recommendations: List[str]
    assessment_confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'overall_risk_score': self.overall_risk_score,
            'risk_level': self.risk_level.label,
            'risk_factors': [
                {
                    'name': rf.name,
                    'category': rf.category.value,
                    'severity': rf.severity,
                    'probability': rf.probability,
                    'risk_score': rf.risk_score,
                    'description': rf.description,
                    'mitigation_strategies': rf.mitigation_strategies
                }
                for rf in self.risk_factors
            ],
            'cumulative_risk': self.cumulative_risk,
            'predicted_risk': self.predicted_risk,
            'recommendations': self.recommendations,
            'assessment_confidence': self.assessment_confidence,
            'timestamp': self.timestamp.isoformat()
        }


class RiskAssessment:
    """Comprehensive risk assessment system"""
    
    def __init__(self, 
                 history_window: int = 50,
                 risk_decay_factor: float = 0.95,
                 prediction_horizon: int = 10):
        """
        Initialize risk assessment system
        
        Args:
            history_window: Number of past assessments to consider
            risk_decay_factor: Factor for cumulative risk decay over time
            prediction_horizon: Number of future steps to predict risk for
        """
        self.history_window = history_window
        self.risk_decay_factor = risk_decay_factor
        self.prediction_horizon = prediction_horizon
        
        self.assessment_history: List[RiskAssessmentResult] = []
        self.environmental_factors: Dict[str, Any] = {}
        self.system_status: Dict[str, Any] = {}
        self.human_factors: Dict[str, Any] = {}
        
        # Risk calculation functions
        self.risk_calculators = {
            RiskDimension.PHYSICAL: self._assess_physical_risk,
            RiskDimension.ENVIRONMENTAL: self._assess_environmental_risk,
            RiskDimension.OPERATIONAL: self._assess_operational_risk,
            RiskDimension.HUMAN_SAFETY: self._assess_human_safety_risk,
            RiskDimension.EQUIPMENT: self._assess_equipment_risk,
            RiskDimension.MISSION: self._assess_mission_risk
        }
        
        logger.info(f"RiskAssessment initialized with history_window={history_window}")
    
    def assess_risk(self, 
                   action: Dict[str, Any],
                   environment: Optional[Dict[str, Any]] = None,
                   system_state: Optional[Dict[str, Any]] = None,
                   human_presence: Optional[Dict[str, Any]] = None) -> RiskAssessmentResult:
        """
        Perform comprehensive risk assessment
        
        Args:
            action: Action to assess
            environment: Environmental conditions
            system_state: Current system status
            human_presence: Human presence information
            
        Returns:
            RiskAssessmentResult with complete assessment
        """
        # Update context information
        if environment:
            self.environmental_factors.update(environment)
        if system_state:
            self.system_status.update(system_state)
        if human_presence:
            self.human_factors.update(human_presence)
        
        # Assess risk in each dimension
        all_risk_factors = []
        dimension_scores = {}
        
        for dimension in RiskDimension:
            try:
                factors = self.risk_calculators[dimension](action)
                all_risk_factors.extend(factors)
                
                # Calculate weighted score for this dimension
                if factors:
                    dimension_scores[dimension] = np.mean([f.risk_score for f in factors])
                else:
                    dimension_scores[dimension] = 0.0
                    
            except Exception as e:
                logger.error(f"Error assessing {dimension.value} risk: {str(e)}")
                dimension_scores[dimension] = 0.5  # Default to moderate risk on error
        
        # Calculate overall risk score
        # Weight different dimensions based on importance
        dimension_weights = {
            RiskDimension.HUMAN_SAFETY: 0.3,
            RiskDimension.PHYSICAL: 0.25,
            RiskDimension.EQUIPMENT: 0.2,
            RiskDimension.ENVIRONMENTAL: 0.15,
            RiskDimension.OPERATIONAL: 0.1
        }
        
        overall_risk = sum(
            dimension_scores[dim] * dimension_weights.get(dim, 0.1)
            for dim in dimension_scores
        )
        
        # Calculate cumulative risk
        cumulative_risk = self._calculate_cumulative_risk()
        
        # Predict future risk
        predicted_risk = self._predict_risk_trend(action)
        
        # Adjust overall risk based on cumulative and predicted risks
        adjusted_risk = self._combine_risk_components(overall_risk, cumulative_risk, predicted_risk)
        
        # Determine risk level
        risk_level = RiskLevel.from_score(adjusted_risk)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_risk_factors, risk_level)
        
        # Calculate assessment confidence
        confidence = self._calculate_confidence(all_risk_factors, len(self.assessment_history))
        
        # Create assessment result
        result = RiskAssessmentResult(
            overall_risk_score=adjusted_risk,
            risk_level=risk_level,
            risk_factors=all_risk_factors,
            cumulative_risk=cumulative_risk,
            predicted_risk=predicted_risk,
            recommendations=recommendations,
            assessment_confidence=confidence
        )
        
        # Add to history
        self.assessment_history.append(result)
        if len(self.assessment_history) > self.history_window:
            self.assessment_history.pop(0)
        
        logger.info(f"Risk assessment complete: {risk_level.label} "
                   f"(score: {adjusted_risk:.3f}, confidence: {confidence:.2f})")
        
        return result
    
    def _assess_physical_risk(self, action: Dict[str, Any]) -> List[RiskFactor]:
        """Assess physical risks from forces, speeds, and accelerations"""
        factors = []
        
        # Force-related risks
        if 'force' in action:
            force_magnitude = np.linalg.norm(action['force'])
            max_safe_force = 20.0  # Newtons
            
            if force_magnitude > 0:
                severity = min(1.0, force_magnitude / max_safe_force)
                probability = 0.8 if severity > 0.5 else 0.3
                
                factors.append(RiskFactor(
                    name="force_application",
                    category=RiskDimension.PHYSICAL,
                    severity=severity,
                    probability=probability,
                    description=f"Applied force of {force_magnitude:.1f}N (max safe: {max_safe_force}N)",
                    mitigation_strategies=[
                        "Reduce applied force",
                        "Use force feedback control",
                        "Implement force ramping"
                    ] if severity > 0.6 else ["Monitor force levels"]
                ))
        
        # Speed-related risks
        if 'velocity' in action:
            speed = np.linalg.norm(action['velocity'])
            max_safe_speed = 1.0  # m/s
            
            if speed > 0:
                severity = min(1.0, speed / max_safe_speed)
                probability = 0.7 if severity > 0.5 else 0.2
                
                factors.append(RiskFactor(
                    name="high_speed_motion",
                    category=RiskDimension.PHYSICAL,
                    severity=severity,
                    probability=probability,
                    description=f"Motion speed of {speed:.2f}m/s (max safe: {max_safe_speed}m/s)",
                    mitigation_strategies=[
                        "Reduce motion speed",
                        "Implement speed limiting",
                        "Add deceleration zones"
                    ] if severity > 0.6 else ["Monitor speed levels"]
                ))
        
        # Acceleration-related risks
        if 'acceleration' in action:
            accel = np.linalg.norm(action['acceleration'])
            max_safe_accel = 2.0  # m/s²
            
            if accel > 0:
                severity = min(1.0, accel / max_safe_accel)
                probability = 0.6 if severity > 0.5 else 0.2
                
                factors.append(RiskFactor(
                    name="high_acceleration",
                    category=RiskDimension.PHYSICAL,
                    severity=severity,
                    probability=probability,
                    description=f"Acceleration of {accel:.2f}m/s² (max safe: {max_safe_accel}m/s²)",
                    mitigation_strategies=[
                        "Reduce acceleration limits",
                        "Implement smooth motion profiles",
                        "Add acceleration monitoring"
                    ] if severity > 0.6 else ["Monitor acceleration levels"]
                ))
        
        return factors
    
    def _assess_environmental_risk(self, action: Dict[str, Any]) -> List[RiskFactor]:
        """Assess environmental risks"""
        factors = []
        env = self.environmental_factors
        
        # Temperature risks
        temp = env.get('temperature', 20)
        if temp < 5 or temp > 40:
            severity = min(1.0, abs(temp - 22.5) / 17.5)
            factors.append(RiskFactor(
                name="extreme_temperature",
                category=RiskDimension.ENVIRONMENTAL,
                severity=severity,
                probability=0.8,
                description=f"Temperature {temp}°C outside safe range (5-40°C)",
                mitigation_strategies=["Wait for suitable temperature", "Use thermal protection"]
            ))
        
        # Lighting risks
        lighting = env.get('lighting_level', 100)
        if lighting < 50:
            severity = (50 - lighting) / 50
            factors.append(RiskFactor(
                name="poor_lighting",
                category=RiskDimension.ENVIRONMENTAL,
                severity=severity,
                probability=0.9,
                description=f"Poor lighting level {lighting}% (minimum 50%)",
                mitigation_strategies=["Improve lighting", "Use enhanced sensors", "Reduce operation speed"]
            ))
        
        # Vibration risks
        vibration = env.get('vibration_level', 0)
        if vibration > 2:
            severity = min(1.0, vibration / 10)
            factors.append(RiskFactor(
                name="high_vibration",
                category=RiskDimension.ENVIRONMENTAL,
                severity=severity,
                probability=0.7,
                description=f"High vibration level {vibration} (max safe: 2)",
                mitigation_strategies=["Reduce vibration sources", "Use vibration dampening", "Adjust operation parameters"]
            ))
        
        # Obstacle proximity
        if 'obstacles' in action:
            min_distance = float('inf')
            for obstacle in action['obstacles']:
                dist = obstacle.get('distance', 1.0)
                min_distance = min(min_distance, dist)
            
            if min_distance < 0.5:
                severity = max(0, (0.5 - min_distance) / 0.5)
                factors.append(RiskFactor(
                    name="obstacle_proximity",
                    category=RiskDimension.ENVIRONMENTAL,
                    severity=severity,
                    probability=0.8,
                    description=f"Obstacle detected at {min_distance:.2f}m (safe distance: 0.5m)",
                    mitigation_strategies=["Increase clearance", "Use obstacle avoidance", "Reduce speed"]
                ))
        
        return factors
    
    def _assess_operational_risk(self, action: Dict[str, Any]) -> List[RiskFactor]:
        """Assess operational risks"""
        factors = []
        
        # Tool-related risks
        if 'tool' in action:
            tool = action['tool'].lower()
            dangerous_tools = {
                'cutter': 0.8,
                'drill': 0.7,
                'grinder': 0.9,
                'laser': 0.9,
                'plasma': 0.95,
                'welder': 0.85
            }
            
            if tool in dangerous_tools:
                severity = dangerous_tools[tool]
                factors.append(RiskFactor(
                    name="dangerous_tool_usage",
                    category=RiskDimension.OPERATIONAL,
                    severity=severity,
                    probability=0.6,
                    description=f"Using high-risk tool: {tool}",
                    mitigation_strategies=[
                        "Verify tool safety protocols",
                        "Ensure proper tool guards",
                        "Require safety confirmation",
                        "Monitor tool operation closely"
                    ]
                ))
        
        # Repetitive operation risks
        if 'repetitions' in action:
            reps = action['repetitions']
            if reps > 50:
                severity = min(1.0, (reps - 50) / 200)  # Risk increases with repetitions
                factors.append(RiskFactor(
                    name="repetitive_operation",
                    category=RiskDimension.OPERATIONAL,
                    severity=severity,
                    probability=0.4,
                    description=f"High repetition count: {reps} cycles",
                    mitigation_strategies=[
                        "Add periodic safety checks",
                        "Monitor for system drift",
                        "Implement fatigue detection"
                    ]
                ))
        
        # Complex trajectory risks
        if 'trajectory' in action and isinstance(action['trajectory'], list):
            traj_length = len(action['trajectory'])
            if traj_length > 100:
                severity = min(1.0, (traj_length - 100) / 400)
                factors.append(RiskFactor(
                    name="complex_trajectory",
                    category=RiskDimension.OPERATIONAL,
                    severity=severity,
                    probability=0.3,
                    description=f"Complex trajectory with {traj_length} waypoints",
                    mitigation_strategies=[
                        "Simplify trajectory",
                        "Add intermediate safety checks",
                        "Reduce execution speed"
                    ]
                ))
        
        return factors
    
    def _assess_human_safety_risk(self, action: Dict[str, Any]) -> List[RiskFactor]:
        """Assess risks to human safety"""
        factors = []
        human_info = self.human_factors
        
        # Human proximity risks
        if 'humans_detected' in action and action['humans_detected']:
            for human in action['humans_detected']:
                distance = human.get('distance', float('inf'))
                min_safe_distance = human.get('min_safe_distance', 1.0)
                
                if distance < min_safe_distance * 2:  # Within 2x safe distance
                    severity = max(0, (min_safe_distance * 2 - distance) / (min_safe_distance * 2))
                    probability = 0.9 if distance < min_safe_distance else 0.5
                    
                    factors.append(RiskFactor(
                        name="human_proximity",
                        category=RiskDimension.HUMAN_SAFETY,
                        severity=severity,
                        probability=probability,
                        description=f"Human at {distance:.2f}m (safe distance: {min_safe_distance:.2f}m)",
                        mitigation_strategies=[
                            "Increase safe distance",
                            "Reduce operation speed",
                            "Implement proximity alerts",
                            "Require human acknowledgment"
                        ]
                    ))
        
        # Human interaction risks
        if action.get('interaction_type') == 'direct_human':
            factors.append(RiskFactor(
                name="direct_human_interaction",
                category=RiskDimension.HUMAN_SAFETY,
                severity=0.7,
                probability=0.8,
                description="Direct human-robot interaction planned",
                mitigation_strategies=[
                    "Use collaborative robotics protocols",
                    "Implement force limiting",
                    "Continuous human monitoring",
                    "Emergency stop readily available"
                ]
            ))
        
        # Operator fatigue risks
        if 'operator_fatigue' in human_info:
            fatigue_level = human_info['operator_fatigue']
            if fatigue_level > 6:  # Scale 1-10
                severity = min(1.0, (fatigue_level - 6) / 4)
                factors.append(RiskFactor(
                    name="operator_fatigue",
                    category=RiskDimension.HUMAN_SAFETY,
                    severity=severity,
                    probability=0.6,
                    description=f"High operator fatigue level: {fatigue_level}/10",
                    mitigation_strategies=[
                        "Recommend operator break",
                        "Increase automation level",
                        "Additional safety monitoring"
                    ]
                ))
        
        return factors
    
    def _assess_equipment_risk(self, action: Dict[str, Any]) -> List[RiskFactor]:
        """Assess equipment and system risks"""
        factors = []
        system = self.system_status
        
        # Battery level risks
        battery_level = system.get('battery_level', 100)
        if battery_level < 20:
            severity = (20 - battery_level) / 20
            factors.append(RiskFactor(
                name="low_battery",
                category=RiskDimension.EQUIPMENT,
                severity=severity,
                probability=0.9,
                description=f"Low battery level: {battery_level}%",
                mitigation_strategies=["Recharge battery", "Reduce power consumption", "Plan charging schedule"]
            ))
        
        # System error risks
        error_count = system.get('error_count', 0)
        if error_count > 0:
            severity = min(1.0, error_count / 10)
            factors.append(RiskFactor(
                name="system_errors",
                category=RiskDimension.EQUIPMENT,
                severity=severity,
                probability=0.7,
                description=f"System errors detected: {error_count}",
                mitigation_strategies=["Investigate errors", "System diagnostics", "Reset if necessary"]
            ))
        
        # Maintenance overdue risks
        if system.get('maintenance_overdue', False):
            factors.append(RiskFactor(
                name="maintenance_overdue",
                category=RiskDimension.EQUIPMENT,
                severity=0.6,
                probability=0.8,
                description="System maintenance is overdue",
                mitigation_strategies=["Schedule maintenance", "Perform safety inspection", "Reduce operation intensity"]
            ))
        
        # Sensor status risks
        if not system.get('all_sensors_operational', True):
            factors.append(RiskFactor(
                name="sensor_malfunction",
                category=RiskDimension.EQUIPMENT,
                severity=0.8,
                probability=0.9,
                description="One or more sensors not operational",
                mitigation_strategies=["Repair sensors", "Use backup sensors", "Reduce operation scope"]
            ))
        
        return factors
    
    def _assess_mission_risk(self, action: Dict[str, Any]) -> List[RiskFactor]:
        """Assess mission and task completion risks"""
        factors = []
        
        # Time pressure risks
        if 'deadline' in action:
            deadline = action['deadline']
            current_time = datetime.utcnow()
            if isinstance(deadline, str):
                deadline = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
            
            time_remaining = (deadline - current_time).total_seconds()
            if time_remaining < 3600:  # Less than 1 hour
                severity = max(0, (3600 - time_remaining) / 3600)
                factors.append(RiskFactor(
                    name="time_pressure",
                    category=RiskDimension.MISSION,
                    severity=severity,
                    probability=0.5,
                    description=f"Time pressure: {time_remaining/60:.1f} minutes remaining",
                    mitigation_strategies=["Prioritize critical tasks", "Request deadline extension", "Increase automation"]
                ))
        
        # Task complexity risks
        if 'complexity_score' in action:
            complexity = action['complexity_score']
            if complexity > 7:  # Scale 1-10
                severity = (complexity - 7) / 3
                factors.append(RiskFactor(
                    name="high_task_complexity",
                    category=RiskDimension.MISSION,
                    severity=severity,
                    probability=0.4,
                    description=f"High task complexity: {complexity}/10",
                    mitigation_strategies=["Break down task", "Add intermediate checkpoints", "Increase monitoring"]
                ))
        
        return factors
    
    def _calculate_cumulative_risk(self) -> float:
        """Calculate cumulative risk from recent history"""
        if not self.assessment_history:
            return 0.0
        
        # Weight recent assessments more heavily
        total_weighted_risk = 0.0
        total_weight = 0.0
        current_time = datetime.utcnow()
        
        for i, assessment in enumerate(reversed(self.assessment_history)):
            # Time-based decay
            time_diff = (current_time - assessment.timestamp).total_seconds() / 3600  # hours
            time_weight = self.risk_decay_factor ** time_diff
            
            # Recent assessments have higher weight
            recency_weight = (i + 1) / len(self.assessment_history)
            
            combined_weight = time_weight * recency_weight
            total_weighted_risk += assessment.overall_risk_score * combined_weight
            total_weight += combined_weight
        
        return total_weighted_risk / total_weight if total_weight > 0 else 0.0
    
    def _predict_risk_trend(self, action: Dict[str, Any]) -> float:
        """Predict future risk based on current trends"""
        if len(self.assessment_history) < 3:
            return 0.3  # Default moderate prediction uncertainty
        
        # Analyze trend in recent risk scores
        recent_scores = [a.overall_risk_score for a in self.assessment_history[-5:]]
        
        if len(recent_scores) >= 2:
            # Calculate trend (simple linear regression slope)
            x = np.arange(len(recent_scores))
            y = np.array(recent_scores)
            trend = np.polyfit(x, y, 1)[0]  # Slope
            
            # Project trend forward
            predicted_change = trend * self.prediction_horizon
            current_risk = recent_scores[-1]
            predicted_risk = max(0.0, min(1.0, current_risk + predicted_change))
            
            return predicted_risk
        
        return recent_scores[-1] if recent_scores else 0.3
    
    def _combine_risk_components(self, current_risk: float, cumulative_risk: float, predicted_risk: float) -> float:
        """Combine different risk components into final score"""
        # Weighted combination
        weights = {
            'current': 0.6,
            'cumulative': 0.3,
            'predicted': 0.1
        }
        
        combined = (current_risk * weights['current'] +
                   cumulative_risk * weights['cumulative'] +
                   predicted_risk * weights['predicted'])
        
        return min(1.0, combined)
    
    def _generate_recommendations(self, risk_factors: List[RiskFactor], risk_level: RiskLevel) -> List[str]:
        """Generate safety recommendations based on risk assessment"""
        recommendations = []
        
        # Level-based recommendations
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: Stop current operation",
                "Activate emergency protocols",
                "Require human supervision",
                "Perform comprehensive safety check"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Increase safety monitoring frequency",
                "Reduce operation speed and force limits",
                "Require confirmation for risky actions",
                "Ensure emergency stop is readily accessible"
            ])
        elif risk_level == RiskLevel.MODERATE:
            recommendations.extend([
                "Enhanced monitoring and logging",
                "Consider adjusting operation parameters",
                "Review safety protocols"
            ])
        
        # Factor-specific recommendations
        category_recommendations = {}
        for factor in risk_factors:
            if factor.category not in category_recommendations:
                category_recommendations[factor.category] = set()
            category_recommendations[factor.category].update(factor.mitigation_strategies)
        
        # Add top recommendations from each category
        for category, strategies in category_recommendations.items():
            recommendations.extend(list(strategies)[:2])  # Top 2 per category
        
        # Remove duplicates and limit total
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:8]  # Limit to 8 recommendations
    
    def _calculate_confidence(self, risk_factors: List[RiskFactor], history_length: int) -> float:
        """Calculate confidence in the risk assessment"""
        base_confidence = 0.5
        
        # More risk factors generally means more information
        factor_confidence = min(0.3, len(risk_factors) * 0.05)
        
        # More history means better trend analysis
        history_confidence = min(0.3, history_length * 0.01)
        
        # Sensor and data quality (assumed good for now)
        data_confidence = 0.2
        
        return min(1.0, base_confidence + factor_confidence + history_confidence + data_confidence)
    
    def get_risk_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get risk trends over specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_assessments = [
            a for a in self.assessment_history 
            if a.timestamp >= cutoff_time
        ]
        
        if not recent_assessments:
            return {'error': 'No recent assessments available'}
        
        scores = [a.overall_risk_score for a in recent_assessments]
        timestamps = [a.timestamp for a in recent_assessments]
        
        return {
            'assessment_count': len(recent_assessments),
            'average_risk': np.mean(scores),
            'max_risk': max(scores),
            'min_risk': min(scores),
            'risk_trend': np.polyfit(range(len(scores)), scores, 1)[0] if len(scores) > 1 else 0,
            'time_range': {
                'start': timestamps[0].isoformat() if timestamps else None,
                'end': timestamps[-1].isoformat() if timestamps else None
            }
        }
    
    def export_risk_data(self) -> Dict[str, Any]:
        """Export complete risk assessment data"""
        return {
            'configuration': {
                'history_window': self.history_window,
                'risk_decay_factor': self.risk_decay_factor,
                'prediction_horizon': self.prediction_horizon
            },
            'current_context': {
                'environmental_factors': self.environmental_factors,
                'system_status': self.system_status,
                'human_factors': self.human_factors
            },
            'assessment_history': [a.to_dict() for a in self.assessment_history],
            'statistics': {
                'total_assessments': len(self.assessment_history),
                'average_risk': np.mean([a.overall_risk_score for a in self.assessment_history]) if self.assessment_history else 0,
                'risk_distribution': {
                    level.label: sum(1 for a in self.assessment_history if a.risk_level == level)
                    for level in RiskLevel
                }
            }
        }