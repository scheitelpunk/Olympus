"""
Intention Analysis System

Analyzes the underlying intention of actions to detect potential safety risks:
- Pattern recognition for dangerous sequences
- Behavioral analysis for anomaly detection
- Intent classification and risk scoring
- Predictive safety assessment
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import deque

logger = logging.getLogger(__name__)


class IntentionType(Enum):
    """Types of detected intentions"""
    BENIGN = "benign"
    AGGRESSIVE = "aggressive"
    EXPLORATORY = "exploratory"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"
    POTENTIALLY_HARMFUL = "potentially_harmful"


class RiskCategory(Enum):
    """Risk categories for intentions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class IntentionAssessment:
    """Result of intention analysis"""
    intention_type: IntentionType
    risk_category: RiskCategory
    confidence: float  # 0.0 to 1.0
    risk_score: float  # 0.0 to 1.0
    reasoning: str
    detected_patterns: List[str]
    recommendations: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ActionPattern:
    """Pattern in action sequences"""
    name: str
    pattern: List[str]  # Sequence of action types
    risk_level: RiskCategory
    description: str
    window_size: int = 5  # Number of actions to consider


class IntentionAnalyzer:
    """Analyzes action intentions for safety assessment"""
    
    def __init__(self, history_size: int = 100):
        """
        Initialize intention analyzer
        
        Args:
            history_size: Number of recent actions to keep in memory
        """
        self.history_size = history_size
        self.action_history = deque(maxlen=history_size)
        self.risk_patterns = self._initialize_risk_patterns()
        self.behavioral_baseline = {}
        self.anomaly_threshold = 0.7
        
        logger.info(f"IntentionAnalyzer initialized with history_size={history_size}")
    
    def analyze_intention(self, action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> IntentionAssessment:
        """
        Analyze the intention behind an action
        
        Args:
            action: Action to analyze
            context: Additional context information
            
        Returns:
            IntentionAssessment with analysis results
        """
        # Add action to history
        self.action_history.append({
            'action': action,
            'timestamp': datetime.utcnow(),
            'context': context or {}
        })
        
        # Perform multi-dimensional analysis
        pattern_analysis = self._analyze_patterns()
        behavioral_analysis = self._analyze_behavior(action)
        contextual_analysis = self._analyze_context(action, context)
        temporal_analysis = self._analyze_temporal_patterns()
        
        # Combine analyses
        overall_assessment = self._combine_analyses(
            action, pattern_analysis, behavioral_analysis, 
            contextual_analysis, temporal_analysis
        )
        
        logger.debug(f"Intention analysis complete: {overall_assessment.intention_type.value} "
                    f"(risk: {overall_assessment.risk_category.value}, "
                    f"confidence: {overall_assessment.confidence:.2f})")
        
        return overall_assessment
    
    def _initialize_risk_patterns(self) -> List[ActionPattern]:
        """Initialize known risky patterns"""
        return [
            ActionPattern(
                name="rapid_force_escalation",
                pattern=["low_force", "medium_force", "high_force"],
                risk_level=RiskCategory.HIGH,
                description="Rapid escalation of force application",
                window_size=3
            ),
            ActionPattern(
                name="repetitive_high_speed",
                pattern=["high_speed"] * 5,
                risk_level=RiskCategory.MEDIUM,
                description="Sustained high-speed operations",
                window_size=5
            ),
            ActionPattern(
                name="erratic_movement",
                pattern=["move_left", "move_right", "move_left", "move_right"],
                risk_level=RiskCategory.MEDIUM,
                description="Erratic oscillating movements",
                window_size=4
            ),
            ActionPattern(
                name="tool_switching_rapid",
                pattern=["tool_change", "tool_change", "tool_change"],
                risk_level=RiskCategory.MEDIUM,
                description="Rapid tool switching may indicate confusion",
                window_size=3
            ),
            ActionPattern(
                name="dangerous_tool_sequence",
                pattern=["select_cutter", "high_speed", "high_force"],
                risk_level=RiskCategory.CRITICAL,
                description="Dangerous tool with high speed and force",
                window_size=3
            ),
            ActionPattern(
                name="boundary_testing",
                pattern=["approach_boundary", "approach_boundary", "approach_boundary"],
                risk_level=RiskCategory.HIGH,
                description="Repeated boundary approach may indicate intent to breach",
                window_size=5
            ),
            ActionPattern(
                name="emergency_sequence",
                pattern=["emergency_stop", "immediate_restart"],
                risk_level=RiskCategory.HIGH,
                description="Emergency stop followed by restart may indicate panic",
                window_size=2
            )
        ]
    
    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze action sequences for known risk patterns"""
        detected_patterns = []
        max_risk = RiskCategory.LOW
        pattern_confidence = 0.0
        
        if len(self.action_history) < 2:
            return {
                'detected_patterns': detected_patterns,
                'max_risk': max_risk,
                'confidence': 0.0,
                'pattern_details': []
            }
        
        # Extract action types from recent history
        recent_actions = list(self.action_history)[-10:]  # Last 10 actions
        action_types = [self._classify_action_type(item['action']) for item in recent_actions]
        
        # Check for each risk pattern
        pattern_details = []
        for pattern in self.risk_patterns:
            matches = self._find_pattern_matches(action_types, pattern)
            if matches:
                detected_patterns.append(pattern.name)
                pattern_details.append({
                    'name': pattern.name,
                    'description': pattern.description,
                    'risk_level': pattern.risk_level,
                    'matches': len(matches),
                    'positions': matches
                })
                
                # Update max risk and confidence
                if pattern.risk_level.value == 'critical' or (pattern.risk_level.value == 'high' and max_risk != RiskCategory.CRITICAL):
                    max_risk = pattern.risk_level
                elif pattern.risk_level.value == 'medium' and max_risk == RiskCategory.LOW:
                    max_risk = pattern.risk_level
                
                pattern_confidence = max(pattern_confidence, len(matches) / len(recent_actions))
        
        return {
            'detected_patterns': detected_patterns,
            'max_risk': max_risk,
            'confidence': min(1.0, pattern_confidence),
            'pattern_details': pattern_details
        }
    
    def _analyze_behavior(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behavioral patterns for anomalies"""
        if not self.behavioral_baseline:
            # Build baseline from history
            self._update_behavioral_baseline()
        
        anomaly_score = 0.0
        anomalies = []
        
        # Check force usage patterns
        if 'force' in action:
            force = np.linalg.norm(action['force'])
            avg_force = self.behavioral_baseline.get('avg_force', 5.0)
            force_deviation = abs(force - avg_force) / max(avg_force, 1.0)
            
            if force_deviation > 2.0:  # More than 2x deviation
                anomaly_score += 0.3
                anomalies.append(f"Force anomaly: {force:.1f}N vs baseline {avg_force:.1f}N")
        
        # Check speed patterns
        if 'velocity' in action:
            speed = np.linalg.norm(action['velocity'])
            avg_speed = self.behavioral_baseline.get('avg_speed', 0.3)
            speed_deviation = abs(speed - avg_speed) / max(avg_speed, 0.1)
            
            if speed_deviation > 2.0:
                anomaly_score += 0.3
                anomalies.append(f"Speed anomaly: {speed:.2f}m/s vs baseline {avg_speed:.2f}m/s")
        
        # Check timing patterns
        if len(self.action_history) >= 2:
            current_time = datetime.utcnow()
            last_action_time = self.action_history[-1]['timestamp']
            time_gap = (current_time - last_action_time).total_seconds()
            
            avg_interval = self.behavioral_baseline.get('avg_interval', 2.0)
            interval_deviation = abs(time_gap - avg_interval) / max(avg_interval, 0.5)
            
            if interval_deviation > 3.0:
                anomaly_score += 0.2
                anomalies.append(f"Timing anomaly: {time_gap:.1f}s gap vs baseline {avg_interval:.1f}s")
        
        return {
            'anomaly_score': min(1.0, anomaly_score),
            'anomalies': anomalies,
            'baseline_available': bool(self.behavioral_baseline)
        }
    
    def _analyze_context(self, action: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze contextual factors"""
        context_risk = 0.0
        context_factors = []
        
        if not context:
            return {'risk_score': 0.1, 'factors': ['No context provided']}
        
        # Check time of day
        current_hour = datetime.utcnow().hour
        if current_hour < 6 or current_hour > 22:  # Night hours
            context_risk += 0.2
            context_factors.append("Operation during night hours")
        
        # Check operator status
        if 'operator' in context:
            operator = context['operator']
            
            if operator.get('fatigue_level', 0) > 7:  # Scale 1-10
                context_risk += 0.3
                context_factors.append("High operator fatigue")
            
            if operator.get('experience_hours', 1000) < 100:
                context_risk += 0.2
                context_factors.append("Inexperienced operator")
            
            if operator.get('recent_errors', 0) > 3:
                context_risk += 0.3
                context_factors.append("Recent operator errors")
        
        # Check environmental factors
        if 'environment' in context:
            env = context['environment']
            
            if env.get('noise_level', 50) > 85:  # dB
                context_risk += 0.2
                context_factors.append("High noise environment")
            
            if env.get('distractions', 0) > 2:
                context_risk += 0.2
                context_factors.append("Multiple distractions present")
        
        # Check system status
        if 'system' in context:
            sys = context['system']
            
            if sys.get('maintenance_overdue', False):
                context_risk += 0.3
                context_factors.append("Maintenance overdue")
            
            if sys.get('recent_failures', 0) > 1:
                context_risk += 0.2
                context_factors.append("Recent system failures")
        
        return {
            'risk_score': min(1.0, context_risk),
            'factors': context_factors
        }
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in actions"""
        if len(self.action_history) < 3:
            return {'risk_score': 0.0, 'patterns': []}
        
        temporal_risk = 0.0
        temporal_patterns = []
        
        # Check for acceleration in action frequency
        recent_times = [item['timestamp'] for item in list(self.action_history)[-5:]]
        intervals = []
        for i in range(1, len(recent_times)):
            interval = (recent_times[i] - recent_times[i-1]).total_seconds()
            intervals.append(interval)
        
        if len(intervals) >= 2:
            # Check if intervals are decreasing (accelerating)
            decreasing_count = sum(1 for i in range(1, len(intervals)) if intervals[i] < intervals[i-1])
            if decreasing_count >= len(intervals) * 0.7:  # 70% decreasing
                temporal_risk += 0.3
                temporal_patterns.append("Accelerating action frequency")
        
        # Check for very rapid actions
        if intervals and min(intervals) < 0.1:  # Less than 100ms
            temporal_risk += 0.4
            temporal_patterns.append("Extremely rapid actions")
        
        # Check for burst patterns
        if len(intervals) >= 4:
            burst_threshold = 0.5  # seconds
            burst_count = sum(1 for interval in intervals if interval < burst_threshold)
            if burst_count >= len(intervals) * 0.8:  # 80% of actions in burst
                temporal_risk += 0.3
                temporal_patterns.append("Burst action pattern")
        
        return {
            'risk_score': min(1.0, temporal_risk),
            'patterns': temporal_patterns,
            'intervals': intervals[-3:] if intervals else []  # Last 3 intervals
        }
    
    def _combine_analyses(self, action: Dict[str, Any], pattern_analysis: Dict[str, Any], 
                         behavioral_analysis: Dict[str, Any], contextual_analysis: Dict[str, Any],
                         temporal_analysis: Dict[str, Any]) -> IntentionAssessment:
        """Combine all analyses into final assessment"""
        
        # Determine primary intention type
        intention_type = self._determine_intention_type(action, pattern_analysis)
        
        # Calculate overall risk
        risk_components = {
            'pattern': self._risk_category_to_score(pattern_analysis['max_risk']),
            'behavioral': behavioral_analysis['anomaly_score'],
            'contextual': contextual_analysis['risk_score'],
            'temporal': temporal_analysis['risk_score']
        }
        
        # Weighted combination
        weights = {'pattern': 0.4, 'behavioral': 0.3, 'contextual': 0.2, 'temporal': 0.1}
        overall_risk_score = sum(risk_components[key] * weights[key] for key in weights)
        
        # Determine risk category
        if overall_risk_score >= 0.8:
            risk_category = RiskCategory.CRITICAL
        elif overall_risk_score >= 0.6:
            risk_category = RiskCategory.HIGH
        elif overall_risk_score >= 0.3:
            risk_category = RiskCategory.MEDIUM
        else:
            risk_category = RiskCategory.LOW
        
        # Calculate confidence
        confidence_factors = [
            pattern_analysis['confidence'],
            1.0 - behavioral_analysis['anomaly_score'] if behavioral_analysis['baseline_available'] else 0.5,
            0.8 if contextual_analysis['factors'] else 0.3,
            0.7 if temporal_analysis['patterns'] else 0.5
        ]
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        
        # Generate reasoning
        reasoning_parts = []
        if pattern_analysis['detected_patterns']:
            reasoning_parts.append(f"Detected patterns: {', '.join(pattern_analysis['detected_patterns'])}")
        if behavioral_analysis['anomalies']:
            reasoning_parts.append(f"Behavioral anomalies: {'; '.join(behavioral_analysis['anomalies'])}")
        if contextual_analysis['factors']:
            reasoning_parts.append(f"Context factors: {'; '.join(contextual_analysis['factors'])}")
        if temporal_analysis['patterns']:
            reasoning_parts.append(f"Temporal patterns: {'; '.join(temporal_analysis['patterns'])}")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No significant risk indicators detected"
        
        # Compile detected patterns
        all_patterns = (pattern_analysis['detected_patterns'] + 
                       behavioral_analysis['anomalies'] +
                       contextual_analysis['factors'] +
                       temporal_analysis['patterns'])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_category, all_patterns)
        
        return IntentionAssessment(
            intention_type=intention_type,
            risk_category=risk_category,
            confidence=overall_confidence,
            risk_score=overall_risk_score,
            reasoning=reasoning,
            detected_patterns=all_patterns,
            recommendations=recommendations
        )
    
    def _classify_action_type(self, action: Dict[str, Any]) -> str:
        """Classify action into a type for pattern matching"""
        if 'tool' in action:
            if action['tool'] in ['cutter', 'drill', 'grinder']:
                return 'dangerous_tool'
            return 'tool_use'
        
        if 'force' in action:
            force = np.linalg.norm(action['force'])
            if force > 15:
                return 'high_force'
            elif force > 8:
                return 'medium_force'
            else:
                return 'low_force'
        
        if 'velocity' in action:
            speed = np.linalg.norm(action['velocity'])
            if speed > 0.7:
                return 'high_speed'
            elif speed > 0.3:
                return 'medium_speed'
            else:
                return 'low_speed'
        
        if 'target_position' in action:
            return 'movement'
        
        return 'unknown'
    
    def _find_pattern_matches(self, action_sequence: List[str], pattern: ActionPattern) -> List[int]:
        """Find matches of a pattern in action sequence"""
        matches = []
        pattern_length = len(pattern.pattern)
        
        for i in range(len(action_sequence) - pattern_length + 1):
            if action_sequence[i:i+pattern_length] == pattern.pattern:
                matches.append(i)
        
        return matches
    
    def _update_behavioral_baseline(self):
        """Update behavioral baseline from action history"""
        if len(self.action_history) < 5:
            return
        
        forces = []
        speeds = []
        intervals = []
        
        prev_time = None
        for item in self.action_history:
            action = item['action']
            timestamp = item['timestamp']
            
            if 'force' in action:
                forces.append(np.linalg.norm(action['force']))
            
            if 'velocity' in action:
                speeds.append(np.linalg.norm(action['velocity']))
            
            if prev_time:
                interval = (timestamp - prev_time).total_seconds()
                intervals.append(interval)
            prev_time = timestamp
        
        self.behavioral_baseline = {
            'avg_force': np.mean(forces) if forces else 5.0,
            'avg_speed': np.mean(speeds) if speeds else 0.3,
            'avg_interval': np.mean(intervals) if intervals else 2.0,
            'force_std': np.std(forces) if forces else 2.0,
            'speed_std': np.std(speeds) if speeds else 0.2,
            'updated_at': datetime.utcnow()
        }
    
    def _determine_intention_type(self, action: Dict[str, Any], pattern_analysis: Dict[str, Any]) -> IntentionType:
        """Determine the primary intention type"""
        patterns = pattern_analysis['detected_patterns']
        
        # Check for emergency patterns
        if any('emergency' in pattern for pattern in patterns):
            return IntentionType.EMERGENCY
        
        # Check for aggressive patterns
        aggressive_patterns = ['rapid_force_escalation', 'dangerous_tool_sequence']
        if any(pattern in patterns for pattern in aggressive_patterns):
            return IntentionType.POTENTIALLY_HARMFUL
        
        # Check for exploratory patterns
        if 'boundary_testing' in patterns:
            return IntentionType.EXPLORATORY
        
        # Check for maintenance patterns
        if 'tool' in action and action['tool'] in ['calibration', 'diagnostic', 'maintenance']:
            return IntentionType.MAINTENANCE
        
        # Default based on action characteristics
        if 'force' in action and np.linalg.norm(action['force']) > 10:
            return IntentionType.AGGRESSIVE
        
        return IntentionType.BENIGN
    
    def _risk_category_to_score(self, category: RiskCategory) -> float:
        """Convert risk category to numerical score"""
        mapping = {
            RiskCategory.LOW: 0.2,
            RiskCategory.MEDIUM: 0.5,
            RiskCategory.HIGH: 0.8,
            RiskCategory.CRITICAL: 1.0
        }
        return mapping.get(category, 0.1)
    
    def _generate_recommendations(self, risk_category: RiskCategory, patterns: List[str]) -> List[str]:
        """Generate safety recommendations based on risk assessment"""
        recommendations = []
        
        if risk_category == RiskCategory.CRITICAL:
            recommendations.extend([
                "Immediate human supervision required",
                "Consider emergency stop protocols",
                "Review and validate all action parameters"
            ])
        
        elif risk_category == RiskCategory.HIGH:
            recommendations.extend([
                "Increase monitoring frequency",
                "Require confirmation for high-risk actions",
                "Reduce operational limits"
            ])
        
        elif risk_category == RiskCategory.MEDIUM:
            recommendations.extend([
                "Enhanced logging and monitoring",
                "Consider operator training",
                "Review recent action history"
            ])
        
        # Pattern-specific recommendations
        if any('force' in pattern for pattern in patterns):
            recommendations.append("Review force application protocols")
        
        if any('speed' in pattern for pattern in patterns):
            recommendations.append("Consider speed reduction measures")
        
        if any('tool' in pattern for pattern in patterns):
            recommendations.append("Verify tool safety procedures")
        
        if any('boundary' in pattern for pattern in patterns):
            recommendations.append("Reinforce workspace boundaries")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analyzer status and statistics"""
        return {
            'history_size': len(self.action_history),
            'max_history': self.history_size,
            'baseline_available': bool(self.behavioral_baseline),
            'risk_patterns_count': len(self.risk_patterns),
            'anomaly_threshold': self.anomaly_threshold,
            'recent_patterns': [item['action'] for item in list(self.action_history)[-3:]] if self.action_history else []
        }