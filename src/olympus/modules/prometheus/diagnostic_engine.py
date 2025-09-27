"""
Diagnostic Engine - Fault Detection, Analysis, and Root Cause Identification

The Diagnostic Engine analyzes system health data, identifies faults, performs
root cause analysis, and provides actionable insights for system repair and
optimization. It serves as the analytical brain of the PROMETHEUS system.

Key Features:
- Fault pattern recognition
- Root cause analysis using correlation and causality
- Symptom clustering and classification
- Historical trend analysis
- Component interaction mapping
- Predictive fault modeling
- Automated diagnostic reasoning
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import statistics
import json


class FaultSeverity(Enum):
    """Fault severity classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class FaultType(Enum):
    """Types of faults that can be detected."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DATA_CORRUPTION = "data_corruption"
    SECURITY_BREACH = "security_breach"
    CONFIGURATION_ERROR = "configuration_error"
    HARDWARE_FAILURE = "hardware_failure"
    NETWORK_FAILURE = "network_failure"
    DEPENDENCY_FAILURE = "dependency_failure"
    UNKNOWN = "unknown"


class DiagnosticStatus(Enum):
    """Status of diagnostic analysis."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass
class Fault:
    """Represents a detected fault in the system."""
    fault_id: str
    fault_type: FaultType
    severity: FaultSeverity
    component: str
    description: str
    timestamp: float
    symptoms: List[str] = field(default_factory=list)
    affected_components: Set[str] = field(default_factory=set)
    metrics: Dict[str, float] = field(default_factory=dict)
    probable_causes: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fault to dictionary for serialization."""
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type.value,
            "severity": self.severity.value,
            "component": self.component,
            "description": self.description,
            "timestamp": self.timestamp,
            "symptoms": self.symptoms,
            "affected_components": list(self.affected_components),
            "metrics": self.metrics,
            "probable_causes": self.probable_causes,
            "confidence": self.confidence
        }


@dataclass
class DiagnosticAnalysis:
    """Results of diagnostic analysis for a fault."""
    analysis_id: str
    fault_id: str
    status: DiagnosticStatus
    root_cause: Optional[str] = None
    root_cause_confidence: float = 0.0
    contributing_factors: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    repair_complexity: str = "unknown"  # simple, moderate, complex, critical
    estimated_repair_time: Optional[float] = None
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary for serialization."""
        return {
            "analysis_id": self.analysis_id,
            "fault_id": self.fault_id,
            "status": self.status.value,
            "root_cause": self.root_cause,
            "root_cause_confidence": self.root_cause_confidence,
            "contributing_factors": self.contributing_factors,
            "recommended_actions": self.recommended_actions,
            "repair_complexity": self.repair_complexity,
            "estimated_repair_time": self.estimated_repair_time,
            "risk_assessment": self.risk_assessment,
            "timestamp": self.timestamp
        }


class FaultPatternMatcher:
    """Identifies known fault patterns using historical data and rules."""
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        self.patterns = self._initialize_patterns()
        self.pattern_history = defaultdict(list)
        
    def _initialize_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize known fault patterns."""
        return {
            "cpu_spike_pattern": {
                "symptoms": ["high_cpu_usage", "slow_response_time", "high_load"],
                "thresholds": {"cpu_usage": 90, "response_time": 2000},
                "fault_type": FaultType.PERFORMANCE_DEGRADATION,
                "confidence_base": 0.8
            },
            "memory_leak_pattern": {
                "symptoms": ["increasing_memory", "declining_performance", "eventual_crash"],
                "thresholds": {"memory_growth_rate": 5, "memory_usage": 95},
                "fault_type": FaultType.RESOURCE_EXHAUSTION,
                "confidence_base": 0.85
            },
            "disk_full_pattern": {
                "symptoms": ["high_disk_usage", "write_failures", "service_errors"],
                "thresholds": {"disk_usage": 95, "disk_free": 1000},  # MB
                "fault_type": FaultType.RESOURCE_EXHAUSTION,
                "confidence_base": 0.9
            },
            "network_degradation_pattern": {
                "symptoms": ["high_latency", "packet_loss", "connection_timeouts"],
                "thresholds": {"latency": 500, "packet_loss": 5, "timeouts": 10},
                "fault_type": FaultType.NETWORK_FAILURE,
                "confidence_base": 0.75
            },
            "dependency_failure_pattern": {
                "symptoms": ["external_service_errors", "cascade_failures", "timeout_errors"],
                "thresholds": {"error_rate": 10, "timeout_rate": 5},
                "fault_type": FaultType.DEPENDENCY_FAILURE,
                "confidence_base": 0.7
            }
        }
    
    async def match_patterns(self, symptoms: List[str], metrics: Dict[str, float]) -> List[Tuple[str, float]]:
        """Match current symptoms and metrics against known patterns."""
        matches = []
        
        for pattern_name, pattern in self.patterns.items():
            confidence = await self._calculate_pattern_confidence(pattern, symptoms, metrics)
            if confidence > 0.5:  # Minimum confidence threshold
                matches.append((pattern_name, confidence))
        
        # Sort by confidence descending
        matches.sort(key=lambda x: x[1], reverse=True)
        
        if self.audit_system:
            await self.audit_system.log_event(
                "pattern_matching_completed",
                {
                    "patterns_checked": len(self.patterns),
                    "matches_found": len(matches),
                    "top_match": matches[0] if matches else None
                }
            )
        
        return matches
    
    async def _calculate_pattern_confidence(self, pattern: Dict[str, Any], 
                                          symptoms: List[str], metrics: Dict[str, float]) -> float:
        """Calculate confidence score for a pattern match."""
        try:
            # Check symptom overlap
            pattern_symptoms = set(pattern["symptoms"])
            current_symptoms = set(symptoms)
            symptom_overlap = len(pattern_symptoms.intersection(current_symptoms))
            symptom_score = symptom_overlap / len(pattern_symptoms)
            
            # Check threshold violations
            thresholds = pattern.get("thresholds", {})
            threshold_violations = 0
            total_thresholds = len(thresholds)
            
            for metric, threshold in thresholds.items():
                if metric in metrics:
                    if metrics[metric] >= threshold:
                        threshold_violations += 1
            
            threshold_score = threshold_violations / total_thresholds if total_thresholds > 0 else 0
            
            # Combine scores with base confidence
            base_confidence = pattern.get("confidence_base", 0.5)
            combined_confidence = base_confidence * (0.6 * symptom_score + 0.4 * threshold_score)
            
            return min(combined_confidence, 1.0)
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "pattern_confidence_calculation_failed",
                    {"pattern": pattern, "error": str(e)}
                )
            return 0.0
    
    async def add_pattern(self, pattern_name: str, pattern_definition: Dict[str, Any]):
        """Add a new fault pattern to the matcher."""
        self.patterns[pattern_name] = pattern_definition
        
        if self.audit_system:
            await self.audit_system.log_event(
                "fault_pattern_added",
                {"pattern_name": pattern_name, "definition": pattern_definition}
            )


class RootCauseAnalyzer:
    """Performs root cause analysis using correlation and causality analysis."""
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        self.component_dependencies = {}
        self.causal_rules = self._initialize_causal_rules()
        self.correlation_history = defaultdict(lambda: deque(maxlen=100))
        
    def _initialize_causal_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize causal analysis rules."""
        return {
            "cpu_high": [
                {
                    "condition": "memory_usage > 90",
                    "cause": "Memory pressure causing CPU overhead",
                    "confidence": 0.8
                },
                {
                    "condition": "disk_io_high",
                    "cause": "High disk I/O causing CPU wait states",
                    "confidence": 0.7
                }
            ],
            "memory_high": [
                {
                    "condition": "process_count > threshold",
                    "cause": "Too many processes consuming memory",
                    "confidence": 0.75
                },
                {
                    "condition": "memory_leak_detected",
                    "cause": "Application memory leak",
                    "confidence": 0.9
                }
            ],
            "network_slow": [
                {
                    "condition": "external_dependency_slow",
                    "cause": "External service degradation",
                    "confidence": 0.8
                },
                {
                    "condition": "bandwidth_saturated",
                    "cause": "Network bandwidth exhaustion",
                    "confidence": 0.85
                }
            ]
        }
    
    async def analyze_root_cause(self, fault: Fault, 
                               historical_data: List[Dict[str, Any]]) -> DiagnosticAnalysis:
        """Perform comprehensive root cause analysis."""
        analysis_id = f"rca_{fault.fault_id}_{int(time.time())}"
        
        try:
            # Perform multiple analysis techniques
            correlation_results = await self._analyze_correlations(fault, historical_data)
            causal_results = await self._apply_causal_rules(fault)
            dependency_results = await self._analyze_dependencies(fault)
            
            # Combine results and determine most likely root cause
            root_cause, confidence = self._combine_analysis_results(
                correlation_results, causal_results, dependency_results
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(fault, root_cause)
            
            # Assess repair complexity and time
            complexity, estimated_time = self._assess_repair_complexity(fault, root_cause)
            
            # Perform risk assessment
            risk_assessment = await self._assess_risks(fault, root_cause)
            
            analysis = DiagnosticAnalysis(
                analysis_id=analysis_id,
                fault_id=fault.fault_id,
                status=DiagnosticStatus.COMPLETED,
                root_cause=root_cause,
                root_cause_confidence=confidence,
                contributing_factors=self._extract_contributing_factors(
                    correlation_results, causal_results, dependency_results
                ),
                recommended_actions=recommendations,
                repair_complexity=complexity,
                estimated_repair_time=estimated_time,
                risk_assessment=risk_assessment
            )
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "root_cause_analysis_completed",
                    {
                        "analysis_id": analysis_id,
                        "fault_id": fault.fault_id,
                        "root_cause": root_cause,
                        "confidence": confidence
                    }
                )
            
            return analysis
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "root_cause_analysis_failed",
                    {"fault_id": fault.fault_id, "error": str(e)}
                )
            
            return DiagnosticAnalysis(
                analysis_id=analysis_id,
                fault_id=fault.fault_id,
                status=DiagnosticStatus.FAILED,
                root_cause=f"Analysis failed: {str(e)}",
                root_cause_confidence=0.0
            )
    
    async def _analyze_correlations(self, fault: Fault, 
                                  historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze correlations between metrics and fault occurrence."""
        correlations = {}
        
        try:
            # Analyze metric correlations around fault time
            fault_time = fault.timestamp
            time_window = 300  # 5 minutes before/after fault
            
            relevant_data = [
                data for data in historical_data
                if abs(data.get("timestamp", 0) - fault_time) <= time_window
            ]
            
            if len(relevant_data) < 3:  # Need minimum data points
                return correlations
            
            # Calculate correlations for each metric
            for metric_name in fault.metrics:
                metric_values = [data.get("metrics", {}).get(metric_name, 0) 
                               for data in relevant_data if metric_name in data.get("metrics", {})]
                
                if len(metric_values) > 2:
                    # Simple correlation with fault occurrence
                    correlation = self._calculate_fault_correlation(metric_values, fault.metrics[metric_name])
                    correlations[metric_name] = correlation
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "correlation_analysis_failed",
                    {"fault_id": fault.fault_id, "error": str(e)}
                )
        
        return correlations
    
    def _calculate_fault_correlation(self, metric_values: List[float], fault_value: float) -> float:
        """Calculate correlation between metric trend and fault."""
        try:
            if not metric_values:
                return 0.0
                
            # Check if fault value is an outlier
            mean_value = statistics.mean(metric_values)
            stdev = statistics.stdev(metric_values) if len(metric_values) > 1 else 0
            
            if stdev == 0:
                return 0.5 if abs(fault_value - mean_value) < 0.01 else 0.0
            
            z_score = abs(fault_value - mean_value) / stdev
            
            # Higher z-score indicates stronger correlation
            return min(z_score / 3.0, 1.0)  # Normalize to 0-1
            
        except Exception:
            return 0.0
    
    async def _apply_causal_rules(self, fault: Fault) -> Dict[str, float]:
        """Apply causal rules to determine potential causes."""
        causal_matches = {}
        
        try:
            for symptom in fault.symptoms:
                if symptom in self.causal_rules:
                    for rule in self.causal_rules[symptom]:
                        # Simplified rule evaluation - would be more sophisticated in practice
                        if self._evaluate_rule_condition(rule["condition"], fault):
                            causal_matches[rule["cause"]] = rule["confidence"]
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "causal_rule_application_failed",
                    {"fault_id": fault.fault_id, "error": str(e)}
                )
        
        return causal_matches
    
    def _evaluate_rule_condition(self, condition: str, fault: Fault) -> bool:
        """Evaluate a causal rule condition against fault data."""
        # Simplified condition evaluation - would parse and evaluate properly
        try:
            if "memory_usage > 90" in condition and "memory_usage_percent" in fault.metrics:
                return fault.metrics["memory_usage_percent"] > 90
            if "disk_io_high" in condition and "disk_io" in fault.metrics:
                return fault.metrics.get("disk_io", 0) > 1000
            # Add more condition evaluations as needed
        except Exception:
            pass
        
        return False
    
    async def _analyze_dependencies(self, fault: Fault) -> Dict[str, float]:
        """Analyze component dependencies to identify cascade failures."""
        dependency_impacts = {}
        
        try:
            component = fault.component
            
            # Check if fault affects dependent components
            if component in self.component_dependencies:
                dependents = self.component_dependencies[component]
                for dependent in dependents:
                    if dependent in fault.affected_components:
                        # Calculate impact score based on dependency strength
                        dependency_impacts[f"cascade_failure_to_{dependent}"] = 0.8
            
            # Check if fault is caused by dependency failure
            for dep_component, dependents in self.component_dependencies.items():
                if component in dependents:
                    dependency_impacts[f"dependency_failure_from_{dep_component}"] = 0.7
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "dependency_analysis_failed",
                    {"fault_id": fault.fault_id, "error": str(e)}
                )
        
        return dependency_impacts
    
    def _combine_analysis_results(self, correlation_results: Dict[str, float],
                                causal_results: Dict[str, float],
                                dependency_results: Dict[str, float]) -> Tuple[str, float]:
        """Combine analysis results to determine most likely root cause."""
        all_causes = {}
        
        # Weight different analysis types
        correlation_weight = 0.3
        causal_weight = 0.5
        dependency_weight = 0.2
        
        # Add weighted correlation results
        for cause, score in correlation_results.items():
            all_causes[f"metric_correlation_{cause}"] = score * correlation_weight
        
        # Add weighted causal results
        for cause, score in causal_results.items():
            all_causes[cause] = all_causes.get(cause, 0) + score * causal_weight
        
        # Add weighted dependency results
        for cause, score in dependency_results.items():
            all_causes[cause] = all_causes.get(cause, 0) + score * dependency_weight
        
        if not all_causes:
            return "Unknown - insufficient analysis data", 0.0
        
        # Return highest scoring cause
        best_cause = max(all_causes.items(), key=lambda x: x[1])
        return best_cause[0], best_cause[1]
    
    def _extract_contributing_factors(self, correlation_results: Dict[str, float],
                                    causal_results: Dict[str, float],
                                    dependency_results: Dict[str, float]) -> List[str]:
        """Extract contributing factors from analysis results."""
        factors = []
        
        # Add high-correlation metrics
        for metric, correlation in correlation_results.items():
            if correlation > 0.6:
                factors.append(f"High correlation with {metric}")
        
        # Add causal factors
        for cause, confidence in causal_results.items():
            if confidence > 0.6:
                factors.append(cause)
        
        # Add dependency factors
        for dep, impact in dependency_results.items():
            if impact > 0.6:
                factors.append(dep)
        
        return factors[:5]  # Limit to top 5 factors
    
    async def _generate_recommendations(self, fault: Fault, root_cause: str) -> List[str]:
        """Generate recommended actions based on fault and root cause."""
        recommendations = []
        
        # Fault type specific recommendations
        if fault.fault_type == FaultType.PERFORMANCE_DEGRADATION:
            recommendations.extend([
                "Monitor resource utilization trends",
                "Consider scaling resources if needed",
                "Review recent code changes or deployments"
            ])
        
        elif fault.fault_type == FaultType.RESOURCE_EXHAUSTION:
            recommendations.extend([
                "Free up resources immediately",
                "Implement resource monitoring alerts",
                "Plan capacity upgrades"
            ])
        
        elif fault.fault_type == FaultType.NETWORK_FAILURE:
            recommendations.extend([
                "Check network connectivity and configuration",
                "Review firewall and routing rules",
                "Contact network provider if external"
            ])
        
        # Root cause specific recommendations
        if "memory" in root_cause.lower():
            recommendations.append("Investigate potential memory leaks")
        
        if "dependency" in root_cause.lower():
            recommendations.append("Check external service status and health")
        
        if "cascade" in root_cause.lower():
            recommendations.append("Implement circuit breakers to prevent cascade failures")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _assess_repair_complexity(self, fault: Fault, root_cause: str) -> Tuple[str, Optional[float]]:
        """Assess repair complexity and estimate time."""
        complexity = "moderate"
        estimated_time = None
        
        try:
            # Assess based on fault severity
            if fault.severity in [FaultSeverity.CRITICAL, FaultSeverity.CATASTROPHIC]:
                complexity = "complex"
                estimated_time = 3600  # 1 hour
            elif fault.severity == FaultSeverity.HIGH:
                complexity = "moderate"
                estimated_time = 1800  # 30 minutes
            else:
                complexity = "simple"
                estimated_time = 600  # 10 minutes
            
            # Adjust based on root cause
            if "dependency" in root_cause.lower():
                complexity = "complex"  # May require external coordination
                estimated_time = estimated_time * 2 if estimated_time else 3600
            
            if "hardware" in root_cause.lower():
                complexity = "critical"  # May require physical intervention
                estimated_time = None  # Cannot estimate without knowing hardware requirements
                
        except Exception:
            pass
        
        return complexity, estimated_time
    
    async def _assess_risks(self, fault: Fault, root_cause: str) -> Dict[str, Any]:
        """Assess risks associated with the fault and potential repairs."""
        risks = {
            "data_loss_risk": "low",
            "service_disruption_risk": "medium",
            "security_risk": "low",
            "cascade_failure_risk": "medium",
            "repair_risks": []
        }
        
        try:
            # Assess based on fault type
            if fault.fault_type == FaultType.DATA_CORRUPTION:
                risks["data_loss_risk"] = "high"
            
            if fault.fault_type == FaultType.SECURITY_BREACH:
                risks["security_risk"] = "high"
            
            if fault.fault_type == FaultType.SERVICE_UNAVAILABLE:
                risks["service_disruption_risk"] = "high"
            
            # Assess cascade failure risk
            if len(fault.affected_components) > 1:
                risks["cascade_failure_risk"] = "high"
            
            # Assess repair risks
            if fault.severity >= FaultSeverity.CRITICAL:
                risks["repair_risks"].extend([
                    "System restart may be required",
                    "Service downtime during repair",
                    "Potential data backup needed"
                ])
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "risk_assessment_failed",
                    {"fault_id": fault.fault_id, "error": str(e)}
                )
        
        return risks
    
    async def set_component_dependencies(self, dependencies: Dict[str, List[str]]):
        """Set component dependency mapping for cascade analysis."""
        self.component_dependencies = dependencies
        
        if self.audit_system:
            await self.audit_system.log_event(
                "component_dependencies_updated",
                {"dependency_count": len(dependencies)}
            )


class SymptomAnalyzer:
    """Analyzes and classifies system symptoms."""
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        self.symptom_clusters = defaultdict(list)
        
    async def analyze_symptoms(self, metrics: Dict[str, float], 
                             alerts: List[Dict[str, Any]]) -> List[str]:
        """Analyze current system state to identify symptoms."""
        symptoms = []
        
        try:
            # Analyze metrics for symptoms
            if metrics.get("cpu_usage_percent", 0) > 80:
                symptoms.append("high_cpu_usage")
            
            if metrics.get("memory_usage_percent", 0) > 85:
                symptoms.append("high_memory_usage")
            
            if metrics.get("disk_usage_percent", 0) > 90:
                symptoms.append("high_disk_usage")
            
            if metrics.get("network_latency_ms", 0) > 200:
                symptoms.append("high_latency")
            
            if metrics.get("error_rate", 0) > 5:
                symptoms.append("high_error_rate")
            
            # Analyze alerts for symptoms
            critical_alerts = [alert for alert in alerts if alert.get("level") == "critical"]
            if critical_alerts:
                symptoms.append("critical_alerts_present")
            
            # Check for performance degradation
            response_time = metrics.get("response_time", 0)
            if response_time > 2000:
                symptoms.append("slow_response_time")
            
            # Check for resource trends
            if self._detect_increasing_trend("memory_usage_percent", metrics):
                symptoms.append("increasing_memory")
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "symptom_analysis_failed",
                    {"error": str(e)}
                )
        
        return symptoms
    
    def _detect_increasing_trend(self, metric_name: str, current_metrics: Dict[str, float]) -> bool:
        """Detect if a metric is showing an increasing trend."""
        # Simplified trend detection - would use historical data in practice
        current_value = current_metrics.get(metric_name, 0)
        return current_value > 70  # Placeholder logic


class DiagnosticEngine:
    """
    Main diagnostic engine that coordinates fault detection, analysis, and diagnosis.
    """
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        
        # Initialize diagnostic components
        self.pattern_matcher = FaultPatternMatcher(audit_system)
        self.root_cause_analyzer = RootCauseAnalyzer(audit_system)
        self.symptom_analyzer = SymptomAnalyzer(audit_system)
        
        # Diagnostic state
        self.is_active = False
        self.detected_faults = {}
        self.completed_analyses = {}
        self.fault_history = deque(maxlen=1000)
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize the diagnostic engine."""
        try:
            self.is_active = True
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "diagnostic_engine_initialized",
                    {"timestamp": time.time()}
                )
            
            self.logger.info("Diagnostic engine initialized")
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "diagnostic_engine_init_failed",
                    {"error": str(e)}
                )
            raise
    
    async def shutdown(self):
        """Shutdown the diagnostic engine."""
        try:
            self.is_active = False
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "diagnostic_engine_shutdown",
                    {
                        "total_faults_analyzed": len(self.fault_history),
                        "active_analyses": len(self.detected_faults)
                    }
                )
            
            self.logger.info("Diagnostic engine shut down")
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "diagnostic_engine_shutdown_failed",
                    {"error": str(e)}
                )
            raise
    
    async def detect_fault(self, component: str, metrics: Dict[str, float], 
                          alerts: List[Dict[str, Any]]) -> Optional[Fault]:
        """Detect and classify faults based on metrics and alerts."""
        if not self.is_active:
            return None
        
        try:
            # Analyze symptoms
            symptoms = await self.symptom_analyzer.analyze_symptoms(metrics, alerts)
            
            if not symptoms:
                return None  # No symptoms detected
            
            # Match against known patterns
            pattern_matches = await self.pattern_matcher.match_patterns(symptoms, metrics)
            
            if not pattern_matches:
                # Unknown fault pattern
                fault_type = FaultType.UNKNOWN
                severity = self._assess_severity_from_metrics(metrics)
                confidence = 0.3
            else:
                # Use best matching pattern
                best_pattern, confidence = pattern_matches[0]
                pattern_info = self.pattern_matcher.patterns[best_pattern]
                fault_type = pattern_info["fault_type"]
                severity = self._assess_severity_from_metrics(metrics)
            
            # Create fault object
            fault_id = f"fault_{component}_{int(time.time())}"
            fault = Fault(
                fault_id=fault_id,
                fault_type=fault_type,
                severity=severity,
                component=component,
                description=self._generate_fault_description(fault_type, symptoms),
                timestamp=time.time(),
                symptoms=symptoms,
                affected_components=self._identify_affected_components(component, symptoms),
                metrics=metrics.copy(),
                probable_causes=self._extract_probable_causes(pattern_matches),
                confidence=confidence
            )
            
            # Store fault
            self.detected_faults[fault_id] = fault
            self.fault_history.append(fault)
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "fault_detected",
                    {
                        "fault_id": fault_id,
                        "component": component,
                        "fault_type": fault_type.value,
                        "severity": severity.value,
                        "confidence": confidence
                    }
                )
            
            self.logger.warning(f"Fault detected: {fault.description}")
            
            return fault
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "fault_detection_failed",
                    {"component": component, "error": str(e)}
                )
            raise
    
    def _assess_severity_from_metrics(self, metrics: Dict[str, float]) -> FaultSeverity:
        """Assess fault severity based on metric values."""
        try:
            severity_score = 0
            
            # CPU severity
            cpu_usage = metrics.get("cpu_usage_percent", 0)
            if cpu_usage > 95:
                severity_score += 3
            elif cpu_usage > 85:
                severity_score += 2
            elif cpu_usage > 75:
                severity_score += 1
            
            # Memory severity
            memory_usage = metrics.get("memory_usage_percent", 0)
            if memory_usage > 95:
                severity_score += 3
            elif memory_usage > 85:
                severity_score += 2
            elif memory_usage > 75:
                severity_score += 1
            
            # Error rate severity
            error_rate = metrics.get("error_rate", 0)
            if error_rate > 20:
                severity_score += 3
            elif error_rate > 10:
                severity_score += 2
            elif error_rate > 5:
                severity_score += 1
            
            # Response time severity
            response_time = metrics.get("response_time", 0)
            if response_time > 5000:
                severity_score += 2
            elif response_time > 2000:
                severity_score += 1
            
            # Map score to severity
            if severity_score >= 8:
                return FaultSeverity.CATASTROPHIC
            elif severity_score >= 6:
                return FaultSeverity.CRITICAL
            elif severity_score >= 4:
                return FaultSeverity.HIGH
            elif severity_score >= 2:
                return FaultSeverity.MEDIUM
            else:
                return FaultSeverity.LOW
                
        except Exception:
            return FaultSeverity.MEDIUM
    
    def _generate_fault_description(self, fault_type: FaultType, symptoms: List[str]) -> str:
        """Generate human-readable fault description."""
        descriptions = {
            FaultType.PERFORMANCE_DEGRADATION: "System performance has degraded",
            FaultType.RESOURCE_EXHAUSTION: "System resources are exhausted",
            FaultType.SERVICE_UNAVAILABLE: "Service is unavailable or unresponsive",
            FaultType.NETWORK_FAILURE: "Network connectivity issues detected",
            FaultType.DEPENDENCY_FAILURE: "External dependency failure detected",
            FaultType.UNKNOWN: "Unknown system fault detected"
        }
        
        base_description = descriptions.get(fault_type, "System fault detected")
        
        if symptoms:
            symptom_text = ", ".join(symptoms[:3])  # Include top 3 symptoms
            return f"{base_description} with symptoms: {symptom_text}"
        
        return base_description
    
    def _identify_affected_components(self, primary_component: str, symptoms: List[str]) -> Set[str]:
        """Identify components that may be affected by the fault."""
        affected = {primary_component}
        
        # Add components based on symptoms
        if "high_cpu_usage" in symptoms:
            affected.add("cpu_scheduler")
        
        if "high_memory_usage" in symptoms:
            affected.add("memory_manager")
        
        if "high_latency" in symptoms:
            affected.add("network_stack")
        
        return affected
    
    def _extract_probable_causes(self, pattern_matches: List[Tuple[str, float]]) -> List[str]:
        """Extract probable causes from pattern matches."""
        causes = []
        
        for pattern_name, confidence in pattern_matches[:3]:  # Top 3 matches
            if confidence > 0.6:
                cause_mapping = {
                    "cpu_spike_pattern": "High computational load or inefficient algorithms",
                    "memory_leak_pattern": "Application memory leak or excessive memory allocation",
                    "disk_full_pattern": "Insufficient disk space or excessive logging",
                    "network_degradation_pattern": "Network congestion or connectivity issues",
                    "dependency_failure_pattern": "External service failure or timeout"
                }
                
                cause = cause_mapping.get(pattern_name, f"Pattern match: {pattern_name}")
                causes.append(cause)
        
        return causes
    
    async def diagnose_fault(self, fault_id: str) -> Optional[DiagnosticAnalysis]:
        """Perform comprehensive diagnosis of a detected fault."""
        if not self.is_active or fault_id not in self.detected_faults:
            return None
        
        try:
            fault = self.detected_faults[fault_id]
            
            # Gather historical data for analysis
            historical_data = await self._gather_historical_data(fault)
            
            # Perform root cause analysis
            analysis = await self.root_cause_analyzer.analyze_root_cause(fault, historical_data)
            
            # Store completed analysis
            self.completed_analyses[analysis.analysis_id] = analysis
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "fault_diagnosis_completed",
                    {
                        "fault_id": fault_id,
                        "analysis_id": analysis.analysis_id,
                        "root_cause": analysis.root_cause,
                        "confidence": analysis.root_cause_confidence
                    }
                )
            
            return analysis
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "fault_diagnosis_failed",
                    {"fault_id": fault_id, "error": str(e)}
                )
            raise
    
    async def _gather_historical_data(self, fault: Fault) -> List[Dict[str, Any]]:
        """Gather historical data relevant to the fault for analysis."""
        # Placeholder implementation - would gather actual historical metrics
        historical_data = []
        
        try:
            # Generate sample historical data for analysis
            base_time = fault.timestamp - 3600  # 1 hour before fault
            
            for i in range(60):  # 60 data points
                data_point = {
                    "timestamp": base_time + (i * 60),  # 1 minute intervals
                    "component": fault.component,
                    "metrics": {
                        metric: value * (0.8 + 0.4 * (i / 60))  # Simulate increasing trend
                        for metric, value in fault.metrics.items()
                    }
                }
                historical_data.append(data_point)
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "historical_data_gathering_failed",
                    {"fault_id": fault.fault_id, "error": str(e)}
                )
        
        return historical_data
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current diagnostic engine status."""
        return {
            "is_active": self.is_active,
            "detected_faults": len(self.detected_faults),
            "completed_analyses": len(self.completed_analyses),
            "fault_history_size": len(self.fault_history),
            "pattern_count": len(self.pattern_matcher.patterns)
        }
    
    async def get_fault(self, fault_id: str) -> Optional[Fault]:
        """Get fault information by ID."""
        return self.detected_faults.get(fault_id)
    
    async def get_analysis(self, analysis_id: str) -> Optional[DiagnosticAnalysis]:
        """Get diagnostic analysis by ID."""
        return self.completed_analyses.get(analysis_id)
    
    async def get_active_faults(self) -> List[Fault]:
        """Get all currently active faults."""
        return list(self.detected_faults.values())
    
    async def resolve_fault(self, fault_id: str):
        """Mark a fault as resolved and remove from active faults."""
        if fault_id in self.detected_faults:
            fault = self.detected_faults.pop(fault_id)
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "fault_resolved",
                    {"fault_id": fault_id, "component": fault.component}
                )
            
            self.logger.info(f"Fault resolved: {fault_id}")
            return True
        
        return False