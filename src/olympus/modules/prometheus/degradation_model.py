"""
Degradation Model - Wear Analysis and Performance Decay Tracking

The Degradation Model analyzes component wear, performance decay, and aging
patterns to predict maintenance needs and optimize component lifecycles.
It provides sophisticated modeling of how components degrade over time.

Key Features:
- Component wear analysis and modeling
- Performance decay trend analysis
- Aging pattern recognition
- Optimal replacement timing calculation
- Resource consumption optimization
- Lifecycle cost analysis
- Predictive wear-out forecasting
"""

import asyncio
import time
import math
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import statistics
import numpy as np


class DegradationType(Enum):
    """Types of degradation that can be modeled."""
    PERFORMANCE_DECAY = "performance_decay"
    WEAR_OUT = "wear_out"
    AGING = "aging"
    RESOURCE_EFFICIENCY_LOSS = "resource_efficiency_loss"
    ERROR_RATE_INCREASE = "error_rate_increase"
    CAPACITY_REDUCTION = "capacity_reduction"
    RELIABILITY_DECLINE = "reliability_decline"


class DegradationSeverity(Enum):
    """Severity levels for degradation."""
    NEGLIGIBLE = "negligible"
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    SEVERE = "severe"
    CRITICAL = "critical"


class ModelType(Enum):
    """Types of degradation models."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    POWER_LAW = "power_law"
    WEIBULL = "weibull"
    BATHTUB = "bathtub"  # Initial failures, stable period, wear-out
    CUSTOM = "custom"


@dataclass
class DegradationPoint:
    """Single data point in degradation analysis."""
    timestamp: float
    metric_value: float
    component_age: float  # Time since component creation
    usage_cycles: int = 0  # Number of usage cycles
    load_factor: float = 1.0  # Current load as fraction of maximum
    environmental_factor: float = 1.0  # Environmental stress factor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "metric_value": self.metric_value,
            "component_age": self.component_age,
            "usage_cycles": self.usage_cycles,
            "load_factor": self.load_factor,
            "environmental_factor": self.environmental_factor
        }


@dataclass
class DegradationModel:
    """Model representing component degradation over time."""
    model_id: str
    component_id: str
    metric_name: str
    degradation_type: DegradationType
    model_type: ModelType
    
    # Model parameters
    parameters: Dict[str, float] = field(default_factory=dict)
    baseline_value: float = 0.0
    current_value: float = 0.0
    
    # Model quality metrics
    r_squared: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    prediction_accuracy: float = 0.0
    
    # Degradation analysis
    degradation_rate: float = 0.0  # Per unit time
    severity: DegradationSeverity = DegradationSeverity.NEGLIGIBLE
    
    # Predictions
    next_maintenance_threshold: Optional[float] = None
    time_to_threshold: Optional[float] = None
    replacement_threshold: Optional[float] = None
    time_to_replacement: Optional[float] = None
    
    # Metadata
    created_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    data_points: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "component_id": self.component_id,
            "metric_name": self.metric_name,
            "degradation_type": self.degradation_type.value,
            "model_type": self.model_type.value,
            "parameters": self.parameters,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "r_squared": self.r_squared,
            "confidence_interval": self.confidence_interval,
            "prediction_accuracy": self.prediction_accuracy,
            "degradation_rate": self.degradation_rate,
            "severity": self.severity.value,
            "next_maintenance_threshold": self.next_maintenance_threshold,
            "time_to_threshold": self.time_to_threshold,
            "replacement_threshold": self.replacement_threshold,
            "time_to_replacement": self.time_to_replacement,
            "created_time": self.created_time,
            "last_updated": self.last_updated,
            "data_points": self.data_points
        }


@dataclass
class WearPattern:
    """Pattern of component wear over time."""
    pattern_id: str
    component_type: str
    wear_factors: List[str] = field(default_factory=list)
    typical_lifespan: float = 0.0  # Expected lifespan in seconds
    wear_curve_type: str = "linear"
    critical_wear_points: List[float] = field(default_factory=list)
    maintenance_intervals: List[float] = field(default_factory=list)
    replacement_indicators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "component_type": self.component_type,
            "wear_factors": self.wear_factors,
            "typical_lifespan": self.typical_lifespan,
            "wear_curve_type": self.wear_curve_type,
            "critical_wear_points": self.critical_wear_points,
            "maintenance_intervals": self.maintenance_intervals,
            "replacement_indicators": self.replacement_indicators
        }


class DegradationModelBuilder:
    """Builds degradation models from historical data."""
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        self.model_templates = self._initialize_model_templates()
        
    def _initialize_model_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize templates for different types of degradation models."""
        return {
            "cpu_performance_decay": {
                "model_type": ModelType.EXPONENTIAL,
                "degradation_type": DegradationType.PERFORMANCE_DECAY,
                "thresholds": {"maintenance": 0.8, "replacement": 0.5},
                "typical_params": {"decay_constant": 0.001}
            },
            "memory_efficiency_loss": {
                "model_type": ModelType.LINEAR,
                "degradation_type": DegradationType.RESOURCE_EFFICIENCY_LOSS,
                "thresholds": {"maintenance": 0.85, "replacement": 0.7},
                "typical_params": {"slope": -0.0001}
            },
            "disk_wear_out": {
                "model_type": ModelType.WEIBULL,
                "degradation_type": DegradationType.WEAR_OUT,
                "thresholds": {"maintenance": 0.9, "replacement": 0.8},
                "typical_params": {"shape": 2.0, "scale": 1000000}  # shape, scale
            },
            "network_reliability_decline": {
                "model_type": ModelType.BATHTUB,
                "degradation_type": DegradationType.RELIABILITY_DECLINE,
                "thresholds": {"maintenance": 0.95, "replacement": 0.85},
                "typical_params": {"initial_rate": 0.001, "stable_rate": 0.0001, "wearout_rate": 0.01}
            },
            "service_error_increase": {
                "model_type": ModelType.EXPONENTIAL,
                "degradation_type": DegradationType.ERROR_RATE_INCREASE,
                "thresholds": {"maintenance": 0.05, "replacement": 0.2},
                "typical_params": {"growth_rate": 0.0005}
            }
        }
    
    async def build_model(self, component_id: str, metric_name: str,
                         data_points: List[DegradationPoint]) -> Optional[DegradationModel]:
        """Build a degradation model from historical data points."""
        try:
            if len(data_points) < 5:  # Need minimum data points
                return None
            
            # Determine best model template
            template_key = self._select_model_template(metric_name, data_points)
            template = self.model_templates.get(template_key, self.model_templates["cpu_performance_decay"])
            
            model_id = f"model_{component_id}_{metric_name}_{int(time.time())}"
            
            # Extract time series data
            times = [point.component_age for point in data_points]
            values = [point.metric_value for point in data_points]
            
            # Fit model based on type
            model_type = template["model_type"]
            parameters, r_squared = await self._fit_model(model_type, times, values)
            
            # Calculate baseline and current values
            baseline_value = values[0] if values else 0.0
            current_value = values[-1] if values else 0.0
            
            # Calculate degradation rate and severity
            degradation_rate = self._calculate_degradation_rate(values, times)
            severity = self._assess_degradation_severity(degradation_rate, baseline_value, current_value)
            
            # Calculate predictions
            thresholds = template.get("thresholds", {})
            maintenance_threshold = thresholds.get("maintenance", 0.8) * baseline_value
            replacement_threshold = thresholds.get("replacement", 0.5) * baseline_value
            
            time_to_maintenance = await self._predict_time_to_value(
                model_type, parameters, current_value, maintenance_threshold, times[-1]
            )
            time_to_replacement = await self._predict_time_to_value(
                model_type, parameters, current_value, replacement_threshold, times[-1]
            )
            
            model = DegradationModel(
                model_id=model_id,
                component_id=component_id,
                metric_name=metric_name,
                degradation_type=template["degradation_type"],
                model_type=model_type,
                parameters=parameters,
                baseline_value=baseline_value,
                current_value=current_value,
                r_squared=r_squared,
                degradation_rate=degradation_rate,
                severity=severity,
                next_maintenance_threshold=maintenance_threshold,
                time_to_threshold=time_to_maintenance,
                replacement_threshold=replacement_threshold,
                time_to_replacement=time_to_replacement,
                data_points=len(data_points)
            )
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "degradation_model_built",
                    {
                        "model_id": model_id,
                        "component_id": component_id,
                        "metric_name": metric_name,
                        "model_type": model_type.value,
                        "r_squared": r_squared,
                        "data_points": len(data_points)
                    }
                )
            
            return model
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "degradation_model_build_failed",
                    {
                        "component_id": component_id,
                        "metric_name": metric_name,
                        "error": str(e)
                    }
                )
            return None
    
    def _select_model_template(self, metric_name: str, data_points: List[DegradationPoint]) -> str:
        """Select the best model template based on metric name and data characteristics."""
        # Simple heuristic based on metric name
        metric_lower = metric_name.lower()
        
        if "cpu" in metric_lower or "performance" in metric_lower:
            return "cpu_performance_decay"
        elif "memory" in metric_lower or "ram" in metric_lower:
            return "memory_efficiency_loss"
        elif "disk" in metric_lower or "storage" in metric_lower:
            return "disk_wear_out"
        elif "network" in metric_lower or "latency" in metric_lower:
            return "network_reliability_decline"
        elif "error" in metric_lower or "failure" in metric_lower:
            return "service_error_increase"
        else:
            return "cpu_performance_decay"  # Default
    
    async def _fit_model(self, model_type: ModelType, times: List[float], 
                        values: List[float]) -> Tuple[Dict[str, float], float]:
        """Fit a specific model type to the data."""
        try:
            if model_type == ModelType.LINEAR:
                return self._fit_linear_model(times, values)
            elif model_type == ModelType.EXPONENTIAL:
                return self._fit_exponential_model(times, values)
            elif model_type == ModelType.LOGARITHMIC:
                return self._fit_logarithmic_model(times, values)
            elif model_type == ModelType.POWER_LAW:
                return self._fit_power_law_model(times, values)
            elif model_type == ModelType.WEIBULL:
                return self._fit_weibull_model(times, values)
            elif model_type == ModelType.BATHTUB:
                return self._fit_bathtub_model(times, values)
            else:
                # Default to linear
                return self._fit_linear_model(times, values)
                
        except Exception:
            # Fallback to simple linear model
            return self._fit_linear_model(times, values)
    
    def _fit_linear_model(self, times: List[float], values: List[float]) -> Tuple[Dict[str, float], float]:
        """Fit a linear model: y = a*x + b."""
        try:
            n = len(times)
            if n < 2:
                return {"slope": 0, "intercept": values[0] if values else 0}, 0.0
            
            # Calculate linear regression
            sum_x = sum(times)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(times, values))
            sum_x2 = sum(x * x for x in times)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return {"slope": 0, "intercept": statistics.mean(values)}, 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
            
            # Calculate R-squared
            y_mean = statistics.mean(values)
            ss_tot = sum((y - y_mean) ** 2 for y in values)
            ss_res = sum((values[i] - (slope * times[i] + intercept)) ** 2 for i in range(n))
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            
            return {"slope": slope, "intercept": intercept}, max(0.0, r_squared)
            
        except Exception:
            return {"slope": 0, "intercept": statistics.mean(values) if values else 0}, 0.0
    
    def _fit_exponential_model(self, times: List[float], values: List[float]) -> Tuple[Dict[str, float], float]:
        """Fit an exponential model: y = a * exp(b*x)."""
        try:
            # Transform to linear: ln(y) = ln(a) + b*x
            log_values = []
            valid_times = []
            
            for i, y in enumerate(values):
                if y > 0:  # Can only take log of positive values
                    log_values.append(math.log(y))
                    valid_times.append(times[i])
            
            if len(log_values) < 2:
                return {"a": values[0] if values else 1, "b": 0}, 0.0
            
            # Fit linear model to log data
            params, r_squared = self._fit_linear_model(valid_times, log_values)
            
            # Convert back to exponential parameters
            a = math.exp(params["intercept"])
            b = params["slope"]
            
            return {"a": a, "b": b}, r_squared
            
        except Exception:
            return {"a": values[0] if values else 1, "b": 0}, 0.0
    
    def _fit_logarithmic_model(self, times: List[float], values: List[float]) -> Tuple[Dict[str, float], float]:
        """Fit a logarithmic model: y = a * ln(x) + b."""
        try:
            # Transform times: x' = ln(x)
            log_times = []
            valid_values = []
            
            for i, x in enumerate(times):
                if x > 0:  # Can only take log of positive values
                    log_times.append(math.log(x))
                    valid_values.append(values[i])
            
            if len(log_times) < 2:
                return {"a": 0, "b": statistics.mean(values) if values else 0}, 0.0
            
            # Fit linear model to log times
            params, r_squared = self._fit_linear_model(log_times, valid_values)
            
            return {"a": params["slope"], "b": params["intercept"]}, r_squared
            
        except Exception:
            return {"a": 0, "b": statistics.mean(values) if values else 0}, 0.0
    
    def _fit_power_law_model(self, times: List[float], values: List[float]) -> Tuple[Dict[str, float], float]:
        """Fit a power law model: y = a * x^b."""
        try:
            # Transform to linear: ln(y) = ln(a) + b*ln(x)
            log_times = []
            log_values = []
            
            for x, y in zip(times, values):
                if x > 0 and y > 0:
                    log_times.append(math.log(x))
                    log_values.append(math.log(y))
            
            if len(log_times) < 2:
                return {"a": values[0] if values else 1, "b": 1}, 0.0
            
            # Fit linear model to log-log data
            params, r_squared = self._fit_linear_model(log_times, log_values)
            
            # Convert back to power law parameters
            a = math.exp(params["intercept"])
            b = params["slope"]
            
            return {"a": a, "b": b}, r_squared
            
        except Exception:
            return {"a": values[0] if values else 1, "b": 1}, 0.0
    
    def _fit_weibull_model(self, times: List[float], values: List[float]) -> Tuple[Dict[str, float], float]:
        """Fit a Weibull distribution model (simplified)."""
        try:
            # Simplified Weibull: y = 1 - exp(-(x/scale)^shape)
            # This is a placeholder - real Weibull fitting requires more sophisticated methods
            
            # Use exponential approximation for now
            params, r_squared = self._fit_exponential_model(times, values)
            
            # Convert to Weibull-like parameters
            shape = 2.0  # Assume Rayleigh distribution (shape=2)
            scale = 1 / abs(params["b"]) if params["b"] != 0 else 1000
            
            return {"shape": shape, "scale": scale}, r_squared * 0.8  # Reduce confidence
            
        except Exception:
            return {"shape": 2.0, "scale": 1000}, 0.0
    
    def _fit_bathtub_model(self, times: List[float], values: List[float]) -> Tuple[Dict[str, float], float]:
        """Fit a bathtub curve model (simplified)."""
        try:
            # Simplified bathtub: three phases with different rates
            # This is a placeholder - real bathtub fitting requires more sophisticated methods
            
            n = len(times)
            if n < 6:  # Need enough points for three phases
                return self._fit_linear_model(times, values)
            
            # Divide into three phases
            phase1_end = n // 3
            phase2_end = 2 * n // 3
            
            # Fit each phase separately (simplified)
            phase1_params, _ = self._fit_linear_model(times[:phase1_end], values[:phase1_end])
            phase2_params, _ = self._fit_linear_model(times[phase1_end:phase2_end], values[phase1_end:phase2_end])
            phase3_params, _ = self._fit_linear_model(times[phase2_end:], values[phase2_end:])
            
            # Combine into bathtub parameters
            initial_rate = abs(phase1_params["slope"])
            stable_rate = abs(phase2_params["slope"])
            wearout_rate = abs(phase3_params["slope"])
            
            # Calculate overall R-squared (simplified)
            r_squared = 0.6  # Conservative estimate
            
            return {
                "initial_rate": initial_rate,
                "stable_rate": stable_rate,
                "wearout_rate": wearout_rate,
                "phase1_end": times[phase1_end],
                "phase2_end": times[phase2_end]
            }, r_squared
            
        except Exception:
            return {"initial_rate": 0.01, "stable_rate": 0.001, "wearout_rate": 0.1}, 0.0
    
    def _calculate_degradation_rate(self, values: List[float], times: List[float]) -> float:
        """Calculate the current rate of degradation."""
        try:
            if len(values) < 2:
                return 0.0
            
            # Use recent data points for current rate
            recent_count = min(10, len(values))
            recent_values = values[-recent_count:]
            recent_times = times[-recent_count:]
            
            if len(recent_values) < 2:
                return 0.0
            
            # Calculate rate of change
            time_diff = recent_times[-1] - recent_times[0]
            value_diff = recent_values[-1] - recent_values[0]
            
            if time_diff == 0:
                return 0.0
            
            return value_diff / time_diff
            
        except Exception:
            return 0.0
    
    def _assess_degradation_severity(self, degradation_rate: float, 
                                   baseline_value: float, current_value: float) -> DegradationSeverity:
        """Assess the severity of degradation."""
        try:
            if baseline_value == 0:
                return DegradationSeverity.NEGLIGIBLE
            
            # Calculate percentage change from baseline
            percent_change = abs((current_value - baseline_value) / baseline_value) * 100
            
            # Also consider rate of change
            rate_factor = abs(degradation_rate) / abs(baseline_value) if baseline_value != 0 else 0
            
            # Combined severity assessment
            if percent_change > 50 or rate_factor > 0.1:
                return DegradationSeverity.CRITICAL
            elif percent_change > 30 or rate_factor > 0.05:
                return DegradationSeverity.SEVERE
            elif percent_change > 20 or rate_factor > 0.02:
                return DegradationSeverity.SIGNIFICANT
            elif percent_change > 10 or rate_factor > 0.01:
                return DegradationSeverity.MODERATE
            elif percent_change > 5 or rate_factor > 0.005:
                return DegradationSeverity.MINOR
            else:
                return DegradationSeverity.NEGLIGIBLE
                
        except Exception:
            return DegradationSeverity.NEGLIGIBLE
    
    async def _predict_time_to_value(self, model_type: ModelType, parameters: Dict[str, float],
                                   current_value: float, target_value: float,
                                   current_time: float) -> Optional[float]:
        """Predict when a metric will reach a target value."""
        try:
            if current_value == target_value:
                return 0.0
            
            if model_type == ModelType.LINEAR:
                slope = parameters.get("slope", 0)
                if slope == 0:
                    return None
                
                time_diff = (target_value - current_value) / slope
                return time_diff if time_diff > 0 else None
            
            elif model_type == ModelType.EXPONENTIAL:
                a = parameters.get("a", 1)
                b = parameters.get("b", 0)
                
                if a == 0 or b == 0 or target_value <= 0:
                    return None
                
                # y = a * exp(b*x) -> x = ln(y/a) / b
                try:
                    target_time = math.log(target_value / a) / b
                    return target_time - current_time if target_time > current_time else None
                except (ValueError, ZeroDivisionError):
                    return None
            
            # Add more model types as needed
            else:
                # Default to linear approximation
                if len(parameters) >= 2:
                    slope = list(parameters.values())[0]  # Use first parameter as slope
                    if slope != 0:
                        time_diff = (target_value - current_value) / slope
                        return time_diff if time_diff > 0 else None
            
            return None
            
        except Exception:
            return None


class WearPatternAnalyzer:
    """Analyzes wear patterns across similar components."""
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        self.wear_patterns = {}
        self.component_wear_data = defaultdict(list)
        
    async def analyze_wear_patterns(self, component_type: str, 
                                  historical_data: List[Tuple[str, List[DegradationPoint]]]) -> WearPattern:
        """Analyze wear patterns for a component type."""
        try:
            pattern_id = f"wear_{component_type}_{int(time.time())}"
            
            # Aggregate data from all components of this type
            all_lifespans = []
            all_degradation_rates = []
            failure_indicators = set()
            
            for component_id, data_points in historical_data:
                if not data_points:
                    continue
                
                # Calculate lifespan
                lifespan = data_points[-1].component_age
                all_lifespans.append(lifespan)
                
                # Calculate average degradation rate
                values = [point.metric_value for point in data_points]
                times = [point.component_age for point in data_points]
                
                if len(values) >= 2:
                    degradation_rate = (values[-1] - values[0]) / (times[-1] - times[0]) if times[-1] != times[0] else 0
                    all_degradation_rates.append(abs(degradation_rate))
                
                # Identify failure indicators (last 10% of data points with significant degradation)
                if len(data_points) >= 10:
                    final_points = data_points[-max(1, len(data_points) // 10):]
                    for point in final_points:
                        if point.load_factor > 0.8:  # High load
                            failure_indicators.add("high_load_factor")
                        if point.environmental_factor > 1.2:  # High environmental stress
                            failure_indicators.add("environmental_stress")
            
            # Calculate typical lifespan
            typical_lifespan = statistics.median(all_lifespans) if all_lifespans else 0.0
            
            # Determine wear curve type
            wear_curve_type = self._determine_wear_curve_type(all_degradation_rates)
            
            # Calculate critical wear points (percentiles of lifespan)
            critical_points = []
            if typical_lifespan > 0:
                critical_points = [
                    typical_lifespan * 0.25,  # 25% of lifespan
                    typical_lifespan * 0.50,  # 50% of lifespan
                    typical_lifespan * 0.75,  # 75% of lifespan
                    typical_lifespan * 0.90   # 90% of lifespan
                ]
            
            # Calculate maintenance intervals
            maintenance_intervals = []
            if typical_lifespan > 0:
                # Suggest maintenance at 20%, 40%, 60%, 80% of lifespan
                maintenance_intervals = [
                    typical_lifespan * 0.20,
                    typical_lifespan * 0.40,
                    typical_lifespan * 0.60,
                    typical_lifespan * 0.80
                ]
            
            # Identify wear factors
            wear_factors = self._identify_wear_factors(component_type)
            
            pattern = WearPattern(
                pattern_id=pattern_id,
                component_type=component_type,
                wear_factors=wear_factors,
                typical_lifespan=typical_lifespan,
                wear_curve_type=wear_curve_type,
                critical_wear_points=critical_points,
                maintenance_intervals=maintenance_intervals,
                replacement_indicators=list(failure_indicators)
            )
            
            self.wear_patterns[component_type] = pattern
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "wear_pattern_analyzed",
                    {
                        "pattern_id": pattern_id,
                        "component_type": component_type,
                        "typical_lifespan": typical_lifespan,
                        "components_analyzed": len(historical_data)
                    }
                )
            
            return pattern
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "wear_pattern_analysis_failed",
                    {"component_type": component_type, "error": str(e)}
                )
            raise
    
    def _determine_wear_curve_type(self, degradation_rates: List[float]) -> str:
        """Determine the type of wear curve from degradation rates."""
        try:
            if not degradation_rates:
                return "linear"
            
            # Analyze distribution of degradation rates
            rates = sorted(degradation_rates)
            
            # Check for exponential pattern (increasing rates)
            if len(rates) >= 5:
                early_avg = statistics.mean(rates[:len(rates)//3])
                late_avg = statistics.mean(rates[-len(rates)//3:])
                
                if late_avg > early_avg * 2:  # Late rates much higher
                    return "exponential"
                elif early_avg > late_avg * 2:  # Early rates much higher
                    return "logarithmic"
            
            # Check for bathtub curve pattern
            if len(rates) >= 10:
                third = len(rates) // 3
                early_avg = statistics.mean(rates[:third])
                middle_avg = statistics.mean(rates[third:2*third])
                late_avg = statistics.mean(rates[2*third:])
                
                if early_avg > middle_avg and late_avg > middle_avg:
                    return "bathtub"
            
            return "linear"  # Default
            
        except Exception:
            return "linear"
    
    def _identify_wear_factors(self, component_type: str) -> List[str]:
        """Identify factors that contribute to component wear."""
        wear_factor_mapping = {
            "cpu": ["temperature", "utilization", "voltage_fluctuation", "thermal_cycling"],
            "memory": ["read_write_cycles", "temperature", "voltage_stress", "data_retention"],
            "disk": ["read_write_cycles", "temperature", "vibration", "power_cycles"],
            "network": ["data_throughput", "connection_frequency", "environmental_interference"],
            "database": ["query_load", "data_growth", "index_fragmentation", "concurrent_connections"],
            "service": ["request_rate", "processing_complexity", "resource_contention", "dependency_failures"]
        }
        
        # Find matching factors
        for component_key, factors in wear_factor_mapping.items():
            if component_key in component_type.lower():
                return factors
        
        # Default factors
        return ["usage_intensity", "environmental_conditions", "maintenance_frequency"]
    
    async def get_wear_pattern(self, component_type: str) -> Optional[WearPattern]:
        """Get wear pattern for a component type."""
        return self.wear_patterns.get(component_type)
    
    async def predict_component_lifespan(self, component_type: str, 
                                       current_age: float, 
                                       current_condition: float) -> Optional[float]:
        """Predict remaining lifespan for a component."""
        try:
            pattern = await self.get_wear_pattern(component_type)
            if not pattern or pattern.typical_lifespan == 0:
                return None
            
            # Adjust prediction based on current condition
            condition_factor = current_condition  # Assume 0-1 scale
            age_factor = current_age / pattern.typical_lifespan
            
            # Simple prediction model
            remaining_lifespan = pattern.typical_lifespan * (1 - age_factor) * condition_factor
            
            return max(0, remaining_lifespan)
            
        except Exception:
            return None


class DegradationModelSystem:
    """
    Main degradation modeling system that coordinates all degradation analysis.
    """
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        
        # Initialize components
        self.model_builder = DegradationModelBuilder(audit_system)
        self.wear_pattern_analyzer = WearPatternAnalyzer(audit_system)
        
        # System state
        self.is_active = False
        self.analysis_interval = 3600.0  # 1 hour
        self.analysis_task = None
        
        # Model storage
        self.active_models = {}  # component_id -> {metric_name -> DegradationModel}
        self.degradation_data = defaultdict(lambda: defaultdict(lambda: deque(maxlen=500)))
        self.model_history = deque(maxlen=1000)
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize the degradation modeling system."""
        try:
            self.is_active = True
            self.analysis_task = asyncio.create_task(self._analysis_loop())
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "degradation_model_system_initialized",
                    {"analysis_interval": self.analysis_interval}
                )
            
            self.logger.info("Degradation modeling system initialized")
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "degradation_model_system_init_failed",
                    {"error": str(e)}
                )
            raise
    
    async def shutdown(self):
        """Shutdown the degradation modeling system."""
        try:
            self.is_active = False
            
            if self.analysis_task:
                self.analysis_task.cancel()
                try:
                    await self.analysis_task
                except asyncio.CancelledError:
                    pass
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "degradation_model_system_shutdown",
                    {
                        "active_models": len(self.active_models),
                        "total_data_points": sum(
                            sum(len(metric_data) for metric_data in component_data.values())
                            for component_data in self.degradation_data.values()
                        )
                    }
                )
            
            self.logger.info("Degradation modeling system shut down")
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "degradation_model_system_shutdown_failed",
                    {"error": str(e)}
                )
            raise
    
    async def add_degradation_data(self, component_id: str, metric_name: str,
                                 metric_value: float, component_age: float,
                                 usage_cycles: int = 0, load_factor: float = 1.0,
                                 environmental_factor: float = 1.0):
        """Add new degradation data point for a component."""
        try:
            data_point = DegradationPoint(
                timestamp=time.time(),
                metric_value=metric_value,
                component_age=component_age,
                usage_cycles=usage_cycles,
                load_factor=load_factor,
                environmental_factor=environmental_factor
            )
            
            self.degradation_data[component_id][metric_name].append(data_point)
            
            # Trigger model update if we have enough data
            data_points = list(self.degradation_data[component_id][metric_name])
            if len(data_points) >= 10 and len(data_points) % 5 == 0:  # Update every 5 new points
                await self._update_model(component_id, metric_name, data_points)
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "degradation_data_add_failed",
                    {
                        "component_id": component_id,
                        "metric_name": metric_name,
                        "error": str(e)
                    }
                )
    
    async def _update_model(self, component_id: str, metric_name: str,
                          data_points: List[DegradationPoint]):
        """Update degradation model for a component metric."""
        try:
            # Build new model
            model = await self.model_builder.build_model(component_id, metric_name, data_points)
            
            if model:
                # Store model
                if component_id not in self.active_models:
                    self.active_models[component_id] = {}
                
                old_model = self.active_models[component_id].get(metric_name)
                self.active_models[component_id][metric_name] = model
                
                # Add to history
                self.model_history.append(model)
                
                # Log significant changes
                if old_model and abs(model.degradation_rate - old_model.degradation_rate) > 0.01:
                    if self.audit_system:
                        await self.audit_system.log_event(
                            "degradation_model_updated",
                            {
                                "component_id": component_id,
                                "metric_name": metric_name,
                                "old_rate": old_model.degradation_rate,
                                "new_rate": model.degradation_rate,
                                "severity": model.severity.value
                            }
                        )
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "degradation_model_update_failed",
                    {
                        "component_id": component_id,
                        "metric_name": metric_name,
                        "error": str(e)
                    }
                )
    
    async def _analysis_loop(self):
        """Periodic analysis loop for degradation modeling."""
        while self.is_active:
            try:
                # Update all models with recent data
                for component_id, component_data in self.degradation_data.items():
                    for metric_name, data_points in component_data.items():
                        if len(data_points) >= 5:  # Minimum data for analysis
                            await self._update_model(component_id, metric_name, list(data_points))
                
                # Analyze wear patterns for component types
                await self._analyze_component_type_patterns()
                
            except Exception as e:
                self.logger.error(f"Error in degradation analysis loop: {e}")
                if self.audit_system:
                    await self.audit_system.log_event(
                        "degradation_analysis_loop_error",
                        {"error": str(e)}
                    )
            
            await asyncio.sleep(self.analysis_interval)
    
    async def _analyze_component_type_patterns(self):
        """Analyze wear patterns across component types."""
        try:
            # Group components by type
            component_types = defaultdict(list)
            
            for component_id, component_data in self.degradation_data.items():
                # Infer component type from ID (would be better to get from component manager)
                component_type = self._infer_component_type(component_id)
                
                # Get all data points for this component
                all_data_points = []
                for metric_name, data_points in component_data.items():
                    all_data_points.extend(data_points)
                
                if all_data_points:
                    component_types[component_type].append((component_id, all_data_points))
            
            # Analyze patterns for each component type
            for component_type, component_data_list in component_types.items():
                if len(component_data_list) >= 3:  # Need multiple components for pattern analysis
                    await self.wear_pattern_analyzer.analyze_wear_patterns(
                        component_type, component_data_list
                    )
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "wear_pattern_analysis_error",
                    {"error": str(e)}
                )
    
    def _infer_component_type(self, component_id: str) -> str:
        """Infer component type from component ID."""
        component_id_lower = component_id.lower()
        
        if "cpu" in component_id_lower or "processor" in component_id_lower:
            return "cpu"
        elif "memory" in component_id_lower or "ram" in component_id_lower:
            return "memory"
        elif "disk" in component_id_lower or "storage" in component_id_lower:
            return "disk"
        elif "network" in component_id_lower or "net" in component_id_lower:
            return "network"
        elif "database" in component_id_lower or "db" in component_id_lower:
            return "database"
        elif "service" in component_id_lower or "svc" in component_id_lower:
            return "service"
        else:
            return "generic"
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current degradation modeling system status."""
        total_models = sum(len(metrics) for metrics in self.active_models.values())
        total_data_points = sum(
            sum(len(metric_data) for metric_data in component_data.values())
            for component_data in self.degradation_data.values()
        )
        
        return {
            "is_active": self.is_active,
            "analysis_interval": self.analysis_interval,
            "active_models": total_models,
            "tracked_components": len(self.degradation_data),
            "total_data_points": total_data_points,
            "wear_patterns": len(self.wear_pattern_analyzer.wear_patterns),
            "model_history_size": len(self.model_history)
        }
    
    async def get_component_models(self, component_id: str) -> Dict[str, DegradationModel]:
        """Get all degradation models for a component."""
        return self.active_models.get(component_id, {})
    
    async def get_degradation_prediction(self, component_id: str, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get degradation prediction for a specific component metric."""
        try:
            if component_id not in self.active_models:
                return None
            
            model = self.active_models[component_id].get(metric_name)
            if not model:
                return None
            
            return {
                "component_id": component_id,
                "metric_name": metric_name,
                "current_value": model.current_value,
                "degradation_rate": model.degradation_rate,
                "severity": model.severity.value,
                "time_to_maintenance": model.time_to_threshold,
                "time_to_replacement": model.time_to_replacement,
                "model_confidence": model.r_squared,
                "prediction_timestamp": time.time()
            }
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "degradation_prediction_failed",
                    {
                        "component_id": component_id,
                        "metric_name": metric_name,
                        "error": str(e)
                    }
                )
            return None
    
    async def get_wear_pattern(self, component_type: str) -> Optional[WearPattern]:
        """Get wear pattern for a component type."""
        return await self.wear_pattern_analyzer.get_wear_pattern(component_type)
    
    async def predict_component_lifespan(self, component_id: str, current_age: float,
                                       current_condition: float) -> Optional[float]:
        """Predict remaining lifespan for a component."""
        try:
            component_type = self._infer_component_type(component_id)
            return await self.wear_pattern_analyzer.predict_component_lifespan(
                component_type, current_age, current_condition
            )
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "lifespan_prediction_failed",
                    {"component_id": component_id, "error": str(e)}
                )
            return None