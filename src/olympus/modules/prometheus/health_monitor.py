"""
Health Monitor - Continuous System Monitoring and Vital Sign Tracking

The Health Monitor continuously tracks system health metrics, vital signs,
and performance indicators across all OLYMPUS components. It serves as the
primary sensor network for the PROMETHEUS self-healing system.

Key Features:
- Real-time vital sign monitoring
- Performance metric collection
- Anomaly detection and alerting
- Health trend analysis
- Component-specific monitoring
- Resource utilization tracking
- Network and connectivity monitoring
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import statistics
from collections import defaultdict, deque


class HealthStatus(Enum):
    """Health status levels for components and overall system."""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric with metadata."""
    name: str
    value: float
    unit: str
    timestamp: float
    component: str
    status: HealthStatus
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None
    description: str = ""


@dataclass
class HealthAlert:
    """Health monitoring alert."""
    alert_id: str
    component: str
    metric: str
    level: AlertLevel
    message: str
    timestamp: float
    value: float
    threshold: Optional[float] = None
    suggested_actions: List[str] = None


class VitalSignsMonitor:
    """Monitors critical system vital signs."""
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        self.vital_signs = {}
        self.history = defaultdict(lambda: deque(maxlen=1000))
        
    async def collect_cpu_metrics(self) -> Dict[str, float]:
        """Collect CPU utilization metrics."""
        try:
            # Simulated CPU metrics - would integrate with actual system monitoring
            return {
                "cpu_usage_percent": 45.2,
                "cpu_load_1min": 0.8,
                "cpu_load_5min": 0.9,
                "cpu_load_15min": 1.1,
                "cpu_temperature": 65.0,
                "cpu_frequency": 2.4
            }
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "cpu_metrics_collection_failed",
                    {"error": str(e)}
                )
            return {}
    
    async def collect_memory_metrics(self) -> Dict[str, float]:
        """Collect memory utilization metrics."""
        try:
            return {
                "memory_usage_percent": 68.5,
                "memory_available_gb": 7.8,
                "memory_used_gb": 24.2,
                "memory_total_gb": 32.0,
                "swap_usage_percent": 12.3,
                "cache_usage_gb": 4.5
            }
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "memory_metrics_collection_failed",
                    {"error": str(e)}
                )
            return {}
    
    async def collect_disk_metrics(self) -> Dict[str, float]:
        """Collect disk utilization metrics."""
        try:
            return {
                "disk_usage_percent": 78.3,
                "disk_free_gb": 125.7,
                "disk_used_gb": 474.3,
                "disk_total_gb": 600.0,
                "disk_read_iops": 150.0,
                "disk_write_iops": 80.0,
                "disk_latency_ms": 2.3
            }
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "disk_metrics_collection_failed",
                    {"error": str(e)}
                )
            return {}
    
    async def collect_network_metrics(self) -> Dict[str, float]:
        """Collect network performance metrics."""
        try:
            return {
                "network_rx_bytes_sec": 1024000.0,
                "network_tx_bytes_sec": 512000.0,
                "network_connections": 25.0,
                "network_errors": 0.0,
                "network_latency_ms": 15.2,
                "network_packet_loss_percent": 0.1
            }
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "network_metrics_collection_failed",
                    {"error": str(e)}
                )
            return {}


class ComponentHealthTracker:
    """Tracks health of individual system components."""
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        self.component_health = {}
        self.component_history = defaultdict(lambda: deque(maxlen=500))
        
    async def track_component(self, component_name: str, metrics: Dict[str, Any]):
        """Track health metrics for a specific component."""
        try:
            timestamp = time.time()
            health_status = self._assess_component_health(component_name, metrics)
            
            health_data = {
                "component": component_name,
                "metrics": metrics,
                "health_status": health_status,
                "timestamp": timestamp,
                "uptime": metrics.get("uptime", 0),
                "error_rate": metrics.get("error_rate", 0),
                "response_time": metrics.get("response_time", 0)
            }
            
            self.component_health[component_name] = health_data
            self.component_history[component_name].append(health_data)
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "component_health_tracked",
                    {
                        "component": component_name,
                        "health_status": health_status.value,
                        "metrics_count": len(metrics)
                    }
                )
                
            return health_data
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "component_tracking_failed",
                    {"component": component_name, "error": str(e)}
                )
            raise
    
    def _assess_component_health(self, component_name: str, metrics: Dict[str, Any]) -> HealthStatus:
        """Assess overall health status for a component."""
        try:
            error_rate = metrics.get("error_rate", 0)
            response_time = metrics.get("response_time", 0)
            cpu_usage = metrics.get("cpu_usage", 0)
            memory_usage = metrics.get("memory_usage", 0)
            
            # Define health assessment criteria
            if error_rate > 10 or response_time > 5000:  # 10% errors or 5s response
                return HealthStatus.CRITICAL
            elif error_rate > 5 or response_time > 2000:  # 5% errors or 2s response
                return HealthStatus.WARNING
            elif cpu_usage > 90 or memory_usage > 90:
                return HealthStatus.WARNING
            elif error_rate > 1 or response_time > 1000:
                return HealthStatus.GOOD
            else:
                return HealthStatus.EXCELLENT
                
        except Exception:
            return HealthStatus.UNKNOWN


class AnomalyDetector:
    """Detects anomalies in health metrics using statistical analysis."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.metric_history = defaultdict(lambda: deque(maxlen=100))
        
    async def detect_anomalies(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect anomalies in current metrics compared to historical data."""
        anomalies = []
        
        for metric_name, value in metrics.items():
            history = self.metric_history[metric_name]
            history.append(value)
            
            if len(history) < 10:  # Need minimum history for analysis
                continue
                
            try:
                mean = statistics.mean(history)
                stdev = statistics.stdev(history)
                
                if stdev == 0:  # No variation in data
                    continue
                    
                z_score = abs(value - mean) / stdev
                
                if z_score > self.sensitivity:
                    anomalies.append({
                        "metric": metric_name,
                        "value": value,
                        "expected_mean": mean,
                        "z_score": z_score,
                        "severity": self._calculate_anomaly_severity(z_score),
                        "timestamp": time.time()
                    })
                    
            except statistics.StatisticsError:
                continue  # Skip metrics with insufficient or invalid data
                
        return anomalies
    
    def _calculate_anomaly_severity(self, z_score: float) -> AlertLevel:
        """Calculate anomaly severity based on z-score."""
        if z_score > 4.0:
            return AlertLevel.CRITICAL
        elif z_score > 3.0:
            return AlertLevel.HIGH
        elif z_score > 2.5:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW


class HealthMonitor:
    """
    Main health monitoring system that coordinates all monitoring activities.
    """
    
    def __init__(self, audit_system=None, alert_callback: Optional[Callable] = None):
        self.audit_system = audit_system
        self.alert_callback = alert_callback
        
        # Initialize monitoring components
        self.vital_signs_monitor = VitalSignsMonitor(audit_system)
        self.component_tracker = ComponentHealthTracker(audit_system)
        self.anomaly_detector = AnomalyDetector()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_interval = 10.0  # seconds
        self.monitoring_task = None
        
        # Alert management
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
        # Health thresholds
        self.health_thresholds = {
            "cpu_usage_percent": {"warning": 80, "critical": 95},
            "memory_usage_percent": {"warning": 85, "critical": 95},
            "disk_usage_percent": {"warning": 85, "critical": 95},
            "network_latency_ms": {"warning": 100, "critical": 500},
            "disk_latency_ms": {"warning": 10, "critical": 50}
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        if self.audit_system:
            await self.audit_system.log_event(
                "health_monitoring_started",
                {"interval": self.monitoring_interval}
            )
        
        self.logger.info("Health monitoring started")
    
    async def stop(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.audit_system:
            await self.audit_system.log_event(
                "health_monitoring_stopped",
                {"total_alerts": len(self.alert_history)}
            )
        
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop that collects metrics and detects issues."""
        while self.is_monitoring:
            try:
                # Collect system vital signs
                cpu_metrics = await self.vital_signs_monitor.collect_cpu_metrics()
                memory_metrics = await self.vital_signs_monitor.collect_memory_metrics()
                disk_metrics = await self.vital_signs_monitor.collect_disk_metrics()
                network_metrics = await self.vital_signs_monitor.collect_network_metrics()
                
                # Combine all metrics
                all_metrics = {**cpu_metrics, **memory_metrics, **disk_metrics, **network_metrics}
                
                # Check for threshold violations
                await self._check_thresholds(all_metrics)
                
                # Detect anomalies
                anomalies = await self.anomaly_detector.detect_anomalies(all_metrics)
                if anomalies:
                    await self._handle_anomalies(anomalies)
                
                # Track overall system component
                await self.component_tracker.track_component("system", {
                    **all_metrics,
                    "uptime": time.time(),
                    "error_rate": 0.0,  # Would be calculated from actual errors
                    "response_time": 50.0  # Average system response time
                })
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                if self.audit_system:
                    await self.audit_system.log_event(
                        "monitoring_loop_error",
                        {"error": str(e)}
                    )
            
            await asyncio.sleep(self.monitoring_interval)
    
    async def _check_thresholds(self, metrics: Dict[str, float]):
        """Check metrics against defined thresholds and generate alerts."""
        for metric_name, value in metrics.items():
            if metric_name not in self.health_thresholds:
                continue
                
            thresholds = self.health_thresholds[metric_name]
            alert_level = None
            
            if value >= thresholds["critical"]:
                alert_level = AlertLevel.CRITICAL
            elif value >= thresholds["warning"]:
                alert_level = AlertLevel.MEDIUM
            
            if alert_level:
                await self._create_alert(
                    component="system",
                    metric=metric_name,
                    level=alert_level,
                    message=f"{metric_name} is {value} (threshold: {thresholds})",
                    value=value,
                    threshold=thresholds["warning"] if alert_level == AlertLevel.MEDIUM else thresholds["critical"]
                )
    
    async def _handle_anomalies(self, anomalies: List[Dict[str, Any]]):
        """Handle detected anomalies by creating appropriate alerts."""
        for anomaly in anomalies:
            await self._create_alert(
                component="system",
                metric=anomaly["metric"],
                level=anomaly["severity"],
                message=f"Anomaly detected in {anomaly['metric']}: {anomaly['value']} (z-score: {anomaly['z_score']:.2f})",
                value=anomaly["value"]
            )
    
    async def _create_alert(self, component: str, metric: str, level: AlertLevel, 
                          message: str, value: float, threshold: Optional[float] = None):
        """Create and manage health alerts."""
        alert_id = f"{component}_{metric}_{int(time.time())}"
        
        alert = HealthAlert(
            alert_id=alert_id,
            component=component,
            metric=metric,
            level=level,
            message=message,
            timestamp=time.time(),
            value=value,
            threshold=threshold,
            suggested_actions=self._get_suggested_actions(metric, level)
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Log to audit system
        if self.audit_system:
            await self.audit_system.log_event(
                "health_alert_created",
                {
                    "alert_id": alert_id,
                    "component": component,
                    "metric": metric,
                    "level": level.value,
                    "value": value
                }
            )
        
        # Call alert callback if configured
        if self.alert_callback:
            try:
                await self.alert_callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        self.logger.warning(f"Health alert: {message}")
    
    def _get_suggested_actions(self, metric: str, level: AlertLevel) -> List[str]:
        """Get suggested actions for specific metric alerts."""
        actions = {
            "cpu_usage_percent": [
                "Check for runaway processes",
                "Consider scaling resources",
                "Review recent deployments"
            ],
            "memory_usage_percent": [
                "Check for memory leaks",
                "Restart high-memory services",
                "Consider increasing memory allocation"
            ],
            "disk_usage_percent": [
                "Clean up old logs and temporary files",
                "Archive old data",
                "Consider adding storage capacity"
            ],
            "network_latency_ms": [
                "Check network connectivity",
                "Review firewall rules",
                "Monitor external dependencies"
            ]
        }
        
        return actions.get(metric, ["Investigate metric anomaly", "Review system logs"])
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current health monitoring status."""
        active_alerts_count = len(self.active_alerts)
        critical_alerts = [alert for alert in self.active_alerts.values() 
                          if alert.level == AlertLevel.CRITICAL]
        
        return {
            "is_monitoring": self.is_monitoring,
            "monitoring_interval": self.monitoring_interval,
            "active_alerts": active_alerts_count,
            "critical_alerts": len(critical_alerts),
            "total_components_tracked": len(self.component_tracker.component_health),
            "alert_history_size": len(self.alert_history),
            "last_monitoring_cycle": time.time() if self.is_monitoring else None
        }
    
    async def get_component_health(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get health information for a specific component."""
        return self.component_tracker.component_health.get(component_name)
    
    async def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[HealthAlert]:
        """Get active alerts, optionally filtered by level."""
        if level:
            return [alert for alert in self.active_alerts.values() if alert.level == level]
        return list(self.active_alerts.values())
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge and remove an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "health_alert_acknowledged",
                    {"alert_id": alert_id, "component": alert.component}
                )
            
            return True
        return False
    
    async def add_component_monitoring(self, component_name: str, metrics: Dict[str, Any]):
        """Add or update monitoring for a specific component."""
        await self.component_tracker.track_component(component_name, metrics)