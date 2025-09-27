"""
Integrity Monitor - Advanced monitoring system for the ethical core

This module provides comprehensive monitoring of the ethical system's integrity,
including real-time health checks, performance monitoring, and security alerting.

Author: OLYMPUS Core Development Team
Version: 1.0.0
Security Level: CRITICAL
"""

import threading
import time
import json
import logging
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
import uuid


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of alerts that can be generated"""
    INTEGRITY_VIOLATION = "integrity_violation"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SECURITY_BREACH = "security_breach"
    SYSTEM_ERROR = "system_error"
    THRESHOLD_EXCEEDED = "threshold_exceeded"


@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: float
    threshold: float
    unit: str = ""
    status: HealthStatus = HealthStatus.HEALTHY
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Determine status based on threshold"""
        if self.value >= self.threshold:
            self.status = HealthStatus.CRITICAL
        elif self.value >= self.threshold * 0.8:
            self.status = HealthStatus.WARNING
        else:
            self.status = HealthStatus.HEALTHY


@dataclass
class SecurityAlert:
    """Security alert information"""
    alert_type: AlertType
    severity: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resolved: bool = False


class IntegrityMonitor:
    """
    Advanced monitoring system for the ethical core that provides:
    - Real-time health monitoring
    - Performance tracking
    - Security alerting
    - Automated diagnostics
    - Trend analysis
    """
    
    def __init__(self, 
                 kernel,
                 monitoring_interval: float = 1.0,
                 alert_callback: Optional[Callable[[SecurityAlert], None]] = None):
        """
        Initialize the integrity monitor
        
        Args:
            kernel: The AsimovKernel instance to monitor
            monitoring_interval: Interval between monitoring checks (seconds)
            alert_callback: Optional callback function for alerts
        """
        self._kernel = kernel
        self._monitoring_interval = monitoring_interval
        self._alert_callback = alert_callback
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread = None
        self._monitor_id = str(uuid.uuid4())
        
        # Health metrics storage
        self._health_metrics: Dict[str, List[HealthMetric]] = {}
        self._alerts: List[SecurityAlert] = []
        
        # Performance tracking
        self._performance_history = []
        self._baseline_metrics = {}
        
        # Thresholds
        self._thresholds = {
            "evaluation_time_ms": 1000.0,  # 1 second max
            "memory_usage_mb": 100.0,      # 100 MB max
            "integrity_check_time_ms": 100.0,  # 100ms max
            "error_rate_percent": 5.0,     # 5% max error rate
            "alert_rate_per_minute": 10.0  # 10 alerts per minute max
        }
        
        # Logging
        self._logger = logging.getLogger(f"IntegrityMonitor.{self._monitor_id}")
        
        # Initialize baseline
        self._establish_baseline()
        
        self._logger.info(f"IntegrityMonitor initialized with ID: {self._monitor_id}")
    
    def start_monitoring(self) -> bool:
        """
        Start the integrity monitoring system
        
        Returns:
            True if monitoring started successfully, False otherwise
        """
        if self._monitoring_active:
            self._logger.warning("Monitoring already active")
            return False
        
        try:
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name=f"IntegrityMonitor.{self._monitor_id}"
            )
            self._monitoring_thread.start()
            
            self._logger.info("Integrity monitoring started")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to start monitoring: {str(e)}")
            self._monitoring_active = False
            return False
    
    def stop_monitoring(self) -> None:
        """Stop the integrity monitoring system"""
        self._monitoring_active = False
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=2.0)
        
        self._logger.info("Integrity monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                start_time = time.time()
                
                # Perform health checks
                health_status = self._perform_health_check()
                
                # Check for performance degradation
                self._check_performance_degradation()
                
                # Verify system integrity
                self._verify_system_integrity()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Calculate monitoring overhead
                monitoring_time = (time.time() - start_time) * 1000
                self._record_metric("monitoring_overhead_ms", monitoring_time, 50.0)
                
                time.sleep(self._monitoring_interval)
                
            except Exception as e:
                self._logger.error(f"Error in monitoring loop: {str(e)}")
                self._generate_alert(
                    AlertType.SYSTEM_ERROR,
                    "critical",
                    f"Monitoring loop error: {str(e)}"
                )
                time.sleep(self._monitoring_interval)
    
    def _perform_health_check(self) -> HealthStatus:
        """
        Perform comprehensive health check
        
        Returns:
            Overall system health status
        """
        try:
            # Get kernel status
            kernel_status = self._kernel.get_system_status()
            
            # Check law integrity
            integrity_start = time.time()
            laws_integrity = self._kernel.verify_law_integrity()
            integrity_time = (time.time() - integrity_start) * 1000
            
            self._record_metric("laws_integrity", 1.0 if laws_integrity else 0.0, 0.5)
            self._record_metric("integrity_check_time_ms", integrity_time, 
                              self._thresholds["integrity_check_time_ms"])
            
            # Check system state
            self._record_metric("emergency_stop_active", 
                              1.0 if kernel_status["emergency_stop_active"] else 0.0, 0.5)
            
            # Check evaluation performance
            if hasattr(self._kernel, '_evaluation_count') and self._kernel._evaluation_count > 0:
                avg_eval_time = self._calculate_average_evaluation_time()
                self._record_metric("avg_evaluation_time_ms", avg_eval_time,
                                  self._thresholds["evaluation_time_ms"])
            
            # Determine overall health
            critical_metrics = [m for metrics in self._health_metrics.values() 
                              for m in metrics[-1:] if m.status == HealthStatus.CRITICAL]
            
            if critical_metrics:
                return HealthStatus.CRITICAL
            
            warning_metrics = [m for metrics in self._health_metrics.values() 
                             for m in metrics[-1:] if m.status == HealthStatus.WARNING]
            
            return HealthStatus.WARNING if warning_metrics else HealthStatus.HEALTHY
            
        except Exception as e:
            self._logger.error(f"Health check failed: {str(e)}")
            return HealthStatus.CRITICAL
    
    def _check_performance_degradation(self) -> None:
        """Check for performance degradation patterns"""
        try:
            # Analyze evaluation time trends
            eval_times = self._get_recent_metrics("avg_evaluation_time_ms", 10)
            
            if len(eval_times) >= 5:
                recent_avg = statistics.mean([m.value for m in eval_times[-3:]])
                baseline_avg = self._baseline_metrics.get("avg_evaluation_time_ms", recent_avg)
                
                if recent_avg > baseline_avg * 2.0:  # 100% degradation
                    self._generate_alert(
                        AlertType.PERFORMANCE_DEGRADATION,
                        "warning",
                        f"Evaluation time degraded: {recent_avg:.2f}ms vs baseline {baseline_avg:.2f}ms"
                    )
            
        except Exception as e:
            self._logger.error(f"Performance degradation check failed: {str(e)}")
    
    def _verify_system_integrity(self) -> None:
        """Verify overall system integrity"""
        try:
            # Check for suspicious patterns
            recent_alerts = [a for a in self._alerts 
                           if (datetime.now(timezone.utc) - a.timestamp).total_seconds() < 300]
            
            if len(recent_alerts) > 20:  # Too many alerts in 5 minutes
                self._generate_alert(
                    AlertType.SECURITY_BREACH,
                    "critical",
                    f"Suspicious alert pattern: {len(recent_alerts)} alerts in 5 minutes"
                )
            
            # Check for integrity violations
            integrity_metrics = self._get_recent_metrics("laws_integrity", 5)
            failed_checks = [m for m in integrity_metrics if m.value < 1.0]
            
            if failed_checks:
                self._generate_alert(
                    AlertType.INTEGRITY_VIOLATION,
                    "critical",
                    f"Law integrity violations detected: {len(failed_checks)} failures"
                )
            
        except Exception as e:
            self._logger.error(f"System integrity verification failed: {str(e)}")
    
    def _record_metric(self, name: str, value: float, threshold: float, unit: str = "") -> None:
        """Record a health metric"""
        metric = HealthMetric(
            name=name,
            value=value,
            threshold=threshold,
            unit=unit
        )
        
        if name not in self._health_metrics:
            self._health_metrics[name] = []
        
        self._health_metrics[name].append(metric)
        
        # Keep only recent metrics (last 1000 entries)
        if len(self._health_metrics[name]) > 1000:
            self._health_metrics[name] = self._health_metrics[name][-500:]
        
        # Generate alert if metric is critical
        if metric.status == HealthStatus.CRITICAL:
            self._generate_alert(
                AlertType.THRESHOLD_EXCEEDED,
                "warning",
                f"Metric {name} exceeded threshold: {value} > {threshold} {unit}"
            )
    
    def _get_recent_metrics(self, name: str, count: int) -> List[HealthMetric]:
        """Get recent metrics for a given name"""
        if name not in self._health_metrics:
            return []
        
        return self._health_metrics[name][-count:]
    
    def _calculate_average_evaluation_time(self) -> float:
        """Calculate average evaluation time from kernel history"""
        try:
            if hasattr(self._kernel, '_evaluation_history'):
                recent_evaluations = self._kernel._evaluation_history[-100:]  # Last 100
                if recent_evaluations:
                    times = [e.get("evaluation_time", 0) * 1000 for e in recent_evaluations
                           if "evaluation_time" in e]
                    return statistics.mean(times) if times else 0.0
            return 0.0
        except Exception:
            return 0.0
    
    def _establish_baseline(self) -> None:
        """Establish baseline metrics for comparison"""
        try:
            # Perform a few test evaluations to establish baseline
            from .asimov_kernel import ActionContext, ActionType
            
            test_context = ActionContext(
                action_type=ActionType.INFORMATION,
                description="Baseline test evaluation",
                risk_level="low"
            )
            
            # Perform baseline evaluations
            times = []
            for _ in range(5):
                start = time.time()
                self._kernel.evaluate_action(test_context)
                times.append((time.time() - start) * 1000)
            
            self._baseline_metrics["avg_evaluation_time_ms"] = statistics.mean(times)
            
            self._logger.info(f"Baseline established: {self._baseline_metrics}")
            
        except Exception as e:
            self._logger.warning(f"Could not establish baseline: {str(e)}")
            self._baseline_metrics["avg_evaluation_time_ms"] = 10.0  # Default baseline
    
    def _generate_alert(self, alert_type: AlertType, severity: str, message: str, 
                       details: Dict[str, Any] = None) -> None:
        """Generate a security alert"""
        alert = SecurityAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details or {}
        )
        
        self._alerts.append(alert)
        
        # Log the alert
        self._logger.warning(f"ALERT - {alert_type.value}: {message}")
        
        # Call alert callback if provided
        if self._alert_callback:
            try:
                self._alert_callback(alert)
            except Exception as e:
                self._logger.error(f"Alert callback failed: {str(e)}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        # Clean up old alerts
        self._alerts = [a for a in self._alerts if a.timestamp > cutoff_time]
        
        # Clean up old performance history
        self._performance_history = [p for p in self._performance_history 
                                   if p.get("timestamp", cutoff_time) > cutoff_time]
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive health report
        
        Returns:
            Dictionary with health report data
        """
        current_health = self._perform_health_check()
        
        # Get recent metrics summary
        metrics_summary = {}
        for name, metrics in self._health_metrics.items():
            if metrics:
                latest = metrics[-1]
                metrics_summary[name] = {
                    "current_value": latest.value,
                    "threshold": latest.threshold,
                    "status": latest.status.value,
                    "unit": latest.unit
                }
        
        # Get recent alerts
        recent_alerts = [
            {
                "type": a.alert_type.value,
                "severity": a.severity,
                "message": a.message,
                "timestamp": a.timestamp.isoformat(),
                "resolved": a.resolved
            }
            for a in self._alerts[-10:]  # Last 10 alerts
        ]
        
        return {
            "monitor_id": self._monitor_id,
            "overall_health": current_health.value,
            "monitoring_active": self._monitoring_active,
            "kernel_status": self._kernel.get_system_status(),
            "metrics_summary": metrics_summary,
            "recent_alerts": recent_alerts,
            "baseline_metrics": self._baseline_metrics,
            "report_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_alerts(self, unresolved_only: bool = True) -> List[SecurityAlert]:
        """
        Get security alerts
        
        Args:
            unresolved_only: If True, return only unresolved alerts
            
        Returns:
            List of security alerts
        """
        if unresolved_only:
            return [a for a in self._alerts if not a.resolved]
        return self._alerts.copy()
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Mark an alert as resolved
        
        Args:
            alert_id: ID of alert to resolve
            
        Returns:
            True if alert was found and resolved, False otherwise
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                self._logger.info(f"Alert {alert_id} marked as resolved")
                return True
        
        return False
    
    def set_threshold(self, metric_name: str, threshold: float) -> None:
        """
        Set threshold for a specific metric
        
        Args:
            metric_name: Name of the metric
            threshold: New threshold value
        """
        self._thresholds[metric_name] = threshold
        self._logger.info(f"Threshold for {metric_name} set to {threshold}")
    
    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()