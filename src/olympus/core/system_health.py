"""
System Health Monitor - Comprehensive Health Monitoring for OLYMPUS
===================================================================

Monitors system health across all components and provides real-time diagnostics.

Key Responsibilities:
- Component health monitoring and reporting
- Performance metrics collection and analysis
- Resource usage tracking and alerting
- System diagnostics and troubleshooting
- Health trend analysis and prediction
- Automated recovery recommendations
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque, defaultdict
import statistics
import json


class HealthStatus(Enum):
    """Health status levels"""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class HealthMetric:
    """Represents a health metric"""
    name: str
    value: float
    unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: HealthStatus = HealthStatus.UNKNOWN


@dataclass
class HealthAlert:
    """Health monitoring alert"""
    id: str
    level: AlertLevel
    component: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class ComponentHealth:
    """Health status for a system component"""
    name: str
    status: HealthStatus
    metrics: List[HealthMetric] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    last_error: Optional[str] = None
    uptime: float = 0.0
    availability: float = 100.0


@dataclass
class SystemDiagnostics:
    """System diagnostic information"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    open_files: int
    process_count: int
    load_average: List[float]
    boot_time: datetime
    uptime: float


class SystemHealth:
    """
    System health monitoring for OLYMPUS
    
    Provides comprehensive health monitoring, metrics collection,
    and diagnostic capabilities for all system components.
    """
    
    def __init__(self, config_manager):
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        
        # Health storage
        self.component_health: Dict[str, ComponentHealth] = {}
        self.system_metrics: deque = deque(maxlen=1000)  # Last 1000 system metrics
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.resolved_alerts: deque = deque(maxlen=500)  # Last 500 resolved alerts
        
        # Health checkers
        self.health_checkers: Dict[str, Callable] = {}
        self.metric_collectors: Dict[str, Callable] = {}
        
        # Configuration
        self.check_interval = 10.0  # seconds
        self.metric_retention_hours = 24
        self.alert_cooldown = 300  # 5 minutes
        self.auto_recovery_enabled = True
        
        # Thresholds
        self.default_thresholds = {
            'cpu_usage': {'warning': 80.0, 'critical': 95.0},
            'memory_usage': {'warning': 85.0, 'critical': 95.0},
            'disk_usage': {'warning': 80.0, 'critical': 90.0},
            'response_time': {'warning': 1000.0, 'critical': 5000.0},  # milliseconds
            'error_rate': {'warning': 5.0, 'critical': 10.0},  # percent
            'availability': {'warning': 95.0, 'critical': 90.0}  # percent
        }
        
        # Performance tracking
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.trend_analysis_window = 50  # metrics for trend analysis
        
        # Internal state
        self._monitoring_active = False
        self._monitoring_tasks: List[asyncio.Task] = []
        self._shutdown_event = threading.Event()
        self._last_alert_times: Dict[str, datetime] = {}
        
        # System diagnostics
        self.system_start_time = datetime.now()
        self.last_diagnostics: Optional[SystemDiagnostics] = None
        
        self.logger.info("System Health Monitor initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the System Health Monitor
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing System Health Monitor...")
            
            # Load configuration
            config = await self.config_manager.get_module_config('system_health')
            self._apply_config(config)
            
            # Register default health checkers
            await self._register_default_checkers()
            
            # Register core components
            await self._register_core_components()
            
            # Start monitoring
            await self._start_monitoring()
            
            # Perform initial health check
            await self._perform_full_health_check()
            
            self.logger.info("System Health Monitor initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"System Health Monitor initialization failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the System Health Monitor"""
        self.logger.info("Shutting down System Health Monitor...")
        
        self._shutdown_event.set()
        self._monitoring_active = False
        
        # Cancel monitoring tasks
        for task in self._monitoring_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Generate final health report
        await self._generate_shutdown_report()
        
        self.logger.info("System Health Monitor shutdown complete")
    
    async def register_component(self, name: str, health_checker: Optional[Callable] = None) -> bool:
        """
        Register a component for health monitoring
        
        Args:
            name: Component name
            health_checker: Optional custom health check function
            
        Returns:
            True if registered successfully
        """
        try:
            # Initialize component health
            self.component_health[name] = ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN
            )
            
            # Register health checker
            if health_checker:
                self.health_checkers[name] = health_checker
            
            self.logger.info(f"Registered component for health monitoring: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register component {name}: {e}")
            return False
    
    async def update_component_health(self, name: str, status: HealthStatus, 
                                    metrics: Optional[List[HealthMetric]] = None,
                                    error_info: Optional[str] = None) -> None:
        """
        Update health status for a component
        
        Args:
            name: Component name
            status: Health status
            metrics: Optional health metrics
            error_info: Optional error information
        """
        try:
            if name not in self.component_health:
                await self.register_component(name)
            
            component = self.component_health[name]
            old_status = component.status
            
            # Update status
            component.status = status
            component.last_check = datetime.now()
            
            # Update metrics
            if metrics:
                component.metrics = metrics
                
                # Store metrics in performance history
                for metric in metrics:
                    self.performance_history[f"{name}.{metric.name}"].append({
                        'timestamp': metric.timestamp,
                        'value': metric.value
                    })
            
            # Update error information
            if error_info:
                component.last_error = error_info
                component.error_count += 1
            elif status == HealthStatus.HEALTHY:
                component.last_error = None
            
            # Calculate uptime and availability
            await self._update_component_availability(component)
            
            # Generate alerts for status changes
            if old_status != status:
                await self._check_status_change_alerts(name, old_status, status)
            
            # Check metric thresholds
            if metrics:
                await self._check_metric_alerts(name, metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to update component health for {name}: {e}")
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """
        Get overall system health summary
        
        Returns:
            Health summary dictionary
        """
        try:
            healthy_count = sum(1 for c in self.component_health.values() 
                              if c.status == HealthStatus.HEALTHY)
            total_count = len(self.component_health)
            
            # Determine overall status
            if total_count == 0:
                overall_status = HealthStatus.UNKNOWN
            elif any(c.status == HealthStatus.CRITICAL for c in self.component_health.values()):
                overall_status = HealthStatus.CRITICAL
            elif any(c.status == HealthStatus.DEGRADED for c in self.component_health.values()):
                overall_status = HealthStatus.DEGRADED
            elif healthy_count == total_count:
                overall_status = HealthStatus.HEALTHY
            else:
                overall_status = HealthStatus.DEGRADED
            
            # Get system diagnostics
            system_diag = await self._collect_system_diagnostics()
            
            return {
                'overall_status': overall_status.value,
                'healthy_components': healthy_count,
                'total_components': total_count,
                'active_alerts': len(self.active_alerts),
                'critical_alerts': len([a for a in self.active_alerts.values() 
                                      if a.level == AlertLevel.CRITICAL]),
                'system_uptime': (datetime.now() - self.system_start_time).total_seconds(),
                'cpu_usage': system_diag.cpu_usage,
                'memory_usage': system_diag.memory_usage,
                'disk_usage': system_diag.disk_usage,
                'load_average': system_diag.load_average,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get health summary: {e}")
            return {'error': str(e)}
    
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """
        Get comprehensive health report
        
        Returns:
            Comprehensive health information
        """
        try:
            summary = await self.get_health_summary()
            
            # Component details
            component_details = {}
            for name, health in self.component_health.items():
                component_details[name] = {
                    'status': health.status.value,
                    'last_check': health.last_check.isoformat(),
                    'error_count': health.error_count,
                    'last_error': health.last_error,
                    'uptime': health.uptime,
                    'availability': health.availability,
                    'metrics': [
                        {
                            'name': m.name,
                            'value': m.value,
                            'unit': m.unit,
                            'status': m.status.value,
                            'timestamp': m.timestamp.isoformat()
                        }
                        for m in health.metrics
                    ]
                }
            
            # Active alerts
            alert_details = {
                alert.id: {
                    'level': alert.level.value,
                    'component': alert.component,
                    'message': alert.message,
                    'details': alert.details,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in self.active_alerts.values()
            }
            
            # System diagnostics
            system_diag = self.last_diagnostics
            diagnostics = {}
            if system_diag:
                diagnostics = {
                    'cpu_usage': system_diag.cpu_usage,
                    'memory_usage': system_diag.memory_usage,
                    'disk_usage': system_diag.disk_usage,
                    'network_io': system_diag.network_io,
                    'open_files': system_diag.open_files,
                    'process_count': system_diag.process_count,
                    'load_average': system_diag.load_average,
                    'uptime': system_diag.uptime
                }
            
            # Performance trends
            trends = await self._analyze_trends()
            
            return {
                **summary,
                'components': component_details,
                'alerts': alert_details,
                'system_diagnostics': diagnostics,
                'performance_trends': trends,
                'recommendations': await self._generate_recommendations()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get comprehensive health: {e}")
            return {'error': str(e)}
    
    async def get_component_health(self, component_name: str) -> Optional[Dict[str, Any]]:
        """
        Get health information for a specific component
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component health information or None
        """
        health = self.component_health.get(component_name)
        if not health:
            return None
        
        return {
            'name': health.name,
            'status': health.status.value,
            'last_check': health.last_check.isoformat(),
            'error_count': health.error_count,
            'last_error': health.last_error,
            'uptime': health.uptime,
            'availability': health.availability,
            'metrics': [
                {
                    'name': m.name,
                    'value': m.value,
                    'unit': m.unit,
                    'status': m.status.value,
                    'threshold_warning': m.threshold_warning,
                    'threshold_critical': m.threshold_critical,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in health.metrics
            ]
        }
    
    async def get_alerts(self, level: Optional[AlertLevel] = None, 
                        component: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get system alerts
        
        Args:
            level: Filter by alert level
            component: Filter by component
            
        Returns:
            List of alerts
        """
        alerts = list(self.active_alerts.values())
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if component:
            alerts = [a for a in alerts if a.component == component]
        
        return [
            {
                'id': alert.id,
                'level': alert.level.value,
                'component': alert.component,
                'message': alert.message,
                'details': alert.details,
                'timestamp': alert.timestamp.isoformat(),
                'resolved': alert.resolved
            }
            for alert in alerts
        ]
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an active alert
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            True if resolved successfully
        """
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolution_time = datetime.now()
                
                # Move to resolved alerts
                self.resolved_alerts.append(alert)
                del self.active_alerts[alert_id]
                
                self.logger.info(f"Alert resolved: {alert_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False
    
    # Private methods
    
    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply configuration settings"""
        self.check_interval = config.get('check_interval', 10.0)
        self.metric_retention_hours = config.get('metric_retention_hours', 24)
        self.alert_cooldown = config.get('alert_cooldown', 300)
        self.auto_recovery_enabled = config.get('auto_recovery_enabled', True)
        
        # Update thresholds
        custom_thresholds = config.get('thresholds', {})
        for metric, thresholds in custom_thresholds.items():
            if metric in self.default_thresholds:
                self.default_thresholds[metric].update(thresholds)
            else:
                self.default_thresholds[metric] = thresholds
    
    async def _register_default_checkers(self) -> None:
        """Register default health checkers"""
        # System health checker
        async def system_health_check():
            diagnostics = await self._collect_system_diagnostics()
            metrics = [
                HealthMetric("cpu_usage", diagnostics.cpu_usage, "%", 
                           self.default_thresholds['cpu_usage']['warning'],
                           self.default_thresholds['cpu_usage']['critical']),
                HealthMetric("memory_usage", diagnostics.memory_usage, "%",
                           self.default_thresholds['memory_usage']['warning'],
                           self.default_thresholds['memory_usage']['critical']),
                HealthMetric("disk_usage", diagnostics.disk_usage, "%",
                           self.default_thresholds['disk_usage']['warning'],
                           self.default_thresholds['disk_usage']['critical'])
            ]
            
            # Determine status
            if diagnostics.cpu_usage > 95 or diagnostics.memory_usage > 95:
                status = HealthStatus.CRITICAL
            elif diagnostics.cpu_usage > 80 or diagnostics.memory_usage > 85:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return status, metrics, None
        
        self.health_checkers['system'] = system_health_check
    
    async def _register_core_components(self) -> None:
        """Register core OLYMPUS components"""
        core_components = [
            'orchestrator',
            'module_manager', 
            'consciousness_kernel',
            'identity_manager',
            'config_manager',
            'system_health'
        ]
        
        for component in core_components:
            await self.register_component(component)
    
    async def _start_monitoring(self) -> None:
        """Start health monitoring tasks"""
        self._monitoring_active = True
        
        # Health check task
        task = asyncio.create_task(self._health_check_loop())
        self._monitoring_tasks.append(task)
        
        # Metric collection task
        task = asyncio.create_task(self._metric_collection_loop())
        self._monitoring_tasks.append(task)
        
        # Alert management task
        task = asyncio.create_task(self._alert_management_loop())
        self._monitoring_tasks.append(task)
        
        self.logger.info("Health monitoring started")
    
    async def _health_check_loop(self) -> None:
        """Main health check monitoring loop"""
        while self._monitoring_active and not self._shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _metric_collection_loop(self) -> None:
        """System metrics collection loop"""
        while self._monitoring_active and not self._shutdown_event.is_set():
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(60.0)  # Collect system metrics every minute
            except Exception as e:
                self.logger.error(f"Metric collection loop error: {e}")
                await asyncio.sleep(60.0)
    
    async def _alert_management_loop(self) -> None:
        """Alert management and cleanup loop"""
        while self._monitoring_active and not self._shutdown_event.is_set():
            try:
                await self._cleanup_old_alerts()
                await asyncio.sleep(300.0)  # Run every 5 minutes
            except Exception as e:
                self.logger.error(f"Alert management loop error: {e}")
                await asyncio.sleep(300.0)
    
    async def _perform_full_health_check(self) -> None:
        """Perform comprehensive health check"""
        await self._perform_health_checks()
        await self._collect_system_metrics()
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks for all components"""
        for component_name in self.component_health.keys():
            try:
                await self._check_component_health(component_name)
            except Exception as e:
                self.logger.error(f"Health check failed for {component_name}: {e}")
                
                # Update component with error status
                await self.update_component_health(
                    component_name, 
                    HealthStatus.CRITICAL,
                    error_info=str(e)
                )
    
    async def _check_component_health(self, component_name: str) -> None:
        """Check health for a specific component"""
        if component_name in self.health_checkers:
            checker = self.health_checkers[component_name]
            status, metrics, error_info = await checker()
            await self.update_component_health(component_name, status, metrics, error_info)
        else:
            # Generic health check - assume healthy if no errors
            component = self.component_health[component_name]
            if component.error_count == 0:
                await self.update_component_health(component_name, HealthStatus.HEALTHY)
    
    async def _collect_system_diagnostics(self) -> SystemDiagnostics:
        """Collect system diagnostic information"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # System info
            open_files = len(psutil.Process().open_files())
            process_count = len(psutil.pids())
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = (datetime.now() - boot_time).total_seconds()
            
            diagnostics = SystemDiagnostics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                open_files=open_files,
                process_count=process_count,
                load_average=load_avg,
                boot_time=boot_time,
                uptime=uptime
            )
            
            self.last_diagnostics = diagnostics
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system diagnostics: {e}")
            # Return default diagnostics
            return SystemDiagnostics(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                open_files=0,
                process_count=0,
                load_average=[0, 0, 0],
                boot_time=datetime.now(),
                uptime=0.0
            )
    
    async def _collect_system_metrics(self) -> None:
        """Collect and store system metrics"""
        diagnostics = await self._collect_system_diagnostics()
        
        # Store system metrics
        metric_entry = {
            'timestamp': datetime.now(),
            'cpu_usage': diagnostics.cpu_usage,
            'memory_usage': diagnostics.memory_usage,
            'disk_usage': diagnostics.disk_usage,
            'load_average': diagnostics.load_average[0] if diagnostics.load_average else 0
        }
        
        self.system_metrics.append(metric_entry)
    
    async def _update_component_availability(self, component: ComponentHealth) -> None:
        """Update component uptime and availability"""
        current_time = datetime.now()
        
        if component.status == HealthStatus.HEALTHY:
            # Component is up
            if component.uptime == 0:
                # First time healthy
                component.uptime = (current_time - self.system_start_time).total_seconds()
            else:
                component.uptime += self.check_interval
        
        # Calculate availability (simple approach)
        total_time = (current_time - self.system_start_time).total_seconds()
        if total_time > 0:
            component.availability = min(100.0, (component.uptime / total_time) * 100)
    
    async def _check_status_change_alerts(self, component_name: str, 
                                        old_status: HealthStatus, 
                                        new_status: HealthStatus) -> None:
        """Check for alerts based on status changes"""
        if old_status == new_status:
            return
        
        alert_level = AlertLevel.INFO
        message = f"Component {component_name} status changed from {old_status.value} to {new_status.value}"
        
        if new_status == HealthStatus.CRITICAL:
            alert_level = AlertLevel.CRITICAL
        elif new_status == HealthStatus.DEGRADED:
            alert_level = AlertLevel.WARNING
        elif new_status == HealthStatus.HEALTHY and old_status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
            alert_level = AlertLevel.INFO
            message = f"Component {component_name} recovered to healthy status"
        
        await self._create_alert(alert_level, component_name, message, {
            'old_status': old_status.value,
            'new_status': new_status.value
        })
    
    async def _check_metric_alerts(self, component_name: str, metrics: List[HealthMetric]) -> None:
        """Check metrics for threshold violations"""
        for metric in metrics:
            if metric.threshold_critical and metric.value >= metric.threshold_critical:
                await self._create_alert(
                    AlertLevel.CRITICAL,
                    component_name,
                    f"Critical threshold exceeded for {metric.name}: {metric.value}{metric.unit}",
                    {'metric': metric.name, 'value': metric.value, 'threshold': metric.threshold_critical}
                )
            elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                await self._create_alert(
                    AlertLevel.WARNING,
                    component_name,
                    f"Warning threshold exceeded for {metric.name}: {metric.value}{metric.unit}",
                    {'metric': metric.name, 'value': metric.value, 'threshold': metric.threshold_warning}
                )
    
    async def _create_alert(self, level: AlertLevel, component: str, 
                          message: str, details: Dict[str, Any]) -> None:
        """Create a new health alert"""
        try:
            alert_key = f"{component}:{level.value}:{message}"
            
            # Check cooldown period
            if alert_key in self._last_alert_times:
                time_since_last = datetime.now() - self._last_alert_times[alert_key]
                if time_since_last.total_seconds() < self.alert_cooldown:
                    return  # Skip duplicate alert
            
            alert_id = f"alert_{int(time.time())}_{len(self.active_alerts)}"
            alert = HealthAlert(
                id=alert_id,
                level=level,
                component=component,
                message=message,
                details=details
            )
            
            self.active_alerts[alert_id] = alert
            self._last_alert_times[alert_key] = datetime.now()
            
            # Log alert
            log_level = logging.CRITICAL if level == AlertLevel.CRITICAL else logging.WARNING
            self.logger.log(log_level, f"HEALTH ALERT [{level.value.upper()}] {component}: {message}")
            
        except Exception as e:
            self.logger.error(f"Failed to create alert: {e}")
    
    async def _cleanup_old_alerts(self) -> None:
        """Clean up old resolved alerts"""
        try:
            # Remove resolved alerts older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            # Clean up resolved alerts deque
            while self.resolved_alerts and self.resolved_alerts[0].resolution_time < cutoff_time:
                self.resolved_alerts.popleft()
            
            # Clean up alert cooldown times
            expired_keys = []
            for alert_key, timestamp in self._last_alert_times.items():
                if datetime.now() - timestamp > timedelta(seconds=self.alert_cooldown * 2):
                    expired_keys.append(alert_key)
            
            for key in expired_keys:
                del self._last_alert_times[key]
            
        except Exception as e:
            self.logger.error(f"Alert cleanup error: {e}")
    
    async def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends"""
        trends = {}
        
        try:
            # Analyze system metrics trends
            if len(self.system_metrics) >= self.trend_analysis_window:
                recent_metrics = list(self.system_metrics)[-self.trend_analysis_window:]
                
                cpu_values = [m['cpu_usage'] for m in recent_metrics]
                memory_values = [m['memory_usage'] for m in recent_metrics]
                
                trends['cpu_usage'] = {
                    'trend': 'increasing' if cpu_values[-1] > cpu_values[0] else 'decreasing',
                    'average': statistics.mean(cpu_values),
                    'slope': (cpu_values[-1] - cpu_values[0]) / len(cpu_values)
                }
                
                trends['memory_usage'] = {
                    'trend': 'increasing' if memory_values[-1] > memory_values[0] else 'decreasing',
                    'average': statistics.mean(memory_values),
                    'slope': (memory_values[-1] - memory_values[0]) / len(memory_values)
                }
            
        except Exception as e:
            self.logger.error(f"Trend analysis error: {e}")
            trends['error'] = str(e)
        
        return trends
    
    async def _generate_recommendations(self) -> List[str]:
        """Generate health recommendations"""
        recommendations = []
        
        try:
            # System resource recommendations
            if self.last_diagnostics:
                if self.last_diagnostics.cpu_usage > 80:
                    recommendations.append("High CPU usage detected. Consider optimizing processes or adding resources.")
                
                if self.last_diagnostics.memory_usage > 85:
                    recommendations.append("High memory usage detected. Check for memory leaks or increase available memory.")
                
                if self.last_diagnostics.disk_usage > 80:
                    recommendations.append("High disk usage detected. Clean up unnecessary files or expand storage.")
            
            # Component-specific recommendations
            critical_components = [c for c in self.component_health.values() 
                                 if c.status == HealthStatus.CRITICAL]
            if critical_components:
                recommendations.append(f"Critical components detected: {[c.name for c in critical_components]}. Immediate attention required.")
            
            degraded_components = [c for c in self.component_health.values() 
                                 if c.status == HealthStatus.DEGRADED]
            if degraded_components:
                recommendations.append(f"Degraded components detected: {[c.name for c in degraded_components]}. Monitor closely.")
            
            # Alert-based recommendations
            critical_alerts = [a for a in self.active_alerts.values() 
                             if a.level == AlertLevel.CRITICAL]
            if critical_alerts:
                recommendations.append(f"Critical alerts active: {len(critical_alerts)}. Address immediately.")
            
            if not recommendations:
                recommendations.append("System health is good. Continue monitoring.")
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}")
            recommendations.append("Unable to generate recommendations due to error.")
        
        return recommendations
    
    async def _generate_shutdown_report(self) -> None:
        """Generate a final health report during shutdown"""
        try:
            report = await self.get_comprehensive_health()
            report['shutdown_time'] = datetime.now().isoformat()
            
            # Save to file
            report_path = Path("logs/health_shutdown_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Shutdown health report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate shutdown report: {e}")