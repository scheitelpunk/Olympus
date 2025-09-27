"""
Comprehensive Audit Logger for Safety Events

Complete logging system for safety-related events and actions:
- Structured safety event logging
- Real-time monitoring and alerts
- Compliance and regulatory reporting
- Forensic analysis capabilities
- Data integrity and tamper detection
"""

import json
import logging
import hashlib
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Iterator
from pathlib import Path
import sqlite3
import queue
import time

logger = logging.getLogger(__name__)


class SafetyEventType(Enum):
    """Types of safety events"""
    ACTION_FILTERED = "action_filtered"
    INTENTION_ANALYZED = "intention_analyzed"
    RISK_ASSESSED = "risk_assessed"
    HUMAN_DETECTED = "human_detected"
    PROXIMITY_ALERT = "proximity_alert"
    FAILSAFE_TRIGGERED = "failsafe_triggered"
    EMERGENCY_STOP = "emergency_stop"
    SYSTEM_START = "system_start"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIGURATION_CHANGED = "configuration_changed"
    SAFETY_VIOLATION = "safety_violation"
    RECOVERY_ATTEMPTED = "recovery_attempted"


class EventSeverity(Enum):
    """Severity levels for safety events"""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class SafetyEvent:
    """Structured safety event record"""
    event_id: str
    event_type: SafetyEventType
    severity: EventSeverity
    timestamp: datetime
    component: str
    description: str
    data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'description': self.description,
            'data': self.data,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'correlation_id': self.correlation_id,
            'tags': self.tags
        }
    
    def calculate_hash(self) -> str:
        """Calculate hash for integrity verification"""
        event_str = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(event_str.encode()).hexdigest()


@dataclass
class AuditConfiguration:
    """Configuration for audit logging"""
    log_directory: str = "logs/safety_audit"
    database_path: str = "logs/safety_audit/audit.db"
    max_log_size_mb: int = 100
    log_retention_days: int = 365
    real_time_alerts: bool = True
    integrity_checking: bool = True
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    encryption_enabled: bool = False
    
    # Performance settings
    batch_size: int = 100
    flush_interval_seconds: float = 5.0
    max_queue_size: int = 10000


class DatabaseManager:
    """Manages SQLite database for audit logs"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_database()
        self._lock = threading.Lock()
    
    def _ensure_db_directory(self):
        """Ensure database directory exists"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS safety_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    component TEXT NOT NULL,
                    description TEXT NOT NULL,
                    data TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    correlation_id TEXT,
                    tags TEXT,
                    hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON safety_events(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type 
                ON safety_events(event_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_severity 
                ON safety_events(severity)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_component 
                ON safety_events(component)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Store database schema version
            conn.execute("""
                INSERT OR REPLACE INTO audit_metadata (key, value, updated_at)
                VALUES ('schema_version', '1.0', ?)
            """, (datetime.utcnow().isoformat(),))
            
            conn.commit()
    
    def insert_event(self, event: SafetyEvent) -> bool:
        """Insert safety event into database"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO safety_events (
                            event_id, event_type, severity, timestamp, component,
                            description, data, user_id, session_id, correlation_id,
                            tags, hash, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.event_type.value,
                        event.severity.value,
                        event.timestamp.isoformat(),
                        event.component,
                        event.description,
                        json.dumps(event.data),
                        event.user_id,
                        event.session_id,
                        event.correlation_id,
                        json.dumps(event.tags) if event.tags else None,
                        event.calculate_hash(),
                        datetime.utcnow().isoformat()
                    ))
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to insert audit event: {str(e)}")
            return False
    
    def insert_events_batch(self, events: List[SafetyEvent]) -> int:
        """Insert multiple events in batch"""
        success_count = 0
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    for event in events:
                        try:
                            conn.execute("""
                                INSERT INTO safety_events (
                                    event_id, event_type, severity, timestamp, component,
                                    description, data, user_id, session_id, correlation_id,
                                    tags, hash, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                event.event_id,
                                event.event_type.value,
                                event.severity.value,
                                event.timestamp.isoformat(),
                                event.component,
                                event.description,
                                json.dumps(event.data),
                                event.user_id,
                                event.session_id,
                                event.correlation_id,
                                json.dumps(event.tags) if event.tags else None,
                                event.calculate_hash(),
                                datetime.utcnow().isoformat()
                            ))
                            success_count += 1
                        except Exception as e:
                            logger.error(f"Failed to insert event {event.event_id}: {str(e)}")
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert event batch: {str(e)}")
        
        return success_count
    
    def query_events(self, 
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    event_types: Optional[List[SafetyEventType]] = None,
                    severities: Optional[List[EventSeverity]] = None,
                    components: Optional[List[str]] = None,
                    limit: int = 1000) -> List[Dict[str, Any]]:
        """Query events with filters"""
        
        conditions = []
        params = []
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())
        
        if event_types:
            placeholders = ','.join('?' for _ in event_types)
            conditions.append(f"event_type IN ({placeholders})")
            params.extend([et.value for et in event_types])
        
        if severities:
            placeholders = ','.join('?' for _ in severities)
            conditions.append(f"severity IN ({placeholders})")
            params.extend([s.value for s in severities])
        
        if components:
            placeholders = ','.join('?' for _ in components)
            conditions.append(f"component IN ({placeholders})")
            params.extend(components)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            SELECT event_id, event_type, severity, timestamp, component,
                   description, data, user_id, session_id, correlation_id,
                   tags, hash, created_at
            FROM safety_events
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                events = []
                for row in cursor:
                    event_data = dict(row)
                    # Parse JSON fields
                    if event_data['data']:
                        event_data['data'] = json.loads(event_data['data'])
                    if event_data['tags']:
                        event_data['tags'] = json.loads(event_data['tags'])
                    events.append(event_data)
                
                return events
        except Exception as e:
            logger.error(f"Failed to query events: {str(e)}")
            return []
    
    def cleanup_old_events(self, retention_days: int) -> int:
        """Remove events older than retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        DELETE FROM safety_events 
                        WHERE timestamp < ?
                    """, (cutoff_date.isoformat(),))
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
                    logger.info(f"Cleaned up {deleted_count} old audit events")
                    return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old events: {str(e)}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Total events
                total = conn.execute("SELECT COUNT(*) as count FROM safety_events").fetchone()['count']
                
                # Events by type
                type_stats = {}
                for row in conn.execute("""
                    SELECT event_type, COUNT(*) as count 
                    FROM safety_events 
                    GROUP BY event_type
                """):
                    type_stats[row['event_type']] = row['count']
                
                # Events by severity
                severity_stats = {}
                for row in conn.execute("""
                    SELECT severity, COUNT(*) as count 
                    FROM safety_events 
                    GROUP BY severity
                """):
                    severity_stats[row['severity']] = row['count']
                
                # Recent activity (last 24 hours)
                yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()
                recent_count = conn.execute("""
                    SELECT COUNT(*) as count 
                    FROM safety_events 
                    WHERE timestamp >= ?
                """, (yesterday,)).fetchone()['count']
                
                return {
                    'total_events': total,
                    'events_by_type': type_stats,
                    'events_by_severity': severity_stats,
                    'recent_24h': recent_count,
                    'database_size_mb': Path(self.db_path).stat().st_size / (1024 * 1024)
                }
        except Exception as e:
            logger.error(f"Failed to get database statistics: {str(e)}")
            return {}


class AuditLogger:
    """Main audit logging system"""
    
    def __init__(self, config: Optional[AuditConfiguration] = None):
        """
        Initialize audit logger
        
        Args:
            config: Audit logging configuration
        """
        self.config = config or AuditConfiguration()
        
        # Initialize database
        self.db_manager = DatabaseManager(self.config.database_path)
        
        # Event processing queue
        self._event_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self._processing_thread = None
        self._running = False
        
        # Alert handlers
        self.alert_handlers: Dict[EventSeverity, List[Callable]] = {
            severity: [] for severity in EventSeverity
        }
        
        # Statistics
        self.stats = {
            'events_logged': 0,
            'events_queued': 0,
            'events_dropped': 0,
            'alerts_sent': 0,
            'integrity_checks_passed': 0,
            'integrity_checks_failed': 0
        }
        
        # Ensure log directory exists
        Path(self.config.log_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AuditLogger initialized with database: {self.config.database_path}")
    
    def start(self):
        """Start the audit logging system"""
        if self._running:
            logger.warning("Audit logger already running")
            return
        
        self._running = True
        self._processing_thread = threading.Thread(target=self._process_events, daemon=True)
        self._processing_thread.start()
        
        # Log system start
        self.log_event(
            event_type=SafetyEventType.SYSTEM_START,
            severity=EventSeverity.INFO,
            component="audit_logger",
            description="Audit logging system started",
            data={"config": asdict(self.config)}
        )
        
        logger.info("Audit logging system started")
    
    def stop(self):
        """Stop the audit logging system"""
        if not self._running:
            return
        
        # Log system shutdown
        self.log_event(
            event_type=SafetyEventType.SYSTEM_SHUTDOWN,
            severity=EventSeverity.INFO,
            component="audit_logger",
            description="Audit logging system shutting down",
            data={"final_stats": self.stats.copy()}
        )
        
        self._running = False
        
        # Wait for processing thread to finish
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)
        
        # Process any remaining events
        self._flush_remaining_events()
        
        logger.info("Audit logging system stopped")
    
    def log_event(self, 
                 event_type: SafetyEventType,
                 severity: EventSeverity,
                 component: str,
                 description: str,
                 data: Dict[str, Any],
                 user_id: Optional[str] = None,
                 session_id: Optional[str] = None,
                 correlation_id: Optional[str] = None,
                 tags: Optional[List[str]] = None) -> str:
        """
        Log a safety event
        
        Args:
            event_type: Type of safety event
            severity: Severity level
            component: Component that generated the event
            description: Human-readable description
            data: Event-specific data
            user_id: Optional user identifier
            session_id: Optional session identifier
            correlation_id: Optional correlation identifier for related events
            tags: Optional tags for categorization
            
        Returns:
            Generated event ID
        """
        event_id = self._generate_event_id()
        
        event = SafetyEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            component=component,
            description=description,
            data=data.copy(),
            user_id=user_id,
            session_id=session_id,
            correlation_id=correlation_id,
            tags=tags or []
        )
        
        # Try to queue the event
        try:
            self._event_queue.put_nowait(event)
            self.stats['events_queued'] += 1
        except queue.Full:
            # Queue is full, drop the event (but log this fact)
            self.stats['events_dropped'] += 1
            logger.error(f"Audit event queue full, dropped event: {event_id}")
        
        return event_id
    
    def log_action_filtered(self, action: Dict[str, Any], filter_result: Dict[str, Any]):
        """Log action filtering event"""
        self.log_event(
            event_type=SafetyEventType.ACTION_FILTERED,
            severity=EventSeverity.INFO if filter_result.get('status') == 'approved' else EventSeverity.WARNING,
            component="action_filter",
            description=f"Action filtered: {filter_result.get('status', 'unknown')}",
            data={
                'original_action': action,
                'filter_result': filter_result
            }
        )
    
    def log_human_proximity(self, human_data: Dict[str, Any], alert_data: Dict[str, Any]):
        """Log human proximity event"""
        self.log_event(
            event_type=SafetyEventType.PROXIMITY_ALERT,
            severity=EventSeverity.WARNING if alert_data.get('alert_level') in ['danger', 'emergency'] else EventSeverity.INFO,
            component="human_protection",
            description=f"Human proximity alert: {alert_data.get('alert_level', 'unknown')}",
            data={
                'human_data': human_data,
                'alert_data': alert_data
            }
        )
    
    def log_failsafe_trigger(self, mechanism_id: str, event_data: Dict[str, Any]):
        """Log fail-safe mechanism trigger"""
        self.log_event(
            event_type=SafetyEventType.FAILSAFE_TRIGGERED,
            severity=EventSeverity.CRITICAL if event_data.get('priority') == 'critical' else EventSeverity.ERROR,
            component="fail_safe_manager",
            description=f"Fail-safe triggered: {mechanism_id}",
            data={
                'mechanism_id': mechanism_id,
                'event_data': event_data
            }
        )
    
    def log_emergency_stop(self, trigger_reason: str, system_state: Dict[str, Any]):
        """Log emergency stop event"""
        self.log_event(
            event_type=SafetyEventType.EMERGENCY_STOP,
            severity=EventSeverity.CRITICAL,
            component="emergency_system",
            description=f"Emergency stop triggered: {trigger_reason}",
            data={
                'trigger_reason': trigger_reason,
                'system_state': system_state
            },
            tags=['emergency', 'critical_safety']
        )
    
    def log_configuration_change(self, component: str, old_config: Dict[str, Any], new_config: Dict[str, Any], user_id: str):
        """Log configuration change"""
        self.log_event(
            event_type=SafetyEventType.CONFIGURATION_CHANGED,
            severity=EventSeverity.WARNING,
            component=component,
            description=f"Configuration changed for {component}",
            data={
                'old_config': old_config,
                'new_config': new_config,
                'changes': self._diff_configs(old_config, new_config)
            },
            user_id=user_id,
            tags=['configuration', 'audit_trail']
        )
    
    def register_alert_handler(self, severity: EventSeverity, handler: Callable[[SafetyEvent], None]):
        """Register alert handler for specific severity level"""
        self.alert_handlers[severity].append(handler)
        logger.info(f"Registered alert handler for {severity.value} events")
    
    def query_events(self, 
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    event_types: Optional[List[SafetyEventType]] = None,
                    severities: Optional[List[EventSeverity]] = None,
                    components: Optional[List[str]] = None,
                    limit: int = 1000) -> List[Dict[str, Any]]:
        """Query audit events with filters"""
        return self.db_manager.query_events(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types,
            severities=severities,
            components=components,
            limit=limit
        )
    
    def generate_report(self, 
                       start_time: datetime,
                       end_time: datetime,
                       report_type: str = "summary") -> Dict[str, Any]:
        """Generate audit report for time period"""
        events = self.query_events(start_time=start_time, end_time=end_time, limit=10000)
        
        if report_type == "summary":
            return self._generate_summary_report(events, start_time, end_time)
        elif report_type == "detailed":
            return self._generate_detailed_report(events, start_time, end_time)
        else:
            return {"error": f"Unknown report type: {report_type}"}
    
    def verify_integrity(self, start_time: Optional[datetime] = None, limit: int = 1000) -> Dict[str, Any]:
        """Verify integrity of audit logs"""
        events = self.query_events(start_time=start_time, limit=limit)
        
        total_checked = 0
        integrity_failures = []
        
        for event_data in events:
            total_checked += 1
            
            # Reconstruct event and verify hash
            event = SafetyEvent(
                event_id=event_data['event_id'],
                event_type=SafetyEventType(event_data['event_type']),
                severity=EventSeverity(event_data['severity']),
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                component=event_data['component'],
                description=event_data['description'],
                data=event_data['data'],
                user_id=event_data['user_id'],
                session_id=event_data['session_id'],
                correlation_id=event_data['correlation_id'],
                tags=event_data['tags'] or []
            )
            
            calculated_hash = event.calculate_hash()
            stored_hash = event_data['hash']
            
            if calculated_hash != stored_hash:
                integrity_failures.append({
                    'event_id': event.event_id,
                    'timestamp': event.timestamp.isoformat(),
                    'calculated_hash': calculated_hash,
                    'stored_hash': stored_hash
                })
                self.stats['integrity_checks_failed'] += 1
            else:
                self.stats['integrity_checks_passed'] += 1
        
        return {
            'total_checked': total_checked,
            'integrity_failures': len(integrity_failures),
            'failure_details': integrity_failures[:10],  # First 10 failures
            'success_rate': (total_checked - len(integrity_failures)) / total_checked if total_checked > 0 else 1.0
        }
    
    def cleanup_old_logs(self) -> Dict[str, Any]:
        """Clean up old audit logs based on retention policy"""
        deleted_count = self.db_manager.cleanup_old_events(self.config.log_retention_days)
        
        return {
            'deleted_events': deleted_count,
            'retention_days': self.config.log_retention_days,
            'cleanup_time': datetime.utcnow().isoformat()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logger statistics"""
        db_stats = self.db_manager.get_statistics()
        
        return {
            'runtime_stats': self.stats.copy(),
            'database_stats': db_stats,
            'configuration': asdict(self.config),
            'system_status': {
                'running': self._running,
                'queue_size': self._event_queue.qsize(),
                'max_queue_size': self.config.max_queue_size,
                'queue_utilization': self._event_queue.qsize() / self.config.max_queue_size
            }
        }
    
    def _process_events(self):
        """Background thread to process events"""
        batch = []
        last_flush = time.time()
        
        while self._running or not self._event_queue.empty():
            try:
                # Get event with timeout
                try:
                    event = self._event_queue.get(timeout=1.0)
                    batch.append(event)
                except queue.Empty:
                    continue
                
                # Check if we should flush the batch
                current_time = time.time()
                should_flush = (
                    len(batch) >= self.config.batch_size or
                    (current_time - last_flush) >= self.config.flush_interval_seconds
                )
                
                if should_flush and batch:
                    self._process_batch(batch)
                    batch.clear()
                    last_flush = current_time
                
            except Exception as e:
                logger.error(f"Error in event processing: {str(e)}")
                time.sleep(0.1)  # Brief pause on error
        
        # Process any remaining events
        if batch:
            self._process_batch(batch)
    
    def _process_batch(self, events: List[SafetyEvent]):
        """Process a batch of events"""
        # Store in database
        success_count = self.db_manager.insert_events_batch(events)
        self.stats['events_logged'] += success_count
        
        # Send alerts for high-severity events
        for event in events:
            if self.config.real_time_alerts:
                self._send_alerts(event)
    
    def _send_alerts(self, event: SafetyEvent):
        """Send alerts for critical events"""
        handlers = self.alert_handlers.get(event.severity, [])
        
        for handler in handlers:
            try:
                handler(event)
                self.stats['alerts_sent'] += 1
            except Exception as e:
                logger.error(f"Error in alert handler: {str(e)}")
    
    def _flush_remaining_events(self):
        """Flush any remaining events in the queue"""
        remaining_events = []
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                remaining_events.append(event)
            except queue.Empty:
                break
        
        if remaining_events:
            self._process_batch(remaining_events)
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return f"evt_{timestamp}_{hash(threading.current_thread()) & 0xFFFF:04x}"
    
    def _diff_configs(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration differences"""
        changes = {}
        
        # Find changed values
        for key in set(old_config.keys()) | set(new_config.keys()):
            old_val = old_config.get(key)
            new_val = new_config.get(key)
            
            if old_val != new_val:
                changes[key] = {
                    'old': old_val,
                    'new': new_val
                }
        
        return changes
    
    def _generate_summary_report(self, events: List[Dict[str, Any]], 
                                start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate summary audit report"""
        if not events:
            return {
                'report_type': 'summary',
                'time_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'total_events': 0,
                'summary': 'No events found in the specified time period'
            }
        
        # Count events by type and severity
        type_counts = {}
        severity_counts = {}
        component_counts = {}
        
        for event in events:
            event_type = event['event_type']
            severity = event['severity']
            component = event['component']
            
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            component_counts[component] = component_counts.get(component, 0) + 1
        
        # Find most active periods
        hourly_counts = {}
        for event in events:
            hour = datetime.fromisoformat(event['timestamp']).replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        
        peak_hour = max(hourly_counts.keys(), key=lambda h: hourly_counts[h]) if hourly_counts else None
        
        return {
            'report_type': 'summary',
            'time_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration_hours': (end_time - start_time).total_seconds() / 3600
            },
            'total_events': len(events),
            'events_by_type': type_counts,
            'events_by_severity': severity_counts,
            'events_by_component': component_counts,
            'peak_activity': {
                'hour': peak_hour.isoformat() if peak_hour else None,
                'event_count': hourly_counts.get(peak_hour, 0) if peak_hour else 0
            },
            'critical_events': [
                event for event in events 
                if event['severity'] in ['critical', 'fatal']
            ][:10]  # Top 10 critical events
        }
    
    def _generate_detailed_report(self, events: List[Dict[str, Any]], 
                                 start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate detailed audit report"""
        summary = self._generate_summary_report(events, start_time, end_time)
        
        return {
            **summary,
            'report_type': 'detailed',
            'all_events': events,
            'timeline': self._generate_timeline(events),
            'correlations': self._find_event_correlations(events)
        }
    
    def _generate_timeline(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate event timeline"""
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e['timestamp'])
        
        timeline = []
        for event in sorted_events:
            timeline.append({
                'timestamp': event['timestamp'],
                'event_id': event['event_id'],
                'event_type': event['event_type'],
                'severity': event['severity'],
                'component': event['component'],
                'description': event['description']
            })
        
        return timeline
    
    def _find_event_correlations(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find correlated events"""
        correlations = []
        
        # Group by correlation_id
        correlation_groups = {}
        for event in events:
            corr_id = event.get('correlation_id')
            if corr_id:
                if corr_id not in correlation_groups:
                    correlation_groups[corr_id] = []
                correlation_groups[corr_id].append(event)
        
        # Report correlations with multiple events
        for corr_id, corr_events in correlation_groups.items():
            if len(corr_events) > 1:
                correlations.append({
                    'correlation_id': corr_id,
                    'event_count': len(corr_events),
                    'time_span': {
                        'start': min(e['timestamp'] for e in corr_events),
                        'end': max(e['timestamp'] for e in corr_events)
                    },
                    'events': [e['event_id'] for e in corr_events]
                })
        
        return correlations
    
    def __del__(self):
        """Cleanup when logger is destroyed"""
        if self._running:
            self.stop()