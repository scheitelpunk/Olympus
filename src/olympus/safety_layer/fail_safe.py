"""
Fail-Safe Mechanisms System

Multi-layered fail-safe system providing:
- Hardware-level emergency stops
- Software watchdog timers
- Communication loss handling
- Power failure protection
- Sensor failure detection
- Graceful degradation strategies
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Set
import threading
import time
from datetime import datetime, timedelta
import logging
import queue
import weakref

logger = logging.getLogger(__name__)


class FailSafeType(Enum):
    """Types of fail-safe mechanisms"""
    HARDWARE_ESTOP = "hardware_estop"
    SOFTWARE_WATCHDOG = "software_watchdog"
    COMMUNICATION_TIMEOUT = "communication_timeout"
    SENSOR_FAILURE = "sensor_failure"
    POWER_FAILURE = "power_failure"
    MOTION_LIMITS = "motion_limits"
    FORCE_LIMITS = "force_limits"
    THERMAL_PROTECTION = "thermal_protection"
    HUMAN_SAFETY = "human_safety"
    SYSTEM_HEALTH = "system_health"


class FailSafeState(Enum):
    """Fail-safe mechanism states"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRIGGERED = "triggered"
    DISABLED = "disabled"
    ERROR = "error"


class FailSafePriority(Enum):
    """Priority levels for fail-safe mechanisms"""
    CRITICAL = 0    # Immediate system shutdown
    HIGH = 1        # Stop current operation
    MEDIUM = 2      # Reduce capabilities
    LOW = 3         # Warning/monitoring only


@dataclass
class FailSafeEvent:
    """Fail-safe event record"""
    mechanism_id: str
    fail_safe_type: FailSafeType
    priority: FailSafePriority
    description: str
    triggered_by: str
    timestamp: datetime
    recovery_actions: List[str]
    system_state: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mechanism_id': self.mechanism_id,
            'fail_safe_type': self.fail_safe_type.value,
            'priority': self.priority.value,
            'description': self.description,
            'triggered_by': self.triggered_by,
            'timestamp': self.timestamp.isoformat(),
            'recovery_actions': self.recovery_actions,
            'system_state': self.system_state
        }


@dataclass
class WatchdogConfig:
    """Configuration for watchdog timer"""
    timeout_seconds: float
    reset_interval: float
    max_missed_resets: int = 3
    auto_recovery: bool = False


class FailSafeMechanism:
    """Individual fail-safe mechanism"""
    
    def __init__(self, 
                 mechanism_id: str,
                 fail_safe_type: FailSafeType,
                 priority: FailSafePriority,
                 check_function: Callable[[], bool],
                 recovery_function: Optional[Callable[[], bool]] = None):
        """
        Initialize fail-safe mechanism
        
        Args:
            mechanism_id: Unique identifier
            fail_safe_type: Type of fail-safe
            priority: Priority level
            check_function: Function to check if fail-safe should trigger
            recovery_function: Optional function to attempt recovery
        """
        self.mechanism_id = mechanism_id
        self.fail_safe_type = fail_safe_type
        self.priority = priority
        self.check_function = check_function
        self.recovery_function = recovery_function
        
        self.state = FailSafeState.ACTIVE
        self.last_check = datetime.utcnow()
        self.trigger_count = 0
        self.last_trigger = None
        self.enabled = True
        
        self._lock = threading.Lock()
    
    def check(self) -> Optional[FailSafeEvent]:
        """Check if fail-safe should trigger"""
        with self._lock:
            if not self.enabled or self.state == FailSafeState.DISABLED:
                return None
            
            try:
                should_trigger = self.check_function()
                self.last_check = datetime.utcnow()
                
                if should_trigger and self.state != FailSafeState.TRIGGERED:
                    return self._trigger()
                elif not should_trigger and self.state == FailSafeState.TRIGGERED:
                    self._reset()
                
                return None
                
            except Exception as e:
                logger.error(f"Error in fail-safe check {self.mechanism_id}: {str(e)}")
                self.state = FailSafeState.ERROR
                return FailSafeEvent(
                    mechanism_id=self.mechanism_id,
                    fail_safe_type=self.fail_safe_type,
                    priority=FailSafePriority.HIGH,
                    description=f"Fail-safe check error: {str(e)}",
                    triggered_by="check_error",
                    timestamp=datetime.utcnow(),
                    recovery_actions=["Investigate fail-safe mechanism", "Reset if safe"],
                    system_state={'error': str(e)}
                )
    
    def _trigger(self) -> FailSafeEvent:
        """Trigger the fail-safe mechanism"""
        self.state = FailSafeState.TRIGGERED
        self.trigger_count += 1
        self.last_trigger = datetime.utcnow()
        
        logger.warning(f"Fail-safe triggered: {self.mechanism_id} ({self.fail_safe_type.value})")
        
        recovery_actions = self._get_recovery_actions()
        
        return FailSafeEvent(
            mechanism_id=self.mechanism_id,
            fail_safe_type=self.fail_safe_type,
            priority=self.priority,
            description=f"Fail-safe mechanism triggered: {self.mechanism_id}",
            triggered_by="safety_check",
            timestamp=self.last_trigger,
            recovery_actions=recovery_actions,
            system_state=self._get_system_state()
        )
    
    def _reset(self):
        """Reset the fail-safe mechanism"""
        if self.state == FailSafeState.TRIGGERED:
            self.state = FailSafeState.ACTIVE
            logger.info(f"Fail-safe reset: {self.mechanism_id}")
    
    def attempt_recovery(self) -> bool:
        """Attempt to recover from fail-safe state"""
        if self.recovery_function and self.state == FailSafeState.TRIGGERED:
            try:
                if self.recovery_function():
                    self._reset()
                    return True
            except Exception as e:
                logger.error(f"Recovery failed for {self.mechanism_id}: {str(e)}")
        
        return False
    
    def _get_recovery_actions(self) -> List[str]:
        """Get recommended recovery actions"""
        actions = {
            FailSafeType.HARDWARE_ESTOP: [
                "Check emergency stop button status",
                "Verify hardware connections",
                "Reset emergency stop circuit"
            ],
            FailSafeType.SOFTWARE_WATCHDOG: [
                "Check system responsiveness",
                "Restart watchdog timer",
                "Investigate system slowdown"
            ],
            FailSafeType.COMMUNICATION_TIMEOUT: [
                "Check network connectivity",
                "Restart communication systems",
                "Verify remote system status"
            ],
            FailSafeType.SENSOR_FAILURE: [
                "Check sensor connections",
                "Calibrate sensors",
                "Switch to backup sensors"
            ],
            FailSafeType.POWER_FAILURE: [
                "Check power supply status",
                "Switch to backup power",
                "Perform graceful shutdown if necessary"
            ]
        }
        
        return actions.get(self.fail_safe_type, ["Investigate and resolve issue", "Reset mechanism"])
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state for logging"""
        return {
            'mechanism_state': self.state.value,
            'trigger_count': self.trigger_count,
            'last_check': self.last_check.isoformat(),
            'enabled': self.enabled
        }
    
    def enable(self):
        """Enable the fail-safe mechanism"""
        with self._lock:
            self.enabled = True
            if self.state == FailSafeState.DISABLED:
                self.state = FailSafeState.ACTIVE
    
    def disable(self):
        """Disable the fail-safe mechanism (use with caution!)"""
        with self._lock:
            self.enabled = False
            self.state = FailSafeState.DISABLED
    
    def get_status(self) -> Dict[str, Any]:
        """Get mechanism status"""
        with self._lock:
            return {
                'mechanism_id': self.mechanism_id,
                'type': self.fail_safe_type.value,
                'priority': self.priority.value,
                'state': self.state.value,
                'enabled': self.enabled,
                'trigger_count': self.trigger_count,
                'last_check': self.last_check.isoformat(),
                'last_trigger': self.last_trigger.isoformat() if self.last_trigger else None
            }


class FailSafeManager:
    """Central manager for all fail-safe mechanisms"""
    
    def __init__(self, check_interval: float = 0.1):  # 100ms default
        """
        Initialize fail-safe manager
        
        Args:
            check_interval: Interval between fail-safe checks in seconds
        """
        self.check_interval = check_interval
        self.mechanisms: Dict[str, FailSafeMechanism] = {}
        self.event_handlers: Dict[FailSafePriority, List[Callable]] = {
            priority: [] for priority in FailSafePriority
        }
        self.event_history: List[FailSafeEvent] = []
        
        self._running = False
        self._check_thread = None
        self._event_queue = queue.Queue()
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'total_triggers': 0,
            'successful_recoveries': 0,
            'active_mechanisms': 0
        }
        
        logger.info(f"FailSafeManager initialized with {check_interval}s check interval")
    
    def register_mechanism(self, mechanism: FailSafeMechanism) -> bool:
        """
        Register a fail-safe mechanism
        
        Args:
            mechanism: Fail-safe mechanism to register
            
        Returns:
            True if registered successfully
        """
        with self._lock:
            if mechanism.mechanism_id in self.mechanisms:
                logger.warning(f"Mechanism {mechanism.mechanism_id} already registered")
                return False
            
            self.mechanisms[mechanism.mechanism_id] = mechanism
            self.stats['active_mechanisms'] += 1
            
            logger.info(f"Registered fail-safe mechanism: {mechanism.mechanism_id} "
                       f"({mechanism.fail_safe_type.value}, priority: {mechanism.priority.value})")
            return True
    
    def unregister_mechanism(self, mechanism_id: str) -> bool:
        """
        Unregister a fail-safe mechanism
        
        Args:
            mechanism_id: ID of mechanism to unregister
            
        Returns:
            True if unregistered successfully
        """
        with self._lock:
            if mechanism_id in self.mechanisms:
                del self.mechanisms[mechanism_id]
                self.stats['active_mechanisms'] -= 1
                logger.info(f"Unregistered fail-safe mechanism: {mechanism_id}")
                return True
            return False
    
    def register_event_handler(self, priority: FailSafePriority, handler: Callable[[FailSafeEvent], None]):
        """
        Register event handler for specific priority level
        
        Args:
            priority: Priority level to handle
            handler: Handler function
        """
        self.event_handlers[priority].append(handler)
        logger.info(f"Registered event handler for {priority.name} priority events")
    
    def start_monitoring(self):
        """Start fail-safe monitoring"""
        if self._running:
            logger.warning("Fail-safe monitoring already running")
            return
        
        self._running = True
        self._check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._check_thread.start()
        
        logger.info("Fail-safe monitoring started")
    
    def stop_monitoring(self):
        """Stop fail-safe monitoring"""
        self._running = False
        if self._check_thread and self._check_thread.is_alive():
            self._check_thread.join(timeout=1.0)
        
        logger.info("Fail-safe monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                self._check_all_mechanisms()
                self._process_events()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in fail-safe monitoring loop: {str(e)}")
                time.sleep(self.check_interval * 2)  # Slower retry on error
    
    def _check_all_mechanisms(self):
        """Check all registered mechanisms"""
        with self._lock:
            mechanisms_list = list(self.mechanisms.values())
        
        for mechanism in mechanisms_list:
            self.stats['total_checks'] += 1
            event = mechanism.check()
            
            if event:
                self.stats['total_triggers'] += 1
                self._event_queue.put(event)
                self.event_history.append(event)
                
                # Limit event history
                if len(self.event_history) > 1000:
                    self.event_history = self.event_history[-500:]
    
    def _process_events(self):
        """Process events from the queue"""
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                self._handle_event(event)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing fail-safe event: {str(e)}")
    
    def _handle_event(self, event: FailSafeEvent):
        """Handle a fail-safe event"""
        logger.warning(f"Processing fail-safe event: {event.mechanism_id} "
                      f"(priority: {event.priority.name})")
        
        # Call registered handlers for this priority level
        handlers = self.event_handlers.get(event.priority, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in fail-safe event handler: {str(e)}")
    
    def trigger_manual_failsafe(self, mechanism_id: str, reason: str) -> bool:
        """
        Manually trigger a specific fail-safe mechanism
        
        Args:
            mechanism_id: ID of mechanism to trigger
            reason: Reason for manual trigger
            
        Returns:
            True if triggered successfully
        """
        with self._lock:
            mechanism = self.mechanisms.get(mechanism_id)
            if not mechanism:
                logger.error(f"Cannot trigger unknown mechanism: {mechanism_id}")
                return False
        
        # Create manual trigger event
        event = FailSafeEvent(
            mechanism_id=mechanism_id,
            fail_safe_type=mechanism.fail_safe_type,
            priority=mechanism.priority,
            description=f"Manual fail-safe trigger: {reason}",
            triggered_by="manual_trigger",
            timestamp=datetime.utcnow(),
            recovery_actions=["Investigate manual trigger reason", "Reset when safe"],
            system_state={'manual_trigger': True, 'reason': reason}
        )
        
        # Force mechanism into triggered state
        mechanism.state = FailSafeState.TRIGGERED
        mechanism.trigger_count += 1
        mechanism.last_trigger = datetime.utcnow()
        
        self._event_queue.put(event)
        self.event_history.append(event)
        
        logger.warning(f"Manually triggered fail-safe: {mechanism_id} - {reason}")
        return True
    
    def attempt_recovery(self, mechanism_id: str) -> bool:
        """
        Attempt recovery for a specific mechanism
        
        Args:
            mechanism_id: ID of mechanism to recover
            
        Returns:
            True if recovery successful
        """
        with self._lock:
            mechanism = self.mechanisms.get(mechanism_id)
            if not mechanism:
                return False
        
        success = mechanism.attempt_recovery()
        if success:
            self.stats['successful_recoveries'] += 1
            logger.info(f"Successfully recovered fail-safe: {mechanism_id}")
        else:
            logger.warning(f"Failed to recover fail-safe: {mechanism_id}")
        
        return success
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall fail-safe system status"""
        with self._lock:
            mechanisms_status = {
                mid: mech.get_status() for mid, mech in self.mechanisms.items()
            }
        
        active_count = sum(1 for status in mechanisms_status.values() if status['state'] == 'active')
        triggered_count = sum(1 for status in mechanisms_status.values() if status['state'] == 'triggered')
        
        return {
            'monitoring_active': self._running,
            'total_mechanisms': len(self.mechanisms),
            'active_mechanisms': active_count,
            'triggered_mechanisms': triggered_count,
            'recent_events': len([e for e in self.event_history 
                                 if (datetime.utcnow() - e.timestamp).total_seconds() < 300]),  # Last 5 minutes
            'statistics': self.stats.copy(),
            'mechanisms': mechanisms_status
        }
    
    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent fail-safe events"""
        recent = self.event_history[-count:] if self.event_history else []
        return [event.to_dict() for event in reversed(recent)]
    
    def reset_all_mechanisms(self, force: bool = False) -> Dict[str, bool]:
        """
        Reset all fail-safe mechanisms
        
        Args:
            force: Force reset even if unsafe
            
        Returns:
            Dictionary of mechanism_id -> success status
        """
        results = {}
        
        with self._lock:
            mechanisms_list = list(self.mechanisms.items())
        
        for mechanism_id, mechanism in mechanisms_list:
            if mechanism.state == FailSafeState.TRIGGERED:
                if force:
                    mechanism._reset()
                    results[mechanism_id] = True
                    logger.warning(f"Force reset fail-safe: {mechanism_id}")
                else:
                    success = mechanism.attempt_recovery()
                    results[mechanism_id] = success
                    if success:
                        self.stats['successful_recoveries'] += 1
            else:
                results[mechanism_id] = True
        
        logger.info(f"Reset {len(mechanisms_list)} fail-safe mechanisms")
        return results
    
    def emergency_shutdown(self, reason: str):
        """
        Trigger emergency shutdown of all systems
        
        Args:
            reason: Reason for emergency shutdown
        """
        logger.critical(f"EMERGENCY SHUTDOWN TRIGGERED: {reason}")
        
        # Trigger all critical fail-safes
        with self._lock:
            critical_mechanisms = [
                (mid, mech) for mid, mech in self.mechanisms.items()
                if mech.priority == FailSafePriority.CRITICAL
            ]
        
        for mechanism_id, mechanism in critical_mechanisms:
            self.trigger_manual_failsafe(mechanism_id, f"Emergency shutdown: {reason}")
        
        # Create emergency event
        emergency_event = FailSafeEvent(
            mechanism_id="system_emergency",
            fail_safe_type=FailSafeType.SYSTEM_HEALTH,
            priority=FailSafePriority.CRITICAL,
            description=f"System emergency shutdown: {reason}",
            triggered_by="emergency_protocol",
            timestamp=datetime.utcnow(),
            recovery_actions=["Investigate emergency cause", "System safety check", "Manual restart"],
            system_state={'emergency': True, 'reason': reason}
        )
        
        self.event_history.append(emergency_event)
        self._event_queue.put(emergency_event)
    
    def create_standard_mechanisms(self) -> List[str]:
        """
        Create standard fail-safe mechanisms
        
        Returns:
            List of created mechanism IDs
        """
        mechanisms = []
        
        # Hardware E-Stop mechanism
        def check_estop():
            # This would interface with actual hardware
            return False  # Placeholder
        
        estop_mechanism = FailSafeMechanism(
            mechanism_id="hardware_estop",
            fail_safe_type=FailSafeType.HARDWARE_ESTOP,
            priority=FailSafePriority.CRITICAL,
            check_function=check_estop
        )
        
        if self.register_mechanism(estop_mechanism):
            mechanisms.append("hardware_estop")
        
        # Communication timeout mechanism
        def check_comm_timeout():
            # This would check actual communication status
            return False  # Placeholder
        
        comm_mechanism = FailSafeMechanism(
            mechanism_id="communication_timeout",
            fail_safe_type=FailSafeType.COMMUNICATION_TIMEOUT,
            priority=FailSafePriority.HIGH,
            check_function=check_comm_timeout
        )
        
        if self.register_mechanism(comm_mechanism):
            mechanisms.append("communication_timeout")
        
        logger.info(f"Created {len(mechanisms)} standard fail-safe mechanisms")
        return mechanisms
    
    def __del__(self):
        """Cleanup when manager is destroyed"""
        self.stop_monitoring()