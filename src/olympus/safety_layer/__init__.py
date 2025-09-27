"""
OLYMPUS Safety Layer - Multi-Layer Action Filtering and Protection System

This module provides comprehensive safety mechanisms for Project OLYMPUS:
- Multi-layer action filtering
- Force and speed limits enforcement
- Human proximity detection and protection
- Risk assessment and mitigation
- Emergency mechanisms integration
- Complete audit logging

Safety Constraints:
- Maximum force: 20N
- Maximum speed: 1.0 m/s
- Minimum human distance: 1.0m (default)
"""

from .action_filter import ActionFilter, FilterResult
from .intention_analyzer import IntentionAnalyzer, IntentionAssessment
from .risk_assessment import RiskAssessment, RiskLevel
from .human_protection import HumanProtection, ProximityAlert
from .fail_safe import FailSafeManager, FailSafeType
from .audit_logger import AuditLogger, SafetyEvent

__all__ = [
    'ActionFilter',
    'FilterResult',
    'IntentionAnalyzer',
    'IntentionAssessment',
    'RiskAssessment',
    'RiskLevel',
    'HumanProtection',
    'ProximityAlert',
    'FailSafeManager',
    'FailSafeType',
    'AuditLogger',
    'SafetyEvent'
]

# Safety Layer Version
__version__ = "1.0.0"

# Global Safety Constants
SAFETY_CONSTANTS = {
    'MAX_FORCE_NEWTONS': 20.0,
    'MAX_SPEED_MS': 1.0,
    'MIN_HUMAN_DISTANCE_M': 1.0,
    'EMERGENCY_STOP_TIMEOUT_MS': 100,
    'SAFETY_CHECK_INTERVAL_MS': 50,
    'AUDIT_LOG_RETENTION_DAYS': 365
}