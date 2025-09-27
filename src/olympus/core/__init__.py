"""
OLYMPUS Core Module
==================

The core orchestrator module for Project OLYMPUS - responsible for system coordination,
consciousness management, and ethical operation routing.

Core Components:
- OlympusOrchestrator: Main system coordinator
- ModuleManager: Lifecycle management for all modules
- ConsciousnessKernel: Self-awareness and cognitive processing
- IdentityManager: Robot identity and authentication
- ConfigManager: System configuration management
- SystemHealth: Health monitoring and diagnostics

All operations flow through the orchestrator with mandatory ethical validation
via the Asimov Kernel and Safety Layer integration.
"""

from .olympus_orchestrator import OlympusOrchestrator
from .module_manager import ModuleManager
from .consciousness_kernel import ConsciousnessKernel
from .identity_manager import IdentityManager
from .config_manager import ConfigManager
from .system_health import SystemHealth

__all__ = [
    'OlympusOrchestrator',
    'ModuleManager',
    'ConsciousnessKernel',
    'IdentityManager',
    'ConfigManager',
    'SystemHealth'
]

__version__ = "1.0.0"
__author__ = "Versino PsiOmega GmbH"
__description__ = "Core orchestrator for Project OLYMPUS"