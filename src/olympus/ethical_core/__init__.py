"""
OLYMPUS Ethical Core Module

This module contains the immutable ethical foundation of the OLYMPUS system,
implementing the Asimov Laws with cryptographic integrity protection.

Security Notice: This module is critical for system safety and must not be modified
without proper authorization and security review.
"""

from .asimov_kernel import AsimovKernel
from .ethical_validator import EthicalValidator
from .integrity_monitor import IntegrityMonitor

__all__ = ['AsimovKernel', 'EthicalValidator', 'IntegrityMonitor']

# Version and integrity information
__version__ = "1.0.0"
__security_level__ = "CRITICAL"
__immutable__ = True