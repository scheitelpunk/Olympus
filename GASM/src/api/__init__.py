"""
GASM-Roboting API Package

This package provides the RESTful API interface for the GASM-Roboting system,
including endpoints for spatial agent control, GASM processing, and system management.

The API is organized into versions (v1) with proper middleware, authentication,
and comprehensive error handling.
"""

from .v1 import router as v1_router
from .middleware import setup_middleware
from .models import *

__version__ = "1.0.0"
__author__ = "GASM-Roboting Team"

# API metadata
API_METADATA = {
    "title": "GASM-Roboting API",
    "description": "Comprehensive API for Geometric Assembly State Machine and Spatial Agents",
    "version": __version__,
    "contact": {
        "name": "GASM-Roboting Team",
        "email": "team@gasm-roboting.dev"
    },
    "license": {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
}

# Export key components for easy import
__all__ = [
    "v1_router",
    "setup_middleware", 
    "API_METADATA"
]