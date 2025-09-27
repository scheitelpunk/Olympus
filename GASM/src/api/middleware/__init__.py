"""
API Middleware Components

Provides authentication, rate limiting, CORS, logging, and other middleware
for the GASM-Roboting API.
"""

from .auth import AuthMiddleware, get_current_user, verify_api_key
from .rate_limit import RateLimitMiddleware, rate_limit_dependency
from .cors import setup_cors
from .logging import setup_request_logging
from .errors import setup_error_handlers

def setup_middleware(app):
    """
    Configure all middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Order matters for middleware setup
    setup_error_handlers(app)
    setup_request_logging(app)
    setup_cors(app)
    
    # Add custom middleware
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)

__all__ = [
    "setup_middleware",
    "AuthMiddleware", 
    "RateLimitMiddleware",
    "get_current_user",
    "verify_api_key",
    "rate_limit_dependency"
]