"""
CORS Middleware Configuration

Configures Cross-Origin Resource Sharing (CORS) for the GASM-Roboting API
with security best practices and development convenience.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)


class CORSConfig:
    """CORS configuration with environment-based settings"""
    
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development").lower()
        
        # Base configuration
        if self.environment == "production":
            self.allowed_origins = self._get_production_origins()
            self.allow_credentials = True
            self.allow_origin_regex = None
        else:
            # Development settings - more permissive
            self.allowed_origins = self._get_development_origins()
            self.allow_credentials = True
            self.allow_origin_regex = r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?"
        
        # Always allowed methods and headers
        self.allowed_methods = [
            "GET",
            "POST", 
            "PUT",
            "DELETE",
            "PATCH",
            "OPTIONS"
        ]
        
        self.allowed_headers = [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "X-Requested-With",
            "X-Request-ID",
            "Cache-Control",
            "Pragma"
        ]
        
        # Headers to expose to the client
        self.expose_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining", 
            "X-RateLimit-Reset",
            "X-Request-ID",
            "X-Response-Time"
        ]
        
        # Cache preflight responses for 1 hour
        self.max_age = 3600
    
    def _get_production_origins(self) -> List[str]:
        """Get allowed origins for production environment"""
        # Read from environment variable
        origins_env = os.getenv("ALLOWED_ORIGINS", "")
        if origins_env:
            origins = [origin.strip() for origin in origins_env.split(",")]
            logger.info(f"Using production CORS origins from environment: {origins}")
            return origins
        
        # Default production origins
        return [
            "https://gasm-roboting.com",
            "https://www.gasm-roboting.com", 
            "https://api.gasm-roboting.com",
            "https://admin.gasm-roboting.com"
        ]
    
    def _get_development_origins(self) -> List[str]:
        """Get allowed origins for development environment"""
        development_origins = [
            "http://localhost:3000",  # React dev server
            "http://localhost:3001", 
            "http://localhost:8000",  # FastAPI docs
            "http://localhost:8080",  # Vue/webpack dev server
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8080",
            "http://0.0.0.0:3000",
            "http://0.0.0.0:8000",
            # Add any additional development origins from environment
        ]
        
        # Add custom development origins from environment
        custom_origins = os.getenv("DEV_ALLOWED_ORIGINS", "")
        if custom_origins:
            additional_origins = [origin.strip() for origin in custom_origins.split(",")]
            development_origins.extend(additional_origins)
        
        logger.info(f"Using development CORS origins: {development_origins}")
        return development_origins


def setup_cors(app: FastAPI, config: Optional[CORSConfig] = None) -> None:
    """
    Configure CORS middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        config: Optional CORS configuration (uses default if None)
    """
    if config is None:
        config = CORSConfig()
    
    # Log CORS configuration
    logger.info("Configuring CORS middleware:")
    logger.info(f"  Environment: {config.environment}")
    logger.info(f"  Allowed origins: {config.allowed_origins}")
    logger.info(f"  Allow credentials: {config.allow_credentials}")
    logger.info(f"  Origin regex: {config.allow_origin_regex}")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_origin_regex=config.allow_origin_regex,
        allow_credentials=config.allow_credentials,
        allow_methods=config.allowed_methods,
        allow_headers=config.allowed_headers,
        expose_headers=config.expose_headers,
        max_age=config.max_age,
    )
    
    logger.info("✅ CORS middleware configured successfully")


def create_cors_config(
    environment: str = None,
    allowed_origins: List[str] = None,
    allow_credentials: bool = True,
    additional_methods: List[str] = None,
    additional_headers: List[str] = None
) -> CORSConfig:
    """
    Create a custom CORS configuration.
    
    Args:
        environment: Environment name ("production", "development", "testing")
        allowed_origins: List of allowed origins (overrides environment defaults)
        allow_credentials: Whether to allow credentials
        additional_methods: Additional HTTP methods to allow
        additional_headers: Additional headers to allow
        
    Returns:
        Configured CORSConfig instance
    """
    config = CORSConfig()
    
    if environment:
        config.environment = environment.lower()
    
    if allowed_origins:
        config.allowed_origins = allowed_origins
    
    config.allow_credentials = allow_credentials
    
    if additional_methods:
        config.allowed_methods.extend(additional_methods)
        config.allowed_methods = list(set(config.allowed_methods))  # Remove duplicates
    
    if additional_headers:
        config.allowed_headers.extend(additional_headers)
        config.allowed_headers = list(set(config.allowed_headers))  # Remove duplicates
    
    return config


def get_cors_headers(origin: str = None) -> dict:
    """
    Get CORS headers for manual responses.
    
    Args:
        origin: Origin to set in Access-Control-Allow-Origin
        
    Returns:
        Dictionary of CORS headers
    """
    headers = {
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS",
        "Access-Control-Allow-Headers": "Accept, Content-Type, Authorization, X-API-Key",
        "Access-Control-Expose-Headers": "X-RateLimit-Limit, X-RateLimit-Remaining, X-Request-ID",
        "Access-Control-Max-Age": "3600"
    }
    
    if origin:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
    
    return headers


def validate_origin(origin: str, allowed_origins: List[str], allow_origin_regex: str = None) -> bool:
    """
    Validate if an origin is allowed.
    
    Args:
        origin: Origin to validate
        allowed_origins: List of explicitly allowed origins
        allow_origin_regex: Regular expression for allowed origins
        
    Returns:
        True if origin is allowed, False otherwise
    """
    # Check explicit origins
    if origin in allowed_origins:
        return True
    
    # Check wildcard
    if "*" in allowed_origins:
        return True
    
    # Check regex pattern
    if allow_origin_regex:
        import re
        try:
            if re.match(allow_origin_regex, origin):
                return True
        except re.error:
            logger.warning(f"Invalid CORS origin regex: {allow_origin_regex}")
    
    return False


# Security utilities

def is_secure_origin(origin: str) -> bool:
    """
    Check if an origin uses secure protocol (HTTPS) or is localhost.
    
    Args:
        origin: Origin URL to check
        
    Returns:
        True if origin is considered secure
    """
    if not origin:
        return False
    
    origin_lower = origin.lower()
    
    # HTTPS is always secure
    if origin_lower.startswith("https://"):
        return True
    
    # Localhost is considered secure for development
    if any(localhost in origin_lower for localhost in ["localhost", "127.0.0.1", "0.0.0.0"]):
        return True
    
    # File protocol for testing
    if origin_lower.startswith("file://"):
        return True
    
    return False


def log_cors_violation(origin: str, reason: str = None):
    """
    Log CORS policy violations for security monitoring.
    
    Args:
        origin: Origin that was rejected
        reason: Reason for rejection
    """
    log_message = f"CORS violation: Origin '{origin}' rejected"
    if reason:
        log_message += f" - {reason}"
    
    logger.warning(log_message)
    
    # In production, you might want to send this to a security monitoring system
    # or increment metrics counters


# Development helpers

def get_development_config() -> CORSConfig:
    """Get permissive CORS configuration for development"""
    config = CORSConfig()
    config.environment = "development"
    config.allowed_origins = ["*"]  # Very permissive for development
    config.allow_origin_regex = None
    config.allow_credentials = False  # Can't use credentials with wildcard origins
    
    logger.warning("🚨 Using permissive development CORS config with wildcard origins")
    
    return config


def get_testing_config() -> CORSConfig:
    """Get CORS configuration for testing"""
    config = CORSConfig()
    config.environment = "testing"
    config.allowed_origins = [
        "http://testserver",
        "http://localhost",
        "http://127.0.0.1"
    ]
    config.allow_credentials = True
    
    return config


# Export main functions
__all__ = [
    "setup_cors",
    "CORSConfig",
    "create_cors_config", 
    "get_cors_headers",
    "validate_origin",
    "is_secure_origin",
    "log_cors_violation",
    "get_development_config",
    "get_testing_config"
]