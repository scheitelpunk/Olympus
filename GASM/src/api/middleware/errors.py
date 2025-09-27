"""
Error Handling Middleware

Comprehensive error handling and exception management for the GASM-Roboting API.
"""

import logging
import traceback
import sys
from typing import Dict, Any, Optional, Union, Type
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..models.base import ErrorResponse, create_error_response

logger = logging.getLogger(__name__)


class ErrorHandlingConfig:
    """Configuration for error handling"""
    
    def __init__(self):
        # Development vs production settings
        self.environment = "development"  # development, production, testing
        self.include_traceback = True     # Include stack traces (dev only)
        self.include_request_info = True  # Include request details in errors
        
        # Error categorization
        self.log_client_errors = False    # Log 4xx errors
        self.log_server_errors = True     # Log 5xx errors
        self.log_validation_errors = True # Log validation errors
        
        # Error reporting
        self.send_error_notifications = False  # Send notifications for critical errors
        self.error_notification_threshold = 500  # Status code threshold for notifications
        
        # Response formatting
        self.standardize_error_responses = True
        self.include_error_codes = True
        self.include_request_id = True


class ErrorHandler:
    """Centralized error handling logic"""
    
    def __init__(self, config: ErrorHandlingConfig = None):
        self.config = config or ErrorHandlingConfig()
        self.error_stats = {
            "total_errors": 0,
            "client_errors": 0,
            "server_errors": 0,
            "validation_errors": 0
        }
    
    def handle_http_exception(self, request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions"""
        self._update_error_stats(exc.status_code)
        
        # Log error if configured
        if self._should_log_error(exc.status_code):
            self._log_error(request, exc, "http_exception")
        
        # Create standardized response
        error_response = self._create_error_response(
            request=request,
            status_code=exc.status_code,
            error=exc.detail,
            error_code=self._get_error_code(exc),
            exception=exc
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response
        )
    
    def handle_validation_error(self, request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle request validation errors"""
        self.error_stats["validation_errors"] += 1
        self.error_stats["total_errors"] += 1
        
        if self.config.log_validation_errors:
            self._log_error(request, exc, "validation_error")
        
        # Format validation errors
        validation_errors = []
        for error in exc.errors():
            validation_errors.append({
                "field": ".".join(str(x) for x in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })
        
        error_response = self._create_error_response(
            request=request,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error="Validation failed",
            error_code="VALIDATION_ERROR",
            details={"validation_errors": validation_errors},
            exception=exc
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response
        )
    
    def handle_general_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle general exceptions"""
        self.error_stats["server_errors"] += 1
        self.error_stats["total_errors"] += 1
        
        # Always log server errors
        self._log_error(request, exc, "general_exception")
        
        # Send notification for critical errors if configured
        if self.config.send_error_notifications:
            self._send_error_notification(request, exc)
        
        # Create error response
        error_message = "Internal server error"
        error_code = "INTERNAL_ERROR"
        
        # In development, include more details
        if self.config.environment == "development":
            error_message = f"{type(exc).__name__}: {str(exc)}"
        
        error_response = self._create_error_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error=error_message,
            error_code=error_code,
            exception=exc
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response
        )
    
    def _create_error_response(
        self,
        request: Request,
        status_code: int,
        error: str,
        error_code: str = None,
        details: Dict[str, Any] = None,
        exception: Exception = None
    ) -> Dict[str, Any]:
        """Create standardized error response"""
        
        response = {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "error": error
        }
        
        # Add error code if configured
        if self.config.include_error_codes and error_code:
            response["error_code"] = error_code
        
        # Add request ID if available
        if (self.config.include_request_id and 
            hasattr(request.state, 'request_id')):
            response["request_id"] = request.state.request_id
        
        # Add additional details
        if details:
            response["details"] = details
        
        # Add request information if configured
        if self.config.include_request_info:
            response["request_info"] = {
                "method": request.method,
                "path": request.url.path,
                "client_ip": self._get_client_ip(request)
            }
        
        # Add traceback for development
        if (self.config.include_traceback and 
            self.config.environment == "development" and 
            exception):
            response["traceback"] = traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        
        return response
    
    def _update_error_stats(self, status_code: int):
        """Update error statistics"""
        self.error_stats["total_errors"] += 1
        
        if 400 <= status_code < 500:
            self.error_stats["client_errors"] += 1
        elif status_code >= 500:
            self.error_stats["server_errors"] += 1
    
    def _should_log_error(self, status_code: int) -> bool:
        """Determine if error should be logged"""
        if status_code >= 500 and self.config.log_server_errors:
            return True
        if 400 <= status_code < 500 and self.config.log_client_errors:
            return True
        return False
    
    def _log_error(self, request: Request, exception: Exception, error_type: str):
        """Log error with context"""
        try:
            error_data = {
                "error_type": error_type,
                "exception_class": type(exception).__name__,
                "exception_message": str(exception),
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent"),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add user info if available
            if hasattr(request.state, 'user') and request.state.user:
                error_data["user_id"] = request.state.user.id
                error_data["username"] = request.state.user.username
            
            # Add request ID if available
            if hasattr(request.state, 'request_id'):
                error_data["request_id"] = request.state.request_id
            
            # Add traceback for server errors
            if error_type == "general_exception":
                error_data["traceback"] = traceback.format_exc()
            
            logger.error(f"API Error: {error_data}")
            
        except Exception as log_error:
            logger.error(f"Failed to log error: {log_error}")
    
    def _send_error_notification(self, request: Request, exception: Exception):
        """Send error notification (implement based on your notification system)"""
        try:
            # This is where you would integrate with your error notification service
            # (e.g., Sentry, email, Slack, etc.)
            logger.critical(f"Critical error notification: {type(exception).__name__}: {str(exception)}")
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _get_error_code(self, exception: HTTPException) -> Optional[str]:
        """Generate error code from HTTP exception"""
        status_code = exception.status_code
        
        # Common HTTP status code mappings
        error_codes = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED", 
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            408: "REQUEST_TIMEOUT",
            409: "CONFLICT",
            410: "GONE",
            413: "PAYLOAD_TOO_LARGE",
            415: "UNSUPPORTED_MEDIA_TYPE",
            422: "UNPROCESSABLE_ENTITY",
            429: "TOO_MANY_REQUESTS",
            500: "INTERNAL_SERVER_ERROR",
            501: "NOT_IMPLEMENTED",
            502: "BAD_GATEWAY",
            503: "SERVICE_UNAVAILABLE",
            504: "GATEWAY_TIMEOUT"
        }
        
        return error_codes.get(status_code)


# Global error handler instance
error_handler = ErrorHandler()


def setup_error_handlers(app: FastAPI, config: ErrorHandlingConfig = None):
    """Setup comprehensive error handling for FastAPI application"""
    
    if config:
        global error_handler
        error_handler = ErrorHandler(config)
    
    # HTTP exceptions (includes FastAPI HTTPException)
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return error_handler.handle_http_exception(request, exc)
    
    # Starlette HTTP exceptions
    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
        # Convert to FastAPI HTTPException for consistent handling
        http_exc = HTTPException(status_code=exc.status_code, detail=exc.detail)
        return error_handler.handle_http_exception(request, http_exc)
    
    # Request validation errors
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return error_handler.handle_validation_error(request, exc)
    
    # Pydantic validation errors
    @app.exception_handler(ValidationError)
    async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
        # Convert to RequestValidationError for consistent handling
        validation_exc = RequestValidationError([{
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"]
        } for error in exc.errors()])
        return error_handler.handle_validation_error(request, validation_exc)
    
    # General exceptions (catch-all)
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return error_handler.handle_general_exception(request, exc)
    
    logger.info("✅ Error handling configured")


def create_error_handling_config(
    environment: str = "development",
    include_traceback: bool = None,
    log_client_errors: bool = False,
    log_server_errors: bool = True
) -> ErrorHandlingConfig:
    """Create custom error handling configuration"""
    
    config = ErrorHandlingConfig()
    config.environment = environment.lower()
    
    # Set environment-specific defaults
    if config.environment == "production":
        config.include_traceback = False
        config.include_request_info = False
        config.log_client_errors = False
    elif config.environment == "development":
        config.include_traceback = True
        config.include_request_info = True
        config.log_client_errors = True
    
    # Override with provided values
    if include_traceback is not None:
        config.include_traceback = include_traceback
    
    config.log_client_errors = log_client_errors
    config.log_server_errors = log_server_errors
    
    return config


def get_error_stats() -> Dict[str, Any]:
    """Get current error statistics"""
    return {
        "stats": error_handler.error_stats.copy(),
        "config": {
            "environment": error_handler.config.environment,
            "include_traceback": error_handler.config.include_traceback,
            "log_client_errors": error_handler.config.log_client_errors,
            "log_server_errors": error_handler.config.log_server_errors
        }
    }


def reset_error_stats():
    """Reset error statistics"""
    error_handler.error_stats = {
        "total_errors": 0,
        "client_errors": 0,
        "server_errors": 0,
        "validation_errors": 0
    }


# Custom exception classes for specific use cases

class GASMAPIException(HTTPException):
    """Base exception for GASM API specific errors"""
    
    def __init__(self, status_code: int, detail: str, error_code: str = None):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code


class ModelNotLoadedException(GASMAPIException):
    """Exception for when GASM model is not loaded"""
    
    def __init__(self):
        super().__init__(
            status_code=503,
            detail="GASM model is not loaded. Please check system status.",
            error_code="MODEL_NOT_LOADED"
        )


class SpatialAgentException(GASMAPIException):
    """Exception for spatial agent errors"""
    
    def __init__(self, detail: str):
        super().__init__(
            status_code=500,
            detail=f"Spatial agent error: {detail}",
            error_code="SPATIAL_AGENT_ERROR"
        )


class ConstraintViolationException(GASMAPIException):
    """Exception for constraint violations"""
    
    def __init__(self, detail: str):
        super().__init__(
            status_code=400,
            detail=f"Constraint violation: {detail}",
            error_code="CONSTRAINT_VIOLATION"
        )


class RateLimitExceededException(GASMAPIException):
    """Exception for rate limit violations"""
    
    def __init__(self, retry_after: int = None):
        detail = "Rate limit exceeded"
        if retry_after:
            detail += f". Retry after {retry_after} seconds."
        
        super().__init__(
            status_code=429,
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED"
        )


# Export main components
__all__ = [
    "setup_error_handlers",
    "ErrorHandlingConfig",
    "ErrorHandler",
    "create_error_handling_config",
    "get_error_stats",
    "reset_error_stats",
    "GASMAPIException",
    "ModelNotLoadedException",
    "SpatialAgentException", 
    "ConstraintViolationException",
    "RateLimitExceededException"
]