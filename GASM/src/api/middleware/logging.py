"""
Request Logging Middleware

Comprehensive logging middleware for API requests, responses, and performance monitoring.
"""

import time
import uuid
import json
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import traceback

logger = logging.getLogger(__name__)


class RequestLoggingConfig:
    """Configuration for request logging middleware"""
    
    def __init__(self):
        # Logging levels
        self.log_requests = True
        self.log_responses = True
        self.log_performance = True
        self.log_errors = True
        
        # Content logging
        self.log_request_body = False  # Security: disabled by default
        self.log_response_body = False  # Performance: disabled by default
        self.max_body_size = 1024  # Max body size to log in bytes
        
        # Performance thresholds
        self.slow_request_threshold = 2.0  # seconds
        self.warn_response_size = 1024 * 1024  # 1MB
        
        # Sensitive data filtering
        self.sensitive_headers = {
            "authorization", "x-api-key", "cookie", "set-cookie",
            "x-auth-token", "x-access-token"
        }
        self.sensitive_fields = {
            "password", "token", "secret", "key", "auth"
        }
        
        # Exclusions
        self.exclude_paths = {"/health", "/metrics", "/favicon.ico"}
        self.exclude_static = True


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware with comprehensive monitoring"""
    
    def __init__(self, app: FastAPI, config: RequestLoggingConfig = None):
        super().__init__(app)
        self.config = config or RequestLoggingConfig()
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.slow_request_count = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with logging"""
        
        # Generate request ID for tracing
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Skip logging for excluded paths
        if self._should_exclude_path(request.url.path):
            return await call_next(request)
        
        # Start timing
        start_time = time.time()
        
        # Log request
        if self.config.log_requests:
            await self._log_request(request, request_id)
        
        response = None
        error = None
        
        try:
            # Process request
            response = await call_next(request)
            
            # Update counters
            self.request_count += 1
            
        except Exception as e:
            error = e
            self.error_count += 1
            
            # Create error response
            response = Response(
                content=json.dumps({
                    "success": False,
                    "error": "Internal server error",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                }),
                status_code=500,
                media_type="application/json"
            )
            
            # Log error
            if self.config.log_errors:
                await self._log_error(request, error, request_id)
        
        # Calculate timing
        end_time = time.time()
        duration = end_time - start_time
        
        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        # Log response and performance
        if self.config.log_responses:
            await self._log_response(request, response, duration, request_id)
        
        if self.config.log_performance:
            await self._log_performance(request, response, duration, request_id)
        
        return response
    
    def _should_exclude_path(self, path: str) -> bool:
        """Check if path should be excluded from logging"""
        # Exact path exclusions
        if path in self.config.exclude_paths:
            return True
        
        # Static file exclusions
        if self.config.exclude_static:
            static_extensions = {".css", ".js", ".png", ".jpg", ".gif", ".ico", ".svg", ".woff", ".woff2"}
            if any(path.endswith(ext) for ext in static_extensions):
                return True
        
        return False
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request"""
        try:
            # Basic request info
            log_data = {
                "event": "request_start",
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent"),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add user info if available
            if hasattr(request.state, 'user') and request.state.user:
                log_data["user_id"] = request.state.user.id
                log_data["username"] = request.state.user.username
            
            # Add filtered headers
            log_data["headers"] = self._filter_headers(dict(request.headers))
            
            # Add request body if configured
            if self.config.log_request_body:
                body = await self._get_request_body(request)
                if body:
                    log_data["body"] = self._filter_sensitive_data(body)
            
            logger.info(json.dumps(log_data))
            
        except Exception as e:
            logger.error(f"Error logging request: {e}")
    
    async def _log_response(self, request: Request, response: Response, duration: float, request_id: str):
        """Log response"""
        try:
            log_data = {
                "event": "request_complete",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration": round(duration, 3),
                "response_size": len(response.body) if hasattr(response, 'body') and response.body else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add user info if available
            if hasattr(request.state, 'user') and request.state.user:
                log_data["user_id"] = request.state.user.id
            
            # Add response headers (filtered)
            log_data["response_headers"] = self._filter_headers(dict(response.headers))
            
            # Add response body if configured and not too large
            if (self.config.log_response_body and 
                hasattr(response, 'body') and 
                response.body and 
                len(response.body) <= self.config.max_body_size):
                try:
                    body_str = response.body.decode('utf-8')
                    log_data["response_body"] = self._filter_sensitive_data(body_str)
                except UnicodeDecodeError:
                    log_data["response_body"] = "<binary_content>"
            
            # Choose log level based on status code
            if response.status_code >= 500:
                logger.error(json.dumps(log_data))
            elif response.status_code >= 400:
                logger.warning(json.dumps(log_data))
            else:
                logger.info(json.dumps(log_data))
            
        except Exception as e:
            logger.error(f"Error logging response: {e}")
    
    async def _log_performance(self, request: Request, response: Response, duration: float, request_id: str):
        """Log performance metrics"""
        try:
            # Check for slow requests
            if duration > self.config.slow_request_threshold:
                self.slow_request_count += 1
                
                log_data = {
                    "event": "slow_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration": round(duration, 3),
                    "threshold": self.config.slow_request_threshold,
                    "status_code": response.status_code,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.warning(json.dumps(log_data))
            
            # Check for large responses
            response_size = len(response.body) if hasattr(response, 'body') and response.body else 0
            if response_size > self.config.warn_response_size:
                log_data = {
                    "event": "large_response",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "response_size": response_size,
                    "size_threshold": self.config.warn_response_size,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.warning(json.dumps(log_data))
            
        except Exception as e:
            logger.error(f"Error logging performance: {e}")
    
    async def _log_error(self, request: Request, error: Exception, request_id: str):
        """Log request error"""
        try:
            log_data = {
                "event": "request_error",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "client_ip": self._get_client_ip(request),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add user info if available
            if hasattr(request.state, 'user') and request.state.user:
                log_data["user_id"] = request.state.user.id
            
            logger.error(json.dumps(log_data))
            
        except Exception as e:
            logger.error(f"Error logging error: {e}")
    
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
    
    def _filter_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Filter sensitive headers"""
        filtered = {}
        
        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower in self.config.sensitive_headers:
                # Mask sensitive headers
                if key_lower == "authorization" and value.startswith("Bearer "):
                    filtered[key] = f"Bearer {value[7:11]}***{value[-4:]}"
                elif key_lower == "x-api-key":
                    filtered[key] = f"{value[:8]}***{value[-4:]}" if len(value) > 12 else "***"
                else:
                    filtered[key] = "***"
            else:
                filtered[key] = value
        
        return filtered
    
    def _filter_sensitive_data(self, data: str) -> str:
        """Filter sensitive data from request/response bodies"""
        try:
            # Try to parse as JSON and filter
            parsed_data = json.loads(data)
            filtered_data = self._filter_json_fields(parsed_data)
            return json.dumps(filtered_data)
        except (json.JSONDecodeError, TypeError):
            # Not JSON, return truncated if too long
            if len(data) > self.config.max_body_size:
                return data[:self.config.max_body_size] + "... (truncated)"
            return data
    
    def _filter_json_fields(self, data: Any) -> Any:
        """Recursively filter sensitive fields from JSON data"""
        if isinstance(data, dict):
            filtered = {}
            for key, value in data.items():
                key_lower = key.lower()
                if any(sensitive in key_lower for sensitive in self.config.sensitive_fields):
                    filtered[key] = "***"
                else:
                    filtered[key] = self._filter_json_fields(value)
            return filtered
        elif isinstance(data, list):
            return [self._filter_json_fields(item) for item in data]
        else:
            return data
    
    async def _get_request_body(self, request: Request) -> Optional[str]:
        """Get request body as string"""
        try:
            if hasattr(request, '_body'):
                body = request._body
            else:
                body = await request.body()
            
            if not body:
                return None
            
            # Limit body size for logging
            if len(body) > self.config.max_body_size:
                body = body[:self.config.max_body_size]
            
            return body.decode('utf-8')
        except Exception:
            return None


def setup_request_logging(app: FastAPI, config: RequestLoggingConfig = None):
    """Setup request logging middleware"""
    if config is None:
        config = RequestLoggingConfig()
    
    app.add_middleware(RequestLoggingMiddleware, config=config)
    logger.info("✅ Request logging middleware configured")


def create_logging_config(
    log_requests: bool = True,
    log_responses: bool = True, 
    log_request_body: bool = False,
    log_response_body: bool = False,
    slow_threshold: float = 2.0
) -> RequestLoggingConfig:
    """Create custom logging configuration"""
    config = RequestLoggingConfig()
    config.log_requests = log_requests
    config.log_responses = log_responses
    config.log_request_body = log_request_body
    config.log_response_body = log_response_body
    config.slow_request_threshold = slow_threshold
    
    return config


def get_development_logging_config() -> RequestLoggingConfig:
    """Get verbose logging configuration for development"""
    config = RequestLoggingConfig()
    config.log_request_body = True
    config.log_response_body = True
    config.max_body_size = 4096  # Larger for development
    config.slow_request_threshold = 1.0  # More sensitive in dev
    
    return config


def get_production_logging_config() -> RequestLoggingConfig:
    """Get optimized logging configuration for production"""
    config = RequestLoggingConfig()
    config.log_request_body = False  # Security
    config.log_response_body = False  # Performance
    config.max_body_size = 512  # Smaller for production
    config.slow_request_threshold = 3.0  # Less noise in production
    
    return config


# Export main components
__all__ = [
    "setup_request_logging",
    "RequestLoggingMiddleware",
    "RequestLoggingConfig",
    "create_logging_config",
    "get_development_logging_config", 
    "get_production_logging_config"
]