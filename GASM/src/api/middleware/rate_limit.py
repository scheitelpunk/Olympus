"""
Rate Limiting Middleware

Implements request rate limiting with sliding window algorithm,
per-user and per-IP limits, and configurable burst handling.
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis
import json

from ..models.auth import RateLimitInfo

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    
    # Default limits (requests per minute)
    default_limit: int = 100
    authenticated_limit: int = 200
    admin_limit: int = 500
    
    # Burst handling
    burst_multiplier: float = 1.5
    burst_window_seconds: int = 60
    
    # Window configuration
    window_seconds: int = 60
    cleanup_interval: int = 300  # 5 minutes
    
    # Redis configuration (optional, falls back to in-memory)
    redis_url: Optional[str] = None
    redis_key_prefix: str = "gasm_rate_limit"
    
    # Exemptions
    exempt_ips: List[str] = field(default_factory=lambda: ["127.0.0.1", "::1"])
    exempt_endpoints: List[str] = field(default_factory=lambda: ["/health", "/docs", "/openapi.json"])


@dataclass
class RequestRecord:
    """Record of a request for rate limiting"""
    timestamp: float
    user_id: Optional[str] = None
    endpoint: str = ""
    method: str = "GET"


class SlidingWindowLimiter:
    """Sliding window rate limiter implementation"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.windows: Dict[str, deque] = defaultdict(lambda: deque())
        self.last_cleanup = time.time()
    
    def is_allowed(self, key: str, limit: int, window_seconds: int = None) -> tuple[bool, RateLimitInfo]:
        """
        Check if request is allowed under rate limit
        
        Args:
            key: Rate limit key (IP, user_id, etc.)
            limit: Requests allowed per window
            window_seconds: Window size in seconds
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        if window_seconds is None:
            window_seconds = self.config.window_seconds
            
        now = time.time()
        window_start = now - window_seconds
        
        # Clean up old entries periodically
        if now - self.last_cleanup > self.config.cleanup_interval:
            self._cleanup_expired_windows(now)
            self.last_cleanup = now
        
        # Get or create window for this key
        window = self.windows[key]
        
        # Remove expired requests from window
        while window and window[0].timestamp < window_start:
            window.popleft()
        
        # Check current count
        current_count = len(window)
        remaining = max(0, limit - current_count)
        
        # Calculate reset time
        reset_at = datetime.fromtimestamp(now + window_seconds)
        
        # Create rate limit info
        rate_limit_info = RateLimitInfo(
            limit=limit,
            remaining=remaining,
            reset_at=reset_at.isoformat(),
            retry_after=None
        )
        
        # Check if limit exceeded
        if current_count >= limit:
            # Calculate retry after time
            if window:
                oldest_request_time = window[0].timestamp
                retry_after = int(oldest_request_time + window_seconds - now) + 1
                rate_limit_info.retry_after = retry_after
            
            return False, rate_limit_info
        
        # Add current request to window
        record = RequestRecord(timestamp=now)
        window.append(record)
        
        # Update remaining count
        rate_limit_info.remaining = remaining - 1
        
        return True, rate_limit_info
    
    def _cleanup_expired_windows(self, now: float):
        """Clean up expired entries from all windows"""
        expired_keys = []
        window_start = now - self.config.window_seconds
        
        for key, window in self.windows.items():
            # Remove expired requests
            while window and window[0].timestamp < window_start:
                window.popleft()
            
            # Mark empty windows for removal
            if not window:
                expired_keys.append(key)
        
        # Remove empty windows
        for key in expired_keys:
            del self.windows[key]
        
        logger.debug(f"Cleaned up {len(expired_keys)} expired rate limit windows")


class RedisRateLimiter:
    """Redis-based rate limiter for distributed deployments"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.redis_client = None
        
        if config.redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(config.redis_url, decode_responses=True)
                # Test connection
                self.redis_client.ping()
                logger.info("Connected to Redis for rate limiting")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory rate limiting.")
                self.redis_client = None
    
    def is_allowed(self, key: str, limit: int, window_seconds: int = None) -> tuple[bool, RateLimitInfo]:
        """Check if request is allowed using Redis sliding window"""
        if not self.redis_client:
            raise RuntimeError("Redis client not available")
        
        if window_seconds is None:
            window_seconds = self.config.window_seconds
        
        now = time.time()
        window_start = now - window_seconds
        redis_key = f"{self.config.redis_key_prefix}:{key}"
        
        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(redis_key, 0, window_start)
            
            # Count current requests
            pipe.zcard(redis_key)
            
            # Add current request
            pipe.zadd(redis_key, {str(now): now})
            
            # Set expiry on key
            pipe.expire(redis_key, window_seconds + 1)
            
            # Execute pipeline
            results = pipe.execute()
            current_count = results[1]  # Count before adding current request
            
            remaining = max(0, limit - current_count - 1)  # -1 for current request
            
            # Calculate reset time
            reset_at = datetime.fromtimestamp(now + window_seconds)
            
            rate_limit_info = RateLimitInfo(
                limit=limit,
                remaining=remaining,
                reset_at=reset_at.isoformat(),
                retry_after=None
            )
            
            # Check if limit exceeded
            if current_count >= limit:
                # Remove the request we just added since it's not allowed
                self.redis_client.zrem(redis_key, str(now))
                
                # Calculate retry after time
                oldest_scores = self.redis_client.zrange(redis_key, 0, 0, withscores=True)
                if oldest_scores:
                    oldest_timestamp = oldest_scores[0][1]
                    retry_after = int(oldest_timestamp + window_seconds - now) + 1
                    rate_limit_info.retry_after = retry_after
                
                return False, rate_limit_info
            
            return True, rate_limit_info
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fall back to allowing the request on Redis errors
            return True, RateLimitInfo(
                limit=limit,
                remaining=limit - 1,
                reset_at=datetime.fromtimestamp(now + window_seconds).isoformat()
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for FastAPI"""
    
    def __init__(self, app, config: RateLimitConfig = None):
        super().__init__(app)
        self.config = config or RateLimitConfig()
        
        # Initialize rate limiter
        if self.config.redis_url:
            try:
                self.limiter = RedisRateLimiter(self.config)
                logger.info("Using Redis-based rate limiting")
            except Exception as e:
                logger.warning(f"Redis rate limiter failed to initialize: {e}. Using in-memory limiter.")
                self.limiter = SlidingWindowLimiter(self.config)
        else:
            self.limiter = SlidingWindowLimiter(self.config)
            logger.info("Using in-memory rate limiting")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting to requests"""
        
        # Skip rate limiting for exempt endpoints
        if self._is_exempt_endpoint(request.url.path):
            return await call_next(request)
        
        # Skip rate limiting for exempt IPs
        client_ip = self._get_client_ip(request)
        if client_ip in self.config.exempt_ips:
            return await call_next(request)
        
        # Determine rate limit key and limit
        rate_key, rate_limit = self._get_rate_limit_params(request)
        
        # Check rate limit
        try:
            is_allowed, rate_info = self._check_rate_limit(rate_key, rate_limit)
            
            if not is_allowed:
                return self._create_rate_limit_response(rate_info)
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            self._add_rate_limit_headers(response, rate_info)
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Allow request on errors to avoid breaking the service
            response = await call_next(request)
            return response
    
    def _is_exempt_endpoint(self, path: str) -> bool:
        """Check if endpoint is exempt from rate limiting"""
        for exempt_path in self.config.exempt_endpoints:
            if path.startswith(exempt_path):
                return True
        return False
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _get_rate_limit_params(self, request: Request) -> tuple[str, int]:
        """Get rate limiting key and limit for request"""
        
        # Check if user is authenticated
        if hasattr(request.state, 'user') and request.state.user:
            user = request.state.user
            
            # Use user-based rate limiting
            rate_key = f"user:{user.id}"
            
            # Determine limit based on user role
            if hasattr(user, 'role'):
                if user.role.value == 'admin':
                    rate_limit = self.config.admin_limit
                else:
                    rate_limit = self.config.authenticated_limit
            else:
                rate_limit = self.config.authenticated_limit
            
            # Check for custom API key limits
            if hasattr(request.state, 'api_key_id'):
                # Could look up custom limits for specific API keys
                pass
                
        else:
            # Use IP-based rate limiting for unauthenticated requests
            client_ip = self._get_client_ip(request)
            rate_key = f"ip:{client_ip}"
            rate_limit = self.config.default_limit
        
        return rate_key, rate_limit
    
    def _check_rate_limit(self, rate_key: str, rate_limit: int) -> tuple[bool, RateLimitInfo]:
        """Check if request is within rate limit"""
        
        # Check burst limit first (if configured)
        burst_limit = int(rate_limit * self.config.burst_multiplier)
        if burst_limit > rate_limit:
            # Check burst window (shorter window, higher limit)
            burst_allowed, burst_info = self.limiter.is_allowed(
                f"burst:{rate_key}", 
                burst_limit, 
                self.config.burst_window_seconds
            )
            
            if not burst_allowed:
                # Burst limit exceeded, return burst info
                return False, burst_info
        
        # Check regular rate limit
        return self.limiter.is_allowed(rate_key, rate_limit)
    
    def _create_rate_limit_response(self, rate_info: RateLimitInfo) -> JSONResponse:
        """Create rate limit exceeded response"""
        content = {
            "success": False,
            "error": "Rate limit exceeded",
            "timestamp": datetime.now().isoformat(),
            "rate_limit": {
                "limit": rate_info.limit,
                "remaining": rate_info.remaining,
                "reset_at": rate_info.reset_at,
                "retry_after": rate_info.retry_after
            }
        }
        
        headers = {
            "X-RateLimit-Limit": str(rate_info.limit),
            "X-RateLimit-Remaining": str(rate_info.remaining),
            "X-RateLimit-Reset": rate_info.reset_at
        }
        
        if rate_info.retry_after:
            headers["Retry-After"] = str(rate_info.retry_after)
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=content,
            headers=headers
        )
    
    def _add_rate_limit_headers(self, response: Response, rate_info: RateLimitInfo):
        """Add rate limit information to response headers"""
        response.headers["X-RateLimit-Limit"] = str(rate_info.limit)
        response.headers["X-RateLimit-Remaining"] = str(rate_info.remaining)
        response.headers["X-RateLimit-Reset"] = rate_info.reset_at


# FastAPI dependency for rate limiting information
async def get_rate_limit_info(request: Request) -> Optional[RateLimitInfo]:
    """Get rate limit information for current request"""
    # This would need access to the middleware instance
    # For now, return None - could be implemented with request state
    return None


def rate_limit_dependency(limit: int = None, window: int = None):
    """
    FastAPI dependency for endpoint-specific rate limiting
    
    Args:
        limit: Custom rate limit for this endpoint
        window: Custom window size in seconds
    """
    async def _check_rate_limit(request: Request) -> bool:
        # This would implement endpoint-specific rate limiting
        # Could be integrated with the middleware or work independently
        return True
    
    return _check_rate_limit


# Utility functions

def create_rate_limit_key(user_id: str = None, ip: str = None, endpoint: str = None) -> str:
    """Create a rate limit key from request parameters"""
    if user_id:
        key_parts = ["user", user_id]
    elif ip:
        key_parts = ["ip", ip]
    else:
        key_parts = ["anonymous"]
    
    if endpoint:
        # Include endpoint for endpoint-specific limits
        key_parts.append("endpoint", endpoint)
    
    return ":".join(key_parts)


def parse_rate_limit_config(config_dict: Dict[str, Any]) -> RateLimitConfig:
    """Parse rate limit configuration from dictionary"""
    return RateLimitConfig(
        default_limit=config_dict.get("default_limit", 100),
        authenticated_limit=config_dict.get("authenticated_limit", 200),
        admin_limit=config_dict.get("admin_limit", 500),
        burst_multiplier=config_dict.get("burst_multiplier", 1.5),
        burst_window_seconds=config_dict.get("burst_window_seconds", 60),
        window_seconds=config_dict.get("window_seconds", 60),
        cleanup_interval=config_dict.get("cleanup_interval", 300),
        redis_url=config_dict.get("redis_url"),
        redis_key_prefix=config_dict.get("redis_key_prefix", "gasm_rate_limit"),
        exempt_ips=config_dict.get("exempt_ips", ["127.0.0.1", "::1"]),
        exempt_endpoints=config_dict.get("exempt_endpoints", ["/health", "/docs", "/openapi.json"])
    )


# Export main components
__all__ = [
    "RateLimitMiddleware",
    "RateLimitConfig", 
    "SlidingWindowLimiter",
    "RedisRateLimiter",
    "rate_limit_dependency",
    "get_rate_limit_info",
    "create_rate_limit_key",
    "parse_rate_limit_config"
]