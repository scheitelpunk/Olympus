"""
Authentication Middleware

Provides JWT-based authentication, API key validation, and user authorization
for the GASM-Roboting API.
"""

import jwt
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from fastapi import HTTPException, Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from datetime import datetime, timedelta
import hashlib
import hmac
import secrets

from ..models.auth import (
    User, AuthToken, TokenPayload, UserSession, SecurityEvent,
    PermissionScope, TokenType, UserRole, check_permission
)

logger = logging.getLogger(__name__)


class AuthConfig:
    """Authentication configuration"""
    
    def __init__(self):
        # JWT configuration
        self.jwt_secret = self._get_jwt_secret()
        self.jwt_algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 30
        
        # API key configuration
        self.api_key_length = 32
        self.api_key_prefix = "gasm_"
        
        # Security configuration
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 15
        self.session_timeout_minutes = 120
        
        # Rate limiting
        self.default_rate_limit = 100  # requests per minute
        self.burst_limit = 200
    
    def _get_jwt_secret(self) -> str:
        """Get or generate JWT secret key"""
        # In production, this should come from environment variables
        import os
        secret = os.getenv("GASM_JWT_SECRET")
        if not secret:
            # Generate a secure random secret for development
            secret = secrets.token_urlsafe(64)
            logger.warning("Using generated JWT secret. Set GASM_JWT_SECRET environment variable in production.")
        return secret


class TokenManager:
    """JWT token management"""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        
    def create_access_token(
        self, 
        user_id: str, 
        scopes: List[PermissionScope],
        session_id: str = None
    ) -> AuthToken:
        """Create JWT access token"""
        expires_in = self.config.access_token_expire_minutes * 60
        expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        payload = {
            "sub": user_id,
            "iat": int(time.time()),
            "exp": int(expires_at.timestamp()),
            "jti": secrets.token_urlsafe(16),
            "type": TokenType.BEARER.value,
            "scopes": [scope.value for scope in scopes],
            "session_id": session_id
        }
        
        token = jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
        
        return AuthToken(
            access_token=token,
            token_type=TokenType.BEARER,
            expires_in=expires_in,
            expires_at=expires_at.isoformat(),
            scopes=scopes
        )
    
    def create_refresh_token(self, user_id: str, session_id: str = None) -> str:
        """Create refresh token"""
        expires_in = self.config.refresh_token_expire_days * 24 * 60 * 60
        expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        payload = {
            "sub": user_id,
            "iat": int(time.time()),
            "exp": int(expires_at.timestamp()),
            "jti": secrets.token_urlsafe(16),
            "type": TokenType.REFRESH.value,
            "session_id": session_id
        }
        
        return jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    def verify_token(self, token: str) -> TokenPayload:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.config.jwt_secret, 
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Convert scopes back to enum
            scopes = [PermissionScope(scope) for scope in payload.get("scopes", [])]
            
            return TokenPayload(
                sub=payload["sub"],
                iat=payload["iat"],
                exp=payload["exp"],
                jti=payload["jti"],
                type=TokenType(payload.get("type", TokenType.BEARER.value)),
                scopes=scopes,
                session_id=payload.get("session_id")
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def generate_api_key(self) -> str:
        """Generate secure API key"""
        random_part = secrets.token_urlsafe(self.config.api_key_length)
        return f"{self.config.api_key_prefix}{random_part}"
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def verify_api_key_hash(self, api_key: str, stored_hash: str) -> bool:
        """Verify API key against stored hash"""
        return hmac.compare_digest(
            self.hash_api_key(api_key),
            stored_hash
        )


class UserStore:
    """In-memory user store for development (replace with database in production)"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}  # key_hash -> key_info
        self.sessions: Dict[str, UserSession] = {}
        self.security_events: List[SecurityEvent] = []
        self.login_attempts: Dict[str, Dict[str, Any]] = {}  # ip -> attempts info
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user for development"""
        admin_user = User(
            id="admin_001",
            username="admin",
            email="admin@gasm-roboting.dev",
            full_name="System Administrator",
            organization="GASM-Roboting",
            role=UserRole.ADMIN,
            scopes=[PermissionScope.ADMIN],
            is_active=True,
            is_verified=True,
            created_at=datetime.now().isoformat(),
            login_count=0
        )
        self.users[admin_user.id] = admin_user
        
        logger.info("Created default admin user (username: admin)")
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        for user in self.users.values():
            if user.username == username.lower():
                return user
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        for user in self.users.values():
            if user.email.lower() == email.lower():
                return user
        return None
    
    def verify_password(self, user: User, password: str) -> bool:
        """Verify user password (simplified for development)"""
        # In production, use proper password hashing (bcrypt, scrypt, etc.)
        # For development, using simple comparison
        if user.username == "admin":
            return password == "admin123"  # Default admin password
        return False
    
    def store_api_key(self, key_hash: str, key_info: Dict[str, Any]):
        """Store API key information"""
        self.api_keys[key_hash] = key_info
    
    def get_api_key_info(self, key_hash: str) -> Optional[Dict[str, Any]]:
        """Get API key information"""
        return self.api_keys.get(key_hash)
    
    def create_session(self, user_id: str, ip_address: str = None) -> UserSession:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(minutes=120)
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now().isoformat(),
            expires_at=expires_at.isoformat(),
            last_activity=datetime.now().isoformat(),
            ip_address=ip_address,
            is_active=True
        )
        
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get user session"""
        return self.sessions.get(session_id)
    
    def update_session_activity(self, session_id: str):
        """Update session last activity"""
        if session_id in self.sessions:
            self.sessions[session_id].last_activity = datetime.now().isoformat()
    
    def record_security_event(self, event: SecurityEvent):
        """Record security event"""
        self.security_events.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
    
    def check_login_attempts(self, ip_address: str) -> bool:
        """Check if IP is not locked out due to failed login attempts"""
        if ip_address not in self.login_attempts:
            return True
        
        attempts_info = self.login_attempts[ip_address]
        if attempts_info["count"] >= 5:
            # Check if lockout has expired
            lockout_end = datetime.fromisoformat(attempts_info["lockout_until"])
            if datetime.now() < lockout_end:
                return False
            else:
                # Reset attempts after lockout period
                del self.login_attempts[ip_address]
        
        return True
    
    def record_login_attempt(self, ip_address: str, success: bool):
        """Record login attempt"""
        if success:
            # Clear failed attempts on successful login
            if ip_address in self.login_attempts:
                del self.login_attempts[ip_address]
            return
        
        # Record failed attempt
        if ip_address not in self.login_attempts:
            self.login_attempts[ip_address] = {"count": 0, "first_attempt": datetime.now().isoformat()}
        
        self.login_attempts[ip_address]["count"] += 1
        
        # Set lockout if too many attempts
        if self.login_attempts[ip_address]["count"] >= 5:
            lockout_until = datetime.now() + timedelta(minutes=15)
            self.login_attempts[ip_address]["lockout_until"] = lockout_until.isoformat()


# Global instances
auth_config = AuthConfig()
token_manager = TokenManager(auth_config)
user_store = UserStore()


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for FastAPI"""
    
    def __init__(self, app, config: AuthConfig = None):
        super().__init__(app)
        self.config = config or auth_config
        self.token_manager = token_manager
        self.user_store = user_store
        
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            "/",
            "/health",
            "/openapi.json",
            "/docs",
            "/redoc",
            "/favicon.ico"
        }
        
        # Endpoints that only require basic authentication (not specific permissions)
        self.basic_auth_endpoints = {
            "/v1/core/info",
            "/v1/core/debug-info"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process authentication for each request"""
        
        # Skip authentication for public endpoints
        if self._is_public_endpoint(request.url.path):
            return await call_next(request)
        
        # Extract and verify authentication
        try:
            user = await self._authenticate_request(request)
            request.state.user = user
            request.state.authenticated = True
            
            # Update session activity
            if hasattr(request.state, 'session_id'):
                self.user_store.update_session_activity(request.state.session_id)
                
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "success": False,
                    "error": e.detail,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        response = await call_next(request)
        return response
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public"""
        return path in self.public_endpoints or path.startswith("/static/")
    
    async def _authenticate_request(self, request: Request) -> User:
        """Authenticate request and return user"""
        
        # Try API key authentication first
        api_key = self._extract_api_key(request)
        if api_key:
            return self._authenticate_api_key(api_key, request)
        
        # Try JWT token authentication
        token = self._extract_bearer_token(request)
        if token:
            return self._authenticate_jwt_token(token, request)
        
        # No authentication provided
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide API key or Bearer token."
        )
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request headers"""
        # Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key
            
        # Check Authorization header with API key scheme
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("ApiKey "):
            return auth_header[7:]  # Remove "ApiKey " prefix
            
        return None
    
    def _extract_bearer_token(self, request: Request) -> Optional[str]:
        """Extract Bearer token from Authorization header"""
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        return None
    
    def _authenticate_api_key(self, api_key: str, request: Request) -> User:
        """Authenticate using API key"""
        # Hash the provided API key
        key_hash = self.token_manager.hash_api_key(api_key)
        
        # Look up API key info
        key_info = self.user_store.get_api_key_info(key_hash)
        if not key_info:
            self._record_auth_event(
                "api_key_auth_failed", 
                None, 
                request, 
                {"reason": "invalid_key"}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        # Check if key is active
        if not key_info.get("is_active", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is inactive"
            )
        
        # Check if key has expired
        if key_info.get("expires_at"):
            expires_at = datetime.fromisoformat(key_info["expires_at"])
            if datetime.now() > expires_at:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key has expired"
                )
        
        # Get user
        user = self.user_store.get_user_by_id(key_info["user_id"])
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive"
            )
        
        # Store API key info in request state
        request.state.api_key_id = key_info["id"]
        request.state.auth_method = "api_key"
        
        self._record_auth_event("api_key_auth_success", user.id, request)
        return user
    
    def _authenticate_jwt_token(self, token: str, request: Request) -> User:
        """Authenticate using JWT token"""
        try:
            # Verify and decode token
            payload = self.token_manager.verify_token(token)
            
            # Check token type
            if payload.type != TokenType.BEARER:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            # Get user
            user = self.user_store.get_user_by_id(payload.sub)
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User account is inactive"
                )
            
            # Check session if provided
            if payload.session_id:
                session = self.user_store.get_session(payload.session_id)
                if not session or not session.is_active:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Session is invalid or expired"
                    )
                request.state.session_id = payload.session_id
            
            # Store token info in request state
            request.state.token_scopes = payload.scopes
            request.state.auth_method = "jwt"
            
            self._record_auth_event("jwt_auth_success", user.id, request)
            return user
            
        except HTTPException:
            self._record_auth_event("jwt_auth_failed", None, request)
            raise
    
    def _record_auth_event(self, event_type: str, user_id: str, request: Request, details: Dict[str, Any] = None):
        """Record authentication event"""
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
            timestamp=datetime.now().isoformat(),
            details=details or {},
            success="success" in event_type
        )
        
        self.user_store.record_security_event(event)


# Dependency functions for FastAPI

async def get_current_user(request: Request) -> User:
    """Get current authenticated user"""
    if not hasattr(request.state, 'user') or not request.state.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return request.state.user


async def get_current_active_user(request: Request) -> User:
    """Get current active user"""
    user = await get_current_user(request)
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    return user


def require_scope(required_scope: PermissionScope):
    """Dependency to require specific permission scope"""
    async def _check_scope(request: Request, user: User = Depends(get_current_active_user)) -> User:
        # Get user scopes (from token or user profile)
        user_scopes = getattr(request.state, 'token_scopes', user.scopes)
        
        if not check_permission(user_scopes, required_scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {required_scope.value}"
            )
        
        return user
    
    return _check_scope


def require_admin():
    """Dependency to require admin privileges"""
    return require_scope(PermissionScope.ADMIN)


async def verify_api_key(api_key: str = None) -> Dict[str, Any]:
    """Verify API key and return key information"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    key_hash = token_manager.hash_api_key(api_key)
    key_info = user_store.get_api_key_info(key_hash)
    
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return key_info


# Export instances and functions
__all__ = [
    "AuthMiddleware",
    "auth_config", 
    "token_manager",
    "user_store",
    "get_current_user",
    "get_current_active_user",
    "require_scope",
    "require_admin",
    "verify_api_key"
]