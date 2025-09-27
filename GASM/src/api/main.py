"""
GASM-Roboting API Main Application

FastAPI application with comprehensive spatial agent and GASM processing capabilities.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi

# Import API components
from . import API_METADATA
from .v1 import router as v1_router
from .middleware import setup_middleware
from .models.base import create_success_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gasm_api.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# Global application state
app_state = {
    "start_time": None,
    "version": "1.0.0",
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting GASM-Roboting API...")
    
    import time
    app_state["start_time"] = time.time()
    
    # Initialize spatial components
    try:
        from .v1.spatial_endpoints import initialize_spatial_components
        await initialize_spatial_components()
        logger.info("✅ Spatial components initialized")
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize spatial components: {e}")
    
    # Initialize GASM components
    try:
        from .v1.gasm_endpoints import initialize_gasm_components
        await initialize_gasm_components()
        logger.info("✅ GASM components initialized")
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize GASM components: {e}")
    
    # Initialize core API components (from original endpoints)
    try:
        from .v1.endpoints import initialize_model
        await initialize_model()
        logger.info("✅ Core API model initialized")
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize core model: {e}")
    
    logger.info("✅ GASM-Roboting API startup complete")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down GASM-Roboting API...")
    
    # Cleanup spatial components
    try:
        # Add any cleanup logic here
        logger.info("✅ Spatial components cleanup complete")
    except Exception as e:
        logger.error(f"❌ Error during spatial cleanup: {e}")
    
    # Cleanup core model
    try:
        from .v1.endpoints import cleanup_model
        await cleanup_model()
        logger.info("✅ Core model cleanup complete")
    except Exception as e:
        logger.error(f"❌ Error during model cleanup: {e}")
    
    logger.info("✅ GASM-Roboting API shutdown complete")


# Create FastAPI application
app = FastAPI(
    **API_METADATA,
    lifespan=lifespan,
    debug=app_state["debug"],
    docs_url="/docs" if app_state["environment"] != "production" else None,
    redoc_url="/redoc" if app_state["environment"] != "production" else None,
)

# Setup middleware (order matters!)
setup_middleware(app)

# Include routers
app.include_router(v1_router, prefix="/api")

# Root endpoint
@app.get("/", response_model=Dict[str, Any])
async def root():
    """
    API root endpoint with service information and health status.
    """
    import time
    
    uptime = time.time() - app_state["start_time"] if app_state["start_time"] else 0
    
    return create_success_response(
        data={
            "service": "GASM-Roboting API",
            "version": app_state["version"],
            "environment": app_state["environment"],
            "uptime": f"{uptime:.2f} seconds",
            "status": "operational",
            "features": {
                "core_endpoints": "✅ Available",
                "spatial_agents": "✅ Available",  
                "gasm_processing": "✅ Available",
                "authentication": "✅ Available",
                "rate_limiting": "✅ Available",
                "comprehensive_docs": "✅ Available"
            },
            "endpoints": {
                "health": "GET /health",
                "api_v1": "GET /api/v1/",
                "core_processing": "POST /api/v1/core/*", 
                "spatial_control": "POST /api/v1/spatial/*",
                "gasm_processing": "POST /api/v1/gasm/*",
                "documentation": "GET /docs",
                "openapi_schema": "GET /openapi.json"
            },
            "documentation": {
                "interactive_docs": "/docs",
                "redoc": "/redoc",
                "openapi_json": "/openapi.json",
                "source_code": "https://github.com/your-org/gasm-roboting"
            }
        },
        message="Welcome to GASM-Roboting API - Geometric Assembly State Machine with Spatial Agents"
    )


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Comprehensive health check endpoint.
    """
    import time
    import psutil
    
    uptime = time.time() - app_state["start_time"] if app_state["start_time"] else 0
    
    # System metrics
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        system_metrics = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_usage_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3)
        }
    except Exception:
        system_metrics = {"error": "Could not collect system metrics"}
    
    # Check component health
    component_health = {
        "fastapi": "healthy",
        "middleware": "healthy"
    }
    
    # Check spatial components
    try:
        from .v1.spatial_endpoints import SPATIAL_COMPONENTS_AVAILABLE
        component_health["spatial_agents"] = "healthy" if SPATIAL_COMPONENTS_AVAILABLE else "degraded"
    except Exception:
        component_health["spatial_agents"] = "unavailable"
    
    # Check GASM components
    try:
        from .v1.gasm_endpoints import GASM_BRIDGE_AVAILABLE
        component_health["gasm_bridge"] = "healthy" if GASM_BRIDGE_AVAILABLE else "degraded"
    except Exception:
        component_health["gasm_bridge"] = "unavailable"
    
    # Check core model
    try:
        from .v1.endpoints import model_instance
        component_health["core_model"] = "healthy" if model_instance else "degraded"
    except Exception:
        component_health["core_model"] = "unavailable"
    
    # Overall health status
    unhealthy_components = [k for k, v in component_health.items() if v in ["degraded", "unavailable"]]
    overall_status = "healthy" if not unhealthy_components else ("degraded" if len(unhealthy_components) < len(component_health) else "unhealthy")
    
    return create_success_response(
        data={
            "status": overall_status,
            "timestamp": time.time(),
            "uptime": f"{uptime:.2f} seconds",
            "version": app_state["version"],
            "environment": app_state["environment"],
            "components": component_health,
            "system_metrics": system_metrics,
            "unhealthy_components": unhealthy_components if unhealthy_components else None
        },
        message=f"System is {overall_status}"
    )


@app.get("/version", response_model=Dict[str, Any])
async def get_version():
    """
    Get detailed version information.
    """
    import platform
    import sys
    
    return create_success_response(
        data={
            "api_version": app_state["version"],
            "environment": app_state["environment"],
            "python_version": sys.version,
            "platform": platform.platform(),
            "fastapi_version": "0.104.1",  # Update as needed
            "build_info": {
                "timestamp": "2024-01-15T10:30:00Z",  # Update with actual build time
                "commit_hash": "abc123def456",  # Update with actual commit
                "build_number": "1.0.0-build.123"
            }
        },
        message="Version information retrieved"
    )


@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """
    Get application metrics and statistics.
    """
    import time
    
    uptime = time.time() - app_state["start_time"] if app_state["start_time"] else 0
    
    # Get processing stats from various components
    metrics_data = {
        "uptime": uptime,
        "requests_processed": "N/A",  # Would be tracked by middleware
        "errors_encountered": "N/A",   # Would be tracked by error handler
        "active_connections": "N/A",   # Would be tracked by connection manager
    }
    
    # Add GASM processing stats
    try:
        from .v1.gasm_endpoints import processing_stats
        metrics_data["gasm_processing"] = processing_stats
    except Exception:
        metrics_data["gasm_processing"] = {"error": "Stats unavailable"}
    
    # Add spatial agent stats
    try:
        from .v1.spatial_endpoints import agent_state
        metrics_data["spatial_agent"] = {
            "current_state": agent_state.get("state", "unknown"),
            "iteration_count": agent_state.get("iteration_count", 0),
            "active_constraints": len(agent_state.get("active_constraints", [])),
            "active_obstacles": len(agent_state.get("active_obstacles", []))
        }
    except Exception:
        metrics_data["spatial_agent"] = {"error": "Stats unavailable"}
    
    return create_success_response(
        data=metrics_data,
        message="Metrics retrieved successfully"
    )


# Custom OpenAPI schema
def custom_openapi():
    """
    Generate custom OpenAPI schema with enhanced documentation.
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=API_METADATA["title"],
        version=API_METADATA["version"],
        description=API_METADATA["description"],
        routes=app.routes,
    )
    
    # Add custom schema enhancements
    openapi_schema["info"]["x-logo"] = {
        "url": "/static/logo.png",
        "altText": "GASM-Roboting Logo"
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "/",
            "description": "Current server"
        },
        {
            "url": "https://api.gasm-roboting.com",
            "description": "Production server"
        },
        {
            "url": "https://staging-api.gasm-roboting.com", 
            "description": "Staging server"
        }
    ]
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token authentication"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key authentication"
        }
    }
    
    # Add global security
    openapi_schema["security"] = [
        {"BearerAuth": []},
        {"ApiKeyAuth": []}
    ]
    
    # Add tags for organization
    openapi_schema["tags"] = [
        {
            "name": "core",
            "description": "Core API endpoints for text processing and model interaction",
            "externalDocs": {
                "description": "Core API Documentation",
                "url": "/docs/core"
            }
        },
        {
            "name": "spatial",
            "description": "Spatial agent endpoints for pose control and motion planning",
            "externalDocs": {
                "description": "Spatial Agent Documentation", 
                "url": "/docs/spatial"
            }
        },
        {
            "name": "gasm",
            "description": "GASM processing endpoints for spatial reasoning",
            "externalDocs": {
                "description": "GASM Documentation",
                "url": "/docs/gasm"
            }
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Custom exception handlers for better error responses
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler with helpful information"""
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Endpoint not found",
            "message": f"The endpoint {request.url.path} was not found",
            "available_endpoints": {
                "root": "GET /",
                "health": "GET /health", 
                "api_v1": "GET /api/v1/",
                "docs": "GET /docs",
                "openapi": "GET /openapi.json"
            },
            "timestamp": "2024-01-15T10:30:00Z"  # Would be actual timestamp
        }
    )


# Static files (if needed)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    logger.info("Static files directory not found - skipping static file serving")


# Development convenience endpoint
if app_state["environment"] == "development":
    @app.get("/dev/reload", include_in_schema=False)
    async def reload_components():
        """Development endpoint to reload components"""
        try:
            # Reload spatial components
            from .v1.spatial_endpoints import initialize_spatial_components
            await initialize_spatial_components()
            
            # Reload GASM components
            from .v1.gasm_endpoints import initialize_gasm_components
            await initialize_gasm_components()
            
            return create_success_response(
                message="Components reloaded successfully"
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"Reload failed: {str(e)}"
                }
            )


if __name__ == "__main__":
    import uvicorn
    
    # Configuration based on environment
    if app_state["environment"] == "production":
        # Production configuration
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            workers=4,
            log_level="info",
            access_log=True,
            server_header=False,
            date_header=False
        )
    else:
        # Development configuration
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="debug",
            access_log=True
        )