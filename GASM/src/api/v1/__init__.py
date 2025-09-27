"""
API Version 1

Main API router and endpoint definitions for GASM-Roboting v1 API.
"""

from fastapi import APIRouter
from .endpoints import router as endpoints_router
from .spatial_endpoints import router as spatial_router
from .gasm_endpoints import router as gasm_router

# Create main v1 router
router = APIRouter(prefix="/v1", tags=["v1"])

# Include sub-routers
router.include_router(endpoints_router, prefix="/core", tags=["core"])
router.include_router(spatial_router, prefix="/spatial", tags=["spatial"])
router.include_router(gasm_router, prefix="/gasm", tags=["gasm"])

__all__ = ["router"]