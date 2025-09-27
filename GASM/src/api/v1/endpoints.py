"""
Core API Endpoints - Migrated from fastapi_endpoint.py

Enhanced version of the original FastAPI endpoints with proper structure,
improved error handling, and integration with spatial agent components.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import torch
import logging
import asyncio
from datetime import datetime
import json
import os
from contextlib import asynccontextmanager

# Import models from the models package
from ..models.base import BaseResponse, ErrorResponse, HealthResponse
from ..models.gasm import (
    TextProcessingRequest, TextProcessingResponse,
    GeometricAnalysisRequest, GeometricAnalysisResponse,
    ComparisonRequest, ComparisonResponse,
    BatchProcessingRequest, BatchProcessingResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router for core endpoints
router = APIRouter()

# Global model instance - will be moved to dependency injection
model_instance = None

# Import GASM components with fallback handling
try:
    from gasm_llm_layer import GASMEnhancedLLM, GASMTokenEmbedding
    from gasm.utils import check_se3_invariance
    from gasm.core import GASM
    GASM_LLM_AVAILABLE = True
    logger.info("✅ GASM LLM layer loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ GASM LLM layer not available: {e}")
    GASM_LLM_AVAILABLE = False

# Import weight persistence utilities
try:
    from tools.dev.utils_weights import handle_gasm_weights, get_weights_info, should_force_regenerate
    WEIGHT_UTILS_AVAILABLE = True
    logger.info("✅ Weight persistence utilities loaded")
except ImportError as e:
    logger.warning(f"⚠️ Weight utilities not available: {e}")
    WEIGHT_UTILS_AVAILABLE = False

def get_model():
    """
    Dependency to get the model instance with proper error handling.
    
    Returns:
        Model instance or raises HTTP 503 if not available
    """
    global model_instance
    if model_instance is None:
        raise HTTPException(
            status_code=503, 
            detail="GASM model not loaded. Please check system status."
        )
    return model_instance

@router.get("/", response_model=Dict[str, Any])
async def root():
    """
    API root endpoint with comprehensive service information.
    
    Returns:
        Service metadata and available endpoints
    """
    return {
        "message": "GASM-Roboting API v1.0",
        "version": "1.0.0",
        "description": "Comprehensive API for Geometric Assembly State Machine and Spatial Agents",
        "status": "operational",
        "features": {
            "gasm_processing": GASM_LLM_AVAILABLE,
            "weight_persistence": WEIGHT_UTILS_AVAILABLE,
            "spatial_agents": True,
            "batch_processing": True
        },
        "endpoints": {
            "health": "GET /health - System health check",
            "info": "GET /info - Model information",
            "process": "POST /process - Process text with GASM enhancement",
            "analyze": "POST /analyze - Geometric analysis of text",
            "compare": "POST /compare - Compare geometric vs standard processing",
            "batch": "POST /batch - Process multiple texts",
            "weights": "GET /download-weights - Download model weights",
            "debug": "GET /debug-info - System debug information"
        },
        "documentation": {
            "openapi": "/openapi.json",
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns:
        System health status with detailed metrics
    """
    global model_instance
    
    # System health metrics
    health_data = {
        "status": "healthy" if model_instance is not None else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_instance is not None,
        "services": {
            "gasm_llm": GASM_LLM_AVAILABLE,
            "weight_utils": WEIGHT_UTILS_AVAILABLE,
            "spatial_agents": True
        }
    }
    
    # Memory usage information
    memory_info = {}
    try:
        if torch.cuda.is_available():
            memory_info["gpu_memory"] = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
                "device": torch.cuda.get_device_name()
            }
        
        # System memory check
        try:
            import psutil
            vm = psutil.virtual_memory()
            memory_info["system_memory"] = {
                "used": vm.used,
                "total": vm.total,
                "percent": vm.percent,
                "available": vm.available
            }
        except ImportError:
            memory_info["system_memory"] = "psutil not available"
            
    except Exception as e:
        memory_info["error"] = str(e)
    
    health_data["memory_usage"] = memory_info
    health_data["device"] = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    return HealthResponse(**health_data)

@router.get("/info", response_model=Dict[str, Any])
async def model_info(model: Any = Depends(get_model)):
    """
    Get detailed model information and configuration.
    
    Returns:
        Model metadata, configuration, and performance metrics
    """
    try:
        # Basic model information
        info = {
            "model_name": getattr(model, 'base_model_name', 'unknown'),
            "geometry_enabled": getattr(model, 'enable_geometry', False),
            "model_type": type(model).__name__,
            "device": str(next(model.parameters()).device) if hasattr(model, 'parameters') else 'unknown'
        }
        
        # Parameter counts
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            
            info.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": round(model_size_mb, 2)
            })
        
        # GASM configuration
        if hasattr(model, 'gasm_embedding') and hasattr(model.gasm_embedding, 'gasm'):
            gasm = model.gasm_embedding.gasm
            info["gasm_config"] = {
                "hidden_dim": getattr(gasm, 'hidden_dim', None),
                "output_dim": getattr(gasm, 'output_dim', None),
                "max_iterations": getattr(gasm, 'max_iterations', None),
                "feature_dim": getattr(gasm, 'feature_dim', None)
            }
        
        # Weight information
        if WEIGHT_UTILS_AVAILABLE:
            weights_info = get_weights_info("gasm_weights.pth")
            info["weights"] = weights_info
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model information: {str(e)}")

@router.post("/process", response_model=TextProcessingResponse)
async def process_text(
    request: TextProcessingRequest,
    model: Any = Depends(get_model)
):
    """
    Process text with GASM-enhanced LLM capabilities.
    
    Args:
        request: Text processing request with configuration
        
    Returns:
        Processed text with embeddings and geometric information
    """
    start_time = datetime.now()
    
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > request.max_length * 10:  # Rough character limit
            raise HTTPException(status_code=400, detail=f"Text too long. Maximum ~{request.max_length * 10} characters")
        
        # Configure model
        if hasattr(model, 'enable_geometry'):
            model.enable_geometry = request.enable_geometry
        
        # Process text with error handling
        try:
            outputs = model.encode_text(
                request.text,
                return_geometry=request.return_geometry
            )
        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Extract and analyze embeddings
        embeddings = outputs.get('last_hidden_state')
        if embeddings is None:
            raise HTTPException(status_code=500, detail="No embeddings generated")
        
        # Calculate embedding statistics
        embedding_stats = {
            "shape": list(embeddings.shape),
            "mean": float(embeddings.mean()),
            "std": float(embeddings.std()),
            "min": float(embeddings.min()),
            "max": float(embeddings.max()),
            "norm": float(torch.norm(embeddings))
        }
        
        # Prepare response data
        response_data = {
            "success": True,
            "timestamp": start_time.isoformat(),
            "processing_time": processing_time,
            "text_length": len(request.text),
            "model_info": {
                "model_name": getattr(model, 'base_model_name', 'unknown'),
                "geometry_enabled": request.enable_geometry,
                "device": str(next(model.parameters()).device) if hasattr(model, 'parameters') else 'unknown'
            },
            "embedding_stats": embedding_stats
        }
        
        # Add embeddings if requested
        if request.return_embeddings:
            response_data["embeddings"] = embeddings.detach().cpu().numpy().tolist()
        
        # Add geometric information if available and requested
        if request.return_geometry and 'geometric_info' in outputs:
            geometric_info = outputs['geometric_info']
            if geometric_info:
                response_data["geometric_info"] = {
                    "num_sequences": len(geometric_info),
                    "has_curvature": any('output' in info for info in geometric_info),
                    "has_constraints": any('constraints' in info for info in geometric_info),
                    "has_relations": any('relations' in info for info in geometric_info)
                }
                
                # Add geometric statistics
                if request.return_geometry:
                    geo_stats = {}
                    for i, info in enumerate(geometric_info):
                        if 'output' in info:
                            geo_output = info['output']
                            geo_stats[f"sequence_{i}"] = {
                                "shape": list(geo_output.shape) if hasattr(geo_output, 'shape') else "unknown",
                                "norm": float(torch.norm(geo_output)) if hasattr(geo_output, 'norm') else 0.0
                            }
                    response_data["geometric_stats"] = geo_stats
        
        return TextProcessingResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing text: {e}")
        return TextProcessingResponse(
            success=False,
            timestamp=start_time.isoformat(),
            processing_time=(datetime.now() - start_time).total_seconds(),
            text_length=len(request.text) if hasattr(request, 'text') else 0,
            model_info={},
            embedding_stats={},
            error=str(e)
        )

@router.post("/analyze", response_model=GeometricAnalysisResponse)
async def analyze_geometry(
    request: GeometricAnalysisRequest,
    model: Any = Depends(get_model)
):
    """
    Perform geometric analysis of text using GASM capabilities.
    
    Args:
        request: Geometric analysis request with parameters
        
    Returns:
        Detailed geometric analysis results
    """
    start_time = datetime.now()
    
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Enable geometry for analysis
        if hasattr(model, 'enable_geometry'):
            model.enable_geometry = True
        
        # Process text with geometric information
        outputs = model.encode_text(request.text, return_geometry=True)
        
        response_data = {
            "success": True,
            "timestamp": start_time.isoformat(),
            "analysis_type": request.analysis_type
        }
        
        # Perform requested analysis types
        if request.analysis_type in ["full", "curvature"]:
            curvature_analysis = await _perform_curvature_analysis(outputs, model)
            response_data["curvature_analysis"] = curvature_analysis
        
        if request.analysis_type in ["full", "invariance"]:
            invariance_results = await _perform_invariance_analysis(
                model, request.num_invariance_tests, request.tolerance
            )
            response_data["invariance_results"] = invariance_results
        
        # Add geometric properties summary
        if 'geometric_info' in outputs and outputs['geometric_info']:
            geometric_properties = {
                "total_sequences": len(outputs['geometric_info']),
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "analysis_complete": True
            }
            response_data["geometric_properties"] = geometric_properties
        
        return GeometricAnalysisResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error in geometric analysis: {e}")
        return GeometricAnalysisResponse(
            success=False,
            timestamp=start_time.isoformat(),
            analysis_type=request.analysis_type,
            error=str(e)
        )

@router.post("/compare", response_model=ComparisonResponse)
async def compare_processing(
    request: ComparisonRequest,
    model: Any = Depends(get_model)
):
    """
    Compare geometric vs standard processing approaches.
    
    Args:
        request: Comparison request with metrics to analyze
        
    Returns:
        Detailed comparison results
    """
    start_time = datetime.now()
    
    try:
        # Process with geometry enabled
        if hasattr(model, 'enable_geometry'):
            model.enable_geometry = True
        geometric_outputs = model.encode_text(request.text, return_geometry=True)
        
        # Process without geometry
        if hasattr(model, 'enable_geometry'):
            model.enable_geometry = False
        standard_outputs = model.encode_text(request.text, return_geometry=False)
        
        # Extract embeddings
        geometric_embeddings = geometric_outputs.get('last_hidden_state')
        standard_embeddings = standard_outputs.get('last_hidden_state')
        
        if geometric_embeddings is None or standard_embeddings is None:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings for comparison")
        
        # Calculate comparison metrics
        comparison_metrics = {}
        
        for metric in request.metrics:
            if metric == "embedding_norm":
                comparison_metrics["embedding_norm"] = {
                    "geometric": float(torch.norm(geometric_embeddings)),
                    "standard": float(torch.norm(standard_embeddings)),
                    "ratio": float(torch.norm(geometric_embeddings) / torch.norm(standard_embeddings))
                }
            
            elif metric == "attention_patterns":
                geo_attention = torch.softmax(
                    geometric_embeddings @ geometric_embeddings.transpose(-2, -1), dim=-1
                )
                std_attention = torch.softmax(
                    standard_embeddings @ standard_embeddings.transpose(-2, -1), dim=-1
                )
                
                comparison_metrics["attention_patterns"] = {
                    "geometric_entropy": float(torch.sum(-geo_attention * torch.log(geo_attention + 1e-9))),
                    "standard_entropy": float(torch.sum(-std_attention * torch.log(std_attention + 1e-9))),
                    "pattern_difference": float(torch.norm(geo_attention - std_attention))
                }
            
            elif metric == "geometric_consistency":
                comparison_metrics["geometric_consistency"] = {
                    "has_geometric_info": 'geometric_info' in geometric_outputs,
                    "embedding_difference": float(torch.norm(geometric_embeddings - standard_embeddings)),
                    "relative_change": float(
                        torch.norm(geometric_embeddings - standard_embeddings) / torch.norm(standard_embeddings)
                    )
                }
        
        return ComparisonResponse(
            success=True,
            timestamp=start_time.isoformat(),
            geometric_results={
                "embedding_stats": {
                    "shape": list(geometric_embeddings.shape),
                    "mean": float(geometric_embeddings.mean()),
                    "std": float(geometric_embeddings.std()),
                    "norm": float(torch.norm(geometric_embeddings))
                }
            },
            standard_results={
                "embedding_stats": {
                    "shape": list(standard_embeddings.shape),
                    "mean": float(standard_embeddings.mean()),
                    "std": float(standard_embeddings.std()),
                    "norm": float(torch.norm(standard_embeddings))
                }
            },
            comparison_metrics=comparison_metrics
        )
        
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        return ComparisonResponse(
            success=False,
            timestamp=start_time.isoformat(),
            geometric_results={},
            standard_results={},
            comparison_metrics={},
            error=str(e)
        )

@router.post("/batch", response_model=BatchProcessingResponse)
async def batch_process(
    request: BatchProcessingRequest,
    model: Any = Depends(get_model)
):
    """
    Process multiple texts in batch with optimization.
    
    Args:
        request: Batch processing request
        
    Returns:
        Batch processing results with statistics
    """
    start_time = datetime.now()
    
    try:
        # Validate batch size
        if len(request.texts) > 100:
            raise HTTPException(status_code=400, detail="Batch size too large. Maximum 100 texts.")
        
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided for processing")
        
        # Configure model
        if hasattr(model, 'enable_geometry'):
            model.enable_geometry = request.enable_geometry
        
        processing_times = []
        individual_results = []
        successful_processes = 0
        
        for i, text in enumerate(request.texts):
            text_start = datetime.now()
            
            try:
                if not text.strip():
                    logger.warning(f"Empty text at index {i}, skipping")
                    continue
                
                outputs = model.encode_text(text, return_geometry=False)
                embeddings = outputs.get('last_hidden_state')
                
                if embeddings is not None:
                    processing_time = (datetime.now() - text_start).total_seconds()
                    processing_times.append(processing_time)
                    successful_processes += 1
                    
                    if not request.return_summary:
                        individual_results.append({
                            "text_index": i,
                            "text_length": len(text),
                            "processing_time": processing_time,
                            "embedding_norm": float(torch.norm(embeddings)),
                            "success": True
                        })
                else:
                    logger.warning(f"No embeddings generated for text {i}")
                    if not request.return_summary:
                        individual_results.append({
                            "text_index": i,
                            "text_length": len(text),
                            "processing_time": 0.0,
                            "success": False,
                            "error": "No embeddings generated"
                        })
                        
            except Exception as text_error:
                logger.error(f"Error processing text {i}: {text_error}")
                if not request.return_summary:
                    individual_results.append({
                        "text_index": i,
                        "text_length": len(text) if text else 0,
                        "processing_time": 0.0,
                        "success": False,
                        "error": str(text_error)
                    })
        
        # Calculate batch summary
        total_time = sum(processing_times)
        batch_summary = {
            "total_texts": len(request.texts),
            "successful_processes": successful_processes,
            "success_rate": successful_processes / len(request.texts) * 100,
            "total_processing_time": total_time,
            "average_processing_time": total_time / max(1, len(processing_times)),
            "texts_per_second": len(processing_times) / max(0.001, total_time),
            "geometry_enabled": request.enable_geometry,
            "total_characters": sum(len(text) for text in request.texts),
            "average_text_length": sum(len(text) for text in request.texts) / len(request.texts)
        }
        
        return BatchProcessingResponse(
            success=successful_processes > 0,
            timestamp=start_time.isoformat(),
            num_texts=len(request.texts),
            processing_times=processing_times,
            batch_summary=batch_summary,
            individual_results=individual_results if not request.return_summary else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return BatchProcessingResponse(
            success=False,
            timestamp=start_time.isoformat(),
            num_texts=len(request.texts) if hasattr(request, 'texts') else 0,
            processing_times=[],
            batch_summary={},
            error=str(e)
        )

@router.get("/download-weights")
async def download_weights():
    """
    Download the current GASM weight file for model persistence.
    
    Returns:
        File download response with model weights
    """
    weight_file = "gasm_weights.pth"
    
    if not os.path.exists(weight_file):
        raise HTTPException(
            status_code=404, 
            detail="Weight file not found. Model may not be initialized yet."
        )
    
    try:
        file_size = os.path.getsize(weight_file)
        logger.info(f"📥 Weight file download requested: {weight_file} ({file_size} bytes)")
        
        return FileResponse(
            weight_file,
            filename="gasm_weights.pth",
            media_type="application/octet-stream",
            headers={
                "Content-Description": "GASM Model Weights",
                "Content-Disposition": "attachment; filename=gasm_weights.pth",
                "X-File-Size": str(file_size)
            }
        )
    except Exception as e:
        logger.error(f"Error downloading weights: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading weights: {str(e)}")

@router.get("/debug-info")
async def debug_info():
    """
    Get comprehensive system debug information.
    
    Returns:
        Detailed system state and configuration
    """
    try:
        import sys
        import platform
        
        debug_data = {
            "system": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "working_directory": os.getcwd(),
                "pytorch_version": torch.__version__ if torch else "not available",
                "cuda_available": torch.cuda.is_available() if torch else False,
                "cuda_version": torch.version.cuda if torch and torch.cuda.is_available() else None
            },
            "model": {
                "loaded": model_instance is not None,
                "type": type(model_instance).__name__ if model_instance else None
            },
            "services": {
                "gasm_llm": GASM_LLM_AVAILABLE,
                "weight_utils": WEIGHT_UTILS_AVAILABLE
            },
            "files": []
        }
        
        # List relevant files
        try:
            for file in sorted(os.listdir(".")):
                if file.endswith(('.pth', '.json', '.py', '.md')):
                    try:
                        file_path = os.path.join(".", file)
                        if os.path.isfile(file_path):
                            stat = os.stat(file_path)
                            debug_data["files"].append({
                                "name": file,
                                "size_bytes": stat.st_size,
                                "size_mb": round(stat.st_size / (1024 * 1024), 3),
                                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                            })
                    except Exception as file_error:
                        logger.warning(f"Error reading file {file}: {file_error}")
        except Exception as list_error:
            debug_data["files_error"] = str(list_error)
        
        # Weight file specific info
        weight_file = "gasm_weights.pth"
        if os.path.exists(weight_file):
            try:
                stat = os.stat(weight_file)
                debug_data["weight_file"] = {
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 3),
                    "full_path": os.path.abspath(weight_file),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            except Exception as weight_error:
                debug_data["weight_file"] = {"exists": True, "error": str(weight_error)}
        else:
            debug_data["weight_file"] = {"exists": False}
        
        return debug_data
        
    except Exception as e:
        logger.error(f"Error generating debug info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting debug info: {str(e)}")

# Helper functions for analysis
async def _perform_curvature_analysis(outputs: Dict[str, Any], model: Any) -> Dict[str, Any]:
    """Perform curvature analysis on geometric outputs."""
    try:
        geometric_info = outputs.get('geometric_info', [])
        if not geometric_info:
            return {"error": "No geometric information available"}
        
        curvature_stats = []
        for i, info in enumerate(geometric_info):
            if 'output' in info:
                geo_output = info['output']
                if hasattr(geo_output, 'norm'):
                    curvature_norm = torch.norm(geo_output, dim=1)
                    curvature_stats.append({
                        "sequence": i,
                        "mean": float(curvature_norm.mean()),
                        "std": float(curvature_norm.std()),
                        "min": float(curvature_norm.min()),
                        "max": float(curvature_norm.max())
                    })
        
        return {
            "per_sequence": curvature_stats,
            "global_stats": {
                "num_sequences": len(curvature_stats),
                "avg_mean_curvature": sum(s["mean"] for s in curvature_stats) / max(1, len(curvature_stats))
            }
        }
        
    except Exception as e:
        logger.error(f"Curvature analysis failed: {e}")
        return {"error": str(e)}

async def _perform_invariance_analysis(model: Any, num_tests: int, tolerance: float) -> Dict[str, Any]:
    """Perform SE(3) invariance analysis."""
    try:
        # Create test data for invariance check
        test_points = torch.randn(10, 3)
        
        # Get feature dimension from model
        feature_dim = 768  # Default for many models
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'config'):
            feature_dim = getattr(model.base_model.config, 'hidden_size', feature_dim)
        
        test_features = torch.randn(10, feature_dim)
        test_relations = torch.randn(10, 10, 16)
        
        # Test with simplified GASM model
        try:
            from gasm.core import GASM
            gasm_model = GASM(
                feature_dim=feature_dim,
                hidden_dim=256,
                output_dim=3
            )
            
            is_invariant = check_se3_invariance(
                gasm_model,
                test_points,
                test_features,
                test_relations,
                num_tests=num_tests,
                tolerance=tolerance
            )
            
            return {
                "is_invariant": is_invariant,
                "num_tests": num_tests,
                "tolerance": tolerance,
                "test_type": "SE(3) invariance"
            }
            
        except ImportError:
            return {
                "is_invariant": None,
                "error": "GASM core not available for invariance testing"
            }
            
    except Exception as e:
        logger.error(f"Invariance analysis failed: {e}")
        return {
            "is_invariant": None,
            "error": str(e)
        }

# Model initialization function - will be called by application lifespan
async def initialize_model():
    """Initialize the GASM model with proper error handling and logging."""
    global model_instance
    
    if not GASM_LLM_AVAILABLE:
        logger.warning("GASM LLM layer not available, skipping model initialization")
        return None
    
    try:
        logger.info("Initializing GASM-enhanced LLM model...")
        
        # Log weight persistence status
        if WEIGHT_UTILS_AVAILABLE:
            weights_info = get_weights_info("gasm_weights.pth")
            force_regen = should_force_regenerate()
            
            logger.info("=" * 60)
            logger.info("🚀 GASM Weight Persistence Status")
            logger.info("=" * 60)
            logger.info(f"📁 Weight file: gasm_weights.pth")
            logger.info(f"✅ Exists: {weights_info['exists']}")
            if weights_info['exists']:
                logger.info(f"📊 Size: {weights_info['size_mb']} MB")
            logger.info(f"🔄 Force regeneration: {force_regen}")
            logger.info("=" * 60)
        
        # Initialize model
        model_instance = GASMEnhancedLLM(
            base_model_name="distilbert-base-uncased",
            gasm_hidden_dim=256,
            gasm_output_dim=128,
            enable_geometry=True
        )
        
        # Apply weight persistence
        if (WEIGHT_UTILS_AVAILABLE and 
            hasattr(model_instance, 'gasm_layer') and 
            hasattr(model_instance.gasm_layer, 'gasm_model')):
            
            device = next(model_instance.gasm_layer.gasm_model.parameters()).device
            weights_handled = handle_gasm_weights(
                model_instance.gasm_layer.gasm_model, 
                device, 
                "gasm_weights.pth"
            )
            if not weights_handled:
                logger.warning("⚠️ Weight persistence failed")
        
        logger.info("✅ GASM model initialized successfully")
        return model_instance
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize GASM model: {e}")
        model_instance = None
        return None

async def cleanup_model():
    """Cleanup model resources."""
    global model_instance
    if model_instance is not None:
        logger.info("Cleaning up GASM model...")
        # Perform any necessary cleanup
        model_instance = None
        logger.info("Model cleanup completed")