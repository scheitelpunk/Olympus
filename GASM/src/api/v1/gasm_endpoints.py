"""
GASM-Specific API Endpoints

Advanced endpoints for GASM processing, geometric analysis, and spatial reasoning.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from datetime import datetime

# Import models
from ..models.gasm import (
    GASMProcessRequest, GASMProcessResponse, GASMAnalysisRequest, GASMAnalysisResponse,
    SpatialEntity, SpatialRelationship, Constraint, SE3Pose
)
from ..models.base import BaseResponse, create_success_response
from ..middleware.auth import get_current_user, require_scope, PermissionScope

# Import GASM components
try:
    from ...spatial_agent.gasm_bridge import GASMBridge, create_bridge, ConstraintType as BridgeConstraintType
    GASM_BRIDGE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"GASM bridge not available: {e}")
    GASM_BRIDGE_AVAILABLE = False

try:
    from gasm_core import GASM, EnhancedGASM  # Fallback to core GASM if available
    GASM_CORE_AVAILABLE = True
except ImportError:
    GASM_CORE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global GASM components
gasm_bridge: Optional[GASMBridge] = None
processing_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "average_processing_time": 0.0,
    "start_time": datetime.now().timestamp()
}


async def initialize_gasm_components():
    """Initialize GASM components"""
    global gasm_bridge
    
    try:
        if GASM_BRIDGE_AVAILABLE:
            gasm_bridge = create_bridge()
            logger.info("✅ GASM bridge initialized")
        else:
            logger.warning("⚠️ GASM bridge not available")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize GASM components: {e}")


def get_gasm_bridge():
    """Dependency to ensure GASM bridge is available"""
    if not GASM_BRIDGE_AVAILABLE or not gasm_bridge:
        raise HTTPException(
            status_code=503,
            detail="GASM processing not available. Please check system configuration."
        )
    return gasm_bridge


@router.get("/status", response_model=Dict[str, Any])
async def get_gasm_status():
    """
    Get GASM system status and processing statistics.
    """
    try:
        uptime = datetime.now().timestamp() - processing_stats["start_time"]
        
        status_data = {
            "gasm_bridge_available": GASM_BRIDGE_AVAILABLE,
            "gasm_core_available": GASM_CORE_AVAILABLE,
            "bridge_initialized": gasm_bridge is not None,
            "processing_stats": {
                **processing_stats,
                "uptime": uptime,
                "success_rate": (processing_stats["successful_requests"] / 
                               max(1, processing_stats["total_requests"]) * 100)
            },
            "supported_constraints": [constraint.value for constraint in BridgeConstraintType] if GASM_BRIDGE_AVAILABLE else [],
            "capabilities": {
                "natural_language_processing": GASM_BRIDGE_AVAILABLE,
                "spatial_reasoning": GASM_BRIDGE_AVAILABLE,
                "constraint_generation": GASM_BRIDGE_AVAILABLE,
                "pose_estimation": GASM_BRIDGE_AVAILABLE,
                "geometric_analysis": GASM_CORE_AVAILABLE
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return create_success_response(
            data=status_data,
            message="GASM status retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting GASM status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GASM status: {str(e)}")


@router.post("/process", response_model=GASMProcessResponse)
async def process_spatial_instruction(
    request: GASMProcessRequest,
    background_tasks: BackgroundTasks,
    user=Depends(require_scope(PermissionScope.GASM_PROCESS)),
    bridge=Depends(get_gasm_bridge)
):
    """
    Process natural language spatial instructions through GASM.
    
    Converts natural language to structured spatial constraints and target poses.
    """
    start_time = datetime.now()
    
    try:
        processing_stats["total_requests"] += 1
        
        # Process through GASM bridge
        gasm_result = bridge.process(request.instruction)
        
        if not gasm_result.get("success", False):
            processing_stats["failed_requests"] += 1
            error_msg = gasm_result.get("error_message", "GASM processing failed")
            
            return GASMProcessResponse(
                success=False,
                timestamp=start_time.isoformat(),
                message=f"GASM processing failed: {error_msg}",
                constraints=[],
                target_poses={},
                confidence=0.0,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=error_msg
            )
        
        # Convert GASM results to API format
        constraints = []
        for constraint_data in gasm_result.get("constraints", []):
            try:
                constraint = Constraint(
                    type=constraint_data.get("type", "distance"),
                    parameters=constraint_data.get("parameters", {}),
                    priority=constraint_data.get("priority", 0.5),
                    tolerance=constraint_data.get("tolerance", {"position": 0.01, "orientation": 0.1})
                )
                constraints.append(constraint)
            except Exception as constraint_error:
                logger.warning(f"Could not convert constraint: {constraint_error}")
        
        # Convert target poses
        target_poses = {}
        for obj_name, pose_data in gasm_result.get("target_poses", {}).items():
            try:
                pose = SE3Pose(
                    position=pose_data.get("position", [0.0, 0.0, 0.0]),
                    orientation=pose_data.get("orientation", [1.0, 0.0, 0.0, 0.0]),
                    frame_id=pose_data.get("frame_id", "world"),
                    confidence=pose_data.get("confidence", 1.0)
                )
                target_poses[obj_name] = pose
            except Exception as pose_error:
                logger.warning(f"Could not convert pose for {obj_name}: {pose_error}")
        
        # Update statistics
        processing_stats["successful_requests"] += 1
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Update average processing time
        total_requests = processing_stats["total_requests"]
        current_avg = processing_stats["average_processing_time"]
        processing_stats["average_processing_time"] = (
            (current_avg * (total_requests - 1) + execution_time) / total_requests
        )
        
        return GASMProcessResponse(
            success=True,
            timestamp=start_time.isoformat(),
            message="Spatial instruction processed successfully",
            constraints=constraints,
            target_poses=target_poses,
            confidence=gasm_result.get("confidence", 0.5),
            execution_time=execution_time,
            debug_info=gasm_result.get("debug_info") if request.return_debug_info else None
        )
        
    except HTTPException:
        processing_stats["failed_requests"] += 1
        raise
    except Exception as e:
        processing_stats["failed_requests"] += 1
        logger.error(f"GASM processing failed: {e}")
        
        return GASMProcessResponse(
            success=False,
            timestamp=start_time.isoformat(),
            message=f"GASM processing failed: {str(e)}",
            constraints=[],
            target_poses={},
            confidence=0.0,
            execution_time=(datetime.now() - start_time).total_seconds(),
            error=str(e)
        )


@router.post("/analyze", response_model=GASMAnalysisResponse)
async def analyze_spatial_scene(
    request: GASMAnalysisRequest,
    user=Depends(require_scope(PermissionScope.GASM_PROCESS)),
    bridge=Depends(get_gasm_bridge)
):
    """
    Analyze spatial scene with entities and relationships.
    
    Performs deep spatial analysis and relationship discovery.
    """
    start_time = datetime.now()
    
    try:
        # Create spatial analysis instruction
        if request.relationships:
            instruction = f"Analyze the spatial arrangement of {', '.join(request.entities)} with relationships: {', '.join(request.relationships)}"
        else:
            instruction = f"Analyze the spatial arrangement of {', '.join(request.entities)}"
        
        # Process through GASM
        gasm_result = bridge.process(instruction)
        
        if not gasm_result.get("success", False):
            return GASMAnalysisResponse(
                success=False,
                timestamp=start_time.isoformat(),
                message=f"GASM analysis failed: {gasm_result.get('error_message', 'Unknown error')}",
                entities=[],
                relationships=[],
                geometric_properties={},
                analysis_depth=request.analysis_depth,
                error=gasm_result.get("error_message", "Analysis failed")
            )
        
        # Convert results to entities
        entities = []
        target_poses = gasm_result.get("target_poses", {})
        
        for entity_name in request.entities:
            if entity_name in target_poses:
                pose_data = target_poses[entity_name]
                pose = SE3Pose(
                    position=pose_data.get("position", [0.0, 0.0, 0.0]),
                    orientation=pose_data.get("orientation", [1.0, 0.0, 0.0, 0.0]),
                    frame_id=pose_data.get("frame_id", "world"),
                    confidence=pose_data.get("confidence", 1.0)
                )
                
                entity = SpatialEntity(
                    name=entity_name,
                    pose=pose,
                    geometry={"type": "unknown", "size": [0.1, 0.1, 0.1]},  # Default geometry
                    properties={"analyzed": True, "confidence": pose.confidence}
                )
                entities.append(entity)
            else:
                # Create default entity if not in poses
                default_pose = SE3Pose(
                    position=[0.0, 0.0, 0.0],
                    orientation=[1.0, 0.0, 0.0, 0.0],
                    confidence=0.1
                )
                entity = SpatialEntity(
                    name=entity_name,
                    pose=default_pose,
                    properties={"analyzed": False, "reason": "not_found_in_analysis"}
                )
                entities.append(entity)
        
        # Extract relationships from constraints
        relationships = []
        for constraint_data in gasm_result.get("constraints", []):
            if constraint_data.get("subject") and constraint_data.get("target"):
                relationship = SpatialRelationship(
                    subject=constraint_data["subject"],
                    predicate=constraint_data.get("type", "unknown"),
                    object=constraint_data["target"],
                    parameters=constraint_data.get("parameters", {}),
                    confidence=constraint_data.get("priority", 0.5)
                )
                relationships.append(relationship)
        
        # Add relationships from the request if they weren't inferred
        if request.relationships:
            for rel_desc in request.relationships:
                # Simple parsing - in a real implementation, this would be more sophisticated
                relationship = SpatialRelationship(
                    subject="entity_1",
                    predicate=rel_desc.lower(),
                    object="entity_2",
                    confidence=0.8
                )
                relationships.append(relationship)
        
        # Calculate geometric properties
        geometric_properties = {
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "analysis_depth": request.analysis_depth,
            "coordinate_system": "cartesian",
            "bounding_box": _calculate_bounding_box(entities),
            "center_of_mass": _calculate_center_of_mass(entities),
            "confidence_score": gasm_result.get("confidence", 0.5),
            "processing_time": (datetime.now() - start_time).total_seconds()
        }
        
        # Add visualizations if requested
        visualizations = None
        if request.include_visualizations:
            visualizations = {
                "scene_graph": {
                    "nodes": [{"id": entity.name, "type": "entity"} for entity in entities],
                    "edges": [
                        {"source": rel.subject, "target": rel.object, "type": rel.predicate}
                        for rel in relationships
                    ]
                },
                "spatial_layout": {
                    "entities": [
                        {
                            "name": entity.name,
                            "position": entity.pose.position,
                            "orientation": entity.pose.orientation
                        }
                        for entity in entities
                    ]
                }
            }
        
        return GASMAnalysisResponse(
            success=True,
            timestamp=start_time.isoformat(),
            message="Spatial analysis completed successfully",
            entities=entities,
            relationships=relationships,
            geometric_properties=geometric_properties,
            analysis_depth=request.analysis_depth,
            visualizations=visualizations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Spatial analysis failed: {e}")
        
        return GASMAnalysisResponse(
            success=False,
            timestamp=start_time.isoformat(),
            message=f"Spatial analysis failed: {str(e)}",
            entities=[],
            relationships=[],
            geometric_properties={},
            analysis_depth=request.analysis_depth,
            error=str(e)
        )


@router.get("/constraints", response_model=Dict[str, Any])
async def get_supported_constraints(
    user=Depends(require_scope(PermissionScope.READ))
):
    """
    Get list of supported spatial constraints and their parameters.
    """
    try:
        if not GASM_BRIDGE_AVAILABLE:
            return create_success_response(
                data={
                    "constraints": [],
                    "message": "GASM bridge not available"
                }
            )
        
        # Get supported constraints from bridge
        if gasm_bridge:
            supported = gasm_bridge.get_supported_constraints()
        else:
            # Default constraints if bridge not initialized
            supported = [constraint.value for constraint in BridgeConstraintType]
        
        constraint_info = {}
        for constraint_type in supported:
            constraint_info[constraint_type] = {
                "description": _get_constraint_description(constraint_type),
                "parameters": _get_constraint_parameters(constraint_type),
                "examples": _get_constraint_examples(constraint_type)
            }
        
        return create_success_response(
            data={
                "supported_constraints": supported,
                "constraint_details": constraint_info,
                "total_count": len(supported)
            },
            message="Supported constraints retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to get supported constraints: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get constraints: {str(e)}")


@router.get("/examples", response_model=Dict[str, Any])
async def get_processing_examples(
    user=Depends(require_scope(PermissionScope.READ))
):
    """
    Get examples of spatial instructions and their expected outputs.
    """
    try:
        examples = {
            "basic_positioning": {
                "instruction": "place the red block above the blue cube",
                "expected_constraints": [
                    {
                        "type": "above",
                        "subject": "red_block",
                        "target": "blue_cube",
                        "parameters": {"vertical_offset": 0.05}
                    }
                ],
                "expected_poses": {
                    "red_block": {
                        "position": [0.0, 0.0, 0.15],
                        "orientation": [1.0, 0.0, 0.0, 0.0]
                    },
                    "blue_cube": {
                        "position": [0.0, 0.0, 0.10],
                        "orientation": [1.0, 0.0, 0.0, 0.0]
                    }
                }
            },
            "distance_constraint": {
                "instruction": "keep the objects 10cm apart",
                "expected_constraints": [
                    {
                        "type": "distance",
                        "subject": "object_a",
                        "target": "object_b",
                        "parameters": {"distance": 0.1, "axis": "euclidean"}
                    }
                ]
            },
            "angular_relationship": {
                "instruction": "rotate the part 45 degrees relative to the reference",
                "expected_constraints": [
                    {
                        "type": "angle",
                        "subject": "part",
                        "target": "reference",
                        "parameters": {"angle": 45.0, "axis": "z", "units": "degrees"}
                    }
                ]
            },
            "alignment": {
                "instruction": "align the components along the x-axis",
                "expected_constraints": [
                    {
                        "type": "aligned",
                        "subject": "component_1",
                        "target": "component_2",
                        "parameters": {"axis": "x", "tolerance": 0.01}
                    }
                ]
            },
            "complex_instruction": {
                "instruction": "position the tool between the workpieces, 5cm above the table",
                "expected_constraints": [
                    {
                        "type": "between",
                        "subject": "tool",
                        "target": "workpiece_1,workpiece_2",
                        "parameters": {"interpolation": 0.5}
                    },
                    {
                        "type": "above",
                        "subject": "tool",
                        "target": "table",
                        "parameters": {"vertical_offset": 0.05}
                    }
                ]
            }
        }
        
        return create_success_response(
            data={
                "examples": examples,
                "usage_tips": [
                    "Use specific object names for better results",
                    "Include units (cm, degrees) when specifying measurements",
                    "Combine multiple constraints for complex arrangements",
                    "Use descriptive spatial terms (above, below, between, etc.)"
                ],
                "supported_units": {
                    "distance": ["cm", "mm", "m", "inches"],
                    "angle": ["degrees", "radians", "deg", "rad"]
                }
            },
            message="Processing examples retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to get examples: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get examples: {str(e)}")


# Helper functions

def _calculate_bounding_box(entities: List[SpatialEntity]) -> Dict[str, List[float]]:
    """Calculate bounding box for entities"""
    if not entities:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    
    positions = [entity.pose.position for entity in entities]
    
    min_pos = [min(pos[i] for pos in positions) for i in range(3)]
    max_pos = [max(pos[i] for pos in positions) for i in range(3)]
    
    return {"min": min_pos, "max": max_pos}


def _calculate_center_of_mass(entities: List[SpatialEntity]) -> List[float]:
    """Calculate center of mass for entities"""
    if not entities:
        return [0.0, 0.0, 0.0]
    
    positions = [entity.pose.position for entity in entities]
    return [sum(pos[i] for pos in positions) / len(positions) for i in range(3)]


def _get_constraint_description(constraint_type: str) -> str:
    """Get description for constraint type"""
    descriptions = {
        "above": "Object A is positioned above object B",
        "below": "Object A is positioned below object B", 
        "left": "Object A is to the left of object B",
        "right": "Object A is to the right of object B",
        "near": "Object A is within proximity of object B",
        "far": "Object A is distant from object B",
        "angle": "Angular relationship between objects",
        "distance": "Specific distance constraint between objects",
        "aligned": "Objects are aligned along a specific axis",
        "parallel": "Objects maintain parallel orientation",
        "perpendicular": "Objects are at 90-degree angles",
        "touching": "Objects are in contact",
        "inside": "Object A is contained within object B",
        "outside": "Object A is outside of object B",
        "between": "Object A is positioned between objects B and C"
    }
    return descriptions.get(constraint_type, "Unknown constraint type")


def _get_constraint_parameters(constraint_type: str) -> Dict[str, Any]:
    """Get parameters for constraint type"""
    parameters = {
        "above": {"vertical_offset": "float", "tolerance": "float"},
        "below": {"vertical_offset": "float", "tolerance": "float"},
        "distance": {"distance": "float", "axis": "str", "tolerance": "float"},
        "angle": {"angle": "float", "axis": "str", "units": "str"},
        "aligned": {"axis": "str", "tolerance": "float"},
        "between": {"interpolation": "float", "tolerance": "float"}
    }
    return parameters.get(constraint_type, {})


def _get_constraint_examples(constraint_type: str) -> List[str]:
    """Get example instructions for constraint type"""
    examples = {
        "above": ["place A above B", "put the block on top of the cube"],
        "below": ["position A below B", "place the part under the fixture"],
        "distance": ["keep A and B 10cm apart", "maintain 5cm spacing"],
        "angle": ["rotate A 90 degrees", "angle the part 45 degrees relative to B"],
        "aligned": ["align A and B along x-axis", "line up the components"],
        "between": ["position A between B and C", "center the tool between workpieces"]
    }
    return examples.get(constraint_type, [])


# Initialize GASM components on module load
try:
    if GASM_BRIDGE_AVAILABLE:
        asyncio.create_task(initialize_gasm_components())
except Exception as e:
    logger.warning(f"Could not initialize GASM components during module load: {e}")