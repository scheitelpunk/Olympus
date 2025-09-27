"""
Spatial Agent API Endpoints

Comprehensive API endpoints for spatial agent control, pose management,
motion planning, and constraint handling.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from datetime import datetime

# Import models
from ..models.spatial import (
    PoseRequest, PoseResponse, MotionPlanRequest, MotionPlanResponse,
    ConstraintRequest, ConstraintResponse, SpatialRequest, SpatialResponse,
    MetricsRequest, MetricsResponse, Pose3D, QuaternionPose,
    SpatialConstraint, Obstacle, PlanningStrategy, AgentState,
    WorkspaceBounds, ToleranceConfig, PlanningConfig
)
from ..models.base import BaseResponse, create_success_response
from ..middleware.auth import get_current_user, require_scope, PermissionScope

# Import spatial agent components
try:
    from ...spatial_agent.gasm_bridge import GASMBridge, create_bridge
    from ...spatial_agent.planner import MotionPlanner, create_default_planner, create_safe_planner
    from ...spatial_agent.metrics import SpatialMetricsCalculator, calculate_pose_error, ToleranceConfig as MetricsToleranceConfig
    SPATIAL_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Spatial agent components not fully available: {e}")
    SPATIAL_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global spatial components (will be properly initialized)
gasm_bridge: Optional[GASMBridge] = None
motion_planner: Optional[MotionPlanner] = None
metrics_calculator: Optional[SpatialMetricsCalculator] = None
agent_state = {
    "current_pose": None,
    "target_pose": None,
    "state": AgentState.IDLE,
    "active_constraints": [],
    "active_obstacles": [],
    "planning_strategy": PlanningStrategy.CONSTRAINED,
    "iteration_count": 0,
    "last_error": None,
    "start_time": datetime.now().timestamp()
}


async def initialize_spatial_components():
    """Initialize spatial agent components"""
    global gasm_bridge, motion_planner, metrics_calculator
    
    try:
        if SPATIAL_COMPONENTS_AVAILABLE:
            # Initialize GASM bridge
            gasm_bridge = create_bridge()
            
            # Initialize motion planner
            motion_planner = create_default_planner()
            
            # Initialize metrics calculator
            metrics_calculator = SpatialMetricsCalculator()
            
            logger.info("✅ Spatial agent components initialized")
        else:
            logger.warning("⚠️ Spatial components not available - using fallback mode")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize spatial components: {e}")


def get_spatial_components():
    """Dependency to ensure spatial components are available"""
    if not SPATIAL_COMPONENTS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Spatial agent components not available. Please check system configuration."
        )
    return {
        "gasm_bridge": gasm_bridge,
        "motion_planner": motion_planner,
        "metrics_calculator": metrics_calculator
    }


@router.get("/status", response_model=Dict[str, Any])
async def get_spatial_status():
    """
    Get current spatial agent status and configuration.
    
    Returns comprehensive status including pose, constraints, and planning state.
    """
    try:
        uptime = datetime.now().timestamp() - agent_state["start_time"]
        
        status_data = {
            "agent_status": {
                "state": agent_state["state"].value,
                "current_pose": agent_state["current_pose"],
                "target_pose": agent_state["target_pose"],
                "active_constraints": len(agent_state["active_constraints"]),
                "active_obstacles": len(agent_state["active_obstacles"]),
                "planning_strategy": agent_state["planning_strategy"].value,
                "iteration_count": agent_state["iteration_count"],
                "last_error": agent_state["last_error"],
                "uptime": uptime
            },
            "components": {
                "gasm_bridge_available": gasm_bridge is not None,
                "motion_planner_available": motion_planner is not None,
                "metrics_calculator_available": metrics_calculator is not None,
                "spatial_components_available": SPATIAL_COMPONENTS_AVAILABLE
            },
            "configuration": {
                "workspace_bounds": "default",
                "planning_config": "default",
                "tolerance_config": "default"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return create_success_response(
            data=status_data,
            message="Spatial agent status retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting spatial status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get spatial status: {str(e)}")


@router.post("/pose", response_model=PoseResponse)
async def set_pose(
    request: PoseRequest,
    user=Depends(require_scope(PermissionScope.SPATIAL_CONTROL)),
    components=Depends(get_spatial_components)
):
    """
    Set or move to target pose with optional path planning.
    
    Supports both immediate pose setting and planned motion with constraints.
    """
    try:
        start_time = datetime.now()
        
        # Update agent state
        agent_state["state"] = AgentState.PLANNING
        
        if request.target_pose:
            # Convert pose to standard format
            if isinstance(request.target_pose, Pose3D):
                target_pose = request.target_pose
                current_pose = agent_state.get("current_pose") or Pose3D()
            else:
                # QuaternionPose - convert for compatibility
                target_pose = Pose3D(
                    x=request.target_pose.position[0],
                    y=request.target_pose.position[1],
                    z=request.target_pose.position[2],
                    # Quaternion to Euler conversion would go here
                    frame_id=request.target_pose.frame_id
                )
                current_pose = agent_state.get("current_pose") or Pose3D()
            
            # Plan motion to target
            planner = components["motion_planner"]
            if planner:
                # Convert to planner format and plan step
                try:
                    from ...spatial_agent.planner import Pose as PlannerPose
                    current_planner_pose = PlannerPose(
                        current_pose.x, current_pose.y, current_pose.z,
                        current_pose.roll, current_pose.pitch, current_pose.yaw
                    )
                    target_planner_pose = PlannerPose(
                        target_pose.x, target_pose.y, target_pose.z,
                        target_pose.roll, target_pose.pitch, target_pose.yaw
                    )
                    
                    planning_result = planner.plan_step(current_planner_pose, target_planner_pose)
                    
                    if planning_result.success and planning_result.next_pose:
                        # Update current pose to next planned pose
                        next_pose = Pose3D(
                            x=planning_result.next_pose.x,
                            y=planning_result.next_pose.y,
                            z=planning_result.next_pose.z,
                            roll=planning_result.next_pose.roll,
                            pitch=planning_result.next_pose.pitch,
                            yaw=planning_result.next_pose.yaw,
                            frame_id=request.frame_id
                        )
                        
                        agent_state["current_pose"] = next_pose
                        agent_state["target_pose"] = target_pose
                        agent_state["state"] = AgentState.EXECUTING
                        
                    else:
                        agent_state["state"] = AgentState.ERROR
                        agent_state["last_error"] = "Planning failed: " + planning_result.reasoning
                        raise HTTPException(
                            status_code=400,
                            detail=f"Motion planning failed: {planning_result.reasoning}"
                        )
                        
                except ImportError:
                    # Fallback if planner components not available
                    agent_state["current_pose"] = target_pose
                    agent_state["target_pose"] = target_pose
                    agent_state["state"] = AgentState.CONVERGED
            else:
                # Direct pose setting without planning
                agent_state["current_pose"] = target_pose
                agent_state["target_pose"] = target_pose
                agent_state["state"] = AgentState.CONVERGED
        
        # Calculate pose error if we have both current and target
        pose_error = None
        is_at_target = True
        
        if (agent_state["current_pose"] and agent_state["target_pose"] and 
            components["metrics_calculator"]):
            try:
                calc = components["metrics_calculator"]
                
                # Convert poses to format expected by metrics calculator
                current_dict = {
                    "position": [agent_state["current_pose"].x, agent_state["current_pose"].y, agent_state["current_pose"].z],
                    "orientation": [agent_state["current_pose"].roll, agent_state["current_pose"].pitch, agent_state["current_pose"].yaw]
                }
                target_dict = {
                    "position": [agent_state["target_pose"].x, agent_state["target_pose"].y, agent_state["target_pose"].z],
                    "orientation": [agent_state["target_pose"].roll, agent_state["target_pose"].pitch, agent_state["target_pose"].yaw]
                }
                
                error_result = calc.pose_error(current_dict, target_dict)
                pose_error = {
                    "position_error": error_result.position_error,
                    "rotation_error": error_result.rotation_error,
                    "total_error": error_result.total_error,
                    "is_converged": error_result.is_converged,
                    "position_vector": error_result.position_vector.tolist(),
                    "rotation_axis": error_result.rotation_axis.tolist()
                }
                is_at_target = error_result.is_converged
                
            except Exception as metrics_error:
                logger.warning(f"Pose error calculation failed: {metrics_error}")
        
        return PoseResponse(
            success=True,
            timestamp=start_time.isoformat(),
            message="Pose operation completed",
            current_pose=agent_state["current_pose"],
            target_pose=agent_state.get("target_pose"),
            pose_error=pose_error,
            is_at_target=is_at_target,
            frame_id=request.frame_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        agent_state["state"] = AgentState.ERROR
        agent_state["last_error"] = str(e)
        logger.error(f"Pose operation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pose operation failed: {str(e)}")


@router.post("/plan", response_model=MotionPlanResponse)
async def plan_motion(
    request: MotionPlanRequest,
    user=Depends(require_scope(PermissionScope.SPATIAL_CONTROL)),
    components=Depends(get_spatial_components)
):
    """
    Plan motion from current pose to target with constraints and obstacles.
    
    Returns detailed planning results including trajectory and performance metrics.
    """
    try:
        start_time = datetime.now()
        
        agent_state["state"] = AgentState.PLANNING
        
        planner = components["motion_planner"]
        if not planner:
            raise HTTPException(status_code=503, detail="Motion planner not available")
        
        # Set planning strategy
        planner.set_strategy(request.strategy)
        agent_state["planning_strategy"] = request.strategy
        
        # Add constraints to planner
        planner.clear_constraints()
        for constraint in request.constraints:
            try:
                from ...spatial_agent.planner import Constraint as PlannerConstraint
                planner_constraint = PlannerConstraint(
                    type=constraint.type.value,
                    params=constraint.parameters,
                    priority=int(constraint.priority * 10)  # Convert 0-1 to integer priority
                )
                planner.add_constraint(planner_constraint)
            except ImportError:
                logger.warning("Could not add constraints - planner constraint class not available")
        
        # Add obstacles to planner
        planner.clear_obstacles()
        for obstacle in request.obstacles:
            try:
                from ...spatial_agent.planner import Obstacle as PlannerObstacle, Pose as PlannerPose
                obstacle_pose = PlannerPose(
                    obstacle.pose.x, obstacle.pose.y, obstacle.pose.z,
                    obstacle.pose.roll, obstacle.pose.pitch, obstacle.pose.yaw
                )
                planner_obstacle = PlannerObstacle(
                    center=obstacle_pose,
                    radius=obstacle.geometry.get("radius", 0.1)
                )
                planner.add_obstacle(planner_obstacle)
            except ImportError:
                logger.warning("Could not add obstacles - planner obstacle class not available")
        
        # Plan single step
        try:
            from ...spatial_agent.planner import Pose as PlannerPose
            
            current_planner_pose = PlannerPose(
                request.current_pose.x if hasattr(request.current_pose, 'x') else request.current_pose.position[0],
                request.current_pose.y if hasattr(request.current_pose, 'y') else request.current_pose.position[1],
                request.current_pose.z if hasattr(request.current_pose, 'z') else request.current_pose.position[2],
                getattr(request.current_pose, 'roll', 0.0),
                getattr(request.current_pose, 'pitch', 0.0),
                getattr(request.current_pose, 'yaw', 0.0)
            )
            
            target_planner_pose = PlannerPose(
                request.target_pose.x if hasattr(request.target_pose, 'x') else request.target_pose.position[0],
                request.target_pose.y if hasattr(request.target_pose, 'y') else request.target_pose.position[1],
                request.target_pose.z if hasattr(request.target_pose, 'z') else request.target_pose.position[2],
                getattr(request.target_pose, 'roll', 0.0),
                getattr(request.target_pose, 'pitch', 0.0),
                getattr(request.target_pose, 'yaw', 0.0)
            )
            
            planning_result = planner.plan_step(current_planner_pose, target_planner_pose)
            
        except ImportError:
            # Fallback planning logic
            planning_result = type('PlanningResult', (), {
                'success': True,
                'next_pose': request.target_pose,
                'step_size': 0.1,
                'constraints_violated': [],
                'obstacles_detected': [],
                'reasoning': "Fallback direct planning",
                'debug_info': {}
            })()
        
        # Update agent state
        if planning_result.success:
            agent_state["state"] = AgentState.EXECUTING
            agent_state["iteration_count"] += 1
        else:
            agent_state["state"] = AgentState.ERROR
            agent_state["last_error"] = planning_result.reasoning
        
        # Calculate planning time
        planning_time = (datetime.now() - start_time).total_seconds()
        
        # Build response
        from ..models.spatial import PlanningResult as ResponsePlanningResult
        
        response_planning_result = ResponsePlanningResult(
            success=planning_result.success,
            next_pose=Pose3D(
                x=planning_result.next_pose.x if hasattr(planning_result.next_pose, 'x') else 0.0,
                y=planning_result.next_pose.y if hasattr(planning_result.next_pose, 'y') else 0.0,
                z=planning_result.next_pose.z if hasattr(planning_result.next_pose, 'z') else 0.0,
                roll=getattr(planning_result.next_pose, 'roll', 0.0),
                pitch=getattr(planning_result.next_pose, 'pitch', 0.0),
                yaw=getattr(planning_result.next_pose, 'yaw', 0.0)
            ) if planning_result.next_pose else None,
            step_size=planning_result.step_size,
            constraints_violated=planning_result.constraints_violated,
            obstacles_detected=planning_result.obstacles_detected,
            reasoning=planning_result.reasoning,
            debug_info=planning_result.debug_info
        )
        
        return MotionPlanResponse(
            success=True,
            timestamp=start_time.isoformat(),
            message="Motion planning completed",
            planning_result=response_planning_result,
            trajectory=None,  # Could generate full trajectory here
            planning_time=planning_time,
            iterations_used=1,
            strategy_used=request.strategy
        )
        
    except HTTPException:
        raise
    except Exception as e:
        agent_state["state"] = AgentState.ERROR
        agent_state["last_error"] = str(e)
        logger.error(f"Motion planning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Motion planning failed: {str(e)}")


@router.post("/constraints", response_model=ConstraintResponse)
async def manage_constraints(
    request: ConstraintRequest,
    user=Depends(require_scope(PermissionScope.SPATIAL_CONTROL)),
    components=Depends(get_spatial_components)
):
    """
    Manage spatial constraints for motion planning.
    
    Supports adding, removing, updating, and listing constraints.
    """
    try:
        if request.action == "add":
            if not request.constraint:
                raise HTTPException(status_code=400, detail="Constraint data required for add action")
            
            agent_state["active_constraints"].append(request.constraint)
            constraint_id = f"constraint_{len(agent_state['active_constraints'])}"
            
            return ConstraintResponse(
                success=True,
                timestamp=datetime.now().isoformat(),
                message="Constraint added successfully",
                action_performed=request.action,
                constraint_id=constraint_id,
                constraints=agent_state["active_constraints"]
            )
            
        elif request.action == "remove":
            if not request.constraint_id:
                raise HTTPException(status_code=400, detail="Constraint ID required for remove action")
            
            # Simple removal by index for now
            try:
                constraint_idx = int(request.constraint_id.split("_")[-1]) - 1
                if 0 <= constraint_idx < len(agent_state["active_constraints"]):
                    removed_constraint = agent_state["active_constraints"].pop(constraint_idx)
                else:
                    raise HTTPException(status_code=404, detail="Constraint not found")
            except (ValueError, IndexError):
                raise HTTPException(status_code=400, detail="Invalid constraint ID")
            
            return ConstraintResponse(
                success=True,
                timestamp=datetime.now().isoformat(),
                message="Constraint removed successfully", 
                action_performed=request.action,
                constraint_id=request.constraint_id,
                constraints=agent_state["active_constraints"]
            )
            
        elif request.action == "clear":
            agent_state["active_constraints"].clear()
            
            return ConstraintResponse(
                success=True,
                timestamp=datetime.now().isoformat(),
                message="All constraints cleared",
                action_performed=request.action,
                constraints=agent_state["active_constraints"]
            )
            
        elif request.action == "list":
            return ConstraintResponse(
                success=True,
                timestamp=datetime.now().isoformat(),
                message="Constraints listed successfully",
                action_performed=request.action,
                constraints=agent_state["active_constraints"]
            )
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported action: {request.action}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Constraint management failed: {e}")
        raise HTTPException(status_code=500, detail=f"Constraint management failed: {str(e)}")


@router.post("/process", response_model=SpatialResponse)
async def process_spatial_instruction(
    request: SpatialRequest,
    background_tasks: BackgroundTasks,
    user=Depends(require_scope(PermissionScope.GASM_PROCESS)),
    components=Depends(get_spatial_components)
):
    """
    Process natural language spatial instructions using GASM.
    
    Converts natural language to spatial constraints and target poses.
    """
    try:
        start_time = datetime.now()
        
        bridge = components.get("gasm_bridge")
        if not bridge:
            raise HTTPException(status_code=503, detail="GASM bridge not available")
        
        # Process instruction through GASM bridge
        try:
            gasm_result = bridge.process(request.instruction)
            
            if not gasm_result.get("success", False):
                raise HTTPException(
                    status_code=400,
                    detail=f"GASM processing failed: {gasm_result.get('error_message', 'Unknown error')}"
                )
            
            # Extract constraints and poses from GASM result
            constraints_data = gasm_result.get("constraints", [])
            target_poses_data = gasm_result.get("target_poses", {})
            
            # Convert to spatial constraints
            spatial_constraints = []
            for constraint_data in constraints_data:
                try:
                    from ..models.spatial import ConstraintType
                    constraint = SpatialConstraint(
                        type=ConstraintType(constraint_data.get("type", "distance")),
                        subject=constraint_data.get("subject", "object"),
                        target=constraint_data.get("target"),
                        parameters=constraint_data.get("parameters", {}),
                        priority=constraint_data.get("priority", 0.5),
                        tolerance=constraint_data.get("tolerance", {"position": 0.01, "orientation": 0.1})
                    )
                    spatial_constraints.append(constraint)
                except Exception as constraint_error:
                    logger.warning(f"Could not convert constraint: {constraint_error}")
            
            # Convert to spatial poses  
            spatial_poses = {}
            for obj_name, pose_data in target_poses_data.items():
                try:
                    pose = Pose3D(
                        x=pose_data.get("position", [0, 0, 0])[0],
                        y=pose_data.get("position", [0, 0, 0])[1], 
                        z=pose_data.get("position", [0, 0, 0])[2],
                        frame_id=pose_data.get("frame_id", "world")
                    )
                    spatial_poses[obj_name] = pose
                except Exception as pose_error:
                    logger.warning(f"Could not convert pose for {obj_name}: {pose_error}")
            
            # Update agent constraints
            agent_state["active_constraints"].extend(spatial_constraints)
            
            # If we have a single target pose, set it as the current target
            if len(spatial_poses) == 1:
                agent_state["target_pose"] = list(spatial_poses.values())[0]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return SpatialResponse(
                success=True,
                timestamp=start_time.isoformat(),
                message="Spatial instruction processed successfully",
                result={
                    "instruction": request.instruction,
                    "constraints_generated": len(spatial_constraints),
                    "poses_generated": len(spatial_poses),
                    "confidence": gasm_result.get("confidence", 0.5),
                    "constraints": [constraint.dict() for constraint in spatial_constraints],
                    "target_poses": {name: pose.dict() for name, pose in spatial_poses.items()}
                },
                constraints_applied=[f"constraint_{i}" for i in range(len(spatial_constraints))],
                processing_time=processing_time,
                debug_info=gasm_result.get("debug_info") if request.options.get("include_debug", False) else None
            )
            
        except Exception as gasm_error:
            logger.error(f"GASM processing error: {gasm_error}")
            
            # Fallback: try to extract basic spatial information
            result = {
                "instruction": request.instruction,
                "constraints_generated": 0,
                "poses_generated": 0,
                "confidence": 0.1,
                "fallback_mode": True,
                "error": str(gasm_error)
            }
            
            return SpatialResponse(
                success=False,
                timestamp=start_time.isoformat(),
                message=f"GASM processing failed, fallback mode: {str(gasm_error)}",
                result=result,
                constraints_applied=[],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Spatial instruction processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Spatial processing failed: {str(e)}")


@router.post("/metrics", response_model=MetricsResponse)
async def calculate_metrics(
    request: MetricsRequest,
    user=Depends(require_scope(PermissionScope.READ)),
    components=Depends(get_spatial_components)
):
    """
    Calculate spatial metrics including pose errors and constraint satisfaction.
    
    Provides detailed analysis of current spatial state and convergence.
    """
    try:
        calculator = components.get("metrics_calculator")
        if not calculator:
            raise HTTPException(status_code=503, detail="Metrics calculator not available")
        
        # Convert poses to calculator format
        current_dict = {
            "position": [request.current_pose.x, request.current_pose.y, request.current_pose.z] if hasattr(request.current_pose, 'x') else request.current_pose.position,
            "orientation": [
                getattr(request.current_pose, 'roll', 0.0),
                getattr(request.current_pose, 'pitch', 0.0), 
                getattr(request.current_pose, 'yaw', 0.0)
            ] if hasattr(request.current_pose, 'roll') else [0.0, 0.0, 0.0]
        }
        
        target_dict = {
            "position": [request.target_pose.x, request.target_pose.y, request.target_pose.z] if hasattr(request.target_pose, 'x') else request.target_pose.position,
            "orientation": [
                getattr(request.target_pose, 'roll', 0.0),
                getattr(request.target_pose, 'pitch', 0.0),
                getattr(request.target_pose, 'yaw', 0.0)
            ] if hasattr(request.target_pose, 'roll') else [0.0, 0.0, 0.0]
        }
        
        # Calculate pose error
        pose_error = calculator.pose_error(current_dict, target_dict)
        
        # Calculate constraint scores
        state_dict = {**request.state, "position": current_dict["position"], "orientation": current_dict["orientation"]}
        constraint_scores = calculator.constraint_score(state_dict, request.constraints)
        
        # Check convergence  
        is_converged, convergence_analysis = calculator.is_done([pose_error], constraint_scores)
        
        # Get accumulated statistics
        statistics = calculator.accumulate_statistics(pose_error, constraint_scores)
        
        # Generate recommendations
        recommendations = []
        if pose_error.position_error > 0.05:
            recommendations.append("Position error is high - consider adjusting control gains")
        if pose_error.rotation_error > 0.1:
            recommendations.append("Rotation error is high - check orientation control")
        
        unsatisfied_constraints = [name for name, score in constraint_scores.items() if not score.is_satisfied]
        if unsatisfied_constraints:
            recommendations.append(f"Unsatisfied constraints: {', '.join(unsatisfied_constraints)}")
        
        if not recommendations:
            recommendations.append("System metrics are within acceptable ranges")
        
        from ..models.spatial import PoseError, ConstraintScore
        
        # Convert pose error to response format
        response_pose_error = PoseError(
            position_error=pose_error.position_error,
            rotation_error=pose_error.rotation_error,
            total_error=pose_error.total_error,
            is_converged=pose_error.is_converged,
            position_vector=pose_error.position_vector.tolist(),
            rotation_axis=pose_error.rotation_axis.tolist()
        )
        
        # Convert constraint scores to response format
        response_constraint_scores = {}
        for name, score in constraint_scores.items():
            response_constraint_scores[name] = ConstraintScore(
                constraint_type=score.constraint_type,
                score=score.score,
                violation_magnitude=score.violation_magnitude,
                is_satisfied=score.is_satisfied,
                tolerance_used=score.tolerance_used,
                additional_info=score.additional_info
            )
        
        return MetricsResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            message="Metrics calculated successfully",
            pose_error=response_pose_error,
            constraint_scores=response_constraint_scores,
            is_converged=is_converged,
            convergence_analysis=convergence_analysis,
            statistics=statistics,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics calculation failed: {str(e)}")


@router.get("/workspace", response_model=Dict[str, Any])
async def get_workspace_info(
    user=Depends(require_scope(PermissionScope.READ))
):
    """Get workspace bounds and configuration."""
    try:
        # Default workspace bounds
        workspace = WorkspaceBounds()
        
        return create_success_response(
            data={
                "workspace_bounds": workspace.dict(),
                "safety_margins": {
                    "position": 0.05,  # 5cm
                    "orientation": 0.1  # ~5.7 degrees
                },
                "coordinate_system": "right_handed_cartesian",
                "units": {
                    "position": "meters", 
                    "orientation": "radians"
                },
                "reference_frame": "world"
            },
            message="Workspace information retrieved"
        )
        
    except Exception as e:
        logger.error(f"Failed to get workspace info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workspace info: {str(e)}")


# Initialize spatial components on module load
import asyncio
try:
    # Try to initialize components
    if SPATIAL_COMPONENTS_AVAILABLE:
        asyncio.create_task(initialize_spatial_components())
except Exception as e:
    logger.warning(f"Could not initialize spatial components during module load: {e}")