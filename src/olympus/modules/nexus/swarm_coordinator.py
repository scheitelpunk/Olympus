"""
Swarm Coordinator - Multi-Robot Coordination System
==================================================

The Swarm Coordinator manages multi-robot coordination, task distribution,
formation control, and collective movement while maintaining individual
robot safety and autonomy.

Features:
- Multi-robot task coordination
- Formation control and management
- Distributed task assignment
- Collision avoidance
- Emergency dispersal protocols
- Individual robot safety monitoring
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class CoordinationMode(Enum):
    """Coordination modes for swarm behavior"""
    FORMATION = "formation"
    DISTRIBUTED = "distributed"
    HIERARCHICAL = "hierarchical"
    AUTONOMOUS = "autonomous"
    EMERGENCY = "emergency"


class FormationType(Enum):
    """Standard formation types"""
    LINE = "line"
    CIRCLE = "circle"
    GRID = "grid"
    WEDGE = "wedge"
    DIAMOND = "diamond"
    CUSTOM = "custom"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class RobotStatus:
    """Status of an individual robot in the swarm"""
    id: str
    position: Tuple[float, float, float]  # x, y, z
    velocity: Tuple[float, float, float]
    battery_level: float
    health_status: str
    current_task: Optional[str]
    capabilities: List[str]
    last_update: datetime
    collision_risk: float = 0.0
    communication_health: float = 1.0


@dataclass
class CoordinatedTask:
    """A task requiring coordination among multiple robots"""
    id: str
    description: str
    required_robots: int
    assigned_robots: Set[str]
    requirements: List[str]
    priority: TaskPriority
    deadline: Optional[datetime]
    status: str
    created_time: datetime
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    progress: float = 0.0


@dataclass
class Formation:
    """Formation configuration for coordinated movement"""
    type: FormationType
    positions: Dict[str, Tuple[float, float, float]]
    center: Tuple[float, float, float]
    scale: float
    rotation: float
    assigned_robots: Set[str]
    leader_robot: Optional[str] = None


class SwarmCoordinator:
    """
    Multi-robot coordination system for NEXUS
    
    Manages coordinated behavior, task distribution, and formation control
    while ensuring individual robot safety and maintaining human authority.
    """
    
    def __init__(self, config):
        self.config = config
        self.coordination_mode = CoordinationMode.AUTONOMOUS
        
        # Robot management
        self.robot_status: Dict[str, RobotStatus] = {}
        self.robot_capabilities: Dict[str, List[str]] = {}
        
        # Task management
        self.coordinated_tasks: Dict[str, CoordinatedTask] = {}
        self.task_queue = asyncio.Queue()
        self.task_assignment_lock = asyncio.Lock()
        
        # Formation management
        self.active_formations: Dict[str, Formation] = {}
        self.formation_templates = {}
        
        # Coordination state
        self.coordination_active = False
        self.emergency_mode = False
        self.human_override_active = False
        
        # Safety systems
        self.collision_avoidance_enabled = True
        self.safety_margins = {
            'min_distance': 2.0,  # meters
            'collision_threshold': 1.5,
            'communication_timeout': 10.0  # seconds
        }
        
        # Coordination metrics
        self.coordination_efficiency = 0.0
        self.task_completion_rate = 0.0
        self.formation_accuracy = 0.0
        
        logger.info("Swarm Coordinator initialized")
    
    async def initialize(self, initial_robots: List[str]) -> bool:
        """Initialize swarm coordination with specified robots"""
        try:
            # Initialize robots
            for robot_id in initial_robots:
                await self._add_robot(robot_id)
            
            # Initialize formation templates
            await self._initialize_formation_templates()
            
            # Start coordination tasks
            self.coordination_active = True
            asyncio.create_task(self._coordination_monitor())
            asyncio.create_task(self._safety_monitor())
            
            logger.info(f"Swarm coordination initialized with {len(initial_robots)} robots")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize swarm coordination: {e}")
            return False
    
    async def add_robot(self, robot_id: str, capabilities: List[str] = None) -> bool:
        """Add a robot to the coordinated swarm"""
        try:
            if robot_id in self.robot_status:
                logger.warning(f"Robot {robot_id} already in swarm")
                return True
            
            await self._add_robot(robot_id, capabilities or [])
            
            # Update coordination parameters
            await self._recalculate_coordination_parameters()
            
            logger.info(f"Robot {robot_id} added to swarm coordination")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add robot {robot_id}: {e}")
            return False
    
    async def remove_robot(self, robot_id: str) -> bool:
        """Remove a robot from coordinated swarm"""
        try:
            if robot_id not in self.robot_status:
                logger.warning(f"Robot {robot_id} not in swarm")
                return True
            
            # Reassign tasks if needed
            await self._reassign_robot_tasks(robot_id)
            
            # Remove from formations
            await self._remove_robot_from_formations(robot_id)
            
            # Remove robot status
            del self.robot_status[robot_id]
            if robot_id in self.robot_capabilities:
                del self.robot_capabilities[robot_id]
            
            logger.info(f"Robot {robot_id} removed from swarm coordination")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove robot {robot_id}: {e}")
            return False
    
    async def execute_coordinated_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a coordinated action across multiple robots"""
        try:
            if self.human_override_active:
                return {"status": "blocked", "reason": "human_override"}
            
            if self.emergency_mode:
                return {"status": "blocked", "reason": "emergency_mode"}
            
            action_type = action.get("type", "unknown")
            
            if action_type == "move_formation":
                return await self._execute_formation_movement(action)
            elif action_type == "coordinate_task":
                return await self._execute_coordinated_task(action)
            elif action_type == "search_pattern":
                return await self._execute_search_pattern(action)
            elif action_type == "collective_action":
                return await self._execute_collective_action(action)
            else:
                return {"status": "error", "reason": f"unknown_action_type: {action_type}"}
            
        except Exception as e:
            logger.error(f"Failed to execute coordinated action: {e}")
            return {"status": "error", "reason": str(e)}
    
    async def create_formation(self, formation_type: FormationType, 
                             robot_ids: List[str], 
                             center: Tuple[float, float, float] = (0, 0, 0),
                             scale: float = 1.0,
                             leader_id: str = None) -> str:
        """Create a formation with specified robots"""
        try:
            formation_id = f"formation_{datetime.now().timestamp()}"
            
            # Generate formation positions
            positions = await self._generate_formation_positions(
                formation_type, robot_ids, center, scale
            )
            
            formation = Formation(
                type=formation_type,
                positions=positions,
                center=center,
                scale=scale,
                rotation=0.0,
                assigned_robots=set(robot_ids),
                leader_robot=leader_id or robot_ids[0]
            )
            
            self.active_formations[formation_id] = formation
            
            # Assign robots to formation positions
            await self._assign_formation_positions(formation_id)
            
            logger.info(f"Formation {formation_id} created with {len(robot_ids)} robots")
            return formation_id
            
        except Exception as e:
            logger.error(f"Failed to create formation: {e}")
            return ""
    
    async def move_formation(self, formation_id: str, 
                           target_center: Tuple[float, float, float],
                           target_rotation: float = None) -> bool:
        """Move an existing formation to a new position"""
        try:
            if formation_id not in self.active_formations:
                logger.error(f"Formation {formation_id} not found")
                return False
            
            formation = self.active_formations[formation_id]
            
            # Calculate movement trajectory
            trajectory = await self._calculate_formation_trajectory(
                formation, target_center, target_rotation
            )
            
            # Execute coordinated movement
            movement_result = await self._execute_formation_trajectory(
                formation_id, trajectory
            )
            
            # Update formation center and rotation
            if movement_result:
                formation.center = target_center
                if target_rotation is not None:
                    formation.rotation = target_rotation
            
            return movement_result
            
        except Exception as e:
            logger.error(f"Failed to move formation {formation_id}: {e}")
            return False
    
    async def assign_coordinated_task(self, task_description: str,
                                    required_robots: int,
                                    requirements: List[str] = None,
                                    priority: TaskPriority = TaskPriority.MEDIUM,
                                    deadline: datetime = None) -> str:
        """Assign a task requiring coordination among multiple robots"""
        try:
            task_id = f"task_{datetime.now().timestamp()}"
            
            # Create coordinated task
            task = CoordinatedTask(
                id=task_id,
                description=task_description,
                required_robots=required_robots,
                assigned_robots=set(),
                requirements=requirements or [],
                priority=priority,
                deadline=deadline,
                status="pending",
                created_time=datetime.now()
            )
            
            self.coordinated_tasks[task_id] = task
            
            # Find and assign suitable robots
            assigned_robots = await self._find_suitable_robots(task)
            
            if len(assigned_robots) >= required_robots:
                task.assigned_robots = set(assigned_robots[:required_robots])
                task.status = "assigned"
                
                # Start task execution
                await self._start_coordinated_task(task_id)
                
                logger.info(f"Coordinated task {task_id} assigned to {len(task.assigned_robots)} robots")
                return task_id
            else:
                task.status = "insufficient_robots"
                logger.warning(f"Insufficient suitable robots for task {task_id}")
                return task_id
                
        except Exception as e:
            logger.error(f"Failed to assign coordinated task: {e}")
            return ""
    
    async def emergency_stop(self) -> bool:
        """Stop all coordinated activities immediately"""
        try:
            logger.critical("Emergency stop activated for swarm coordination")
            
            self.emergency_mode = True
            self.coordination_mode = CoordinationMode.EMERGENCY
            
            # Stop all active tasks
            for task_id, task in self.coordinated_tasks.items():
                if task.status == "active":
                    task.status = "emergency_stopped"
            
            # Break all formations
            for formation_id in list(self.active_formations.keys()):
                await self._break_formation(formation_id)
            
            # Send emergency stop to all robots
            for robot_id in self.robot_status:
                await self._send_emergency_stop_to_robot(robot_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False
    
    async def execute_dispersal(self) -> bool:
        """Execute emergency dispersal of the swarm"""
        try:
            logger.critical("Executing emergency dispersal")
            
            # Calculate safe dispersal positions
            dispersal_positions = await self._calculate_dispersal_positions()
            
            # Send dispersal commands to all robots
            dispersal_tasks = []
            for robot_id, position in dispersal_positions.items():
                if robot_id in self.robot_status:
                    task = self._send_dispersal_command(robot_id, position)
                    dispersal_tasks.append(task)
            
            # Wait for dispersal completion
            results = await asyncio.gather(*dispersal_tasks, return_exceptions=True)
            
            # Check success rate
            successful = sum(1 for r in results if r is True)
            total = len(dispersal_tasks)
            
            logger.info(f"Dispersal completed: {successful}/{total} robots successfully dispersed")
            return successful >= total * 0.8  # 80% success threshold
            
        except Exception as e:
            logger.error(f"Emergency dispersal failed: {e}")
            return False
    
    async def execute_human_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute direct human command with override authority"""
        try:
            self.human_override_active = True
            
            command_type = command.get("type", "unknown")
            
            if command_type == "stop_all":
                return await self._execute_stop_all()
            elif command_type == "move_robot":
                return await self._execute_move_robot_command(command)
            elif command_type == "break_formation":
                return await self._execute_break_formation_command(command)
            elif command_type == "assign_task":
                return await self._execute_assign_task_command(command)
            elif command_type == "resume_coordination":
                return await self._execute_resume_coordination()
            else:
                return {"status": "error", "reason": f"unknown_command: {command_type}"}
                
        except Exception as e:
            logger.error(f"Failed to execute human command: {e}")
            return {"status": "error", "reason": str(e)}
    
    async def suspend_coordination(self) -> bool:
        """Suspend coordination activities"""
        try:
            self.coordination_active = False
            
            # Pause all active tasks
            for task in self.coordinated_tasks.values():
                if task.status == "active":
                    task.status = "suspended"
            
            # Hold current formations
            for formation in self.active_formations.values():
                await self._hold_formation(formation)
            
            logger.warning("Swarm coordination suspended")
            return True
            
        except Exception as e:
            logger.error(f"Failed to suspend coordination: {e}")
            return False
    
    async def resume_coordination(self) -> bool:
        """Resume coordination activities"""
        try:
            self.coordination_active = True
            self.human_override_active = False
            self.emergency_mode = False
            self.coordination_mode = CoordinationMode.AUTONOMOUS
            
            # Resume suspended tasks
            for task in self.coordinated_tasks.values():
                if task.status == "suspended":
                    task.status = "active"
            
            logger.info("Swarm coordination resumed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume coordination: {e}")
            return False
    
    async def get_active_robots(self) -> List[str]:
        """Get list of active robots in the swarm"""
        return list(self.robot_status.keys())
    
    async def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status"""
        active_robots = len(self.robot_status)
        active_tasks = sum(1 for task in self.coordinated_tasks.values() 
                          if task.status == "active")
        active_formations = len(self.active_formations)
        
        return {
            "coordination_active": self.coordination_active,
            "coordination_mode": self.coordination_mode.value,
            "emergency_mode": self.emergency_mode,
            "human_override": self.human_override_active,
            "active_robots": active_robots,
            "active_tasks": active_tasks,
            "active_formations": active_formations,
            "coordination_efficiency": self.coordination_efficiency,
            "task_completion_rate": self.task_completion_rate,
            "formation_accuracy": self.formation_accuracy,
            "safety_status": {
                "collision_avoidance_enabled": self.collision_avoidance_enabled,
                "robots_at_risk": sum(1 for robot in self.robot_status.values() 
                                    if robot.collision_risk > 0.7)
            }
        }
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown swarm coordination"""
        try:
            logger.info("Shutting down swarm coordination")
            
            # Stop all coordination activities
            self.coordination_active = False
            
            # Complete or cancel all tasks
            for task in self.coordinated_tasks.values():
                if task.status == "active":
                    task.status = "cancelled"
                    task.completed_time = datetime.now()
            
            # Break all formations safely
            for formation_id in list(self.active_formations.keys()):
                await self._break_formation(formation_id)
            
            # Clear all state
            self.robot_status.clear()
            self.coordinated_tasks.clear()
            self.active_formations.clear()
            
            logger.info("Swarm coordination shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Swarm coordination shutdown failed: {e}")
            return False
    
    # Private helper methods
    
    async def _add_robot(self, robot_id: str, capabilities: List[str] = None):
        """Add a robot to the coordination system"""
        self.robot_status[robot_id] = RobotStatus(
            id=robot_id,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            battery_level=1.0,
            health_status="healthy",
            current_task=None,
            capabilities=capabilities or [],
            last_update=datetime.now()
        )
        self.robot_capabilities[robot_id] = capabilities or []
    
    async def _initialize_formation_templates(self):
        """Initialize standard formation templates"""
        self.formation_templates = {
            FormationType.LINE: self._generate_line_formation,
            FormationType.CIRCLE: self._generate_circle_formation,
            FormationType.GRID: self._generate_grid_formation,
            FormationType.WEDGE: self._generate_wedge_formation,
            FormationType.DIAMOND: self._generate_diamond_formation
        }
    
    async def _generate_formation_positions(self, formation_type: FormationType,
                                          robot_ids: List[str],
                                          center: Tuple[float, float, float],
                                          scale: float) -> Dict[str, Tuple[float, float, float]]:
        """Generate positions for a formation"""
        if formation_type in self.formation_templates:
            return await self.formation_templates[formation_type](robot_ids, center, scale)
        else:
            # Default to circle formation
            return await self._generate_circle_formation(robot_ids, center, scale)
    
    async def _generate_circle_formation(self, robot_ids: List[str], 
                                       center: Tuple[float, float, float], 
                                       scale: float) -> Dict[str, Tuple[float, float, float]]:
        """Generate circular formation positions"""
        positions = {}
        n = len(robot_ids)
        radius = scale * 5.0  # Base radius of 5 meters
        
        for i, robot_id in enumerate(robot_ids):
            angle = 2 * math.pi * i / n
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            z = center[2]
            positions[robot_id] = (x, y, z)
        
        return positions
    
    async def _generate_line_formation(self, robot_ids: List[str],
                                     center: Tuple[float, float, float],
                                     scale: float) -> Dict[str, Tuple[float, float, float]]:
        """Generate line formation positions"""
        positions = {}
        n = len(robot_ids)
        spacing = scale * 3.0  # 3 meters spacing
        start_x = center[0] - (n - 1) * spacing / 2
        
        for i, robot_id in enumerate(robot_ids):
            x = start_x + i * spacing
            y = center[1]
            z = center[2]
            positions[robot_id] = (x, y, z)
        
        return positions
    
    async def _generate_grid_formation(self, robot_ids: List[str],
                                     center: Tuple[float, float, float],
                                     scale: float) -> Dict[str, Tuple[float, float, float]]:
        """Generate grid formation positions"""
        positions = {}
        n = len(robot_ids)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        spacing = scale * 4.0
        
        start_x = center[0] - (cols - 1) * spacing / 2
        start_y = center[1] - (rows - 1) * spacing / 2
        
        for i, robot_id in enumerate(robot_ids):
            row = i // cols
            col = i % cols
            x = start_x + col * spacing
            y = start_y + row * spacing
            z = center[2]
            positions[robot_id] = (x, y, z)
        
        return positions
    
    async def _generate_wedge_formation(self, robot_ids: List[str],
                                      center: Tuple[float, float, float],
                                      scale: float) -> Dict[str, Tuple[float, float, float]]:
        """Generate wedge formation positions"""
        positions = {}
        n = len(robot_ids)
        spacing = scale * 3.0
        
        # Leader at front
        positions[robot_ids[0]] = center
        
        # Arrange others in wedge pattern
        for i in range(1, n):
            side = 1 if i % 2 == 1 else -1
            rank = (i + 1) // 2
            x = center[0] - rank * spacing
            y = center[1] + side * rank * spacing * 0.5
            z = center[2]
            positions[robot_ids[i]] = (x, y, z)
        
        return positions
    
    async def _generate_diamond_formation(self, robot_ids: List[str],
                                        center: Tuple[float, float, float],
                                        scale: float) -> Dict[str, Tuple[float, float, float]]:
        """Generate diamond formation positions"""
        positions = {}
        n = len(robot_ids)
        spacing = scale * 4.0
        
        if n >= 4:
            # Diamond corners
            positions[robot_ids[0]] = (center[0], center[1] + spacing, center[2])  # Front
            positions[robot_ids[1]] = (center[0] + spacing, center[1], center[2])  # Right
            positions[robot_ids[2]] = (center[0], center[1] - spacing, center[2])  # Back
            positions[robot_ids[3]] = (center[0] - spacing, center[1], center[2])  # Left
            
            # Additional robots inside diamond
            for i in range(4, n):
                angle = 2 * math.pi * (i - 4) / max(1, n - 4)
                r = spacing * 0.5
                x = center[0] + r * math.cos(angle)
                y = center[1] + r * math.sin(angle)
                z = center[2]
                positions[robot_ids[i]] = (x, y, z)
        else:
            # Fall back to circle for fewer robots
            return await self._generate_circle_formation(robot_ids, center, scale)
        
        return positions
    
    async def _coordination_monitor(self):
        """Background task to monitor coordination health"""
        while self.coordination_active:
            try:
                await self._update_coordination_metrics()
                await self._check_task_progress()
                await self._monitor_formation_integrity()
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Coordination monitor error: {e}")
    
    async def _safety_monitor(self):
        """Background task to monitor safety conditions"""
        while self.coordination_active:
            try:
                await self._check_collision_risks()
                await self._monitor_communication_health()
                await self._check_robot_health()
                await asyncio.sleep(0.5)  # Faster safety monitoring
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Safety monitor error: {e}")
    
    async def _update_coordination_metrics(self):
        """Update coordination performance metrics"""
        try:
            # Calculate coordination efficiency
            total_robots = len(self.robot_status)
            active_robots = sum(1 for robot in self.robot_status.values() 
                              if robot.current_task is not None)
            
            if total_robots > 0:
                self.coordination_efficiency = active_robots / total_robots
            
            # Calculate task completion rate
            completed_tasks = sum(1 for task in self.coordinated_tasks.values() 
                                if task.status == "completed")
            total_tasks = len(self.coordinated_tasks)
            
            if total_tasks > 0:
                self.task_completion_rate = completed_tasks / total_tasks
            
            # Calculate formation accuracy
            formation_errors = []
            for formation in self.active_formations.values():
                error = await self._calculate_formation_error(formation)
                formation_errors.append(error)
            
            if formation_errors:
                self.formation_accuracy = 1.0 - np.mean(formation_errors)
            
        except Exception as e:
            logger.error(f"Error updating coordination metrics: {e}")
    
    # Additional helper methods would be implemented here for:
    # - Task assignment and execution
    # - Formation movement and control  
    # - Collision avoidance
    # - Communication monitoring
    # - Emergency procedures
    # - Human command execution
    
    # These methods follow similar patterns to the ones shown above