"""
Distributed Learning - Collective Skill Acquisition System
=========================================================

The Distributed Learning module enables the robot swarm to learn collectively,
sharing training data, model updates, and learned behaviors across all members
while maintaining individual learning autonomy and preventing harmful patterns.

Features:
- Federated learning across the swarm
- Distributed model training and updates
- Skill transfer and adaptation
- Collective behavior learning
- Learning validation and safety checks
- Performance optimization through shared experiences
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
import hashlib
from collections import defaultdict, deque
import pickle
import copy

logger = logging.getLogger(__name__)


class LearningType(Enum):
    """Types of distributed learning"""
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    IMITATION = "imitation"
    TRANSFER = "transfer"
    META_LEARNING = "meta_learning"
    BEHAVIORAL = "behavioral"


class LearningScope(Enum):
    """Scope of learning propagation"""
    INDIVIDUAL = "individual"
    LOCAL_CLUSTER = "local_cluster"
    SWARM_WIDE = "swarm_wide"
    GLOBAL = "global"


class ModelUpdateType(Enum):
    """Types of model updates"""
    WEIGHT_UPDATE = "weight_update"
    GRADIENT_UPDATE = "gradient_update"
    FULL_MODEL = "full_model"
    SKILL_PATTERN = "skill_pattern"
    BEHAVIORAL_RULE = "behavioral_rule"


@dataclass
class LearningExperience:
    """A learning experience to be shared"""
    id: str
    robot_id: str
    learning_type: LearningType
    task_context: Dict[str, Any]
    state_data: np.ndarray
    action_data: np.ndarray
    reward_data: np.ndarray
    outcome_data: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: datetime
    confidence: float = 1.0
    validation_count: int = 0
    validation_score: float = 0.0


@dataclass
class ModelUpdate:
    """Model update package for distribution"""
    id: str
    source_robot: str
    model_id: str
    update_type: ModelUpdateType
    update_data: bytes  # Serialized model data
    metadata: Dict[str, Any]
    performance_improvement: float
    validation_metrics: Dict[str, float]
    timestamp: datetime
    propagation_scope: LearningScope
    applied_robots: Set[str] = field(default_factory=set)
    safety_validated: bool = False


@dataclass
class LearningTask:
    """A distributed learning task"""
    id: str
    task_name: str
    task_type: LearningType
    objective: str
    participating_robots: Set[str]
    coordinator_robot: str
    data_requirements: Dict[str, Any]
    model_architecture: Dict[str, Any]
    convergence_criteria: Dict[str, float]
    safety_constraints: List[str]
    status: str
    created_time: datetime
    completion_time: Optional[datetime] = None
    performance_history: List[Dict[str, float]] = field(default_factory=list)


class DistributedLearning:
    """
    Distributed learning system for robot swarms
    
    Enables collective learning through federated training, shared experiences,
    and distributed model updates while maintaining safety and individual autonomy.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Learning state
        self.robot_models: Dict[str, Dict[str, Any]] = {}
        self.robot_learning_profiles: Dict[str, Dict[str, Any]] = {}
        self.robot_performance_history: Dict[str, List[Dict[str, float]]] = {}
        
        # Experience and update sharing
        self.experience_buffer: Dict[str, LearningExperience] = {}
        self.model_updates: Dict[str, ModelUpdate] = {}
        self.pending_updates: deque = deque()
        
        # Active learning tasks
        self.learning_tasks: Dict[str, LearningTask] = {}
        
        # Processing queues
        self.experience_queue = asyncio.Queue()
        self.update_queue = asyncio.Queue()
        self.validation_queue = asyncio.Queue()
        
        # Background tasks
        self.processing_task = None
        self.aggregation_task = None
        self.validation_task = None
        
        # Learning metrics
        self.swarm_learning_rate = 0.0
        self.convergence_rate = 0.0
        self.knowledge_transfer_efficiency = 0.0
        self.safety_violation_count = 0
        
        # Safety and validation
        self.safety_validators: List[Callable] = []
        self.performance_thresholds: Dict[str, float] = {}
        self.banned_patterns: Set[str] = set()
        
        # Federated learning parameters
        self.aggregation_rounds = 0
        self.min_participants = 3
        self.convergence_tolerance = 0.001
        
        logger.info("Distributed Learning system initialized")
    
    async def initialize(self) -> bool:
        """Initialize the distributed learning system"""
        try:
            # Initialize safety validators
            await self._initialize_safety_validators()
            
            # Set performance thresholds
            await self._initialize_performance_thresholds()
            
            # Start background processing
            self.processing_task = asyncio.create_task(self._process_learning_updates())
            self.aggregation_task = asyncio.create_task(self._aggregate_learning())
            self.validation_task = asyncio.create_task(self._validate_learning())
            
            logger.info("Distributed Learning system initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Distributed Learning initialization failed: {e}")
            return False
    
    async def register_learner(self, robot_id: str, capabilities: List[str],
                             learning_preferences: Dict[str, Any] = None) -> bool:
        """Register a robot as a distributed learner"""
        try:
            self.robot_models[robot_id] = {}
            self.robot_learning_profiles[robot_id] = {
                "capabilities": capabilities,
                "preferences": learning_preferences or {},
                "learning_rate": self.config.learning_rate,
                "specializations": [],
                "learning_history": []
            }
            self.robot_performance_history[robot_id] = []
            
            logger.info(f"Robot {robot_id} registered as distributed learner")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register learner {robot_id}: {e}")
            return False
    
    async def share_learning_experience(self, robot_id: str, learning_type: LearningType,
                                      task_context: Dict[str, Any],
                                      state_data: np.ndarray,
                                      action_data: np.ndarray,
                                      reward_data: np.ndarray,
                                      outcome_data: Dict[str, Any],
                                      performance_metrics: Dict[str, float]) -> str:
        """Share a learning experience with the swarm"""
        try:
            if robot_id not in self.robot_learning_profiles:
                logger.error(f"Robot {robot_id} not registered as learner")
                return ""
            
            # Create experience record
            experience_id = f"exp_{robot_id}_{datetime.now().timestamp()}"
            experience = LearningExperience(
                id=experience_id,
                robot_id=robot_id,
                learning_type=learning_type,
                task_context=task_context,
                state_data=state_data,
                action_data=action_data,
                reward_data=reward_data,
                outcome_data=outcome_data,
                performance_metrics=performance_metrics,
                timestamp=datetime.now(),
                confidence=self._calculate_experience_confidence(performance_metrics)
            )
            
            # Safety validation
            if not await self._validate_experience_safety(experience):
                logger.warning(f"Experience {experience_id} failed safety validation")
                return ""
            
            # Store experience
            self.experience_buffer[experience_id] = experience
            
            # Queue for processing
            await self.experience_queue.put(experience_id)
            
            logger.info(f"Learning experience shared by {robot_id}: {learning_type.value}")
            return experience_id
            
        except Exception as e:
            logger.error(f"Failed to share learning experience: {e}")
            return ""
    
    async def distribute_model_update(self, robot_id: str, model_id: str,
                                    update_type: ModelUpdateType,
                                    update_data: Any,
                                    performance_improvement: float,
                                    validation_metrics: Dict[str, float],
                                    scope: LearningScope = LearningScope.SWARM_WIDE) -> str:
        """Distribute a model update across the swarm"""
        try:
            if robot_id not in self.robot_learning_profiles:
                logger.error(f"Robot {robot_id} not registered as learner")
                return ""
            
            # Serialize update data
            serialized_data = pickle.dumps(update_data)
            
            # Create model update
            update_id = f"update_{robot_id}_{datetime.now().timestamp()}"
            update = ModelUpdate(
                id=update_id,
                source_robot=robot_id,
                model_id=model_id,
                update_type=update_type,
                update_data=serialized_data,
                metadata={
                    "data_size": len(serialized_data),
                    "compression": "pickle",
                    "model_version": "1.0"
                },
                performance_improvement=performance_improvement,
                validation_metrics=validation_metrics,
                timestamp=datetime.now(),
                propagation_scope=scope
            )
            
            # Safety validation
            if not await self._validate_update_safety(update):
                logger.warning(f"Model update {update_id} failed safety validation")
                return ""
            
            update.safety_validated = True
            
            # Store update
            self.model_updates[update_id] = update
            
            # Queue for distribution
            await self.update_queue.put(update_id)
            
            logger.info(f"Model update distributed by {robot_id}: {update_type.value}")
            return update_id
            
        except Exception as e:
            logger.error(f"Failed to distribute model update: {e}")
            return ""
    
    async def start_federated_learning_task(self, coordinator_robot: str,
                                          task_name: str,
                                          learning_type: LearningType,
                                          objective: str,
                                          participating_robots: List[str],
                                          model_architecture: Dict[str, Any],
                                          data_requirements: Dict[str, Any],
                                          safety_constraints: List[str] = None) -> str:
        """Start a federated learning task across multiple robots"""
        try:
            if coordinator_robot not in self.robot_learning_profiles:
                logger.error(f"Coordinator robot {coordinator_robot} not registered")
                return ""
            
            # Validate participants
            valid_participants = []
            for robot_id in participating_robots:
                if robot_id in self.robot_learning_profiles:
                    valid_participants.append(robot_id)
                else:
                    logger.warning(f"Robot {robot_id} not registered - excluding from task")
            
            if len(valid_participants) < self.min_participants:
                logger.error(f"Insufficient participants for federated learning task")
                return ""
            
            # Create learning task
            task_id = f"fedlearn_{coordinator_robot}_{datetime.now().timestamp()}"
            task = LearningTask(
                id=task_id,
                task_name=task_name,
                task_type=learning_type,
                objective=objective,
                participating_robots=set(valid_participants),
                coordinator_robot=coordinator_robot,
                data_requirements=data_requirements,
                model_architecture=model_architecture,
                convergence_criteria={"tolerance": self.convergence_tolerance, "max_rounds": 100},
                safety_constraints=safety_constraints or [],
                status="initializing",
                created_time=datetime.now()
            )
            
            self.learning_tasks[task_id] = task
            
            # Initialize federated learning
            await self._initialize_federated_task(task_id)
            
            logger.info(f"Federated learning task started: {task_name}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to start federated learning task: {e}")
            return ""
    
    async def apply_model_update(self, robot_id: str, update_id: str) -> bool:
        """Apply a model update to a robot's local models"""
        try:
            if robot_id not in self.robot_learning_profiles:
                logger.error(f"Robot {robot_id} not registered as learner")
                return False
            
            if update_id not in self.model_updates:
                logger.error(f"Model update {update_id} not found")
                return False
            
            update = self.model_updates[update_id]
            
            # Check if update is safe and validated
            if not update.safety_validated:
                logger.warning(f"Model update {update_id} not safety validated")
                return False
            
            # Check if robot should receive this update (scope filtering)
            if not self._should_receive_update(robot_id, update):
                return True  # Not an error, just not applicable
            
            # Deserialize and apply update
            try:
                update_data = pickle.loads(update.update_data)
                success = await self._apply_update_to_robot(robot_id, update.model_id, 
                                                          update.update_type, update_data)
                
                if success:
                    update.applied_robots.add(robot_id)
                    
                    # Update robot's performance history
                    performance_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "update_id": update_id,
                        "improvement": update.performance_improvement,
                        "metrics": update.validation_metrics
                    }
                    self.robot_performance_history[robot_id].append(performance_entry)
                    
                    logger.info(f"Model update {update_id} applied to robot {robot_id}")
                    return True
                else:
                    logger.error(f"Failed to apply update {update_id} to robot {robot_id}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error deserializing/applying update {update_id}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply model update: {e}")
            return False
    
    async def get_learning_recommendations(self, robot_id: str, 
                                         current_task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get personalized learning recommendations for a robot"""
        try:
            if robot_id not in self.robot_learning_profiles:
                return []
            
            recommendations = []
            
            # Find relevant experiences from other robots
            relevant_experiences = await self._find_relevant_experiences(robot_id, current_task)
            
            for exp_id, relevance_score in relevant_experiences[:5]:  # Top 5
                experience = self.experience_buffer[exp_id]
                recommendation = {
                    "type": "experience_learning",
                    "experience_id": exp_id,
                    "source_robot": experience.robot_id,
                    "learning_type": experience.learning_type.value,
                    "relevance_score": relevance_score,
                    "performance_potential": experience.confidence,
                    "description": f"Learn from {experience.robot_id}'s experience with {experience.learning_type.value}"
                }
                recommendations.append(recommendation)
            
            # Find applicable model updates
            applicable_updates = await self._find_applicable_updates(robot_id)
            
            for update_id, applicability_score in applicable_updates[:3]:  # Top 3
                update = self.model_updates[update_id]
                recommendation = {
                    "type": "model_update",
                    "update_id": update_id,
                    "source_robot": update.source_robot,
                    "update_type": update.update_type.value,
                    "applicability_score": applicability_score,
                    "performance_improvement": update.performance_improvement,
                    "description": f"Apply {update.update_type.value} from {update.source_robot}"
                }
                recommendations.append(recommendation)
            
            # Sort by potential benefit
            recommendations.sort(key=lambda x: x.get("relevance_score", 0) + x.get("applicability_score", 0), 
                               reverse=True)
            
            return recommendations[:10]  # Top 10 recommendations
            
        except Exception as e:
            logger.error(f"Failed to get learning recommendations: {e}")
            return []
    
    async def update_from_experience(self, action: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Update distributed learning from a swarm action experience"""
        try:
            # Extract learning data from action and result
            learning_data = await self._extract_learning_from_action(action, result)
            
            if not learning_data:
                return True  # No learning data to extract
            
            # Create synthetic experience for swarm-level learning
            for robot_id in learning_data.get("involved_robots", []):
                if robot_id in self.robot_learning_profiles:
                    await self.share_learning_experience(
                        robot_id=robot_id,
                        learning_type=LearningType.REINFORCEMENT,
                        task_context={"swarm_action": action.get("type", "unknown")},
                        state_data=np.array(learning_data.get("state", [])),
                        action_data=np.array(learning_data.get("action", [])),
                        reward_data=np.array([learning_data.get("reward", 0.0)]),
                        outcome_data=result,
                        performance_metrics=learning_data.get("metrics", {})
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update from experience: {e}")
            return False
    
    async def get_progress(self) -> Dict[str, Any]:
        """Get learning progress across the swarm"""
        try:
            total_experiences = len(self.experience_buffer)
            total_updates = len(self.model_updates)
            active_tasks = len([t for t in self.learning_tasks.values() if t.status == "active"])
            
            # Calculate average performance improvement
            all_improvements = []
            for robot_history in self.robot_performance_history.values():
                improvements = [entry["improvement"] for entry in robot_history]
                all_improvements.extend(improvements)
            
            avg_improvement = np.mean(all_improvements) if all_improvements else 0.0
            
            # Calculate convergence metrics
            convergence_metrics = await self._calculate_convergence_metrics()
            
            return {
                "total_experiences_shared": total_experiences,
                "total_model_updates": total_updates,
                "active_learning_tasks": active_tasks,
                "completed_tasks": len([t for t in self.learning_tasks.values() if t.status == "completed"]),
                "registered_learners": len(self.robot_learning_profiles),
                "average_performance_improvement": avg_improvement,
                "swarm_learning_rate": self.swarm_learning_rate,
                "convergence_rate": self.convergence_rate,
                "knowledge_transfer_efficiency": self.knowledge_transfer_efficiency,
                "safety_violations": self.safety_violation_count,
                "aggregation_rounds_completed": self.aggregation_rounds,
                "convergence_metrics": convergence_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get learning progress: {e}")
            return {}
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown distributed learning system"""
        try:
            logger.info("Shutting down Distributed Learning system")
            
            # Stop background tasks
            if self.processing_task:
                self.processing_task.cancel()
            if self.aggregation_task:
                self.aggregation_task.cancel()
            if self.validation_task:
                self.validation_task.cancel()
            
            # Complete active learning tasks
            for task in self.learning_tasks.values():
                if task.status == "active":
                    task.status = "shutdown"
                    task.completion_time = datetime.now()
            
            # Save learning state
            await self._save_learning_state()
            
            logger.info("Distributed Learning system shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Distributed Learning shutdown failed: {e}")
            return False
    
    # Private helper methods
    
    async def _process_learning_updates(self):
        """Background task to process learning updates"""
        while True:
            try:
                # Process experiences
                try:
                    exp_id = await asyncio.wait_for(self.experience_queue.get(), timeout=0.5)
                    await self._process_experience(exp_id)
                except asyncio.TimeoutError:
                    pass
                
                # Process model updates
                try:
                    update_id = await asyncio.wait_for(self.update_queue.get(), timeout=0.5)
                    await self._distribute_update(update_id)
                except asyncio.TimeoutError:
                    pass
                
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing learning updates: {e}")
    
    async def _aggregate_learning(self):
        """Background task for federated learning aggregation"""
        while True:
            try:
                # Perform aggregation rounds for active tasks
                for task_id, task in self.learning_tasks.items():
                    if task.status == "active":
                        await self._perform_aggregation_round(task_id)
                
                # Update swarm-wide learning metrics
                await self._update_learning_metrics()
                
                await asyncio.sleep(30.0)  # Aggregation every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in learning aggregation: {e}")
    
    async def _validate_learning(self):
        """Background task for learning validation"""
        while True:
            try:
                # Validate recent experiences and updates
                await self._validate_recent_learning()
                
                # Check for harmful patterns
                await self._detect_harmful_patterns()
                
                await asyncio.sleep(60.0)  # Validation every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in learning validation: {e}")
    
    def _calculate_experience_confidence(self, performance_metrics: Dict[str, float]) -> float:
        """Calculate confidence score for an experience"""
        if not performance_metrics:
            return 0.5
        
        # Use average of performance metrics as confidence
        return np.clip(np.mean(list(performance_metrics.values())), 0.0, 1.0)
    
    async def _validate_experience_safety(self, experience: LearningExperience) -> bool:
        """Validate that an experience is safe to share"""
        try:
            # Check against banned patterns
            experience_hash = hashlib.md5(
                str(experience.task_context).encode() + 
                experience.state_data.tobytes() + 
                experience.action_data.tobytes()
            ).hexdigest()
            
            if experience_hash in self.banned_patterns:
                return False
            
            # Check performance metrics for anomalies
            for metric_name, value in experience.performance_metrics.items():
                if metric_name in self.performance_thresholds:
                    if value < self.performance_thresholds[metric_name]:
                        return False
            
            # Run custom safety validators
            for validator in self.safety_validators:
                if not validator(experience):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating experience safety: {e}")
            return False
    
    async def _validate_update_safety(self, update: ModelUpdate) -> bool:
        """Validate that a model update is safe to distribute"""
        try:
            # Check update size (prevent memory attacks)
            if len(update.update_data) > 100 * 1024 * 1024:  # 100MB limit
                logger.warning(f"Update {update.id} exceeds size limit")
                return False
            
            # Check performance improvement (prevent degradation)
            if update.performance_improvement < -0.5:  # Allow small degradation
                logger.warning(f"Update {update.id} shows significant performance degradation")
                return False
            
            # Validate metadata
            if not update.validation_metrics:
                logger.warning(f"Update {update.id} lacks validation metrics")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating update safety: {e}")
            return False
    
    def _should_receive_update(self, robot_id: str, update: ModelUpdate) -> bool:
        """Check if a robot should receive a specific update"""
        # Don't send back to source
        if robot_id == update.source_robot:
            return False
        
        # Check scope
        if update.propagation_scope == LearningScope.INDIVIDUAL:
            return False
        
        # Check if already applied
        if robot_id in update.applied_robots:
            return False
        
        # Check robot capabilities (simplified)
        robot_profile = self.robot_learning_profiles.get(robot_id, {})
        robot_capabilities = robot_profile.get("capabilities", [])
        
        # Basic compatibility check (in real implementation, this would be more sophisticated)
        return len(robot_capabilities) > 0
    
    async def _apply_update_to_robot(self, robot_id: str, model_id: str, 
                                   update_type: ModelUpdateType, update_data: Any) -> bool:
        """Apply a model update to a specific robot"""
        try:
            # Initialize robot models if not exists
            if robot_id not in self.robot_models:
                self.robot_models[robot_id] = {}
            
            if model_id not in self.robot_models[robot_id]:
                self.robot_models[robot_id][model_id] = {"weights": {}, "metadata": {}}
            
            robot_model = self.robot_models[robot_id][model_id]
            
            # Apply update based on type
            if update_type == ModelUpdateType.WEIGHT_UPDATE:
                # Merge weights
                if "weights" in robot_model:
                    robot_model["weights"].update(update_data)
                else:
                    robot_model["weights"] = update_data
            
            elif update_type == ModelUpdateType.GRADIENT_UPDATE:
                # Apply gradient update (simplified)
                if "weights" in robot_model:
                    for key, gradient in update_data.items():
                        if key in robot_model["weights"]:
                            robot_model["weights"][key] += gradient * self.config.learning_rate
            
            elif update_type == ModelUpdateType.FULL_MODEL:
                # Replace entire model
                robot_model.update(update_data)
            
            elif update_type == ModelUpdateType.SKILL_PATTERN:
                # Add skill pattern
                if "skills" not in robot_model:
                    robot_model["skills"] = {}
                robot_model["skills"].update(update_data)
            
            elif update_type == ModelUpdateType.BEHAVIORAL_RULE:
                # Add behavioral rule
                if "behaviors" not in robot_model:
                    robot_model["behaviors"] = []
                robot_model["behaviors"].extend(update_data)
            
            # Update metadata
            robot_model["metadata"]["last_update"] = datetime.now().isoformat()
            robot_model["metadata"]["update_count"] = robot_model["metadata"].get("update_count", 0) + 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying update to robot {robot_id}: {e}")
            return False
    
    async def _find_relevant_experiences(self, robot_id: str, 
                                       current_task: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Find experiences relevant to a robot's current task"""
        relevant = []
        
        robot_profile = self.robot_learning_profiles.get(robot_id, {})
        robot_capabilities = set(robot_profile.get("capabilities", []))
        
        for exp_id, experience in self.experience_buffer.items():
            if experience.robot_id == robot_id:  # Don't recommend own experiences
                continue
            
            # Calculate relevance score
            relevance = 0.0
            
            # Task context similarity
            context_similarity = self._calculate_dict_similarity(
                experience.task_context, current_task
            )
            relevance += context_similarity * 0.4
            
            # Capability overlap
            exp_robot_profile = self.robot_learning_profiles.get(experience.robot_id, {})
            exp_capabilities = set(exp_robot_profile.get("capabilities", []))
            capability_overlap = len(robot_capabilities.intersection(exp_capabilities))
            if robot_capabilities:
                relevance += (capability_overlap / len(robot_capabilities)) * 0.3
            
            # Experience quality
            relevance += experience.confidence * 0.3
            
            if relevance > 0.3:  # Minimum relevance threshold
                relevant.append((exp_id, relevance))
        
        relevant.sort(key=lambda x: x[1], reverse=True)
        return relevant
    
    async def _find_applicable_updates(self, robot_id: str) -> List[Tuple[str, float]]:
        """Find model updates applicable to a robot"""
        applicable = []
        
        for update_id, update in self.model_updates.items():
            if not self._should_receive_update(robot_id, update):
                continue
            
            # Calculate applicability score
            applicability = 0.0
            
            # Performance improvement potential
            applicability += min(update.performance_improvement, 1.0) * 0.5
            
            # Validation quality
            if update.validation_metrics:
                avg_validation = np.mean(list(update.validation_metrics.values()))
                applicability += avg_validation * 0.3
            
            # Safety validation
            if update.safety_validated:
                applicability += 0.2
            
            if applicability > 0.4:  # Minimum applicability threshold
                applicable.append((update_id, applicability))
        
        applicable.sort(key=lambda x: x[1], reverse=True)
        return applicable
    
    def _calculate_dict_similarity(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> float:
        """Calculate similarity between two dictionaries"""
        if not dict1 or not dict2:
            return 0.0
        
        common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
        if not common_keys:
            return 0.0
        
        matches = sum(1 for key in common_keys if dict1[key] == dict2[key])
        return matches / len(common_keys)
    
    async def _extract_learning_from_action(self, action: Dict[str, Any], 
                                          result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract learning data from a swarm action and result"""
        try:
            learning_data = {}
            
            # Extract involved robots
            involved_robots = action.get("involved_robots", [])
            if not involved_robots and "robot_id" in action:
                involved_robots = [action["robot_id"]]
            
            learning_data["involved_robots"] = involved_robots
            
            # Extract state representation
            state_features = []
            if "environment" in action:
                state_features.extend(list(action["environment"].values()))
            if "parameters" in action:
                state_features.extend(list(action["parameters"].values()))
            
            learning_data["state"] = state_features[:10]  # Limit to 10 features
            
            # Extract action representation
            action_features = [hash(str(action.get("type", ""))) % 1000]  # Simple action encoding
            learning_data["action"] = action_features
            
            # Calculate reward based on result
            if result.get("status") == "success":
                reward = 1.0
            elif result.get("status") == "partial_success":
                reward = 0.5
            else:
                reward = -0.5
            
            learning_data["reward"] = reward
            
            # Extract performance metrics
            metrics = {}
            if "execution_time" in result:
                metrics["execution_time"] = float(result["execution_time"])
            if "success_rate" in result:
                metrics["success_rate"] = float(result["success_rate"])
            if "efficiency" in result:
                metrics["efficiency"] = float(result["efficiency"])
            
            learning_data["metrics"] = metrics
            
            return learning_data if learning_data["involved_robots"] else None
            
        except Exception as e:
            logger.error(f"Error extracting learning from action: {e}")
            return None
    
    async def _initialize_safety_validators(self):
        """Initialize safety validation functions"""
        def basic_safety_validator(experience: LearningExperience) -> bool:
            # Check for reasonable action bounds
            if experience.action_data.size > 0:
                if np.any(np.abs(experience.action_data) > 100):  # Arbitrary large action
                    return False
            
            # Check for reasonable rewards
            if experience.reward_data.size > 0:
                if np.any(np.abs(experience.reward_data) > 1000):  # Arbitrary large reward
                    return False
            
            return True
        
        self.safety_validators.append(basic_safety_validator)
    
    async def _initialize_performance_thresholds(self):
        """Initialize performance thresholds for safety"""
        self.performance_thresholds = {
            "success_rate": 0.1,  # Minimum 10% success rate
            "efficiency": 0.05,   # Minimum 5% efficiency
            "safety_score": 0.8   # Minimum 80% safety score
        }
    
    async def _initialize_federated_task(self, task_id: str):
        """Initialize a federated learning task"""
        try:
            task = self.learning_tasks[task_id]
            
            # Initialize model on all participants
            for robot_id in task.participating_robots:
                if task.model_id not in self.robot_models.get(robot_id, {}):
                    # Initialize with base model architecture
                    await self._initialize_robot_model(robot_id, task.model_id, task.model_architecture)
            
            task.status = "active"
            logger.info(f"Federated learning task {task_id} initialized")
            
        except Exception as e:
            logger.error(f"Error initializing federated task: {e}")
    
    async def _initialize_robot_model(self, robot_id: str, model_id: str, architecture: Dict[str, Any]):
        """Initialize a model for a robot"""
        if robot_id not in self.robot_models:
            self.robot_models[robot_id] = {}
        
        self.robot_models[robot_id][model_id] = {
            "architecture": architecture,
            "weights": self._generate_initial_weights(architecture),
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": "1.0",
                "update_count": 0
            }
        }
    
    def _generate_initial_weights(self, architecture: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate initial random weights for a model"""
        weights = {}
        
        # Simple weight initialization based on architecture
        layer_count = architecture.get("layers", 3)
        input_size = architecture.get("input_size", 10)
        hidden_size = architecture.get("hidden_size", 64)
        output_size = architecture.get("output_size", 1)
        
        for i in range(layer_count):
            if i == 0:
                weights[f"layer_{i}"] = np.random.normal(0, 0.1, (input_size, hidden_size))
            elif i == layer_count - 1:
                weights[f"layer_{i}"] = np.random.normal(0, 0.1, (hidden_size, output_size))
            else:
                weights[f"layer_{i}"] = np.random.normal(0, 0.1, (hidden_size, hidden_size))
        
        return weights
    
    async def _perform_aggregation_round(self, task_id: str):
        """Perform one round of federated aggregation"""
        try:
            task = self.learning_tasks[task_id]
            
            # Collect model updates from participants
            participant_weights = {}
            for robot_id in task.participating_robots:
                if robot_id in self.robot_models and task.model_id in self.robot_models[robot_id]:
                    participant_weights[robot_id] = self.robot_models[robot_id][task.model_id]["weights"]
            
            if len(participant_weights) < self.min_participants:
                return  # Not enough participants
            
            # Simple federated averaging
            aggregated_weights = self._federated_average(participant_weights)
            
            # Distribute aggregated weights back to participants
            for robot_id in task.participating_robots:
                if robot_id in self.robot_models and task.model_id in self.robot_models[robot_id]:
                    self.robot_models[robot_id][task.model_id]["weights"] = copy.deepcopy(aggregated_weights)
            
            self.aggregation_rounds += 1
            
            # Check convergence
            if await self._check_task_convergence(task_id):
                task.status = "completed"
                task.completion_time = datetime.now()
                logger.info(f"Federated learning task {task_id} converged")
            
        except Exception as e:
            logger.error(f"Error in aggregation round for task {task_id}: {e}")
    
    def _federated_average(self, participant_weights: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Compute federated average of participant weights"""
        if not participant_weights:
            return {}
        
        # Get all weight keys from first participant
        first_participant = next(iter(participant_weights.values()))
        weight_keys = first_participant.keys()
        
        averaged_weights = {}
        for key in weight_keys:
            # Stack all participant weights for this layer
            weight_arrays = [weights[key] for weights in participant_weights.values() if key in weights]
            
            if weight_arrays:
                # Simple average
                averaged_weights[key] = np.mean(weight_arrays, axis=0)
        
        return averaged_weights
    
    async def _check_task_convergence(self, task_id: str) -> bool:
        """Check if a federated learning task has converged"""
        try:
            task = self.learning_tasks[task_id]
            
            # Simple convergence check based on aggregation rounds
            max_rounds = task.convergence_criteria.get("max_rounds", 100)
            
            if self.aggregation_rounds >= max_rounds:
                return True
            
            # In a real implementation, this would check weight changes, performance metrics, etc.
            return False
            
        except Exception as e:
            logger.error(f"Error checking task convergence: {e}")
            return False
    
    async def _calculate_convergence_metrics(self) -> Dict[str, float]:
        """Calculate convergence metrics for active learning tasks"""
        active_tasks = [t for t in self.learning_tasks.values() if t.status == "active"]
        
        if not active_tasks:
            return {"convergence_rate": 0.0, "average_rounds": 0.0}
        
        total_rounds = sum(t.performance_history for t in active_tasks if hasattr(t, 'performance_history'))
        avg_rounds = total_rounds / len(active_tasks) if active_tasks else 0
        
        return {
            "convergence_rate": self.convergence_rate,
            "average_rounds_to_convergence": avg_rounds,
            "active_tasks": len(active_tasks)
        }
    
    async def _update_learning_metrics(self):
        """Update swarm-wide learning metrics"""
        try:
            # Calculate swarm learning rate
            recent_experiences = [exp for exp in self.experience_buffer.values()
                                if (datetime.now() - exp.timestamp).seconds < 3600]  # Last hour
            
            self.swarm_learning_rate = len(recent_experiences) / max(1, len(self.robot_learning_profiles))
            
            # Calculate knowledge transfer efficiency
            total_updates_applied = sum(len(update.applied_robots) for update in self.model_updates.values())
            total_updates_available = len(self.model_updates) * len(self.robot_learning_profiles)
            
            if total_updates_available > 0:
                self.knowledge_transfer_efficiency = total_updates_applied / total_updates_available
            
            # Update convergence rate
            completed_tasks = len([t for t in self.learning_tasks.values() if t.status == "completed"])
            total_tasks = len(self.learning_tasks)
            
            if total_tasks > 0:
                self.convergence_rate = completed_tasks / total_tasks
            
        except Exception as e:
            logger.error(f"Error updating learning metrics: {e}")
    
    async def _process_experience(self, exp_id: str):
        """Process a shared learning experience"""
        try:
            if exp_id not in self.experience_buffer:
                return
            
            experience = self.experience_buffer[exp_id]
            
            # Add to robot's learning history
            robot_profile = self.robot_learning_profiles.get(experience.robot_id, {})
            if "learning_history" not in robot_profile:
                robot_profile["learning_history"] = []
            
            robot_profile["learning_history"].append({
                "experience_id": exp_id,
                "timestamp": experience.timestamp.isoformat(),
                "type": experience.learning_type.value,
                "confidence": experience.confidence
            })
            
            # Maintain history size
            if len(robot_profile["learning_history"]) > 1000:
                robot_profile["learning_history"] = robot_profile["learning_history"][-500:]
            
        except Exception as e:
            logger.error(f"Error processing experience {exp_id}: {e}")
    
    async def _distribute_update(self, update_id: str):
        """Distribute a model update to relevant robots"""
        try:
            if update_id not in self.model_updates:
                return
            
            update = self.model_updates[update_id]
            
            # Find robots that should receive this update
            target_robots = []
            for robot_id in self.robot_learning_profiles:
                if self._should_receive_update(robot_id, update):
                    target_robots.append(robot_id)
            
            logger.info(f"Distributing update {update_id} to {len(target_robots)} robots")
            
            # In a real implementation, this would actually send the update to robots
            
        except Exception as e:
            logger.error(f"Error distributing update {update_id}: {e}")
    
    async def _validate_recent_learning(self):
        """Validate recent learning activities for safety"""
        try:
            current_time = datetime.now()
            
            # Validate recent experiences
            recent_experiences = [
                exp for exp in self.experience_buffer.values()
                if (current_time - exp.timestamp).seconds < 3600  # Last hour
            ]
            
            for experience in recent_experiences:
                if not await self._validate_experience_safety(experience):
                    # Mark as unsafe and potentially remove
                    logger.warning(f"Unsafe experience detected: {experience.id}")
                    self.safety_violation_count += 1
            
        except Exception as e:
            logger.error(f"Error validating recent learning: {e}")
    
    async def _detect_harmful_patterns(self):
        """Detect and prevent harmful learning patterns"""
        try:
            # Simple pattern detection based on performance degradation
            for robot_id, history in self.robot_performance_history.items():
                if len(history) < 5:
                    continue
                
                recent_improvements = [entry["improvement"] for entry in history[-5:]]
                avg_recent_improvement = np.mean(recent_improvements)
                
                if avg_recent_improvement < -0.3:  # Significant degradation
                    logger.warning(f"Harmful pattern detected for robot {robot_id}")
                    # In a real implementation, this would trigger corrective actions
            
        except Exception as e:
            logger.error(f"Error detecting harmful patterns: {e}")
    
    async def _save_learning_state(self):
        """Save learning state for persistence"""
        try:
            # In production, this would save to persistent storage
            logger.info(f"Saving learning state with {len(self.experience_buffer)} experiences")
            
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")