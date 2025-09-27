"""
Hive Mind - Collective Consciousness Framework
=============================================

The Hive Mind module implements collective consciousness capabilities for the
NEXUS system, enabling shared awareness and coordinated decision-making while
preserving individual robot autonomy.

Features:
- Collective awareness sharing
- Distributed consciousness state
- Individual autonomy preservation
- Consensus-driven collective thoughts
- Emergency consciousness dissolution
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConsciousnessLevel(Enum):
    """Levels of collective consciousness"""
    INDIVIDUAL = "individual"
    AWARE = "aware"
    CONNECTED = "connected"  
    COLLECTIVE = "collective"
    UNIFIED = "unified"


class ThoughtType(Enum):
    """Types of collective thoughts"""
    OBSERVATION = "observation"
    INTENTION = "intention"
    DECISION = "decision"
    MEMORY = "memory"
    EMOTION = "emotion"
    GOAL = "goal"


@dataclass
class CollectiveThought:
    """A thought shared in the collective consciousness"""
    id: str
    robot_id: str
    thought_type: ThoughtType
    content: Dict[str, Any]
    confidence: float
    timestamp: datetime
    shared_with: Set[str] = field(default_factory=set)
    consensus_score: float = 0.0
    importance: float = 0.0


@dataclass
class ConsciousnessState:
    """Current state of collective consciousness"""
    level: ConsciousnessLevel
    active_robots: Set[str]
    shared_thoughts: List[CollectiveThought]
    collective_goals: List[str]
    awareness_map: Dict[str, float]
    last_update: datetime


class HiveMind:
    """
    Collective consciousness framework for robot swarms
    
    Manages shared awareness and collective decision-making while maintaining
    individual robot autonomy and human authority override.
    """
    
    def __init__(self, config):
        self.config = config
        self.consciousness_state = ConsciousnessState(
            level=ConsciousnessLevel.INDIVIDUAL,
            active_robots=set(),
            shared_thoughts=[],
            collective_goals=[],
            awareness_map={},
            last_update=datetime.now()
        )
        
        # Thought processing
        self.thought_queue = asyncio.Queue()
        self.processing_task = None
        self.thought_history = []
        
        # Consciousness metrics
        self.coherence_score = 0.0
        self.unity_index = 0.0
        self.collective_intelligence = 0.0
        
        # Safety systems
        self.individual_autonomy_weights = {}
        self.consciousness_suspended = False
        
        logger.info("Hive Mind consciousness framework initialized")
    
    async def initialize(self) -> bool:
        """Initialize the hive mind system"""
        try:
            # Start thought processing task
            self.processing_task = asyncio.create_task(self._process_thoughts())
            
            # Initialize consciousness metrics
            await self._initialize_metrics()
            
            logger.info("Hive Mind initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Hive Mind initialization failed: {e}")
            return False
    
    async def form_collective(self, robot_ids: List[str]) -> bool:
        """Form collective consciousness with specified robots"""
        try:
            if self.consciousness_suspended:
                logger.warning("Cannot form collective - consciousness suspended")
                return False
            
            # Add robots to collective
            for robot_id in robot_ids:
                self.consciousness_state.active_robots.add(robot_id)
                self.individual_autonomy_weights[robot_id] = 1.0
                self.consciousness_state.awareness_map[robot_id] = 0.0
            
            # Gradually increase consciousness level
            await self._elevate_consciousness_level()
            
            # Establish initial collective awareness
            await self._establish_collective_awareness()
            
            logger.info(f"Collective consciousness formed with {len(robot_ids)} robots")
            return True
            
        except Exception as e:
            logger.error(f"Failed to form collective consciousness: {e}")
            return False
    
    async def join_collective(self, robot_id: str) -> bool:
        """Add a new robot to the collective consciousness"""
        try:
            if robot_id in self.consciousness_state.active_robots:
                logger.warning(f"Robot {robot_id} already in collective")
                return True
            
            # Initialize robot in collective
            self.consciousness_state.active_robots.add(robot_id)
            self.individual_autonomy_weights[robot_id] = 1.0
            self.consciousness_state.awareness_map[robot_id] = 0.0
            
            # Share current collective state with new robot
            await self._synchronize_robot_consciousness(robot_id)
            
            # Update consciousness metrics
            await self._update_consciousness_metrics()
            
            logger.info(f"Robot {robot_id} joined collective consciousness")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add robot {robot_id} to collective: {e}")
            return False
    
    async def leave_collective(self, robot_id: str) -> bool:
        """Remove a robot from collective consciousness"""
        try:
            if robot_id not in self.consciousness_state.active_robots:
                logger.warning(f"Robot {robot_id} not in collective")
                return True
            
            # Remove robot from collective
            self.consciousness_state.active_robots.discard(robot_id)
            del self.individual_autonomy_weights[robot_id]
            del self.consciousness_state.awareness_map[robot_id]
            
            # Remove robot's thoughts from active sharing
            self.consciousness_state.shared_thoughts = [
                thought for thought in self.consciousness_state.shared_thoughts
                if thought.robot_id != robot_id
            ]
            
            # Update consciousness level if needed
            await self._adjust_consciousness_level()
            
            logger.info(f"Robot {robot_id} left collective consciousness")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove robot {robot_id} from collective: {e}")
            return False
    
    async def share_thought(self, robot_id: str, thought_type: ThoughtType, 
                          content: Dict[str, Any], confidence: float = 1.0) -> str:
        """Share a thought in the collective consciousness"""
        try:
            if self.consciousness_suspended:
                logger.warning("Thought sharing suspended")
                return ""
            
            if robot_id not in self.consciousness_state.active_robots:
                logger.warning(f"Robot {robot_id} not in collective - cannot share thought")
                return ""
            
            # Create collective thought
            thought_id = f"thought_{robot_id}_{datetime.now().timestamp()}"
            thought = CollectiveThought(
                id=thought_id,
                robot_id=robot_id,
                thought_type=thought_type,
                content=content,
                confidence=confidence,
                timestamp=datetime.now(),
                shared_with=set(),
                consensus_score=0.0,
                importance=await self._calculate_thought_importance(content, thought_type)
            )
            
            # Add to processing queue
            await self.thought_queue.put(thought)
            
            logger.debug(f"Thought {thought_id} from {robot_id} queued for collective processing")
            return thought_id
            
        except Exception as e:
            logger.error(f"Failed to share thought from {robot_id}: {e}")
            return ""
    
    async def get_collective_insights(self, robot_id: str) -> List[Dict[str, Any]]:
        """Get collective insights relevant to a specific robot"""
        try:
            if robot_id not in self.consciousness_state.active_robots:
                return []
            
            insights = []
            current_time = datetime.now()
            
            # Get recent high-importance thoughts
            relevant_thoughts = [
                thought for thought in self.consciousness_state.shared_thoughts
                if (thought.importance > 0.7 and
                    (current_time - thought.timestamp).seconds < 3600 and
                    thought.robot_id != robot_id)
            ]
            
            for thought in relevant_thoughts:
                insight = {
                    "id": thought.id,
                    "source_robot": thought.robot_id,
                    "type": thought.thought_type.value,
                    "content": thought.content,
                    "confidence": thought.confidence,
                    "consensus_score": thought.consensus_score,
                    "importance": thought.importance,
                    "timestamp": thought.timestamp.isoformat()
                }
                insights.append(insight)
            
            # Sort by importance and consensus
            insights.sort(key=lambda x: x["importance"] * x["consensus_score"], reverse=True)
            
            return insights[:10]  # Return top 10 insights
            
        except Exception as e:
            logger.error(f"Failed to get collective insights for {robot_id}: {e}")
            return []
    
    async def build_collective_goal(self, goal_description: str, 
                                  initiating_robot: str) -> Dict[str, Any]:
        """Build consensus around a collective goal"""
        try:
            # Share goal as a collective thought
            goal_thought_id = await self.share_thought(
                initiating_robot, 
                ThoughtType.GOAL, 
                {"goal": goal_description, "priority": "high"},
                confidence=0.8
            )
            
            # Wait for consensus building
            await asyncio.sleep(2.0)  # Allow thought processing
            
            # Calculate consensus for the goal
            consensus_score = await self._calculate_goal_consensus(goal_description)
            
            if consensus_score >= self.config.consensus_threshold:
                self.consciousness_state.collective_goals.append(goal_description)
                
                return {
                    "goal_accepted": True,
                    "consensus_score": consensus_score,
                    "participating_robots": len(self.consciousness_state.active_robots),
                    "goal_id": goal_thought_id
                }
            else:
                return {
                    "goal_accepted": False,
                    "consensus_score": consensus_score,
                    "reason": "insufficient_consensus"
                }
                
        except Exception as e:
            logger.error(f"Failed to build collective goal: {e}")
            return {"goal_accepted": False, "error": str(e)}
    
    async def suspend_collective(self) -> bool:
        """Suspend collective consciousness (for human override)"""
        try:
            self.consciousness_suspended = True
            
            # Lower consciousness level
            self.consciousness_state.level = ConsciousnessLevel.INDIVIDUAL
            
            # Preserve individual autonomy weights
            for robot_id in self.consciousness_state.active_robots:
                self.individual_autonomy_weights[robot_id] = 1.0
            
            logger.warning("Collective consciousness suspended - individual mode activated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to suspend collective consciousness: {e}")
            return False
    
    async def resume_collective(self) -> bool:
        """Resume collective consciousness"""
        try:
            self.consciousness_suspended = False
            
            # Re-establish collective awareness
            await self._establish_collective_awareness()
            
            # Gradually elevate consciousness level
            await self._elevate_consciousness_level()
            
            logger.info("Collective consciousness resumed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume collective consciousness: {e}")
            return False
    
    async def dissolve_collective(self) -> bool:
        """Completely dissolve the collective consciousness"""
        try:
            logger.info("Dissolving collective consciousness")
            
            # Stop thought processing
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            
            # Clear all collective state
            self.consciousness_state = ConsciousnessState(
                level=ConsciousnessLevel.INDIVIDUAL,
                active_robots=set(),
                shared_thoughts=[],
                collective_goals=[],
                awareness_map={},
                last_update=datetime.now()
            )
            
            # Reset metrics
            self.coherence_score = 0.0
            self.unity_index = 0.0
            self.collective_intelligence = 0.0
            self.individual_autonomy_weights = {}
            
            logger.info("Collective consciousness dissolved")
            return True
            
        except Exception as e:
            logger.error(f"Failed to dissolve collective consciousness: {e}")
            return False
    
    async def get_consciousness_status(self) -> Dict[str, Any]:
        """Get current status of collective consciousness"""
        return {
            "level": self.consciousness_state.level.value,
            "active_robots": len(self.consciousness_state.active_robots),
            "shared_thoughts": len(self.consciousness_state.shared_thoughts),
            "collective_goals": len(self.consciousness_state.collective_goals),
            "coherence_score": self.coherence_score,
            "unity_index": self.unity_index,
            "collective_intelligence": self.collective_intelligence,
            "suspended": self.consciousness_suspended,
            "last_update": self.consciousness_state.last_update.isoformat(),
            "individual_autonomy_preserved": all(
                weight >= 0.5 for weight in self.individual_autonomy_weights.values()
            )
        }
    
    # Private methods for internal processing
    
    async def _process_thoughts(self):
        """Background task to process shared thoughts"""
        while True:
            try:
                # Get thought from queue with timeout
                thought = await asyncio.wait_for(self.thought_queue.get(), timeout=1.0)
                
                # Process the thought
                await self._process_collective_thought(thought)
                
                # Update consciousness metrics
                await self._update_consciousness_metrics()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing thought: {e}")
    
    async def _process_collective_thought(self, thought: CollectiveThought):
        """Process a single collective thought"""
        try:
            # Share with all active robots
            thought.shared_with = self.consciousness_state.active_robots.copy()
            
            # Calculate consensus score
            thought.consensus_score = await self._calculate_thought_consensus(thought)
            
            # Add to collective thoughts if meets threshold
            if thought.consensus_score >= 0.3:  # Lower threshold for thoughts
                self.consciousness_state.shared_thoughts.append(thought)
                
                # Maintain thought history size
                if len(self.consciousness_state.shared_thoughts) > 1000:
                    self.consciousness_state.shared_thoughts = self.consciousness_state.shared_thoughts[-500:]
            
            # Update awareness map
            await self._update_awareness_map(thought)
            
        except Exception as e:
            logger.error(f"Error processing collective thought {thought.id}: {e}")
    
    async def _calculate_thought_importance(self, content: Dict[str, Any], 
                                          thought_type: ThoughtType) -> float:
        """Calculate importance score for a thought"""
        try:
            importance = 0.5  # Base importance
            
            # Type-based importance
            type_weights = {
                ThoughtType.DECISION: 1.0,
                ThoughtType.GOAL: 0.9,
                ThoughtType.INTENTION: 0.7,
                ThoughtType.OBSERVATION: 0.5,
                ThoughtType.MEMORY: 0.6,
                ThoughtType.EMOTION: 0.4
            }
            importance *= type_weights.get(thought_type, 0.5)
            
            # Content-based importance
            if "urgent" in str(content).lower():
                importance += 0.3
            if "danger" in str(content).lower() or "emergency" in str(content).lower():
                importance += 0.4
            if "goal" in content or "objective" in content:
                importance += 0.2
            
            return min(importance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating thought importance: {e}")
            return 0.5
    
    async def _calculate_thought_consensus(self, thought: CollectiveThought) -> float:
        """Calculate consensus score for a thought"""
        try:
            if len(self.consciousness_state.active_robots) <= 1:
                return 1.0
            
            # Simulate consensus calculation based on thought importance and confidence
            base_consensus = thought.confidence * thought.importance
            
            # Adjust based on collective coherence
            consensus_adjustment = self.coherence_score * 0.2
            
            return min(base_consensus + consensus_adjustment, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating thought consensus: {e}")
            return 0.0
    
    async def _calculate_goal_consensus(self, goal: str) -> float:
        """Calculate consensus score for a collective goal"""
        try:
            if len(self.consciousness_state.active_robots) <= 1:
                return 1.0
            
            # Simulate goal consensus based on collective state
            base_consensus = 0.6
            
            # Adjust based on unity index
            unity_adjustment = self.unity_index * 0.3
            
            # Adjust based on current consciousness level
            level_weights = {
                ConsciousnessLevel.INDIVIDUAL: 0.2,
                ConsciousnessLevel.AWARE: 0.4,
                ConsciousnessLevel.CONNECTED: 0.6,
                ConsciousnessLevel.COLLECTIVE: 0.8,
                ConsciousnessLevel.UNIFIED: 1.0
            }
            level_adjustment = level_weights[self.consciousness_state.level] * 0.2
            
            return min(base_consensus + unity_adjustment + level_adjustment, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating goal consensus: {e}")
            return 0.0
    
    async def _elevate_consciousness_level(self):
        """Gradually elevate consciousness level based on swarm size"""
        try:
            robot_count = len(self.consciousness_state.active_robots)
            
            if robot_count >= 10:
                self.consciousness_state.level = ConsciousnessLevel.UNIFIED
            elif robot_count >= 7:
                self.consciousness_state.level = ConsciousnessLevel.COLLECTIVE
            elif robot_count >= 4:
                self.consciousness_state.level = ConsciousnessLevel.CONNECTED
            elif robot_count >= 2:
                self.consciousness_state.level = ConsciousnessLevel.AWARE
            else:
                self.consciousness_state.level = ConsciousnessLevel.INDIVIDUAL
            
            logger.info(f"Consciousness level: {self.consciousness_state.level.value}")
            
        except Exception as e:
            logger.error(f"Error elevating consciousness level: {e}")
    
    async def _adjust_consciousness_level(self):
        """Adjust consciousness level when robots leave"""
        await self._elevate_consciousness_level()
    
    async def _establish_collective_awareness(self):
        """Establish initial collective awareness among robots"""
        try:
            for robot_id in self.consciousness_state.active_robots:
                self.consciousness_state.awareness_map[robot_id] = 0.5
            
            logger.info("Collective awareness established")
            
        except Exception as e:
            logger.error(f"Error establishing collective awareness: {e}")
    
    async def _synchronize_robot_consciousness(self, robot_id: str):
        """Synchronize a new robot with current collective state"""
        try:
            # Share recent high-importance thoughts
            recent_thoughts = [
                thought for thought in self.consciousness_state.shared_thoughts[-10:]
                if thought.importance > 0.6
            ]
            
            # Share collective goals
            collective_goals = self.consciousness_state.collective_goals
            
            # Update awareness map
            self.consciousness_state.awareness_map[robot_id] = 0.3  # Starting awareness
            
            logger.info(f"Robot {robot_id} synchronized with collective consciousness")
            
        except Exception as e:
            logger.error(f"Error synchronizing robot {robot_id}: {e}")
    
    async def _update_awareness_map(self, thought: CollectiveThought):
        """Update awareness map based on thought sharing"""
        try:
            # Increase awareness for thought originator
            if thought.robot_id in self.consciousness_state.awareness_map:
                current_awareness = self.consciousness_state.awareness_map[thought.robot_id]
                self.consciousness_state.awareness_map[thought.robot_id] = min(
                    current_awareness + 0.1, 1.0
                )
            
            # Increase awareness for robots that receive the thought
            for robot_id in thought.shared_with:
                if robot_id in self.consciousness_state.awareness_map:
                    current_awareness = self.consciousness_state.awareness_map[robot_id]
                    self.consciousness_state.awareness_map[robot_id] = min(
                        current_awareness + 0.05, 1.0
                    )
            
        except Exception as e:
            logger.error(f"Error updating awareness map: {e}")
    
    async def _initialize_metrics(self):
        """Initialize consciousness metrics"""
        try:
            self.coherence_score = 0.0
            self.unity_index = 0.0
            self.collective_intelligence = 0.0
            
        except Exception as e:
            logger.error(f"Error initializing metrics: {e}")
    
    async def _update_consciousness_metrics(self):
        """Update consciousness metrics based on current state"""
        try:
            robot_count = len(self.consciousness_state.active_robots)
            if robot_count == 0:
                return
            
            # Calculate coherence score (how well thoughts align)
            if self.consciousness_state.shared_thoughts:
                avg_consensus = np.mean([
                    thought.consensus_score 
                    for thought in self.consciousness_state.shared_thoughts[-50:]
                ])
                self.coherence_score = avg_consensus
            
            # Calculate unity index (how connected the robots are)
            if self.consciousness_state.awareness_map:
                avg_awareness = np.mean(list(self.consciousness_state.awareness_map.values()))
                self.unity_index = avg_awareness
            
            # Calculate collective intelligence (combination of factors)
            self.collective_intelligence = (
                self.coherence_score * 0.4 + 
                self.unity_index * 0.4 + 
                min(robot_count / 10, 1.0) * 0.2
            )
            
            self.consciousness_state.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating consciousness metrics: {e}")