"""
NEXUS - Collective Intelligence and Swarm Coordination System
===========================================================

The NEXUS system enables collective consciousness and coordinated behavior
across multiple robotic units while maintaining individual autonomy and
human authority override capabilities.

Core Components:
- Hive Mind: Collective consciousness framework
- Swarm Coordinator: Multi-robot coordination
- Consensus Engine: Democratic decision making
- Knowledge Sharing: Experience distribution
- Distributed Learning: Collective skill acquisition
- Communication Protocol: Robot-to-robot messaging
- Collective Memory: Shared memory pool
- Swarm Ethics: Collective behavior validation

Safety Features:
- Human authority always overrides collective decisions
- Individual robot autonomy preserved
- Ethical validation of all collective actions
- Emergency dispersal capabilities
- Complete transparency of collective processes
"""

from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Core imports
from .hive_mind import HiveMind
from .swarm_coordinator import SwarmCoordinator
from .consensus_engine import ConsensusEngine
from .knowledge_sharing import KnowledgeSharing
from .distributed_learning import DistributedLearning
from .communication_protocol import CommunicationProtocol
from .collective_memory import CollectiveMemory
from .swarm_ethics import SwarmEthics

__version__ = "1.0.0"
__author__ = "Project OLYMPUS"

logger = logging.getLogger(__name__)


class SwarmState(Enum):
    """Current state of the swarm collective"""
    DORMANT = "dormant"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    COORDINATING = "coordinating"
    LEARNING = "learning"
    DISPERSING = "dispersing"
    EMERGENCY = "emergency"


@dataclass
class SwarmConfiguration:
    """Configuration for NEXUS swarm operations"""
    max_swarm_size: int = 100
    consensus_threshold: float = 0.67
    learning_rate: float = 0.1
    communication_range: float = 1000.0  # meters
    memory_retention_hours: int = 24
    ethics_validation_required: bool = True
    human_override_enabled: bool = True
    emergency_dispersal_enabled: bool = True
    transparency_logging: bool = True
    individual_autonomy_preserved: bool = True


class NEXUSCore:
    """
    Core NEXUS system orchestrating collective intelligence
    
    This class manages all aspects of swarm coordination while ensuring
    safety, ethics, and human authority override capabilities.
    """
    
    def __init__(self, config: Optional[SwarmConfiguration] = None):
        self.config = config or SwarmConfiguration()
        self.state = SwarmState.DORMANT
        self.swarm_id = f"nexus_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize core components
        self.hive_mind = HiveMind(self.config)
        self.coordinator = SwarmCoordinator(self.config)
        self.consensus = ConsensusEngine(self.config)
        self.knowledge = KnowledgeSharing(self.config)
        self.learning = DistributedLearning(self.config)
        self.communication = CommunicationProtocol(self.config)
        self.memory = CollectiveMemory(self.config)
        self.ethics = SwarmEthics(self.config)
        
        # Safety systems
        self.human_override_active = False
        self.emergency_stop_triggered = False
        
        logger.info(f"NEXUS Core initialized with swarm ID: {self.swarm_id}")
    
    async def initialize_swarm(self, initial_robots: List[str]) -> bool:
        """Initialize the swarm collective with specified robots"""
        try:
            self.state = SwarmState.INITIALIZING
            
            # Initialize all subsystems
            await self.hive_mind.initialize()
            await self.coordinator.initialize(initial_robots)
            await self.consensus.initialize()
            await self.knowledge.initialize()
            await self.learning.initialize()
            await self.communication.initialize()
            await self.memory.initialize()
            await self.ethics.initialize()
            
            # Establish collective consciousness
            await self.hive_mind.form_collective(initial_robots)
            
            self.state = SwarmState.ACTIVE
            logger.info(f"NEXUS swarm initialized with {len(initial_robots)} robots")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize NEXUS swarm: {e}")
            self.state = SwarmState.DORMANT
            return False
    
    async def coordinate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate a collective action across the swarm"""
        if self.human_override_active:
            return {"status": "blocked", "reason": "human_override_active"}
        
        if self.emergency_stop_triggered:
            return {"status": "blocked", "reason": "emergency_stop"}
        
        # Ethical validation first
        ethical_approval = await self.ethics.validate_action(action)
        if not ethical_approval["approved"]:
            return {"status": "blocked", "reason": "ethics_violation", 
                   "details": ethical_approval["reasons"]}
        
        # Build consensus
        consensus_result = await self.consensus.build_consensus(action)
        if not consensus_result["achieved"]:
            return {"status": "no_consensus", "details": consensus_result}
        
        # Execute coordinated action
        self.state = SwarmState.COORDINATING
        result = await self.coordinator.execute_coordinated_action(action)
        
        # Share knowledge and learn from result
        await self.knowledge.share_experience(action, result)
        await self.learning.update_from_experience(action, result)
        
        self.state = SwarmState.ACTIVE
        return result
    
    async def emergency_dispersal(self, reason: str = "emergency") -> bool:
        """Trigger emergency dispersal of the swarm"""
        try:
            self.state = SwarmState.EMERGENCY
            self.emergency_stop_triggered = True
            
            logger.critical(f"Emergency dispersal triggered: {reason}")
            
            # Stop all coordinated activities
            await self.coordinator.emergency_stop()
            await self.consensus.abort_all_processes()
            
            # Disperse robots to safe positions
            dispersal_result = await self.coordinator.execute_dispersal()
            
            self.state = SwarmState.DISPERSING
            return dispersal_result
            
        except Exception as e:
            logger.error(f"Emergency dispersal failed: {e}")
            return False
    
    async def human_override(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Process human override command with absolute authority"""
        self.human_override_active = True
        
        logger.warning("Human override activated")
        
        try:
            # Stop all collective processes
            await self.hive_mind.suspend_collective()
            await self.coordinator.suspend_coordination()
            
            # Execute human command directly
            result = await self.coordinator.execute_human_command(command)
            
            # Log for transparency
            await self.memory.store_event({
                "type": "human_override",
                "command": command,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Human override execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def release_human_override(self) -> bool:
        """Release human override and return control to collective"""
        self.human_override_active = False
        logger.info("Human override released - collective control restored")
        return True
    
    async def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the swarm collective"""
        return {
            "swarm_id": self.swarm_id,
            "state": self.state.value,
            "human_override_active": self.human_override_active,
            "emergency_stop": self.emergency_stop_triggered,
            "active_robots": await self.coordinator.get_active_robots(),
            "consensus_status": await self.consensus.get_status(),
            "collective_memory_size": await self.memory.get_size(),
            "learning_progress": await self.learning.get_progress(),
            "ethics_violations": await self.ethics.get_recent_violations(),
            "communication_health": await self.communication.get_health()
        }
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown the NEXUS system"""
        try:
            logger.info("Initiating NEXUS shutdown sequence")
            
            # Suspend collective consciousness
            await self.hive_mind.dissolve_collective()
            
            # Shutdown all subsystems
            await self.coordinator.shutdown()
            await self.consensus.shutdown()
            await self.knowledge.shutdown()
            await self.learning.shutdown()
            await self.communication.shutdown()
            await self.memory.persist_state()
            await self.ethics.shutdown()
            
            self.state = SwarmState.DORMANT
            logger.info("NEXUS shutdown completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"NEXUS shutdown failed: {e}")
            return False


# Export main classes and utilities
__all__ = [
    'NEXUSCore',
    'SwarmState', 
    'SwarmConfiguration',
    'HiveMind',
    'SwarmCoordinator',
    'ConsensusEngine', 
    'KnowledgeSharing',
    'DistributedLearning',
    'CommunicationProtocol',
    'CollectiveMemory',
    'SwarmEthics'
]