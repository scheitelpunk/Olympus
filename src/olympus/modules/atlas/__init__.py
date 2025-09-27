"""
ATLAS - Advanced Transfer Learning and Adaptation System

A comprehensive transfer learning framework enabling safe knowledge transfer 
across domains with ethical validation and reality gap mitigation.

Components:
- Knowledge Transfer: Cross-domain knowledge sharing
- Domain Adaptation: Strategy adaptation across environments
- Sim2Real Bridge: Simulation to reality transfer
- Meta Learning: Learn-to-learn capabilities  
- Few-Shot Learning: Rapid adaptation with minimal data
- Task Encoding: Knowledge representation and encoding
- Skill Library: Reusable skill management
- Transfer Validator: Safety and ethics validation
"""

from .knowledge_transfer import KnowledgeTransfer
from .domain_adaptation import DomainAdapter
from .sim2real_bridge import Sim2RealBridge
from .meta_learning import MetaLearner
from .few_shot_learner import FewShotLearner
from .task_encoder import TaskEncoder
from .skill_library import SkillLibrary
from .transfer_validator import TransferValidator

__version__ = "1.0.0"
__author__ = "Project OLYMPUS"

class ATLAS:
    """Main ATLAS system coordinator"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Initialize all components
        self.knowledge_transfer = KnowledgeTransfer(config)
        self.domain_adapter = DomainAdapter(config)
        self.sim2real_bridge = Sim2RealBridge(config)
        self.meta_learner = MetaLearner(config)
        self.few_shot_learner = FewShotLearner(config)
        self.task_encoder = TaskEncoder(config)
        self.skill_library = SkillLibrary(config)
        self.transfer_validator = TransferValidator(config)
        
    async def transfer_knowledge(self, source_domain, target_domain, knowledge, safety_level="high"):
        """Safely transfer knowledge between domains"""
        # Validate transfer safety first
        validation_result = await self.transfer_validator.validate_transfer(
            source_domain, target_domain, knowledge, safety_level
        )
        
        if not validation_result.is_safe:
            raise ValueError(f"Transfer rejected: {validation_result.reason}")
            
        # Perform knowledge transfer
        return await self.knowledge_transfer.transfer(
            source_domain, target_domain, knowledge
        )
        
    async def adapt_to_domain(self, source_skills, target_domain):
        """Adapt skills to new domain"""
        return await self.domain_adapter.adapt(source_skills, target_domain)
        
    async def bridge_sim_to_real(self, simulation_policy, real_world_context):
        """Bridge simulation to reality"""
        return await self.sim2real_bridge.transfer(simulation_policy, real_world_context)

__all__ = [
    'ATLAS',
    'KnowledgeTransfer', 
    'DomainAdapter',
    'Sim2RealBridge',
    'MetaLearner',
    'FewShotLearner', 
    'TaskEncoder',
    'SkillLibrary',
    'TransferValidator'
]