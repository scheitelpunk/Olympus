"""
Domain Adaptation Module

Handles adaptation of skills and knowledge across different domains with
safety validation and performance monitoring.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class AdaptationStrategy(Enum):
    ADVERSARIAL = "adversarial"
    FEATURE_ALIGNMENT = "feature_alignment"
    DOMAIN_CONFUSION = "domain_confusion"
    GRADUAL_ADAPTATION = "gradual_adaptation"
    MULTI_DOMAIN = "multi_domain"

@dataclass
class DomainProfile:
    """Profile of a domain's characteristics"""
    name: str
    features: Dict[str, Any]
    constraints: List[str]
    safety_requirements: str
    adaptation_difficulty: float
    
@dataclass
class AdaptationResult:
    """Result of domain adaptation"""
    success: bool
    adapted_skills: Any
    confidence: float
    adaptation_loss: float
    safety_validated: bool
    warnings: List[str]
    adaptation_metadata: Dict[str, Any]

class DomainAdapter:
    """Domain adaptation system with safety validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adaptation_threshold = config.get('adaptation_threshold', 0.7)
        self.safety_margin = config.get('safety_margin', 0.2)
        self.known_domains = {}
        self.adaptation_history = []
        
        # Adaptation strategies
        self.strategies = {
            AdaptationStrategy.ADVERSARIAL: self._adversarial_adaptation,
            AdaptationStrategy.FEATURE_ALIGNMENT: self._feature_alignment_adaptation,
            AdaptationStrategy.DOMAIN_CONFUSION: self._domain_confusion_adaptation,
            AdaptationStrategy.GRADUAL_ADAPTATION: self._gradual_adaptation,
            AdaptationStrategy.MULTI_DOMAIN: self._multi_domain_adaptation
        }
        
    async def adapt(self, source_skills: Any, target_domain: str, 
                   strategy: AdaptationStrategy = None) -> AdaptationResult:
        """Adapt skills to target domain with safety validation"""
        
        logger.info(f"Starting domain adaptation to: {target_domain}")
        
        # Get or create target domain profile
        target_profile = await self._get_domain_profile(target_domain)
        
        # Analyze adaptation requirements
        adaptation_analysis = await self._analyze_adaptation_requirements(
            source_skills, target_profile
        )
        
        # Select optimal strategy if not provided
        if strategy is None:
            strategy = await self._select_adaptation_strategy(adaptation_analysis)
            
        # Pre-adaptation safety check
        safety_check = await self._pre_adaptation_safety_check(
            source_skills, target_profile
        )
        
        if not safety_check.is_safe:
            return AdaptationResult(
                success=False,
                adapted_skills=None,
                confidence=0.0,
                adaptation_loss=1.0,
                safety_validated=False,
                warnings=[f"Pre-adaptation safety check failed: {safety_check.reason}"],
                adaptation_metadata={}
            )
        
        try:
            # Execute adaptation strategy
            adaptation_func = self.strategies[strategy]
            result = await adaptation_func(source_skills, target_profile)
            
            # Post-adaptation validation
            validation_result = await self._validate_adaptation(result, target_profile)
            
            # Record adaptation
            await self._record_adaptation(source_skills, target_domain, strategy, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Domain adaptation failed: {e}")
            return AdaptationResult(
                success=False,
                adapted_skills=None,
                confidence=0.0,
                adaptation_loss=1.0,
                safety_validated=False,
                warnings=[f"Adaptation execution failed: {str(e)}"],
                adaptation_metadata={}
            )
    
    async def _adversarial_adaptation(self, source_skills: Any, 
                                    target_profile: DomainProfile) -> AdaptationResult:
        """Adversarial domain adaptation using domain discriminator"""
        
        # Initialize adversarial adaptation components
        feature_extractor = await self._initialize_feature_extractor(source_skills)
        domain_discriminator = await self._initialize_domain_discriminator(target_profile)
        
        # Adversarial training loop
        adaptation_loss = 1.0
        epochs = self.config.get('adversarial_epochs', 100)
        
        for epoch in range(epochs):
            # Extract features
            features = await self._extract_features(source_skills, feature_extractor)
            
            # Train discriminator to distinguish domains
            discriminator_loss = await self._train_discriminator(
                features, target_profile, domain_discriminator
            )
            
            # Train feature extractor to confuse discriminator
            extractor_loss = await self._train_feature_extractor(
                features, domain_discriminator, feature_extractor
            )
            
            # Calculate adaptation loss
            adaptation_loss = discriminator_loss + extractor_loss
            
            # Early stopping if adaptation converges
            if adaptation_loss < 0.1:
                break
                
            # Safety monitoring during training
            if epoch % 10 == 0:
                safety_status = await self._monitor_adaptation_safety(features, target_profile)
                if not safety_status.is_safe:
                    return AdaptationResult(
                        success=False,
                        adapted_skills=None,
                        confidence=0.0,
                        adaptation_loss=adaptation_loss,
                        safety_validated=False,
                        warnings=[f"Safety violation during adversarial training: {safety_status.reason}"],
                        adaptation_metadata={"epoch": epoch}
                    )
        
        # Extract adapted skills
        adapted_skills = await self._extract_adapted_skills(feature_extractor, target_profile)
        
        # Calculate confidence
        confidence = max(0.0, 1.0 - adaptation_loss)
        
        return AdaptationResult(
            success=True,
            adapted_skills=adapted_skills,
            confidence=confidence,
            adaptation_loss=adaptation_loss,
            safety_validated=True,
            warnings=[],
            adaptation_metadata={
                "strategy": "adversarial",
                "epochs_trained": epoch + 1,
                "final_loss": adaptation_loss
            }
        )
    
    async def _feature_alignment_adaptation(self, source_skills: Any,
                                          target_profile: DomainProfile) -> AdaptationResult:
        """Feature alignment based domain adaptation"""
        
        # Extract source domain features
        source_features = await self._extract_domain_features(source_skills)
        
        # Get target domain feature distribution
        target_feature_distribution = await self._get_target_feature_distribution(target_profile)
        
        # Align features using statistical matching
        aligned_features = await self._align_features(
            source_features, target_feature_distribution
        )
        
        # Calculate alignment quality
        alignment_score = await self._calculate_alignment_score(
            aligned_features, target_feature_distribution
        )
        
        # Reconstruct skills with aligned features
        adapted_skills = await self._reconstruct_skills(aligned_features, target_profile)
        
        return AdaptationResult(
            success=alignment_score > self.adaptation_threshold,
            adapted_skills=adapted_skills,
            confidence=alignment_score,
            adaptation_loss=1.0 - alignment_score,
            safety_validated=True,
            warnings=[],
            adaptation_metadata={
                "strategy": "feature_alignment",
                "alignment_score": alignment_score
            }
        )
    
    async def _domain_confusion_adaptation(self, source_skills: Any,
                                         target_profile: DomainProfile) -> AdaptationResult:
        """Domain confusion based adaptation"""
        
        # Create domain-invariant representations
        invariant_features = await self._create_domain_invariant_features(source_skills)
        
        # Apply domain confusion loss
        confusion_loss = await self._apply_domain_confusion(invariant_features, target_profile)
        
        # Adapt skills using confused features
        adapted_skills = await self._adapt_with_confused_features(
            invariant_features, target_profile
        )
        
        # Validate domain confusion effectiveness
        confusion_effectiveness = await self._validate_domain_confusion(
            adapted_skills, target_profile
        )
        
        return AdaptationResult(
            success=confusion_effectiveness > self.adaptation_threshold,
            adapted_skills=adapted_skills,
            confidence=confusion_effectiveness,
            adaptation_loss=confusion_loss,
            safety_validated=True,
            warnings=[],
            adaptation_metadata={
                "strategy": "domain_confusion",
                "confusion_loss": confusion_loss,
                "effectiveness": confusion_effectiveness
            }
        )
    
    async def _gradual_adaptation(self, source_skills: Any,
                                target_profile: DomainProfile) -> AdaptationResult:
        """Gradual adaptation with incremental domain shifts"""
        
        # Create adaptation path
        adaptation_path = await self._create_adaptation_path(source_skills, target_profile)
        
        adapted_skills = source_skills
        total_adaptation_loss = 0.0
        
        # Gradual adaptation along path
        for step, intermediate_domain in enumerate(adaptation_path):
            step_result = await self._adapt_single_step(adapted_skills, intermediate_domain)
            
            if not step_result.success:
                return AdaptationResult(
                    success=False,
                    adapted_skills=None,
                    confidence=0.0,
                    adaptation_loss=1.0,
                    safety_validated=False,
                    warnings=[f"Gradual adaptation failed at step {step}"],
                    adaptation_metadata={"failed_step": step}
                )
            
            adapted_skills = step_result.adapted_skills
            total_adaptation_loss += step_result.adaptation_loss
            
            # Safety check at each step
            safety_status = await self._check_step_safety(adapted_skills, intermediate_domain)
            if not safety_status.is_safe:
                return AdaptationResult(
                    success=False,
                    adapted_skills=None,
                    confidence=0.0,
                    adaptation_loss=total_adaptation_loss,
                    safety_validated=False,
                    warnings=[f"Safety violation at step {step}: {safety_status.reason}"],
                    adaptation_metadata={"failed_step": step}
                )
        
        # Final validation
        final_confidence = max(0.0, 1.0 - (total_adaptation_loss / len(adaptation_path)))
        
        return AdaptationResult(
            success=True,
            adapted_skills=adapted_skills,
            confidence=final_confidence,
            adaptation_loss=total_adaptation_loss,
            safety_validated=True,
            warnings=[],
            adaptation_metadata={
                "strategy": "gradual_adaptation",
                "adaptation_steps": len(adaptation_path),
                "total_loss": total_adaptation_loss
            }
        )
    
    async def _multi_domain_adaptation(self, source_skills: Any,
                                     target_profile: DomainProfile) -> AdaptationResult:
        """Multi-domain adaptation using multiple source domains"""
        
        # Get related domains for multi-domain learning
        related_domains = await self._find_related_domains(target_profile)
        
        if not related_domains:
            # Fall back to single domain adaptation
            return await self._feature_alignment_adaptation(source_skills, target_profile)
        
        # Collect skills from multiple domains
        multi_domain_skills = {"source": source_skills}
        for domain in related_domains:
            domain_skills = await self._get_domain_skills(domain)
            multi_domain_skills[domain] = domain_skills
        
        # Learn domain-shared representations
        shared_representations = await self._learn_shared_representations(multi_domain_skills)
        
        # Adapt shared representations to target domain
        adapted_skills = await self._adapt_shared_to_target(
            shared_representations, target_profile
        )
        
        # Validate multi-domain adaptation
        validation_score = await self._validate_multi_domain_adaptation(
            adapted_skills, target_profile
        )
        
        return AdaptationResult(
            success=validation_score > self.adaptation_threshold,
            adapted_skills=adapted_skills,
            confidence=validation_score,
            adaptation_loss=1.0 - validation_score,
            safety_validated=True,
            warnings=[],
            adaptation_metadata={
                "strategy": "multi_domain",
                "related_domains": related_domains,
                "validation_score": validation_score
            }
        )
    
    async def _get_domain_profile(self, domain_name: str) -> DomainProfile:
        """Get or create domain profile"""
        if domain_name in self.known_domains:
            return self.known_domains[domain_name]
        
        # Create new domain profile
        profile = DomainProfile(
            name=domain_name,
            features=await self._analyze_domain_features(domain_name),
            constraints=await self._identify_domain_constraints(domain_name),
            safety_requirements=await self._assess_domain_safety_requirements(domain_name),
            adaptation_difficulty=await self._estimate_adaptation_difficulty(domain_name)
        )
        
        self.known_domains[domain_name] = profile
        return profile
    
    async def _analyze_adaptation_requirements(self, source_skills: Any, 
                                             target_profile: DomainProfile) -> Dict[str, Any]:
        """Analyze what adaptations are needed"""
        return {
            "skill_complexity": await self._assess_skill_complexity(source_skills),
            "domain_gap": await self._calculate_domain_gap(source_skills, target_profile),
            "safety_considerations": await self._identify_safety_considerations(target_profile),
            "adaptation_feasibility": await self._assess_adaptation_feasibility(source_skills, target_profile)
        }
    
    async def _select_adaptation_strategy(self, analysis: Dict[str, Any]) -> AdaptationStrategy:
        """Select optimal adaptation strategy"""
        domain_gap = analysis.get("domain_gap", 0.5)
        skill_complexity = analysis.get("skill_complexity", 0.5)
        safety_level = analysis.get("safety_considerations", "standard")
        
        if safety_level == "critical":
            return AdaptationStrategy.GRADUAL_ADAPTATION
        elif domain_gap > 0.8:
            return AdaptationStrategy.MULTI_DOMAIN
        elif skill_complexity > 0.7:
            return AdaptationStrategy.ADVERSARIAL
        elif domain_gap > 0.6:
            return AdaptationStrategy.DOMAIN_CONFUSION
        else:
            return AdaptationStrategy.FEATURE_ALIGNMENT
    
    async def _pre_adaptation_safety_check(self, source_skills: Any, 
                                         target_profile: DomainProfile):
        """Perform safety check before adaptation"""
        class SafetyCheck:
            def __init__(self, is_safe: bool = True, reason: str = ""):
                self.is_safe = is_safe
                self.reason = reason
        
        # Check for critical safety requirements
        if target_profile.safety_requirements == "critical":
            # Perform enhanced safety validation
            skill_safety = await self._validate_skill_safety(source_skills)
            if not skill_safety:
                return SafetyCheck(False, "Source skills contain unsafe elements")
        
        return SafetyCheck(True, "")
    
    async def _validate_adaptation(self, result: AdaptationResult, 
                                 target_profile: DomainProfile) -> AdaptationResult:
        """Validate adaptation result"""
        if not result.success:
            return result
        
        # Perform comprehensive validation
        performance_validation = await self._validate_adapted_performance(
            result.adapted_skills, target_profile
        )
        
        safety_validation = await self._validate_adapted_safety(
            result.adapted_skills, target_profile
        )
        
        # Update result with validation
        result.safety_validated = safety_validation
        if not performance_validation or not safety_validation:
            result.success = False
            result.warnings.append("Post-adaptation validation failed")
        
        return result
    
    async def _record_adaptation(self, source_skills: Any, target_domain: str,
                               strategy: AdaptationStrategy, result: AdaptationResult):
        """Record adaptation for audit trail"""
        adaptation_record = {
            "timestamp": datetime.now().isoformat(),
            "target_domain": target_domain,
            "strategy": strategy.value,
            "success": result.success,
            "confidence": result.confidence,
            "adaptation_loss": result.adaptation_loss,
            "safety_validated": result.safety_validated,
            "warnings": result.warnings
        }
        
        self.adaptation_history.append(adaptation_record)
        logger.info(f"Adaptation recorded: {adaptation_record}")
    
    # Helper methods (mock implementations for brevity)
    async def _initialize_feature_extractor(self, source_skills): return {}
    async def _initialize_domain_discriminator(self, target_profile): return {}
    async def _extract_features(self, skills, extractor): return {}
    async def _train_discriminator(self, features, profile, discriminator): return 0.1
    async def _train_feature_extractor(self, features, discriminator, extractor): return 0.1
    async def _monitor_adaptation_safety(self, features, profile): 
        class SafetyStatus:
            def __init__(self): 
                self.is_safe = True
                self.reason = ""
        return SafetyStatus()
    async def _extract_adapted_skills(self, extractor, profile): return {}
    async def _extract_domain_features(self, skills): return {}
    async def _get_target_feature_distribution(self, profile): return {}
    async def _align_features(self, source_features, target_distribution): return {}
    async def _calculate_alignment_score(self, aligned_features, target_distribution): return 0.8
    async def _reconstruct_skills(self, features, profile): return {}
    async def _create_domain_invariant_features(self, skills): return {}
    async def _apply_domain_confusion(self, features, profile): return 0.1
    async def _adapt_with_confused_features(self, features, profile): return {}
    async def _validate_domain_confusion(self, skills, profile): return 0.8
    async def _create_adaptation_path(self, skills, profile): return ["intermediate_domain"]
    async def _adapt_single_step(self, skills, domain): 
        return AdaptationResult(True, {}, 0.8, 0.1, True, [], {})
    async def _check_step_safety(self, skills, domain):
        class SafetyStatus:
            def __init__(self): 
                self.is_safe = True
                self.reason = ""
        return SafetyStatus()
    async def _find_related_domains(self, profile): return ["related_domain_1"]
    async def _get_domain_skills(self, domain): return {}
    async def _learn_shared_representations(self, multi_domain_skills): return {}
    async def _adapt_shared_to_target(self, representations, profile): return {}
    async def _validate_multi_domain_adaptation(self, skills, profile): return 0.8
    async def _analyze_domain_features(self, domain_name): return {}
    async def _identify_domain_constraints(self, domain_name): return []
    async def _assess_domain_safety_requirements(self, domain_name): return "standard"
    async def _estimate_adaptation_difficulty(self, domain_name): return 0.5
    async def _assess_skill_complexity(self, skills): return 0.5
    async def _calculate_domain_gap(self, skills, profile): return 0.4
    async def _identify_safety_considerations(self, profile): return "standard"
    async def _assess_adaptation_feasibility(self, skills, profile): return 0.8
    async def _validate_skill_safety(self, skills): return True
    async def _validate_adapted_performance(self, skills, profile): return True
    async def _validate_adapted_safety(self, skills, profile): return True

    async def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get adaptation history for auditing"""
        return self.adaptation_history.copy()
    
    async def get_known_domains(self) -> Dict[str, DomainProfile]:
        """Get all known domain profiles"""
        return self.known_domains.copy()