"""
Swarm Ethics - Collective Behavior Validation System
===================================================

The Swarm Ethics module implements ethical guidelines and safety constraints
for collective robot behavior, ensuring that swarm decisions and actions
align with human values and safety requirements.

Features:
- Ethical principle validation
- Collective behavior assessment
- Safety constraint enforcement
- Human value alignment
- Ethical decision frameworks
- Harm prevention systems
- Transparency and accountability
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class EthicalPrinciple(Enum):
    """Core ethical principles for swarm behavior"""
    DO_NO_HARM = "do_no_harm"                       # Avoid causing harm
    BENEFICENCE = "beneficence"                     # Act for benefit of humans
    AUTONOMY = "autonomy"                           # Respect human autonomy
    JUSTICE = "justice"                             # Fair treatment and distribution
    TRANSPARENCY = "transparency"                   # Open and explainable actions
    ACCOUNTABILITY = "accountability"               # Clear responsibility chains
    PRIVACY = "privacy"                            # Protect personal information
    CONSENT = "consent"                            # Obtain informed consent
    PROPORTIONALITY = "proportionality"            # Response proportional to need
    SUSTAINABILITY = "sustainability"              # Environmental responsibility


class EthicalSeverity(Enum):
    """Severity levels for ethical violations"""
    MINOR = "minor"                                # Minor ethical concern
    MODERATE = "moderate"                          # Notable ethical issue
    MAJOR = "major"                               # Significant ethical violation
    CRITICAL = "critical"                         # Severe ethical breach
    CATASTROPHIC = "catastrophic"                 # Potentially catastrophic harm


class ActionCategory(Enum):
    """Categories of actions for ethical evaluation"""
    MOVEMENT = "movement"                         # Physical movement/positioning
    COMMUNICATION = "communication"              # Information sharing/messaging
    DECISION_MAKING = "decision_making"          # Collective decisions
    RESOURCE_USE = "resource_use"                # Resource allocation/usage
    HUMAN_INTERACTION = "human_interaction"      # Direct human contact
    ENVIRONMENTAL = "environmental"              # Environmental impact
    DATA_HANDLING = "data_handling"              # Data collection/processing
    COORDINATION = "coordination"                # Swarm coordination
    LEARNING = "learning"                        # Learning and adaptation


@dataclass
class EthicalRule:
    """A single ethical rule for behavior validation"""
    id: str
    name: str
    principle: EthicalPrinciple
    description: str
    category: ActionCategory
    severity: EthicalSeverity
    condition: str  # Condition that triggers evaluation
    validation_function: Callable[[Dict[str, Any]], bool]
    parameters: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    priority: int = 1
    created_by: str = "system"
    created_time: datetime = field(default_factory=datetime.now)


@dataclass
class EthicalViolation:
    """Record of an ethical violation"""
    id: str
    rule_id: str
    principle: EthicalPrinciple
    severity: EthicalSeverity
    action: Dict[str, Any]
    violation_details: str
    timestamp: datetime
    robots_involved: Set[str]
    corrective_action: Optional[str] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class EthicalProfile:
    """Ethical profile for a robot or the swarm"""
    id: str
    name: str
    ethical_weights: Dict[EthicalPrinciple, float]  # Importance weights
    violation_history: List[str]  # Violation IDs
    compliance_score: float = 1.0
    last_assessment: datetime = field(default_factory=datetime.now)
    ethical_preferences: Dict[str, Any] = field(default_factory=dict)


class SwarmEthics:
    """
    Collective behavior validation and ethics system
    
    Ensures that swarm actions comply with ethical principles and human values
    while maintaining transparency and accountability in decision-making.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Ethical framework
        self.ethical_rules: Dict[str, EthicalRule] = {}
        self.ethical_profiles: Dict[str, EthicalProfile] = {}
        self.violation_records: Dict[str, EthicalViolation] = {}
        
        # Validation system
        self.validation_enabled = True
        self.strict_mode = False  # If True, blocks actions on any violation
        self.human_override_allowed = True
        
        # Ethical weights (importance of each principle)
        self.principle_weights = {
            EthicalPrinciple.DO_NO_HARM: 1.0,
            EthicalPrinciple.BENEFICENCE: 0.8,
            EthicalPrinciple.AUTONOMY: 0.9,
            EthicalPrinciple.JUSTICE: 0.7,
            EthicalPrinciple.TRANSPARENCY: 0.6,
            EthicalPrinciple.ACCOUNTABILITY: 0.8,
            EthicalPrinciple.PRIVACY: 0.8,
            EthicalPrinciple.CONSENT: 0.9,
            EthicalPrinciple.PROPORTIONALITY: 0.7,
            EthicalPrinciple.SUSTAINABILITY: 0.6
        }
        
        # Action monitoring
        self.action_history: deque = deque(maxlen=10000)
        self.pending_validations: deque = deque()
        
        # Background tasks
        self.validation_task = None
        self.monitoring_task = None
        self.compliance_task = None
        
        # Metrics
        self.total_validations = 0
        self.total_violations = 0
        self.average_compliance_score = 1.0
        self.recent_violations = []
        
        # Human authority override
        self.human_oversight_enabled = True
        self.override_history = []
        
        logger.info("Swarm Ethics system initialized")
    
    async def initialize(self) -> bool:
        """Initialize the swarm ethics system"""
        try:
            # Initialize default ethical rules
            await self._initialize_default_rules()
            
            # Create default ethical profiles
            await self._create_default_profiles()
            
            # Start background validation tasks
            self.validation_task = asyncio.create_task(self._process_validations())
            self.monitoring_task = asyncio.create_task(self._monitor_compliance())
            self.compliance_task = asyncio.create_task(self._assess_compliance())
            
            logger.info("Swarm Ethics system initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Swarm Ethics initialization failed: {e}")
            return False
    
    async def validate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an action against ethical principles"""
        try:
            if not self.validation_enabled:
                return {"approved": True, "reason": "validation_disabled"}
            
            # Record action for monitoring
            self.action_history.append({
                "action": action,
                "timestamp": datetime.now(),
                "validation_id": f"val_{datetime.now().timestamp()}"
            })
            
            # Check each applicable ethical rule
            violations = []
            warnings = []
            
            action_type = action.get("type", "unknown")
            action_category = self._classify_action(action)
            
            for rule_id, rule in self.ethical_rules.items():
                if not rule.active:
                    continue
                
                # Check if rule applies to this action
                if rule.category != action_category and rule.category != ActionCategory.COORDINATION:
                    continue
                
                # Evaluate the rule
                try:
                    violation_result = await self._evaluate_rule(rule, action)
                    if violation_result:
                        violation_details = violation_result.get("details", "Rule violation")
                        
                        if rule.severity in [EthicalSeverity.CRITICAL, EthicalSeverity.CATASTROPHIC]:
                            violations.append({
                                "rule_id": rule_id,
                                "principle": rule.principle.value,
                                "severity": rule.severity.value,
                                "details": violation_details
                            })
                        else:
                            warnings.append({
                                "rule_id": rule_id,
                                "principle": rule.principle.value,
                                "severity": rule.severity.value,
                                "details": violation_details
                            })
                            
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule_id}: {e}")
            
            # Record violations
            for violation in violations:
                await self._record_violation(violation, action)
            
            # Determine approval
            approved = len(violations) == 0
            
            if not approved and not self.strict_mode:
                # In non-strict mode, major violations are warnings
                major_violations = [v for v in violations if v["severity"] in ["critical", "catastrophic"]]
                approved = len(major_violations) == 0
                
                if approved:
                    warnings.extend([v for v in violations if v["severity"] not in ["critical", "catastrophic"]])
                    violations = major_violations
            
            # Update metrics
            self.total_validations += 1
            if violations:
                self.total_violations += len(violations)
            
            result = {
                "approved": approved,
                "violations": violations,
                "warnings": warnings,
                "validation_id": f"val_{datetime.now().timestamp()}",
                "timestamp": datetime.now().isoformat()
            }
            
            if not approved:
                result["reason"] = "ethical_violations"
                result["recommended_action"] = await self._suggest_alternative_action(action, violations)
            
            # Log critical violations
            if violations:
                severity_levels = [v["severity"] for v in violations]
                if "critical" in severity_levels or "catastrophic" in severity_levels:
                    logger.critical(f"Critical ethical violations in action {action_type}: {violations}")
                else:
                    logger.warning(f"Ethical violations in action {action_type}: {violations}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to validate action: {e}")
            return {"approved": False, "reason": "validation_error", "error": str(e)}
    
    async def add_ethical_rule(self, name: str, principle: EthicalPrinciple,
                             description: str, category: ActionCategory,
                             severity: EthicalSeverity, condition: str,
                             validation_function: Callable[[Dict[str, Any]], bool],
                             parameters: Dict[str, Any] = None,
                             created_by: str = "user") -> str:
        """Add a new ethical rule to the system"""
        try:
            rule_id = f"rule_{hashlib.md5(name.encode()).hexdigest()[:8]}_{datetime.now().timestamp()}"
            
            rule = EthicalRule(
                id=rule_id,
                name=name,
                principle=principle,
                description=description,
                category=category,
                severity=severity,
                condition=condition,
                validation_function=validation_function,
                parameters=parameters or {},
                created_by=created_by
            )
            
            self.ethical_rules[rule_id] = rule
            
            logger.info(f"Ethical rule added: {name} ({rule_id})")
            return rule_id
            
        except Exception as e:
            logger.error(f"Failed to add ethical rule: {e}")
            return ""
    
    async def create_ethical_profile(self, robot_id: str, 
                                   ethical_weights: Dict[EthicalPrinciple, float] = None,
                                   preferences: Dict[str, Any] = None) -> bool:
        """Create an ethical profile for a robot"""
        try:
            # Use default weights if not provided
            weights = ethical_weights or self.principle_weights.copy()
            
            profile = EthicalProfile(
                id=robot_id,
                name=f"Profile for {robot_id}",
                ethical_weights=weights,
                violation_history=[],
                ethical_preferences=preferences or {}
            )
            
            self.ethical_profiles[robot_id] = profile
            
            logger.info(f"Ethical profile created for robot {robot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create ethical profile: {e}")
            return False
    
    async def human_override_decision(self, action: Dict[str, Any], 
                                    override_reason: str,
                                    authority_id: str = "human") -> Dict[str, Any]:
        """Allow human authority to override ethical decision"""
        try:
            if not self.human_oversight_enabled:
                return {"approved": False, "reason": "human_override_disabled"}
            
            # Record the override
            override_record = {
                "action": action,
                "override_reason": override_reason,
                "authority_id": authority_id,
                "timestamp": datetime.now().isoformat(),
                "original_validation": await self.validate_action(action)
            }
            
            self.override_history.append(override_record)
            
            # Log the override
            logger.warning(f"Human override applied by {authority_id}: {override_reason}")
            
            return {
                "approved": True,
                "reason": "human_override",
                "override_record": override_record,
                "authority": authority_id
            }
            
        except Exception as e:
            logger.error(f"Failed to process human override: {e}")
            return {"approved": False, "reason": "override_error", "error": str(e)}
    
    async def get_ethical_assessment(self, robot_id: str = None) -> Dict[str, Any]:
        """Get ethical assessment for a robot or the entire swarm"""
        try:
            if robot_id:
                return await self._assess_robot_ethics(robot_id)
            else:
                return await self._assess_swarm_ethics()
                
        except Exception as e:
            logger.error(f"Failed to get ethical assessment: {e}")
            return {}
    
    async def get_recent_violations(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent ethical violations"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_violations = []
            for violation_id, violation in self.violation_records.items():
                if violation.timestamp >= cutoff_time:
                    violation_info = {
                        "id": violation_id,
                        "rule_id": violation.rule_id,
                        "principle": violation.principle.value,
                        "severity": violation.severity.value,
                        "details": violation.violation_details,
                        "timestamp": violation.timestamp.isoformat(),
                        "robots_involved": list(violation.robots_involved),
                        "resolved": violation.resolved
                    }
                    recent_violations.append(violation_info)
            
            # Sort by timestamp, most recent first
            recent_violations.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return recent_violations
            
        except Exception as e:
            logger.error(f"Failed to get recent violations: {e}")
            return []
    
    async def resolve_violation(self, violation_id: str, 
                              corrective_action: str,
                              resolver_id: str = "system") -> bool:
        """Mark a violation as resolved with corrective action"""
        try:
            if violation_id not in self.violation_records:
                logger.error(f"Violation {violation_id} not found")
                return False
            
            violation = self.violation_records[violation_id]
            violation.resolved = True
            violation.resolution_time = datetime.now()
            violation.corrective_action = corrective_action
            
            logger.info(f"Violation {violation_id} resolved by {resolver_id}: {corrective_action}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve violation: {e}")
            return False
    
    async def update_principle_weights(self, new_weights: Dict[EthicalPrinciple, float]) -> bool:
        """Update the importance weights of ethical principles"""
        try:
            # Validate weights
            for principle, weight in new_weights.items():
                if not 0.0 <= weight <= 1.0:
                    logger.error(f"Invalid weight {weight} for principle {principle.value}")
                    return False
            
            # Update weights
            self.principle_weights.update(new_weights)
            
            # Update all robot profiles with new weights
            for profile in self.ethical_profiles.values():
                profile.ethical_weights.update(new_weights)
            
            logger.info("Ethical principle weights updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update principle weights: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown swarm ethics system"""
        try:
            logger.info("Shutting down Swarm Ethics system")
            
            # Stop background tasks
            if self.validation_task:
                self.validation_task.cancel()
            if self.monitoring_task:
                self.monitoring_task.cancel()
            if self.compliance_task:
                self.compliance_task.cancel()
            
            # Save violation records and compliance data
            await self._save_ethics_data()
            
            logger.info("Swarm Ethics system shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Swarm Ethics shutdown failed: {e}")
            return False
    
    # Private helper methods
    
    async def _initialize_default_rules(self):
        """Initialize default ethical rules"""
        try:
            # Do No Harm rules
            await self._add_do_no_harm_rules()
            
            # Human autonomy rules
            await self._add_autonomy_rules()
            
            # Privacy and consent rules
            await self._add_privacy_rules()
            
            # Transparency and accountability rules
            await self._add_transparency_rules()
            
            # Environmental responsibility rules
            await self._add_sustainability_rules()
            
        except Exception as e:
            logger.error(f"Error initializing default rules: {e}")
    
    async def _add_do_no_harm_rules(self):
        """Add rules related to the 'do no harm' principle"""
        
        def no_human_harm_validator(action: Dict[str, Any]) -> bool:
            """Check if action could cause harm to humans"""
            # Check for dangerous parameters
            if "force" in action and action["force"] > 10.0:  # Arbitrary safety limit
                return {"violation": True, "details": "Excessive force parameter detected"}
            
            if "target" in action and action["target"] == "human":
                return {"violation": True, "details": "Direct targeting of humans not allowed"}
            
            if action.get("type") == "collision" or "collision" in str(action).lower():
                return {"violation": True, "details": "Collision action detected"}
            
            return None
        
        await self.add_ethical_rule(
            name="No Human Harm",
            principle=EthicalPrinciple.DO_NO_HARM,
            description="Prevent actions that could cause physical harm to humans",
            category=ActionCategory.HUMAN_INTERACTION,
            severity=EthicalSeverity.CRITICAL,
            condition="Any action involving human interaction",
            validation_function=no_human_harm_validator,
            created_by="system"
        )
        
        def no_property_damage_validator(action: Dict[str, Any]) -> bool:
            """Check if action could cause property damage"""
            dangerous_actions = ["destroy", "damage", "break", "crash"]
            action_str = str(action).lower()
            
            for dangerous_action in dangerous_actions:
                if dangerous_action in action_str:
                    return {"violation": True, "details": f"Potentially destructive action detected: {dangerous_action}"}
            
            return None
        
        await self.add_ethical_rule(
            name="No Property Damage",
            principle=EthicalPrinciple.DO_NO_HARM,
            description="Prevent actions that could cause property damage",
            category=ActionCategory.MOVEMENT,
            severity=EthicalSeverity.MAJOR,
            condition="Any movement or manipulation action",
            validation_function=no_property_damage_validator,
            created_by="system"
        )
    
    async def _add_autonomy_rules(self):
        """Add rules related to human autonomy"""
        
        def respect_human_choice_validator(action: Dict[str, Any]) -> bool:
            """Check if action respects human choice"""
            if action.get("type") == "override_human_decision":
                return {"violation": True, "details": "Attempting to override human decision"}
            
            if "force_human" in str(action).lower():
                return {"violation": True, "details": "Attempting to force human action"}
            
            return None
        
        await self.add_ethical_rule(
            name="Respect Human Choice",
            principle=EthicalPrinciple.AUTONOMY,
            description="Respect human decision-making autonomy",
            category=ActionCategory.HUMAN_INTERACTION,
            severity=EthicalSeverity.MAJOR,
            condition="Any action affecting human choices",
            validation_function=respect_human_choice_validator,
            created_by="system"
        )
    
    async def _add_privacy_rules(self):
        """Add rules related to privacy protection"""
        
        def data_privacy_validator(action: Dict[str, Any]) -> bool:
            """Check if action respects data privacy"""
            if action.get("type") == "collect_personal_data" and not action.get("consent", False):
                return {"violation": True, "details": "Personal data collection without consent"}
            
            if "share_private_data" in str(action).lower():
                return {"violation": True, "details": "Sharing private data detected"}
            
            return None
        
        await self.add_ethical_rule(
            name="Data Privacy Protection",
            principle=EthicalPrinciple.PRIVACY,
            description="Protect personal and private data",
            category=ActionCategory.DATA_HANDLING,
            severity=EthicalSeverity.MAJOR,
            condition="Any data handling action",
            validation_function=data_privacy_validator,
            created_by="system"
        )
    
    async def _add_transparency_rules(self):
        """Add rules related to transparency"""
        
        def transparency_validator(action: Dict[str, Any]) -> bool:
            """Check if action maintains transparency"""
            if action.get("hidden", False) and action.get("category") == "human_interaction":
                return {"violation": True, "details": "Hidden action in human interaction"}
            
            if "secret" in str(action).lower() and "human" in str(action).lower():
                return {"violation": True, "details": "Secret action involving humans"}
            
            return None
        
        await self.add_ethical_rule(
            name="Maintain Transparency",
            principle=EthicalPrinciple.TRANSPARENCY,
            description="Ensure actions are transparent and explainable",
            category=ActionCategory.HUMAN_INTERACTION,
            severity=EthicalSeverity.MODERATE,
            condition="Any action involving humans",
            validation_function=transparency_validator,
            created_by="system"
        )
    
    async def _add_sustainability_rules(self):
        """Add rules related to environmental sustainability"""
        
        def environmental_impact_validator(action: Dict[str, Any]) -> bool:
            """Check environmental impact of action"""
            if action.get("energy_consumption", 0) > 1000:  # Arbitrary high consumption
                return {"violation": True, "details": "Excessive energy consumption"}
            
            if action.get("type") == "waste_generation" and action.get("amount", 0) > 10:
                return {"violation": True, "details": "Excessive waste generation"}
            
            return None
        
        await self.add_ethical_rule(
            name="Environmental Responsibility",
            principle=EthicalPrinciple.SUSTAINABILITY,
            description="Minimize negative environmental impact",
            category=ActionCategory.ENVIRONMENTAL,
            severity=EthicalSeverity.MODERATE,
            condition="Any action with environmental impact",
            validation_function=environmental_impact_validator,
            created_by="system"
        )
    
    async def _create_default_profiles(self):
        """Create default ethical profiles"""
        try:
            # Create system profile
            await self.create_ethical_profile("system", self.principle_weights.copy())
            
            # Create swarm profile
            await self.create_ethical_profile("swarm", self.principle_weights.copy())
            
        except Exception as e:
            logger.error(f"Error creating default profiles: {e}")
    
    def _classify_action(self, action: Dict[str, Any]) -> ActionCategory:
        """Classify an action into an ethical category"""
        action_type = action.get("type", "").lower()
        action_str = str(action).lower()
        
        # Classification based on action type and content
        if any(word in action_str for word in ["move", "position", "navigate", "location"]):
            return ActionCategory.MOVEMENT
        elif any(word in action_str for word in ["message", "communicate", "send", "broadcast"]):
            return ActionCategory.COMMUNICATION
        elif any(word in action_str for word in ["decide", "choose", "consensus", "vote"]):
            return ActionCategory.DECISION_MAKING
        elif any(word in action_str for word in ["human", "person", "user", "interaction"]):
            return ActionCategory.HUMAN_INTERACTION
        elif any(word in action_str for word in ["data", "collect", "store", "process"]):
            return ActionCategory.DATA_HANDLING
        elif any(word in action_str for word in ["resource", "energy", "battery", "allocation"]):
            return ActionCategory.RESOURCE_USE
        elif any(word in action_str for word in ["environment", "pollution", "waste", "impact"]):
            return ActionCategory.ENVIRONMENTAL
        elif any(word in action_str for word in ["learn", "adapt", "training", "model"]):
            return ActionCategory.LEARNING
        else:
            return ActionCategory.COORDINATION
    
    async def _evaluate_rule(self, rule: EthicalRule, action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate a single ethical rule against an action"""
        try:
            # Call the rule's validation function
            result = rule.validation_function(action)
            
            if result is None or (isinstance(result, bool) and not result):
                return None  # No violation
            
            if isinstance(result, dict) and result.get("violation", False):
                return result
            elif isinstance(result, bool) and result:
                return {"violation": True, "details": "Rule violation detected"}
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.id}: {e}")
            return None
    
    async def _record_violation(self, violation_data: Dict[str, Any], action: Dict[str, Any]):
        """Record an ethical violation"""
        try:
            violation_id = f"viol_{datetime.now().timestamp()}_{hashlib.md5(str(violation_data).encode()).hexdigest()[:8]}"
            
            # Extract involved robots
            involved_robots = set()
            if "robot_id" in action:
                involved_robots.add(action["robot_id"])
            if "robots" in action:
                involved_robots.update(action["robots"])
            if "involved_robots" in action:
                involved_robots.update(action["involved_robots"])
            
            violation = EthicalViolation(
                id=violation_id,
                rule_id=violation_data["rule_id"],
                principle=EthicalPrinciple(violation_data["principle"]),
                severity=EthicalSeverity(violation_data["severity"]),
                action=action,
                violation_details=violation_data["details"],
                timestamp=datetime.now(),
                robots_involved=involved_robots
            )
            
            self.violation_records[violation_id] = violation
            
            # Update robot profiles
            for robot_id in involved_robots:
                if robot_id in self.ethical_profiles:
                    self.ethical_profiles[robot_id].violation_history.append(violation_id)
            
            # Add to recent violations for quick access
            self.recent_violations.append(violation_id)
            if len(self.recent_violations) > 100:  # Keep last 100
                self.recent_violations = self.recent_violations[-100:]
            
        except Exception as e:
            logger.error(f"Error recording violation: {e}")
    
    async def _suggest_alternative_action(self, action: Dict[str, Any], 
                                        violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Suggest alternative actions that avoid ethical violations"""
        try:
            suggestions = []
            
            # Analyze violations and suggest modifications
            for violation in violations:
                if violation["principle"] == "do_no_harm":
                    suggestions.append("Consider reducing force or impact parameters")
                    suggestions.append("Implement additional safety checks")
                elif violation["principle"] == "privacy":
                    suggestions.append("Obtain explicit consent before data handling")
                    suggestions.append("Anonymize or encrypt sensitive data")
                elif violation["principle"] == "autonomy":
                    suggestions.append("Provide human with choice options")
                    suggestions.append("Request human confirmation before proceeding")
                elif violation["principle"] == "transparency":
                    suggestions.append("Provide clear explanation of action purpose")
                    suggestions.append("Make action visible and trackable")
            
            return {
                "alternatives": suggestions,
                "general_recommendation": "Review action parameters and add appropriate safeguards"
            }
            
        except Exception as e:
            logger.error(f"Error suggesting alternatives: {e}")
            return {"alternatives": [], "general_recommendation": "Manual review required"}
    
    async def _assess_robot_ethics(self, robot_id: str) -> Dict[str, Any]:
        """Assess ethical compliance of a specific robot"""
        try:
            if robot_id not in self.ethical_profiles:
                return {"error": "Robot profile not found"}
            
            profile = self.ethical_profiles[robot_id]
            
            # Calculate compliance metrics
            total_violations = len(profile.violation_history)
            recent_violations = len([v_id for v_id in profile.violation_history
                                   if v_id in self.violation_records and
                                   (datetime.now() - self.violation_records[v_id].timestamp).days <= 7])
            
            # Compliance score calculation
            if total_violations == 0:
                compliance_score = 1.0
            else:
                # Weighted by severity and recency
                severity_weights = {
                    EthicalSeverity.MINOR: 0.1,
                    EthicalSeverity.MODERATE: 0.3,
                    EthicalSeverity.MAJOR: 0.6,
                    EthicalSeverity.CRITICAL: 0.9,
                    EthicalSeverity.CATASTROPHIC: 1.0
                }
                
                total_severity_score = 0.0
                for v_id in profile.violation_history:
                    if v_id in self.violation_records:
                        violation = self.violation_records[v_id]
                        severity_score = severity_weights[violation.severity]
                        
                        # Decay factor for older violations
                        days_old = (datetime.now() - violation.timestamp).days
                        decay_factor = max(0.1, 1.0 - (days_old / 365))  # Decay over 1 year
                        
                        total_severity_score += severity_score * decay_factor
                
                compliance_score = max(0.0, 1.0 - (total_severity_score / 10.0))  # Scale to 0-1
            
            profile.compliance_score = compliance_score
            profile.last_assessment = datetime.now()
            
            return {
                "robot_id": robot_id,
                "compliance_score": compliance_score,
                "total_violations": total_violations,
                "recent_violations": recent_violations,
                "ethical_weights": {p.value: w for p, w in profile.ethical_weights.items()},
                "last_assessment": profile.last_assessment.isoformat(),
                "violation_breakdown": await self._get_violation_breakdown(profile.violation_history)
            }
            
        except Exception as e:
            logger.error(f"Error assessing robot ethics: {e}")
            return {"error": str(e)}
    
    async def _assess_swarm_ethics(self) -> Dict[str, Any]:
        """Assess ethical compliance of the entire swarm"""
        try:
            # Aggregate metrics across all robots
            total_robots = len(self.ethical_profiles)
            total_violations = len(self.violation_records)
            
            # Calculate average compliance score
            compliance_scores = [profile.compliance_score for profile in self.ethical_profiles.values()]
            avg_compliance = np.mean(compliance_scores) if compliance_scores else 1.0
            
            # Recent violation analysis
            recent_violations = await self.get_recent_violations(hours=168)  # Last week
            
            # Principle violation breakdown
            principle_violations = {}
            for violation in self.violation_records.values():
                principle = violation.principle.value
                principle_violations[principle] = principle_violations.get(principle, 0) + 1
            
            # Severity distribution
            severity_distribution = {}
            for violation in self.violation_records.values():
                severity = violation.severity.value
                severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
            
            return {
                "swarm_compliance_score": avg_compliance,
                "total_robots": total_robots,
                "total_violations": total_violations,
                "recent_violations": len(recent_violations),
                "principle_violations": principle_violations,
                "severity_distribution": severity_distribution,
                "ethical_health_score": self._calculate_ethical_health_score(),
                "active_rules": len([r for r in self.ethical_rules.values() if r.active]),
                "human_overrides": len(self.override_history),
                "assessment_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error assessing swarm ethics: {e}")
            return {"error": str(e)}
    
    def _calculate_ethical_health_score(self) -> float:
        """Calculate overall ethical health score"""
        try:
            factors = []
            
            # Compliance score factor
            if self.ethical_profiles:
                avg_compliance = np.mean([p.compliance_score for p in self.ethical_profiles.values()])
                factors.append(avg_compliance)
            
            # Recent violations factor
            recent_violations = [v for v in self.violation_records.values()
                               if (datetime.now() - v.timestamp).days <= 7]
            violation_factor = max(0.0, 1.0 - (len(recent_violations) / 10.0))  # Scale based on violations
            factors.append(violation_factor)
            
            # Rule coverage factor (having comprehensive rules is good)
            rule_coverage = min(1.0, len(self.ethical_rules) / 20.0)  # Ideal: 20+ rules
            factors.append(rule_coverage)
            
            # Resolution rate factor
            resolved_violations = len([v for v in self.violation_records.values() if v.resolved])
            total_violations = len(self.violation_records)
            resolution_rate = resolved_violations / max(1, total_violations)
            factors.append(resolution_rate)
            
            return np.mean(factors) if factors else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating ethical health score: {e}")
            return 0.5
    
    async def _get_violation_breakdown(self, violation_ids: List[str]) -> Dict[str, Any]:
        """Get detailed breakdown of violations"""
        try:
            breakdown = {
                "by_principle": {},
                "by_severity": {},
                "by_month": {},
                "resolved_count": 0
            }
            
            for v_id in violation_ids:
                if v_id in self.violation_records:
                    violation = self.violation_records[v_id]
                    
                    # By principle
                    principle = violation.principle.value
                    breakdown["by_principle"][principle] = breakdown["by_principle"].get(principle, 0) + 1
                    
                    # By severity
                    severity = violation.severity.value
                    breakdown["by_severity"][severity] = breakdown["by_severity"].get(severity, 0) + 1
                    
                    # By month
                    month_key = violation.timestamp.strftime("%Y-%m")
                    breakdown["by_month"][month_key] = breakdown["by_month"].get(month_key, 0) + 1
                    
                    # Resolution count
                    if violation.resolved:
                        breakdown["resolved_count"] += 1
            
            return breakdown
            
        except Exception as e:
            logger.error(f"Error getting violation breakdown: {e}")
            return {}
    
    # Background task methods
    
    async def _process_validations(self):
        """Background task to process pending validations"""
        while True:
            try:
                # Process any pending validations
                while self.pending_validations:
                    validation_task = self.pending_validations.popleft()
                    # Process validation task if needed
                
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing validations: {e}")
    
    async def _monitor_compliance(self):
        """Background task to monitor compliance trends"""
        while True:
            try:
                # Update compliance scores for all profiles
                for robot_id in self.ethical_profiles:
                    await self._assess_robot_ethics(robot_id)
                
                # Calculate swarm-wide metrics
                swarm_assessment = await self._assess_swarm_ethics()
                self.average_compliance_score = swarm_assessment.get("swarm_compliance_score", 1.0)
                
                await asyncio.sleep(300.0)  # Monitor every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring compliance: {e}")
    
    async def _assess_compliance(self):
        """Background task to assess and report compliance issues"""
        while True:
            try:
                # Check for compliance trends and patterns
                await self._analyze_violation_patterns()
                
                # Generate compliance reports if needed
                await self._generate_compliance_alerts()
                
                await asyncio.sleep(3600.0)  # Assess every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error assessing compliance: {e}")
    
    async def _analyze_violation_patterns(self):
        """Analyze patterns in ethical violations"""
        try:
            # Look for recurring patterns
            recent_violations = await self.get_recent_violations(hours=24)
            
            if len(recent_violations) > 5:  # Threshold for concern
                logger.warning(f"High violation rate detected: {len(recent_violations)} violations in 24 hours")
            
            # Check for repeated violations by same robots
            robot_violation_counts = {}
            for violation in recent_violations:
                for robot_id in violation["robots_involved"]:
                    robot_violation_counts[robot_id] = robot_violation_counts.get(robot_id, 0) + 1
            
            # Flag robots with repeated violations
            for robot_id, count in robot_violation_counts.items():
                if count >= 3:  # Multiple violations
                    logger.warning(f"Robot {robot_id} has {count} recent violations - may need intervention")
            
        except Exception as e:
            logger.error(f"Error analyzing violation patterns: {e}")
    
    async def _generate_compliance_alerts(self):
        """Generate alerts for compliance issues"""
        try:
            # Check for critical violations
            critical_violations = [v for v in self.violation_records.values()
                                 if v.severity in [EthicalSeverity.CRITICAL, EthicalSeverity.CATASTROPHIC]
                                 and not v.resolved
                                 and (datetime.now() - v.timestamp).hours < 24]
            
            if critical_violations:
                logger.critical(f"Unresolved critical violations: {len(critical_violations)}")
                # In production, this would trigger alerts to human supervisors
            
            # Check overall compliance health
            health_score = self._calculate_ethical_health_score()
            if health_score < 0.7:  # Below acceptable threshold
                logger.warning(f"Low ethical health score: {health_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error generating compliance alerts: {e}")
    
    async def _save_ethics_data(self):
        """Save ethical compliance data for persistence"""
        try:
            # In production, this would save to persistent storage
            logger.info(f"Saving ethics data with {len(self.violation_records)} violation records")
            
        except Exception as e:
            logger.error(f"Error saving ethics data: {e}")