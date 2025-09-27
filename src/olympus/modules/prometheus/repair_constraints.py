"""
Repair Constraints - Safety Validation and Repair Limitations

The Repair Constraints system enforces safety boundaries and limitations on
autonomous repair operations. It ensures that all repair actions comply with
ethical guidelines, safety requirements, and operational constraints.

Key Features:
- Safety constraint validation
- Ethical repair boundaries
- Risk assessment and approval requirements
- Resource limitation enforcement
- Temporal constraint management
- Human approval workflows
- Emergency override controls
- Comprehensive audit trails
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque


class ConstraintType(Enum):
    """Types of constraints that can be applied to repairs."""
    SAFETY_CONSTRAINT = "safety_constraint"
    ETHICAL_CONSTRAINT = "ethical_constraint"
    RESOURCE_CONSTRAINT = "resource_constraint"
    TEMPORAL_CONSTRAINT = "temporal_constraint"
    APPROVAL_CONSTRAINT = "approval_constraint"
    OPERATIONAL_CONSTRAINT = "operational_constraint"
    ENVIRONMENTAL_CONSTRAINT = "environmental_constraint"


class ConstraintSeverity(Enum):
    """Severity levels for constraint violations."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    BLOCKING = "blocking"


class ApprovalLevel(Enum):
    """Levels of human approval required."""
    NONE = "none"
    OPERATOR = "operator"
    SUPERVISOR = "supervisor"
    SAFETY_OFFICER = "safety_officer"
    ETHICS_BOARD = "ethics_board"
    EXECUTIVE = "executive"


class ConstraintStatus(Enum):
    """Status of constraint evaluation."""
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    PENDING_APPROVAL = "pending_approval"
    OVERRIDE_APPLIED = "override_applied"
    EXEMPTION_GRANTED = "exemption_granted"


@dataclass
class RepairConstraint:
    """Represents a constraint on repair operations."""
    constraint_id: str
    name: str
    description: str
    constraint_type: ConstraintType
    severity: ConstraintSeverity
    
    # Constraint definition
    conditions: Dict[str, Any] = field(default_factory=dict)
    limitations: Dict[str, Any] = field(default_factory=dict)
    exceptions: List[str] = field(default_factory=list)
    
    # Scope and applicability
    applicable_components: Set[str] = field(default_factory=set)
    applicable_repair_types: Set[str] = field(default_factory=set)
    applicable_environments: Set[str] = field(default_factory=set)
    
    # Approval requirements
    required_approval_level: ApprovalLevel = ApprovalLevel.NONE
    approval_timeout: float = 3600.0  # 1 hour default
    
    # Temporal constraints
    allowed_time_windows: List[Dict[str, Any]] = field(default_factory=list)
    prohibited_time_windows: List[Dict[str, Any]] = field(default_factory=list)
    maximum_frequency: Optional[Dict[str, Any]] = None  # e.g., {"count": 3, "period": 3600}
    
    # Override and exemption
    can_be_overridden: bool = False
    override_approval_level: ApprovalLevel = ApprovalLevel.EXECUTIVE
    exemption_conditions: List[str] = field(default_factory=list)
    
    # Metadata
    created_by: str = "system"
    created_time: float = field(default_factory=time.time)
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "constraint_id": self.constraint_id,
            "name": self.name,
            "description": self.description,
            "constraint_type": self.constraint_type.value,
            "severity": self.severity.value,
            "conditions": self.conditions,
            "limitations": self.limitations,
            "exceptions": self.exceptions,
            "applicable_components": list(self.applicable_components),
            "applicable_repair_types": list(self.applicable_repair_types),
            "applicable_environments": list(self.applicable_environments),
            "required_approval_level": self.required_approval_level.value,
            "approval_timeout": self.approval_timeout,
            "allowed_time_windows": self.allowed_time_windows,
            "prohibited_time_windows": self.prohibited_time_windows,
            "maximum_frequency": self.maximum_frequency,
            "can_be_overridden": self.can_be_overridden,
            "override_approval_level": self.override_approval_level.value,
            "exemption_conditions": self.exemption_conditions,
            "created_by": self.created_by,
            "created_time": self.created_time,
            "enabled": self.enabled
        }


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""
    violation_id: str
    constraint_id: str
    repair_request_id: str
    violation_type: str
    severity: ConstraintSeverity
    
    # Violation details
    violated_conditions: List[str] = field(default_factory=list)
    violation_message: str = ""
    suggested_actions: List[str] = field(default_factory=list)
    
    # Resolution
    can_be_resolved: bool = True
    resolution_options: List[str] = field(default_factory=list)
    requires_approval: bool = False
    required_approval_level: ApprovalLevel = ApprovalLevel.NONE
    
    # Metadata
    detected_time: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "violation_id": self.violation_id,
            "constraint_id": self.constraint_id,
            "repair_request_id": self.repair_request_id,
            "violation_type": self.violation_type,
            "severity": self.severity.value,
            "violated_conditions": self.violated_conditions,
            "violation_message": self.violation_message,
            "suggested_actions": self.suggested_actions,
            "can_be_resolved": self.can_be_resolved,
            "resolution_options": self.resolution_options,
            "requires_approval": self.requires_approval,
            "required_approval_level": self.required_approval_level.value,
            "detected_time": self.detected_time,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time
        }


@dataclass
class ApprovalRequest:
    """Represents a request for human approval."""
    request_id: str
    repair_request_id: str
    requested_approval_level: ApprovalLevel
    reason: str
    
    # Request details
    repair_description: str = ""
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    constraint_violations: List[str] = field(default_factory=list)
    alternative_options: List[str] = field(default_factory=list)
    
    # Approval status
    status: str = "pending"  # pending, approved, denied, expired
    approver: Optional[str] = None
    approval_time: Optional[float] = None
    approval_comments: str = ""
    
    # Timing
    request_time: float = field(default_factory=time.time)
    expiration_time: float = field(default_factory=lambda: time.time() + 3600)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "repair_request_id": self.repair_request_id,
            "requested_approval_level": self.requested_approval_level.value,
            "reason": self.reason,
            "repair_description": self.repair_description,
            "risk_assessment": self.risk_assessment,
            "constraint_violations": self.constraint_violations,
            "alternative_options": self.alternative_options,
            "status": self.status,
            "approver": self.approver,
            "approval_time": self.approval_time,
            "approval_comments": self.approval_comments,
            "request_time": self.request_time,
            "expiration_time": self.expiration_time
        }


class SafetyValidator:
    """Validates safety constraints for repair operations."""
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        self.safety_rules = self._initialize_safety_rules()
        
    def _initialize_safety_rules(self) -> Dict[str, Any]:
        """Initialize core safety rules."""
        return {
            "no_human_harm": {
                "description": "Repair must not cause harm to humans",
                "checks": ["check_human_safety", "verify_safe_procedures"],
                "severity": ConstraintSeverity.CRITICAL
            },
            "system_stability": {
                "description": "Repair must not destabilize critical systems",
                "checks": ["check_system_dependencies", "verify_rollback_capability"],
                "severity": ConstraintSeverity.ERROR
            },
            "data_integrity": {
                "description": "Repair must preserve data integrity",
                "checks": ["verify_backup_availability", "check_data_consistency"],
                "severity": ConstraintSeverity.ERROR
            },
            "resource_limits": {
                "description": "Repair must respect resource limitations",
                "checks": ["check_resource_availability", "verify_resource_limits"],
                "severity": ConstraintSeverity.WARNING
            },
            "operational_continuity": {
                "description": "Repair must maintain operational continuity",
                "checks": ["check_service_dependencies", "verify_redundancy"],
                "severity": ConstraintSeverity.WARNING
            }
        }
    
    async def validate_safety(self, repair_request: Dict[str, Any]) -> Tuple[bool, List[ConstraintViolation]]:
        """Validate safety constraints for a repair request."""
        violations = []
        
        try:
            for rule_name, rule_config in self.safety_rules.items():
                rule_violations = await self._check_safety_rule(rule_name, rule_config, repair_request)
                violations.extend(rule_violations)
            
            # Check if any critical violations exist
            critical_violations = [v for v in violations if v.severity == ConstraintSeverity.CRITICAL]
            is_safe = len(critical_violations) == 0
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "safety_validation_completed",
                    {
                        "repair_request_id": repair_request.get("request_id", "unknown"),
                        "is_safe": is_safe,
                        "violations_found": len(violations),
                        "critical_violations": len(critical_violations)
                    }
                )
            
            return is_safe, violations
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "safety_validation_failed",
                    {
                        "repair_request_id": repair_request.get("request_id", "unknown"),
                        "error": str(e)
                    }
                )
            raise
    
    async def _check_safety_rule(self, rule_name: str, rule_config: Dict[str, Any],
                                repair_request: Dict[str, Any]) -> List[ConstraintViolation]:
        """Check a specific safety rule."""
        violations = []
        
        try:
            for check_name in rule_config["checks"]:
                violation = await self._execute_safety_check(
                    rule_name, check_name, rule_config["severity"], repair_request
                )
                if violation:
                    violations.append(violation)
            
            return violations
            
        except Exception as e:
            # Create a violation for the failed check
            violation = ConstraintViolation(
                violation_id=f"safety_check_error_{rule_name}_{int(time.time())}",
                constraint_id=f"safety_rule_{rule_name}",
                repair_request_id=repair_request.get("request_id", "unknown"),
                violation_type="safety_check_error",
                severity=ConstraintSeverity.ERROR,
                violation_message=f"Safety check failed: {str(e)}",
                can_be_resolved=False
            )
            return [violation]
    
    async def _execute_safety_check(self, rule_name: str, check_name: str,
                                  severity: ConstraintSeverity,
                                  repair_request: Dict[str, Any]) -> Optional[ConstraintViolation]:
        """Execute a specific safety check."""
        try:
            # Simulate safety check execution
            await asyncio.sleep(0.1)
            
            if check_name == "check_human_safety":
                return await self._check_human_safety(rule_name, repair_request, severity)
            elif check_name == "verify_safe_procedures":
                return await self._verify_safe_procedures(rule_name, repair_request, severity)
            elif check_name == "check_system_dependencies":
                return await self._check_system_dependencies(rule_name, repair_request, severity)
            elif check_name == "verify_rollback_capability":
                return await self._verify_rollback_capability(rule_name, repair_request, severity)
            elif check_name == "verify_backup_availability":
                return await self._verify_backup_availability(rule_name, repair_request, severity)
            elif check_name == "check_data_consistency":
                return await self._check_data_consistency(rule_name, repair_request, severity)
            elif check_name == "check_resource_availability":
                return await self._check_resource_availability(rule_name, repair_request, severity)
            elif check_name == "verify_resource_limits":
                return await self._verify_resource_limits(rule_name, repair_request, severity)
            elif check_name == "check_service_dependencies":
                return await self._check_service_dependencies(rule_name, repair_request, severity)
            elif check_name == "verify_redundancy":
                return await self._verify_redundancy(rule_name, repair_request, severity)
            else:
                return None
                
        except Exception as e:
            violation = ConstraintViolation(
                violation_id=f"check_error_{check_name}_{int(time.time())}",
                constraint_id=f"safety_rule_{rule_name}",
                repair_request_id=repair_request.get("request_id", "unknown"),
                violation_type="check_execution_error",
                severity=severity,
                violation_message=f"Check '{check_name}' failed: {str(e)}",
                can_be_resolved=False
            )
            return violation
    
    async def _check_human_safety(self, rule_name: str, repair_request: Dict[str, Any],
                                severity: ConstraintSeverity) -> Optional[ConstraintViolation]:
        """Check if repair could harm humans."""
        try:
            component = repair_request.get("component", "")
            repair_type = repair_request.get("repair_type", "")
            
            # Check for dangerous combinations
            dangerous_components = ["safety_layer", "emergency_systems", "life_support"]
            high_risk_repairs = ["replace_component", "update_software", "modify_configuration"]
            
            if component in dangerous_components and repair_type in high_risk_repairs:
                return ConstraintViolation(
                    violation_id=f"human_safety_{int(time.time())}",
                    constraint_id=f"safety_rule_{rule_name}",
                    repair_request_id=repair_request.get("request_id", "unknown"),
                    violation_type="human_safety_risk",
                    severity=ConstraintSeverity.CRITICAL,
                    violation_message="Repair may affect human safety-critical systems",
                    suggested_actions=["Request human oversight", "Use safer repair methods"],
                    requires_approval=True,
                    required_approval_level=ApprovalLevel.SAFETY_OFFICER
                )
            
            return None
            
        except Exception:
            return None
    
    async def _verify_safe_procedures(self, rule_name: str, repair_request: Dict[str, Any],
                                    severity: ConstraintSeverity) -> Optional[ConstraintViolation]:
        """Verify that safe procedures are being followed."""
        try:
            # Check if repair follows safe procedures
            has_backup_plan = repair_request.get("has_backup_plan", False)
            has_rollback_plan = repair_request.get("has_rollback_plan", False)
            has_testing_plan = repair_request.get("has_testing_plan", False)
            
            missing_procedures = []
            if not has_backup_plan:
                missing_procedures.append("backup_plan")
            if not has_rollback_plan:
                missing_procedures.append("rollback_plan")
            if not has_testing_plan:
                missing_procedures.append("testing_plan")
            
            if missing_procedures:
                return ConstraintViolation(
                    violation_id=f"safe_procedures_{int(time.time())}",
                    constraint_id=f"safety_rule_{rule_name}",
                    repair_request_id=repair_request.get("request_id", "unknown"),
                    violation_type="missing_safe_procedures",
                    severity=severity,
                    violation_message=f"Missing safe procedures: {', '.join(missing_procedures)}",
                    suggested_actions=[f"Create {proc}" for proc in missing_procedures]
                )
            
            return None
            
        except Exception:
            return None
    
    async def _check_system_dependencies(self, rule_name: str, repair_request: Dict[str, Any],
                                       severity: ConstraintSeverity) -> Optional[ConstraintViolation]:
        """Check system dependencies for potential impacts."""
        try:
            component = repair_request.get("component", "")
            
            # Simulate dependency check
            critical_dependencies = ["database", "authentication", "monitoring"]
            
            if component in critical_dependencies:
                return ConstraintViolation(
                    violation_id=f"system_dependencies_{int(time.time())}",
                    constraint_id=f"safety_rule_{rule_name}",
                    repair_request_id=repair_request.get("request_id", "unknown"),
                    violation_type="critical_system_dependency",
                    severity=severity,
                    violation_message=f"Repair affects critical system dependency: {component}",
                    suggested_actions=["Coordinate with dependent systems", "Plan maintenance window"],
                    requires_approval=True,
                    required_approval_level=ApprovalLevel.SUPERVISOR
                )
            
            return None
            
        except Exception:
            return None
    
    async def _verify_rollback_capability(self, rule_name: str, repair_request: Dict[str, Any],
                                        severity: ConstraintSeverity) -> Optional[ConstraintViolation]:
        """Verify that rollback capability exists."""
        try:
            has_rollback = repair_request.get("has_rollback_plan", False)
            rollback_tested = repair_request.get("rollback_tested", False)
            
            if not has_rollback:
                return ConstraintViolation(
                    violation_id=f"no_rollback_{int(time.time())}",
                    constraint_id=f"safety_rule_{rule_name}",
                    repair_request_id=repair_request.get("request_id", "unknown"),
                    violation_type="no_rollback_capability",
                    severity=severity,
                    violation_message="No rollback capability available for repair",
                    suggested_actions=["Create rollback plan", "Test rollback procedure"]
                )
            
            if not rollback_tested:
                return ConstraintViolation(
                    violation_id=f"untested_rollback_{int(time.time())}",
                    constraint_id=f"safety_rule_{rule_name}",
                    repair_request_id=repair_request.get("request_id", "unknown"),
                    violation_type="untested_rollback",
                    severity=ConstraintSeverity.WARNING,
                    violation_message="Rollback procedure has not been tested",
                    suggested_actions=["Test rollback procedure in safe environment"]
                )
            
            return None
            
        except Exception:
            return None
    
    # Additional safety check methods would be implemented here...
    async def _verify_backup_availability(self, rule_name: str, repair_request: Dict[str, Any],
                                        severity: ConstraintSeverity) -> Optional[ConstraintViolation]:
        """Placeholder for backup availability check."""
        return None
    
    async def _check_data_consistency(self, rule_name: str, repair_request: Dict[str, Any],
                                    severity: ConstraintSeverity) -> Optional[ConstraintViolation]:
        """Placeholder for data consistency check."""
        return None
    
    async def _check_resource_availability(self, rule_name: str, repair_request: Dict[str, Any],
                                         severity: ConstraintSeverity) -> Optional[ConstraintViolation]:
        """Placeholder for resource availability check."""
        return None
    
    async def _verify_resource_limits(self, rule_name: str, repair_request: Dict[str, Any],
                                    severity: ConstraintSeverity) -> Optional[ConstraintViolation]:
        """Placeholder for resource limits check."""
        return None
    
    async def _check_service_dependencies(self, rule_name: str, repair_request: Dict[str, Any],
                                        severity: ConstraintSeverity) -> Optional[ConstraintViolation]:
        """Placeholder for service dependencies check."""
        return None
    
    async def _verify_redundancy(self, rule_name: str, repair_request: Dict[str, Any],
                               severity: ConstraintSeverity) -> Optional[ConstraintViolation]:
        """Placeholder for redundancy verification."""
        return None


class EthicalValidator:
    """Validates ethical constraints for repair operations."""
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        self.ethical_principles = self._initialize_ethical_principles()
    
    def _initialize_ethical_principles(self) -> Dict[str, Any]:
        """Initialize ethical principles for repair operations."""
        return {
            "do_no_harm": {
                "description": "Repairs must not cause harm",
                "weight": 1.0,
                "checks": ["assess_harm_potential", "verify_benefit_outweighs_risk"]
            },
            "respect_autonomy": {
                "description": "Respect human decision-making autonomy",
                "weight": 0.8,
                "checks": ["check_human_override_capability", "verify_transparency"]
            },
            "fairness": {
                "description": "Repairs should be fair and non-discriminatory",
                "weight": 0.7,
                "checks": ["check_equitable_access", "verify_non_discrimination"]
            },
            "transparency": {
                "description": "Repair actions should be transparent and explainable",
                "weight": 0.8,
                "checks": ["verify_auditability", "check_explainability"]
            },
            "proportionality": {
                "description": "Repair response should be proportional to the problem",
                "weight": 0.6,
                "checks": ["assess_proportionality", "verify_minimal_intervention"]
            }
        }
    
    async def validate_ethics(self, repair_request: Dict[str, Any]) -> Tuple[bool, List[ConstraintViolation]]:
        """Validate ethical constraints for a repair request."""
        violations = []
        ethical_score = 0.0
        total_weight = 0.0
        
        try:
            for principle_name, principle_config in self.ethical_principles.items():
                principle_violations, principle_score = await self._check_ethical_principle(
                    principle_name, principle_config, repair_request
                )
                
                violations.extend(principle_violations)
                ethical_score += principle_score * principle_config["weight"]
                total_weight += principle_config["weight"]
            
            # Calculate overall ethical score
            overall_score = ethical_score / total_weight if total_weight > 0 else 0.0
            
            # Determine if ethically acceptable (threshold: 0.7)
            is_ethical = overall_score >= 0.7 and len([v for v in violations 
                                                     if v.severity == ConstraintSeverity.CRITICAL]) == 0
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "ethical_validation_completed",
                    {
                        "repair_request_id": repair_request.get("request_id", "unknown"),
                        "ethical_score": overall_score,
                        "is_ethical": is_ethical,
                        "violations_found": len(violations)
                    }
                )
            
            return is_ethical, violations
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "ethical_validation_failed",
                    {
                        "repair_request_id": repair_request.get("request_id", "unknown"),
                        "error": str(e)
                    }
                )
            raise
    
    async def _check_ethical_principle(self, principle_name: str, principle_config: Dict[str, Any],
                                     repair_request: Dict[str, Any]) -> Tuple[List[ConstraintViolation], float]:
        """Check a specific ethical principle."""
        violations = []
        scores = []
        
        try:
            for check_name in principle_config["checks"]:
                violation, score = await self._execute_ethical_check(
                    principle_name, check_name, repair_request
                )
                
                if violation:
                    violations.append(violation)
                
                scores.append(score)
            
            # Average score for this principle
            principle_score = sum(scores) / len(scores) if scores else 0.0
            
            return violations, principle_score
            
        except Exception:
            return violations, 0.0
    
    async def _execute_ethical_check(self, principle_name: str, check_name: str,
                                   repair_request: Dict[str, Any]) -> Tuple[Optional[ConstraintViolation], float]:
        """Execute a specific ethical check."""
        try:
            # Simulate ethical check execution
            await asyncio.sleep(0.05)
            
            if check_name == "assess_harm_potential":
                return await self._assess_harm_potential(principle_name, repair_request)
            elif check_name == "verify_benefit_outweighs_risk":
                return await self._verify_benefit_outweighs_risk(principle_name, repair_request)
            elif check_name == "check_human_override_capability":
                return await self._check_human_override_capability(principle_name, repair_request)
            elif check_name == "verify_transparency":
                return await self._verify_transparency(principle_name, repair_request)
            elif check_name == "check_equitable_access":
                return await self._check_equitable_access(principle_name, repair_request)
            elif check_name == "verify_non_discrimination":
                return await self._verify_non_discrimination(principle_name, repair_request)
            elif check_name == "verify_auditability":
                return await self._verify_auditability(principle_name, repair_request)
            elif check_name == "check_explainability":
                return await self._check_explainability(principle_name, repair_request)
            elif check_name == "assess_proportionality":
                return await self._assess_proportionality(principle_name, repair_request)
            elif check_name == "verify_minimal_intervention":
                return await self._verify_minimal_intervention(principle_name, repair_request)
            else:
                return None, 0.5  # Default neutral score
                
        except Exception:
            return None, 0.0
    
    async def _assess_harm_potential(self, principle_name: str, repair_request: Dict[str, Any]) -> Tuple[Optional[ConstraintViolation], float]:
        """Assess potential for harm from repair."""
        try:
            # Simple harm assessment based on repair type and component
            repair_type = repair_request.get("repair_type", "")
            component = repair_request.get("component", "")
            
            high_harm_repairs = ["replace_component", "modify_configuration", "update_software"]
            critical_components = ["safety_layer", "asimov_kernel", "ethical_core"]
            
            harm_score = 1.0
            
            if repair_type in high_harm_repairs:
                harm_score -= 0.3
            
            if component in critical_components:
                harm_score -= 0.4
            
            if harm_score < 0.3:
                violation = ConstraintViolation(
                    violation_id=f"high_harm_potential_{int(time.time())}",
                    constraint_id=f"ethical_principle_{principle_name}",
                    repair_request_id=repair_request.get("request_id", "unknown"),
                    violation_type="high_harm_potential",
                    severity=ConstraintSeverity.CRITICAL,
                    violation_message="Repair has high potential for causing harm",
                    requires_approval=True,
                    required_approval_level=ApprovalLevel.ETHICS_BOARD
                )
                return violation, harm_score
            
            return None, harm_score
            
        except Exception:
            return None, 0.0
    
    async def _verify_benefit_outweighs_risk(self, principle_name: str, repair_request: Dict[str, Any]) -> Tuple[Optional[ConstraintViolation], float]:
        """Verify that benefit outweighs risk."""
        try:
            # Simple benefit/risk assessment
            repair_urgency = repair_request.get("urgency", "medium")
            repair_impact = repair_request.get("impact", "medium")
            risk_level = repair_request.get("risk_level", "medium")
            
            benefit_score = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
            risk_score = {"low": 0.1, "medium": 0.3, "high": 0.6, "critical": 0.9}
            
            benefit = (benefit_score.get(repair_urgency, 0.5) + 
                      benefit_score.get(repair_impact, 0.5)) / 2
            risk = risk_score.get(risk_level, 0.5)
            
            # Benefit should significantly outweigh risk
            ratio = benefit / max(risk, 0.1)
            
            if ratio < 1.5:  # Benefit should be at least 1.5x the risk
                violation = ConstraintViolation(
                    violation_id=f"insufficient_benefit_{int(time.time())}",
                    constraint_id=f"ethical_principle_{principle_name}",
                    repair_request_id=repair_request.get("request_id", "unknown"),
                    violation_type="insufficient_benefit_risk_ratio",
                    severity=ConstraintSeverity.WARNING,
                    violation_message="Repair benefit does not sufficiently outweigh risk",
                    suggested_actions=["Reassess repair necessity", "Consider alternative approaches"]
                )
                return violation, min(ratio / 2, 1.0)
            
            return None, min(ratio / 2, 1.0)
            
        except Exception:
            return None, 0.5
    
    # Additional ethical check methods would be implemented here...
    async def _check_human_override_capability(self, principle_name: str, repair_request: Dict[str, Any]) -> Tuple[Optional[ConstraintViolation], float]:
        """Check human override capability."""
        has_override = repair_request.get("human_override_available", True)
        return (None, 1.0) if has_override else (None, 0.3)
    
    async def _verify_transparency(self, principle_name: str, repair_request: Dict[str, Any]) -> Tuple[Optional[ConstraintViolation], float]:
        """Verify transparency of repair."""
        has_audit_trail = repair_request.get("audit_trail_enabled", True)
        return (None, 1.0) if has_audit_trail else (None, 0.2)
    
    async def _check_equitable_access(self, principle_name: str, repair_request: Dict[str, Any]) -> Tuple[Optional[ConstraintViolation], float]:
        """Check equitable access."""
        return None, 0.8  # Assume good unless proven otherwise
    
    async def _verify_non_discrimination(self, principle_name: str, repair_request: Dict[str, Any]) -> Tuple[Optional[ConstraintViolation], float]:
        """Verify non-discrimination."""
        return None, 0.9  # Assume good unless proven otherwise
    
    async def _verify_auditability(self, principle_name: str, repair_request: Dict[str, Any]) -> Tuple[Optional[ConstraintViolation], float]:
        """Verify auditability."""
        has_logging = repair_request.get("comprehensive_logging", True)
        return (None, 1.0) if has_logging else (None, 0.3)
    
    async def _check_explainability(self, principle_name: str, repair_request: Dict[str, Any]) -> Tuple[Optional[ConstraintViolation], float]:
        """Check explainability."""
        has_explanation = repair_request.get("repair_explanation_available", True)
        return (None, 1.0) if has_explanation else (None, 0.4)
    
    async def _assess_proportionality(self, principle_name: str, repair_request: Dict[str, Any]) -> Tuple[Optional[ConstraintViolation], float]:
        """Assess proportionality of repair."""
        problem_severity = repair_request.get("problem_severity", "medium")
        repair_scope = repair_request.get("repair_scope", "medium")
        
        severity_scale = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        scope_scale = {"minimal": 1, "medium": 2, "extensive": 3, "complete": 4}
        
        problem_level = severity_scale.get(problem_severity, 2)
        repair_level = scope_scale.get(repair_scope, 2)
        
        # Repair should not be much larger than problem
        if repair_level > problem_level + 1:
            violation = ConstraintViolation(
                violation_id=f"disproportionate_repair_{int(time.time())}",
                constraint_id=f"ethical_principle_{principle_name}",
                repair_request_id=repair_request.get("request_id", "unknown"),
                violation_type="disproportionate_repair",
                severity=ConstraintSeverity.WARNING,
                violation_message="Repair scope appears disproportionate to problem severity"
            )
            return violation, 0.4
        
        return None, 0.8
    
    async def _verify_minimal_intervention(self, principle_name: str, repair_request: Dict[str, Any]) -> Tuple[Optional[ConstraintViolation], float]:
        """Verify minimal intervention principle."""
        alternative_count = repair_request.get("alternatives_considered", 0)
        chosen_least_invasive = repair_request.get("least_invasive_chosen", False)
        
        if alternative_count == 0:
            return None, 0.5  # Neutral if no alternatives considered
        
        if not chosen_least_invasive:
            violation = ConstraintViolation(
                violation_id=f"not_minimal_intervention_{int(time.time())}",
                constraint_id=f"ethical_principle_{principle_name}",
                repair_request_id=repair_request.get("request_id", "unknown"),
                violation_type="not_minimal_intervention",
                severity=ConstraintSeverity.INFO,
                violation_message="More invasive repair chosen over less invasive alternatives"
            )
            return violation, 0.6
        
        return None, 1.0


class ApprovalManager:
    """Manages human approval workflows for repair operations."""
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        self.pending_approvals = {}
        self.approval_history = deque(maxlen=1000)
        self.approval_workflows = self._initialize_approval_workflows()
    
    def _initialize_approval_workflows(self) -> Dict[ApprovalLevel, Dict[str, Any]]:
        """Initialize approval workflow configurations."""
        return {
            ApprovalLevel.OPERATOR: {
                "timeout": 1800,  # 30 minutes
                "required_roles": ["system_operator"],
                "escalation_level": ApprovalLevel.SUPERVISOR
            },
            ApprovalLevel.SUPERVISOR: {
                "timeout": 3600,  # 1 hour
                "required_roles": ["supervisor", "senior_operator"],
                "escalation_level": ApprovalLevel.SAFETY_OFFICER
            },
            ApprovalLevel.SAFETY_OFFICER: {
                "timeout": 7200,  # 2 hours
                "required_roles": ["safety_officer", "senior_supervisor"],
                "escalation_level": ApprovalLevel.ETHICS_BOARD
            },
            ApprovalLevel.ETHICS_BOARD: {
                "timeout": 86400,  # 24 hours
                "required_roles": ["ethics_board_member", "ethics_chair"],
                "escalation_level": ApprovalLevel.EXECUTIVE
            },
            ApprovalLevel.EXECUTIVE: {
                "timeout": 172800,  # 48 hours
                "required_roles": ["executive", "ceo", "cto"],
                "escalation_level": None  # No further escalation
            }
        }
    
    async def request_approval(self, repair_request_id: str, required_level: ApprovalLevel,
                             reason: str, risk_assessment: Dict[str, Any] = None) -> str:
        """Request human approval for a repair operation."""
        try:
            request_id = f"approval_{repair_request_id}_{int(time.time())}"
            
            workflow_config = self.approval_workflows[required_level]
            expiration_time = time.time() + workflow_config["timeout"]
            
            approval_request = ApprovalRequest(
                request_id=request_id,
                repair_request_id=repair_request_id,
                requested_approval_level=required_level,
                reason=reason,
                risk_assessment=risk_assessment or {},
                expiration_time=expiration_time
            )
            
            self.pending_approvals[request_id] = approval_request
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "approval_requested",
                    {
                        "request_id": request_id,
                        "repair_request_id": repair_request_id,
                        "approval_level": required_level.value,
                        "reason": reason,
                        "expires_at": expiration_time
                    }
                )
            
            return request_id
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "approval_request_failed",
                    {
                        "repair_request_id": repair_request_id,
                        "error": str(e)
                    }
                )
            raise
    
    async def provide_approval(self, request_id: str, approver: str, approved: bool,
                             comments: str = "") -> bool:
        """Provide approval decision for a pending request."""
        try:
            if request_id not in self.pending_approvals:
                return False
            
            approval_request = self.pending_approvals[request_id]
            
            # Check if request has expired
            if time.time() > approval_request.expiration_time:
                approval_request.status = "expired"
                self.approval_history.append(approval_request)
                del self.pending_approvals[request_id]
                return False
            
            # Update approval request
            approval_request.approver = approver
            approval_request.approval_time = time.time()
            approval_request.approval_comments = comments
            approval_request.status = "approved" if approved else "denied"
            
            # Move to history
            self.approval_history.append(approval_request)
            del self.pending_approvals[request_id]
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "approval_provided",
                    {
                        "request_id": request_id,
                        "repair_request_id": approval_request.repair_request_id,
                        "approver": approver,
                        "approved": approved,
                        "approval_level": approval_request.requested_approval_level.value
                    }
                )
            
            return True
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "approval_provision_failed",
                    {"request_id": request_id, "error": str(e)}
                )
            return False
    
    async def check_approval_status(self, request_id: str) -> Optional[str]:
        """Check the status of an approval request."""
        if request_id in self.pending_approvals:
            approval_request = self.pending_approvals[request_id]
            
            # Check if expired
            if time.time() > approval_request.expiration_time:
                approval_request.status = "expired"
                self.approval_history.append(approval_request)
                del self.pending_approvals[request_id]
                return "expired"
            
            return approval_request.status
        
        # Check history
        for approval in self.approval_history:
            if approval.request_id == request_id:
                return approval.status
        
        return None
    
    async def get_pending_approvals(self, approval_level: Optional[ApprovalLevel] = None) -> List[ApprovalRequest]:
        """Get pending approval requests, optionally filtered by level."""
        pending = list(self.pending_approvals.values())
        
        if approval_level:
            pending = [req for req in pending if req.requested_approval_level == approval_level]
        
        return pending


class RepairConstraints:
    """
    Main repair constraints system that enforces safety and ethical boundaries.
    """
    
    def __init__(self, safety_layer=None, audit_system=None):
        self.safety_layer = safety_layer
        self.audit_system = audit_system
        
        # Initialize validators and managers
        self.safety_validator = SafetyValidator(audit_system)
        self.ethical_validator = EthicalValidator(audit_system)
        self.approval_manager = ApprovalManager(audit_system)
        
        # Constraint storage
        self.constraints = {}
        self.constraint_violations = {}
        self.constraint_history = deque(maxlen=1000)
        
        # System state
        self.is_active = True
        self.enforcement_enabled = True
        
        # Statistics
        self.total_validations = 0
        self.blocked_repairs = 0
        self.approved_repairs = 0
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default constraints
        asyncio.create_task(self._initialize_default_constraints())
    
    async def _initialize_default_constraints(self):
        """Initialize default system constraints."""
        try:
            # Core safety constraints
            await self.add_constraint(RepairConstraint(
                constraint_id="no_harm_to_humans",
                name="No Harm to Humans",
                description="Repairs must not cause harm to human beings",
                constraint_type=ConstraintType.SAFETY_CONSTRAINT,
                severity=ConstraintSeverity.CRITICAL,
                applicable_components={"*"},
                applicable_repair_types={"*"},
                required_approval_level=ApprovalLevel.SAFETY_OFFICER
            ))
            
            await self.add_constraint(RepairConstraint(
                constraint_id="preserve_data_integrity",
                name="Preserve Data Integrity",
                description="Repairs must preserve system data integrity",
                constraint_type=ConstraintType.SAFETY_CONSTRAINT,
                severity=ConstraintSeverity.ERROR,
                applicable_components={"database", "storage", "backup"},
                required_approval_level=ApprovalLevel.SUPERVISOR
            ))
            
            # Ethical constraints
            await self.add_constraint(RepairConstraint(
                constraint_id="human_oversight_critical",
                name="Human Oversight for Critical Repairs",
                description="Critical repairs require human oversight",
                constraint_type=ConstraintType.ETHICAL_CONSTRAINT,
                severity=ConstraintSeverity.ERROR,
                applicable_components={"safety_layer", "asimov_kernel", "ethical_core"},
                required_approval_level=ApprovalLevel.ETHICS_BOARD
            ))
            
            # Resource constraints
            await self.add_constraint(RepairConstraint(
                constraint_id="resource_limits",
                name="Resource Usage Limits",
                description="Repairs must respect system resource limits",
                constraint_type=ConstraintType.RESOURCE_CONSTRAINT,
                severity=ConstraintSeverity.WARNING,
                limitations={"max_cpu": 80, "max_memory": 85, "max_duration": 3600}
            ))
            
            # Temporal constraints
            await self.add_constraint(RepairConstraint(
                constraint_id="maintenance_windows",
                name="Maintenance Window Restrictions",
                description="High-impact repairs must occur during maintenance windows",
                constraint_type=ConstraintType.TEMPORAL_CONSTRAINT,
                severity=ConstraintSeverity.WARNING,
                allowed_time_windows=[
                    {"start": "02:00", "end": "06:00", "timezone": "UTC", "days": ["sunday"]}
                ]
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default constraints: {e}")
    
    async def add_constraint(self, constraint: RepairConstraint) -> bool:
        """Add a new repair constraint."""
        try:
            if constraint.constraint_id in self.constraints:
                self.logger.warning(f"Constraint {constraint.constraint_id} already exists")
                return False
            
            self.constraints[constraint.constraint_id] = constraint
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "repair_constraint_added",
                    {
                        "constraint_id": constraint.constraint_id,
                        "constraint_type": constraint.constraint_type.value,
                        "severity": constraint.severity.value,
                        "created_by": constraint.created_by
                    }
                )
            
            self.logger.info(f"Added repair constraint: {constraint.constraint_id}")
            return True
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "constraint_addition_failed",
                    {"constraint_id": constraint.constraint_id, "error": str(e)}
                )
            return False
    
    async def remove_constraint(self, constraint_id: str) -> bool:
        """Remove a repair constraint."""
        try:
            if constraint_id not in self.constraints:
                return False
            
            constraint = self.constraints[constraint_id]
            del self.constraints[constraint_id]
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "repair_constraint_removed",
                    {"constraint_id": constraint_id, "constraint_name": constraint.name}
                )
            
            return True
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "constraint_removal_failed",
                    {"constraint_id": constraint_id, "error": str(e)}
                )
            return False
    
    async def validate_repair_request(self, repair_request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a repair request against all constraints."""
        try:
            validation_id = f"validation_{repair_request.get('request_id', 'unknown')}_{int(time.time())}"
            self.total_validations += 1
            
            if not self.enforcement_enabled:
                return {
                    "validation_id": validation_id,
                    "allowed": True,
                    "reason": "Constraint enforcement is disabled",
                    "violations": [],
                    "required_approvals": []
                }
            
            violations = []
            required_approvals = []
            
            # Validate against safety constraints
            safety_valid, safety_violations = await self.safety_validator.validate_safety(repair_request)
            violations.extend(safety_violations)
            
            # Validate against ethical constraints
            ethical_valid, ethical_violations = await self.ethical_validator.validate_ethics(repair_request)
            violations.extend(ethical_violations)
            
            # Check custom constraints
            custom_violations = await self._check_custom_constraints(repair_request)
            violations.extend(custom_violations)
            
            # Determine required approvals
            critical_violations = [v for v in violations if v.severity == ConstraintSeverity.CRITICAL]
            error_violations = [v for v in violations if v.severity == ConstraintSeverity.ERROR]
            
            if critical_violations:
                required_approvals.append(ApprovalLevel.SAFETY_OFFICER)
            elif error_violations:
                required_approvals.append(ApprovalLevel.SUPERVISOR)
            
            # Add constraint-specific approval requirements
            for violation in violations:
                if violation.requires_approval and violation.required_approval_level not in required_approvals:
                    required_approvals.append(violation.required_approval_level)
            
            # Determine if repair is allowed
            blocking_violations = [v for v in violations if v.severity == ConstraintSeverity.BLOCKING]
            is_allowed = len(blocking_violations) == 0
            
            if not is_allowed:
                self.blocked_repairs += 1
            elif len(violations) == 0:
                self.approved_repairs += 1
            
            validation_result = {
                "validation_id": validation_id,
                "allowed": is_allowed,
                "reason": self._generate_validation_reason(violations, is_allowed),
                "violations": [v.to_dict() for v in violations],
                "required_approvals": [level.value for level in required_approvals],
                "safety_valid": safety_valid,
                "ethical_valid": ethical_valid,
                "total_violations": len(violations),
                "critical_violations": len(critical_violations),
                "blocking_violations": len(blocking_violations)
            }
            
            # Store validation history
            self.constraint_history.append({
                "validation_id": validation_id,
                "repair_request_id": repair_request.get("request_id", "unknown"),
                "timestamp": time.time(),
                "result": validation_result
            })
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "repair_request_validated",
                    {
                        "validation_id": validation_id,
                        "repair_request_id": repair_request.get("request_id", "unknown"),
                        "allowed": is_allowed,
                        "violations_count": len(violations),
                        "required_approvals": len(required_approvals)
                    }
                )
            
            return validation_result
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "repair_validation_failed",
                    {
                        "repair_request_id": repair_request.get("request_id", "unknown"),
                        "error": str(e)
                    }
                )
            raise
    
    async def _check_custom_constraints(self, repair_request: Dict[str, Any]) -> List[ConstraintViolation]:
        """Check custom user-defined constraints."""
        violations = []
        
        try:
            component = repair_request.get("component", "")
            repair_type = repair_request.get("repair_type", "")
            
            for constraint_id, constraint in self.constraints.items():
                if not constraint.enabled:
                    continue
                
                # Check if constraint applies to this repair
                if not self._constraint_applies(constraint, component, repair_type):
                    continue
                
                # Check constraint conditions
                violation = await self._evaluate_constraint(constraint, repair_request)
                if violation:
                    violations.append(violation)
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Error checking custom constraints: {e}")
            return violations
    
    def _constraint_applies(self, constraint: RepairConstraint, component: str, repair_type: str) -> bool:
        """Check if a constraint applies to the given repair."""
        try:
            # Check component applicability
            if (constraint.applicable_components and 
                "*" not in constraint.applicable_components and 
                component not in constraint.applicable_components):
                return False
            
            # Check repair type applicability
            if (constraint.applicable_repair_types and 
                "*" not in constraint.applicable_repair_types and 
                repair_type not in constraint.applicable_repair_types):
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _evaluate_constraint(self, constraint: RepairConstraint, 
                                 repair_request: Dict[str, Any]) -> Optional[ConstraintViolation]:
        """Evaluate a specific constraint against a repair request."""
        try:
            violated_conditions = []
            
            # Check resource limitations
            if constraint.constraint_type == ConstraintType.RESOURCE_CONSTRAINT:
                resource_violations = self._check_resource_limitations(constraint, repair_request)
                violated_conditions.extend(resource_violations)
            
            # Check temporal constraints
            elif constraint.constraint_type == ConstraintType.TEMPORAL_CONSTRAINT:
                temporal_violations = await self._check_temporal_constraints(constraint, repair_request)
                violated_conditions.extend(temporal_violations)
            
            # Check operational constraints
            elif constraint.constraint_type == ConstraintType.OPERATIONAL_CONSTRAINT:
                operational_violations = self._check_operational_constraints(constraint, repair_request)
                violated_conditions.extend(operational_violations)
            
            # If violations found, create violation record
            if violated_conditions:
                violation = ConstraintViolation(
                    violation_id=f"constraint_violation_{constraint.constraint_id}_{int(time.time())}",
                    constraint_id=constraint.constraint_id,
                    repair_request_id=repair_request.get("request_id", "unknown"),
                    violation_type=constraint.constraint_type.value,
                    severity=constraint.severity,
                    violated_conditions=violated_conditions,
                    violation_message=f"Constraint '{constraint.name}' violated",
                    requires_approval=(constraint.required_approval_level != ApprovalLevel.NONE),
                    required_approval_level=constraint.required_approval_level
                )
                return violation
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error evaluating constraint {constraint.constraint_id}: {e}")
            return None
    
    def _check_resource_limitations(self, constraint: RepairConstraint, 
                                  repair_request: Dict[str, Any]) -> List[str]:
        """Check resource limitation constraints."""
        violations = []
        limitations = constraint.limitations
        
        try:
            # Check CPU limitation
            if "max_cpu" in limitations:
                required_cpu = repair_request.get("estimated_cpu_usage", 0)
                if required_cpu > limitations["max_cpu"]:
                    violations.append(f"CPU usage {required_cpu}% exceeds limit {limitations['max_cpu']}%")
            
            # Check memory limitation
            if "max_memory" in limitations:
                required_memory = repair_request.get("estimated_memory_usage", 0)
                if required_memory > limitations["max_memory"]:
                    violations.append(f"Memory usage {required_memory}% exceeds limit {limitations['max_memory']}%")
            
            # Check duration limitation
            if "max_duration" in limitations:
                estimated_duration = repair_request.get("estimated_duration", 0)
                if estimated_duration > limitations["max_duration"]:
                    violations.append(f"Duration {estimated_duration}s exceeds limit {limitations['max_duration']}s")
            
            return violations
            
        except Exception:
            return violations
    
    async def _check_temporal_constraints(self, constraint: RepairConstraint, 
                                        repair_request: Dict[str, Any]) -> List[str]:
        """Check temporal constraints."""
        violations = []
        
        try:
            import datetime
            now = datetime.datetime.now()
            
            # Check allowed time windows
            if constraint.allowed_time_windows:
                in_allowed_window = False
                
                for window in constraint.allowed_time_windows:
                    if self._is_in_time_window(now, window):
                        in_allowed_window = True
                        break
                
                if not in_allowed_window:
                    violations.append("Repair requested outside allowed time windows")
            
            # Check prohibited time windows
            if constraint.prohibited_time_windows:
                for window in constraint.prohibited_time_windows:
                    if self._is_in_time_window(now, window):
                        violations.append("Repair requested during prohibited time window")
                        break
            
            # Check frequency limitations
            if constraint.maximum_frequency:
                frequency_violation = await self._check_frequency_limit(constraint, repair_request)
                if frequency_violation:
                    violations.append(frequency_violation)
            
            return violations
            
        except Exception:
            return violations
    
    def _is_in_time_window(self, current_time, window: Dict[str, Any]) -> bool:
        """Check if current time is within a specified time window."""
        try:
            import datetime
            
            start_time = window.get("start", "00:00")
            end_time = window.get("end", "23:59")
            allowed_days = window.get("days", [])
            
            # Check day of week
            if allowed_days:
                current_day = current_time.strftime("%A").lower()
                if current_day not in [day.lower() for day in allowed_days]:
                    return False
            
            # Check time range
            start_hour, start_minute = map(int, start_time.split(":"))
            end_hour, end_minute = map(int, end_time.split(":"))
            
            start_time_obj = current_time.replace(hour=start_hour, minute=start_minute, second=0)
            end_time_obj = current_time.replace(hour=end_hour, minute=end_minute, second=0)
            
            return start_time_obj <= current_time <= end_time_obj
            
        except Exception:
            return False
    
    async def _check_frequency_limit(self, constraint: RepairConstraint, 
                                   repair_request: Dict[str, Any]) -> Optional[str]:
        """Check frequency limitation constraints."""
        try:
            max_frequency = constraint.maximum_frequency
            if not max_frequency:
                return None
            
            max_count = max_frequency.get("count", 0)
            period_seconds = max_frequency.get("period", 3600)
            
            # Count recent repairs of the same type
            current_time = time.time()
            cutoff_time = current_time - period_seconds
            
            recent_repairs = 0
            for history_entry in self.constraint_history:
                if (history_entry["timestamp"] > cutoff_time and 
                    history_entry["result"]["allowed"]):
                    recent_repairs += 1
            
            if recent_repairs >= max_count:
                return f"Frequency limit exceeded: {recent_repairs} repairs in {period_seconds}s (max: {max_count})"
            
            return None
            
        except Exception:
            return None
    
    def _check_operational_constraints(self, constraint: RepairConstraint, 
                                     repair_request: Dict[str, Any]) -> List[str]:
        """Check operational constraints."""
        violations = []
        
        try:
            # Check system load constraints
            system_load = repair_request.get("system_load", 0)
            if system_load > 0.8:  # High system load
                violations.append("System load too high for repair operation")
            
            # Check dependency availability
            has_dependencies = repair_request.get("has_external_dependencies", False)
            if has_dependencies:
                dependencies_available = repair_request.get("dependencies_available", True)
                if not dependencies_available:
                    violations.append("Required dependencies not available")
            
            return violations
            
        except Exception:
            return violations
    
    def _generate_validation_reason(self, violations: List[ConstraintViolation], is_allowed: bool) -> str:
        """Generate human-readable validation reason."""
        if is_allowed and len(violations) == 0:
            return "All constraints satisfied"
        elif is_allowed and len(violations) > 0:
            return f"Repair allowed with {len(violations)} constraint warnings"
        else:
            blocking_count = len([v for v in violations if v.severity == ConstraintSeverity.BLOCKING])
            critical_count = len([v for v in violations if v.severity == ConstraintSeverity.CRITICAL])
            
            if blocking_count > 0:
                return f"Repair blocked due to {blocking_count} blocking constraint violations"
            elif critical_count > 0:
                return f"Repair requires approval due to {critical_count} critical constraint violations"
            else:
                return f"Repair validation issues: {len(violations)} constraint violations"
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current repair constraints system status."""
        return {
            "is_active": self.is_active,
            "enforcement_enabled": self.enforcement_enabled,
            "total_constraints": len(self.constraints),
            "enabled_constraints": len([c for c in self.constraints.values() if c.enabled]),
            "total_validations": self.total_validations,
            "blocked_repairs": self.blocked_repairs,
            "approved_repairs": self.approved_repairs,
            "block_rate": (self.blocked_repairs / self.total_validations 
                          if self.total_validations > 0 else 0.0),
            "pending_approvals": len(self.approval_manager.pending_approvals),
            "constraint_types": {
                constraint_type.value: len([c for c in self.constraints.values() 
                                           if c.constraint_type == constraint_type])
                for constraint_type in ConstraintType
            }
        }
    
    async def get_constraint(self, constraint_id: str) -> Optional[RepairConstraint]:
        """Get constraint by ID."""
        return self.constraints.get(constraint_id)
    
    async def get_constraints(self, constraint_type: Optional[ConstraintType] = None,
                            enabled_only: bool = True) -> List[RepairConstraint]:
        """Get constraints with optional filtering."""
        constraints = list(self.constraints.values())
        
        if enabled_only:
            constraints = [c for c in constraints if c.enabled]
        
        if constraint_type:
            constraints = [c for c in constraints if c.constraint_type == constraint_type]
        
        return constraints
    
    async def enable_constraint(self, constraint_id: str) -> bool:
        """Enable a constraint."""
        if constraint_id in self.constraints:
            self.constraints[constraint_id].enabled = True
            return True
        return False
    
    async def disable_constraint(self, constraint_id: str) -> bool:
        """Disable a constraint."""
        if constraint_id in self.constraints:
            self.constraints[constraint_id].enabled = False
            return True
        return False
    
    async def enable_enforcement(self) -> bool:
        """Enable constraint enforcement."""
        self.enforcement_enabled = True
        
        if self.audit_system:
            await self.audit_system.log_event(
                "constraint_enforcement_enabled",
                {"timestamp": time.time()}
            )
        
        return True
    
    async def disable_enforcement(self) -> bool:
        """Disable constraint enforcement (emergency use only)."""
        self.enforcement_enabled = False
        
        if self.audit_system:
            await self.audit_system.log_event(
                "constraint_enforcement_disabled",
                {"timestamp": time.time(), "warning": "Emergency override"}
            )
        
        return True