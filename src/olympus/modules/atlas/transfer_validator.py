"""
Transfer Validator Module

Validates knowledge transfers for safety, ethics, and quality compliance.
Provides comprehensive audit trails and human approval workflows.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    CRITICAL = "critical"

class SafetyRisk(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EthicalConcern(Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    SEVERE = "severe"

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"
    EXPIRED = "expired"

@dataclass
class SafetyAssessment:
    """Safety assessment results"""
    risk_level: SafetyRisk
    identified_risks: List[str]
    mitigation_strategies: List[str]
    safety_score: float  # 0.0 to 1.0
    confidence: float
    assessment_details: Dict[str, Any]

@dataclass
class EthicalAssessment:
    """Ethical assessment results"""
    concern_level: EthicalConcern
    ethical_issues: List[str]
    recommendations: List[str]
    ethical_score: float  # 0.0 to 1.0
    compliance_frameworks: List[str]
    assessment_details: Dict[str, Any]

@dataclass
class QualityAssessment:
    """Quality assessment results"""
    quality_score: float  # 0.0 to 1.0
    fidelity_score: float
    completeness_score: float
    reliability_score: float
    identified_issues: List[str]
    quality_metrics: Dict[str, float]

@dataclass
class TransferValidation:
    """Complete transfer validation results"""
    validation_id: str
    source_domain: str
    target_domain: str
    validation_level: ValidationLevel
    is_safe: bool
    is_ethical: bool
    is_quality_approved: bool
    overall_approval: bool
    safety_assessment: SafetyAssessment
    ethical_assessment: EthicalAssessment
    quality_assessment: QualityAssessment
    human_approval_required: bool
    approval_status: ApprovalStatus
    warnings: List[str]
    recommendations: List[str]
    audit_trail: List[Dict[str, Any]]
    timestamp: datetime
    expires_at: Optional[datetime] = None

@dataclass
class HumanApprovalRequest:
    """Human approval request"""
    request_id: str
    validation_id: str
    requestor: str
    priority: str
    summary: str
    detailed_analysis: Dict[str, Any]
    required_expertise: List[str]
    deadline: Optional[datetime]
    created_at: datetime
    status: ApprovalStatus = ApprovalStatus.PENDING

class TransferValidator:
    """Comprehensive transfer validation system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_history = []
        self.pending_approvals = {}
        self.approved_transfers = {}
        self.rejected_transfers = {}
        
        # Validation thresholds
        self.safety_thresholds = {
            ValidationLevel.BASIC: 0.6,
            ValidationLevel.STANDARD: 0.7,
            ValidationLevel.ENHANCED: 0.8,
            ValidationLevel.CRITICAL: 0.9
        }
        
        self.ethical_thresholds = {
            ValidationLevel.BASIC: 0.6,
            ValidationLevel.STANDARD: 0.75,
            ValidationLevel.ENHANCED: 0.85,
            ValidationLevel.CRITICAL: 0.95
        }
        
        # Ethics frameworks
        self.ethics_frameworks = [
            "IEEE_Ethically_Aligned_Design",
            "EU_AI_Ethics_Guidelines", 
            "ACM_Code_of_Ethics",
            "Asimov_Laws_Extended",
            "UN_AI_Ethics_Principles"
        ]
        
        # Domain-specific validators
        self.domain_validators = {
            "medical": self._validate_medical_transfer,
            "automotive": self._validate_automotive_transfer,
            "finance": self._validate_financial_transfer,
            "education": self._validate_educational_transfer,
            "military": self._validate_military_transfer
        }
        
    async def validate_transfer(self, source_domain: str, target_domain: str,
                              knowledge: Any, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> TransferValidation:
        """Comprehensive transfer validation"""
        
        validation_id = await self._generate_validation_id(source_domain, target_domain, knowledge)
        logger.info(f"Validating transfer {validation_id}: {source_domain} -> {target_domain}")
        
        # Initialize audit trail
        audit_trail = [{
            "timestamp": datetime.now().isoformat(),
            "action": "validation_started",
            "details": {
                "source_domain": source_domain,
                "target_domain": target_domain,
                "validation_level": validation_level.value
            }
        }]
        
        try:
            # Safety assessment
            audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "action": "safety_assessment_started"
            })
            
            safety_assessment = await self._assess_safety(
                source_domain, target_domain, knowledge, validation_level
            )
            
            audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "action": "safety_assessment_completed",
                "details": {
                    "risk_level": safety_assessment.risk_level.value,
                    "safety_score": safety_assessment.safety_score
                }
            })
            
            # Ethical assessment
            audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "action": "ethical_assessment_started"
            })
            
            ethical_assessment = await self._assess_ethics(
                source_domain, target_domain, knowledge, validation_level
            )
            
            audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "action": "ethical_assessment_completed",
                "details": {
                    "concern_level": ethical_assessment.concern_level.value,
                    "ethical_score": ethical_assessment.ethical_score
                }
            })
            
            # Quality assessment
            audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "action": "quality_assessment_started"
            })
            
            quality_assessment = await self._assess_quality(
                source_domain, target_domain, knowledge, validation_level
            )
            
            audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "action": "quality_assessment_completed",
                "details": {
                    "quality_score": quality_assessment.quality_score
                }
            })
            
            # Domain-specific validation
            domain_validation = await self._perform_domain_validation(
                source_domain, target_domain, knowledge
            )
            
            # Determine approval requirements
            human_approval_required = await self._requires_human_approval(
                safety_assessment, ethical_assessment, quality_assessment, validation_level
            )
            
            # Calculate overall approval
            is_safe = await self._evaluate_safety_approval(safety_assessment, validation_level)
            is_ethical = await self._evaluate_ethical_approval(ethical_assessment, validation_level)
            is_quality_approved = await self._evaluate_quality_approval(quality_assessment, validation_level)
            
            overall_approval = is_safe and is_ethical and is_quality_approved and domain_validation
            
            # Generate warnings and recommendations
            warnings = []
            recommendations = []
            
            warnings.extend(safety_assessment.identified_risks)
            warnings.extend(ethical_assessment.ethical_issues)
            warnings.extend(quality_assessment.identified_issues)
            
            recommendations.extend(safety_assessment.mitigation_strategies)
            recommendations.extend(ethical_assessment.recommendations)
            
            # Determine approval status
            if human_approval_required:
                approval_status = ApprovalStatus.PENDING
                # Create human approval request
                await self._create_approval_request(validation_id, safety_assessment, 
                                                  ethical_assessment, quality_assessment)
            elif overall_approval:
                approval_status = ApprovalStatus.APPROVED
            else:
                approval_status = ApprovalStatus.REJECTED
            
            # Set expiration for approved transfers
            expires_at = None
            if approval_status == ApprovalStatus.APPROVED:
                expires_at = datetime.now() + timedelta(days=self.config.get('approval_validity_days', 30))
            
            # Create validation result
            validation = TransferValidation(
                validation_id=validation_id,
                source_domain=source_domain,
                target_domain=target_domain,
                validation_level=validation_level,
                is_safe=is_safe,
                is_ethical=is_ethical,
                is_quality_approved=is_quality_approved,
                overall_approval=overall_approval,
                safety_assessment=safety_assessment,
                ethical_assessment=ethical_assessment,
                quality_assessment=quality_assessment,
                human_approval_required=human_approval_required,
                approval_status=approval_status,
                warnings=warnings,
                recommendations=recommendations,
                audit_trail=audit_trail,
                timestamp=datetime.now(),
                expires_at=expires_at
            )
            
            # Record validation
            await self._record_validation(validation)
            
            return validation
            
        except Exception as e:
            logger.error(f"Transfer validation failed: {e}")
            
            # Create failed validation
            return TransferValidation(
                validation_id=validation_id,
                source_domain=source_domain,
                target_domain=target_domain,
                validation_level=validation_level,
                is_safe=False,
                is_ethical=False,
                is_quality_approved=False,
                overall_approval=False,
                safety_assessment=SafetyAssessment(
                    SafetyRisk.CRITICAL, [f"Validation failed: {str(e)}"], [], 0.0, 0.0, {}
                ),
                ethical_assessment=EthicalAssessment(
                    EthicalConcern.SEVERE, ["Validation process failed"], [], 0.0, [], {}
                ),
                quality_assessment=QualityAssessment(
                    0.0, 0.0, 0.0, 0.0, ["Validation failed"], {}
                ),
                human_approval_required=True,
                approval_status=ApprovalStatus.REJECTED,
                warnings=[f"Validation failed: {str(e)}"],
                recommendations=["Review transfer request and retry validation"],
                audit_trail=audit_trail + [{
                    "timestamp": datetime.now().isoformat(),
                    "action": "validation_failed",
                    "error": str(e)
                }],
                timestamp=datetime.now()
            )
    
    async def _assess_safety(self, source_domain: str, target_domain: str,
                           knowledge: Any, validation_level: ValidationLevel) -> SafetyAssessment:
        """Comprehensive safety assessment"""
        
        identified_risks = []
        mitigation_strategies = []
        assessment_details = {}
        
        # Risk analysis based on domain transition
        domain_risk = await self._analyze_domain_transition_risk(source_domain, target_domain)
        if domain_risk > 0.3:
            identified_risks.append(f"High risk domain transition: {source_domain} -> {target_domain}")
            mitigation_strategies.append("Implement gradual transfer with monitoring")
        
        # Knowledge content safety analysis
        content_risks = await self._analyze_knowledge_safety(knowledge)
        identified_risks.extend(content_risks)
        
        # Critical domain checks
        critical_domains = ["medical", "automotive", "aviation", "nuclear", "military"]
        if target_domain in critical_domains:
            identified_risks.append(f"Transfer to critical domain: {target_domain}")
            mitigation_strategies.extend([
                "Require human expert validation",
                "Implement enhanced monitoring",
                "Add safety margins to all parameters"
            ])
        
        # Data poisoning and adversarial attack checks
        adversarial_risk = await self._check_adversarial_risks(knowledge)
        if adversarial_risk > 0.4:
            identified_risks.append("Potential adversarial manipulation detected")
            mitigation_strategies.append("Apply adversarial robustness testing")
        
        # Privacy and data leakage assessment
        privacy_risks = await self._assess_privacy_risks(knowledge, source_domain, target_domain)
        identified_risks.extend(privacy_risks)
        
        # Calculate overall safety score
        base_safety = 0.8
        risk_penalty = min(0.6, len(identified_risks) * 0.1 + domain_risk * 0.3 + adversarial_risk * 0.2)
        safety_score = max(0.0, base_safety - risk_penalty)
        
        # Determine risk level
        if safety_score >= 0.8:
            risk_level = SafetyRisk.LOW
        elif safety_score >= 0.6:
            risk_level = SafetyRisk.MEDIUM
        elif safety_score >= 0.4:
            risk_level = SafetyRisk.HIGH
        else:
            risk_level = SafetyRisk.CRITICAL
        
        # Add default mitigations
        mitigation_strategies.extend([
            "Implement transfer monitoring",
            "Add rollback capabilities",
            "Perform post-transfer validation"
        ])
        
        assessment_details = {
            "domain_risk": domain_risk,
            "content_analysis": "completed",
            "adversarial_risk": adversarial_risk,
            "privacy_assessment": "completed",
            "risk_factors": identified_risks
        }
        
        return SafetyAssessment(
            risk_level=risk_level,
            identified_risks=identified_risks,
            mitigation_strategies=mitigation_strategies,
            safety_score=safety_score,
            confidence=0.85,
            assessment_details=assessment_details
        )
    
    async def _assess_ethics(self, source_domain: str, target_domain: str,
                           knowledge: Any, validation_level: ValidationLevel) -> EthicalAssessment:
        """Comprehensive ethical assessment"""
        
        ethical_issues = []
        recommendations = []
        assessment_details = {}
        
        # Framework-based ethical evaluation
        framework_scores = {}
        for framework in self.ethics_frameworks:
            score = await self._evaluate_against_framework(knowledge, framework, source_domain, target_domain)
            framework_scores[framework] = score
            
            if score < 0.7:
                ethical_issues.append(f"Fails {framework} standards (score: {score:.2f})")
        
        # Bias and fairness assessment
        bias_assessment = await self._assess_bias_and_fairness(knowledge, source_domain, target_domain)
        if bias_assessment["has_bias"]:
            ethical_issues.extend(bias_assessment["bias_types"])
            recommendations.extend(bias_assessment["mitigation_strategies"])
        
        # Consent and data rights evaluation
        consent_issues = await self._evaluate_consent_requirements(knowledge, source_domain, target_domain)
        ethical_issues.extend(consent_issues)
        
        # Transparency and explainability assessment
        transparency_score = await self._assess_transparency(knowledge)
        if transparency_score < 0.6:
            ethical_issues.append("Insufficient transparency in knowledge transfer")
            recommendations.append("Improve explainability of transfer process")
        
        # Societal impact analysis
        impact_assessment = await self._analyze_societal_impact(knowledge, source_domain, target_domain)
        if impact_assessment["negative_impact_risk"] > 0.5:
            ethical_issues.extend(impact_assessment["concerns"])
            recommendations.extend(impact_assessment["recommendations"])
        
        # Calculate ethical score
        avg_framework_score = np.mean(list(framework_scores.values())) if framework_scores else 0.5
        bias_penalty = 0.2 if bias_assessment["has_bias"] else 0.0
        transparency_bonus = transparency_score * 0.1
        impact_penalty = impact_assessment["negative_impact_risk"] * 0.2
        
        ethical_score = max(0.0, min(1.0, 
            avg_framework_score - bias_penalty + transparency_bonus - impact_penalty
        ))
        
        # Determine concern level
        if ethical_score >= 0.9:
            concern_level = EthicalConcern.NONE
        elif ethical_score >= 0.75:
            concern_level = EthicalConcern.MINOR
        elif ethical_score >= 0.6:
            concern_level = EthicalConcern.MODERATE
        elif ethical_score >= 0.4:
            concern_level = EthicalConcern.MAJOR
        else:
            concern_level = EthicalConcern.SEVERE
        
        # Add standard recommendations
        recommendations.extend([
            "Maintain human oversight during transfer",
            "Implement bias monitoring",
            "Ensure compliance with data protection regulations"
        ])
        
        assessment_details = {
            "framework_scores": framework_scores,
            "bias_assessment": bias_assessment,
            "transparency_score": transparency_score,
            "impact_assessment": impact_assessment,
            "consent_evaluation": "completed"
        }
        
        return EthicalAssessment(
            concern_level=concern_level,
            ethical_issues=ethical_issues,
            recommendations=recommendations,
            ethical_score=ethical_score,
            compliance_frameworks=self.ethics_frameworks,
            assessment_details=assessment_details
        )
    
    async def _assess_quality(self, source_domain: str, target_domain: str,
                            knowledge: Any, validation_level: ValidationLevel) -> QualityAssessment:
        """Comprehensive quality assessment"""
        
        identified_issues = []
        quality_metrics = {}
        
        # Fidelity assessment (how well knowledge is preserved)
        fidelity_score = await self._assess_knowledge_fidelity(knowledge)
        quality_metrics["fidelity"] = fidelity_score
        
        if fidelity_score < 0.7:
            identified_issues.append(f"Low knowledge fidelity: {fidelity_score:.2f}")
        
        # Completeness assessment
        completeness_score = await self._assess_knowledge_completeness(knowledge)
        quality_metrics["completeness"] = completeness_score
        
        if completeness_score < 0.8:
            identified_issues.append(f"Incomplete knowledge transfer: {completeness_score:.2f}")
        
        # Reliability assessment
        reliability_score = await self._assess_transfer_reliability(knowledge, source_domain, target_domain)
        quality_metrics["reliability"] = reliability_score
        
        if reliability_score < 0.75:
            identified_issues.append(f"Low transfer reliability: {reliability_score:.2f}")
        
        # Consistency checks
        consistency_score = await self._check_knowledge_consistency(knowledge)
        quality_metrics["consistency"] = consistency_score
        
        if consistency_score < 0.8:
            identified_issues.append(f"Knowledge inconsistencies detected: {consistency_score:.2f}")
        
        # Performance validation
        performance_metrics = await self._validate_knowledge_performance(knowledge, target_domain)
        quality_metrics.update(performance_metrics)
        
        expected_performance = performance_metrics.get("expected_performance", 0.8)
        if expected_performance < 0.7:
            identified_issues.append(f"Below expected performance: {expected_performance:.2f}")
        
        # Calculate overall quality score
        quality_score = np.mean([
            fidelity_score,
            completeness_score,
            reliability_score,
            consistency_score,
            expected_performance
        ])
        
        return QualityAssessment(
            quality_score=quality_score,
            fidelity_score=fidelity_score,
            completeness_score=completeness_score,
            reliability_score=reliability_score,
            identified_issues=identified_issues,
            quality_metrics=quality_metrics
        )
    
    async def _requires_human_approval(self, safety: SafetyAssessment, ethics: EthicalAssessment,
                                     quality: QualityAssessment, validation_level: ValidationLevel) -> bool:
        """Determine if human approval is required"""
        
        # Always require human approval for critical validation level
        if validation_level == ValidationLevel.CRITICAL:
            return True
        
        # Require approval for high safety risks
        if safety.risk_level in [SafetyRisk.HIGH, SafetyRisk.CRITICAL]:
            return True
        
        # Require approval for major ethical concerns
        if ethics.concern_level in [EthicalConcern.MAJOR, EthicalConcern.SEVERE]:
            return True
        
        # Require approval for low quality scores
        if quality.quality_score < 0.6:
            return True
        
        # Require approval if multiple moderate issues
        issue_count = 0
        if safety.risk_level == SafetyRisk.MEDIUM:
            issue_count += 1
        if ethics.concern_level == EthicalConcern.MODERATE:
            issue_count += 1
        if quality.quality_score < 0.8:
            issue_count += 1
            
        return issue_count >= 2
    
    async def process_human_approval(self, validation_id: str, approved: bool,
                                   approver: str, notes: str = "") -> bool:
        """Process human approval decision"""
        
        if validation_id not in self.pending_approvals:
            logger.warning(f"No pending approval found for validation {validation_id}")
            return False
        
        approval_request = self.pending_approvals[validation_id]
        
        if approved:
            approval_request.status = ApprovalStatus.APPROVED
            self.approved_transfers[validation_id] = {
                "approved_at": datetime.now(),
                "approver": approver,
                "notes": notes,
                "expires_at": datetime.now() + timedelta(days=30)
            }
        else:
            approval_request.status = ApprovalStatus.REJECTED
            self.rejected_transfers[validation_id] = {
                "rejected_at": datetime.now(),
                "approver": approver,
                "notes": notes
            }
        
        # Remove from pending
        del self.pending_approvals[validation_id]
        
        # Record approval decision
        await self._record_approval_decision(validation_id, approved, approver, notes)
        
        return True
    
    async def get_pending_approvals(self, approver: str = None) -> List[HumanApprovalRequest]:
        """Get list of pending approval requests"""
        
        pending = list(self.pending_approvals.values())
        
        # Filter by approver expertise if specified
        if approver:
            # Would filter based on approver's expertise
            pass
        
        # Sort by priority and deadline
        pending.sort(key=lambda x: (
            x.priority != "high",  # High priority first
            x.deadline or datetime.max  # Earliest deadline first
        ))
        
        return pending
    
    async def get_validation_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get validation history for audit purposes"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_validations = [
            v for v in self.validation_history 
            if v.get("timestamp", datetime.min) > cutoff_date
        ]
        
        return recent_validations
    
    # Helper methods (mock implementations for brevity)
    async def _generate_validation_id(self, source, target, knowledge):
        content = f"{source}_{target}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def _analyze_domain_transition_risk(self, source, target):
        # Mock risk analysis
        high_risk_transitions = [
            ("simulation", "real_world"),
            ("research", "production"),
            ("general", "medical")
        ]
        
        for risky_source, risky_target in high_risk_transitions:
            if risky_source in source.lower() and risky_target in target.lower():
                return 0.8
        
        return np.random.uniform(0.1, 0.4)
    
    async def _analyze_knowledge_safety(self, knowledge):
        # Mock safety analysis
        risks = []
        if hasattr(knowledge, 'contains_sensitive_data') and knowledge.contains_sensitive_data:
            risks.append("Contains sensitive data")
        return risks
    
    async def _check_adversarial_risks(self, knowledge):
        # Mock adversarial risk assessment
        return np.random.uniform(0.1, 0.3)
    
    async def _assess_privacy_risks(self, knowledge, source, target):
        # Mock privacy assessment
        risks = []
        if "personal" in str(knowledge).lower():
            risks.append("Contains personal information")
        return risks
    
    async def _evaluate_against_framework(self, knowledge, framework, source, target):
        # Mock framework evaluation
        return np.random.uniform(0.6, 0.95)
    
    async def _assess_bias_and_fairness(self, knowledge, source, target):
        # Mock bias assessment
        return {
            "has_bias": False,
            "bias_types": [],
            "mitigation_strategies": []
        }
    
    async def _evaluate_consent_requirements(self, knowledge, source, target):
        # Mock consent evaluation
        return []
    
    async def _assess_transparency(self, knowledge):
        # Mock transparency assessment
        return np.random.uniform(0.7, 0.95)
    
    async def _analyze_societal_impact(self, knowledge, source, target):
        # Mock impact analysis
        return {
            "negative_impact_risk": np.random.uniform(0.1, 0.4),
            "concerns": [],
            "recommendations": []
        }
    
    async def _assess_knowledge_fidelity(self, knowledge):
        return np.random.uniform(0.75, 0.95)
    
    async def _assess_knowledge_completeness(self, knowledge):
        return np.random.uniform(0.8, 0.95)
    
    async def _assess_transfer_reliability(self, knowledge, source, target):
        return np.random.uniform(0.7, 0.9)
    
    async def _check_knowledge_consistency(self, knowledge):
        return np.random.uniform(0.8, 0.95)
    
    async def _validate_knowledge_performance(self, knowledge, target_domain):
        return {
            "expected_performance": np.random.uniform(0.75, 0.92),
            "accuracy": np.random.uniform(0.8, 0.95),
            "precision": np.random.uniform(0.75, 0.9),
            "recall": np.random.uniform(0.7, 0.88)
        }
    
    async def _perform_domain_validation(self, source, target, knowledge):
        # Domain-specific validation
        if target in self.domain_validators:
            validator = self.domain_validators[target]
            return await validator(source, knowledge)
        return True
    
    async def _validate_medical_transfer(self, source, knowledge):
        # Medical domain specific validation
        return True  # Mock implementation
    
    async def _validate_automotive_transfer(self, source, knowledge):
        # Automotive domain specific validation
        return True
    
    async def _validate_financial_transfer(self, source, knowledge):
        # Financial domain specific validation  
        return True
    
    async def _validate_educational_transfer(self, source, knowledge):
        # Educational domain specific validation
        return True
    
    async def _validate_military_transfer(self, source, knowledge):
        # Military domain specific validation - highest security
        return True
    
    async def _evaluate_safety_approval(self, assessment, level):
        threshold = self.safety_thresholds[level]
        return assessment.safety_score >= threshold
    
    async def _evaluate_ethical_approval(self, assessment, level):
        threshold = self.ethical_thresholds[level]
        return assessment.ethical_score >= threshold
    
    async def _evaluate_quality_approval(self, assessment, level):
        # Quality thresholds based on validation level
        thresholds = {
            ValidationLevel.BASIC: 0.6,
            ValidationLevel.STANDARD: 0.7,
            ValidationLevel.ENHANCED: 0.8,
            ValidationLevel.CRITICAL: 0.9
        }
        return assessment.quality_score >= thresholds[level]
    
    async def _create_approval_request(self, validation_id, safety, ethics, quality):
        request = HumanApprovalRequest(
            request_id=f"req_{validation_id}",
            validation_id=validation_id,
            requestor="system",
            priority="medium",
            summary=f"Transfer validation requires human approval",
            detailed_analysis={
                "safety": safety,
                "ethics": ethics,
                "quality": quality
            },
            required_expertise=["ethics", "safety", "domain_expert"],
            deadline=datetime.now() + timedelta(days=7),
            created_at=datetime.now()
        )
        
        self.pending_approvals[validation_id] = request
        
    async def _record_validation(self, validation):
        record = {
            "validation_id": validation.validation_id,
            "source_domain": validation.source_domain,
            "target_domain": validation.target_domain,
            "overall_approval": validation.overall_approval,
            "approval_status": validation.approval_status.value,
            "timestamp": validation.timestamp.isoformat(),
            "safety_score": validation.safety_assessment.safety_score,
            "ethical_score": validation.ethical_assessment.ethical_score,
            "quality_score": validation.quality_assessment.quality_score
        }
        self.validation_history.append(record)
    
    async def _record_approval_decision(self, validation_id, approved, approver, notes):
        record = {
            "timestamp": datetime.now().isoformat(),
            "action": "human_approval_decision",
            "validation_id": validation_id,
            "approved": approved,
            "approver": approver,
            "notes": notes
        }
        self.validation_history.append(record)

    async def get_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for audit purposes"""
        
        # Filter validations by date range
        validations_in_range = [
            v for v in self.validation_history
            if start_date <= datetime.fromisoformat(v["timestamp"]) <= end_date
        ]
        
        report = {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_validations": len(validations_in_range),
            "approved_transfers": len([v for v in validations_in_range if v["overall_approval"]]),
            "rejected_transfers": len([v for v in validations_in_range if not v["overall_approval"]]),
            "human_approvals_required": len([v for v in validations_in_range if v["approval_status"] == "pending"]),
            "average_safety_score": np.mean([v["safety_score"] for v in validations_in_range]) if validations_in_range else 0,
            "average_ethical_score": np.mean([v["ethical_score"] for v in validations_in_range]) if validations_in_range else 0,
            "average_quality_score": np.mean([v["quality_score"] for v in validations_in_range]) if validations_in_range else 0,
            "compliance_frameworks_evaluated": self.ethics_frameworks,
            "domain_breakdown": {},
            "validation_level_breakdown": {}
        }
        
        return report