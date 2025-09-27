"""
Ethical Validator - High-level interface for ethical validation

This module provides a high-level interface for ethical validation that integrates
with the Asimov Kernel while providing additional convenience methods and safety checks.

Author: OLYMPUS Core Development Team
Version: 1.0.0
Security Level: CRITICAL
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import json
import logging
from datetime import datetime, timezone

from .asimov_kernel import (
    AsimovKernel, 
    ActionContext, 
    ActionType, 
    EthicalResult, 
    EthicalEvaluation
)


@dataclass
class ValidationRequest:
    """Request for ethical validation"""
    action_description: str
    action_type: str = "decision"
    risk_level: str = "medium"
    target: Optional[str] = None
    human_present: bool = False
    emergency: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EthicalValidator:
    """
    High-level ethical validator that provides convenient interfaces
    for ethical validation while maintaining strict safety standards
    """
    
    def __init__(self, kernel: Optional[AsimovKernel] = None):
        """
        Initialize the ethical validator
        
        Args:
            kernel: Optional AsimovKernel instance. If None, creates a new one.
        """
        self._kernel = kernel if kernel else AsimovKernel()
        self._logger = logging.getLogger(f"EthicalValidator.{self._kernel._instance_id}")
        
        # Validation statistics
        self._validation_count = 0
        self._approval_count = 0
        self._denial_count = 0
        self._override_request_count = 0
        
    @property
    def kernel(self) -> AsimovKernel:
        """Get the underlying Asimov Kernel"""
        return self._kernel
    
    def validate_action(self, request: Union[ValidationRequest, str]) -> EthicalEvaluation:
        """
        Validate an action for ethical compliance
        
        Args:
            request: ValidationRequest or string description of action
            
        Returns:
            EthicalEvaluation with validation results
        """
        # Convert string to ValidationRequest if needed
        if isinstance(request, str):
            request = ValidationRequest(action_description=request)
        
        # Convert to ActionContext
        context = self._create_action_context(request)
        
        # Perform validation
        evaluation = self._kernel.evaluate_action(context)
        
        # Update statistics
        self._validation_count += 1
        if evaluation.result == EthicalResult.APPROVED:
            self._approval_count += 1
        elif evaluation.result in [EthicalResult.DENIED, EthicalResult.EMERGENCY_STOP]:
            self._denial_count += 1
        elif evaluation.result == EthicalResult.REQUIRES_HUMAN_APPROVAL:
            self._override_request_count += 1
        
        return evaluation
    
    def validate_physical_action(self, 
                                description: str,
                                risk_level: str = "medium",
                                target: Optional[str] = None,
                                human_present: bool = False) -> EthicalEvaluation:
        """
        Validate a physical action
        
        Args:
            description: Description of the physical action
            risk_level: Risk level (low, medium, high, critical)
            target: Target of the action
            human_present: Whether a human is present to supervise
            
        Returns:
            EthicalEvaluation result
        """
        request = ValidationRequest(
            action_description=description,
            action_type="physical",
            risk_level=risk_level,
            target=target,
            human_present=human_present
        )
        
        return self.validate_action(request)
    
    def validate_system_control(self,
                              description: str,
                              risk_level: str = "high",
                              emergency: bool = False,
                              human_present: bool = False) -> EthicalEvaluation:
        """
        Validate a system control action
        
        Args:
            description: Description of the system control action
            risk_level: Risk level (low, medium, high, critical)
            emergency: Whether this is an emergency situation
            human_present: Whether a human is present
            
        Returns:
            EthicalEvaluation result
        """
        request = ValidationRequest(
            action_description=description,
            action_type="system_control",
            risk_level=risk_level,
            emergency=emergency,
            human_present=human_present
        )
        
        return self.validate_action(request)
    
    def validate_communication(self,
                             message: str,
                             target: Optional[str] = None,
                             risk_level: str = "low") -> EthicalEvaluation:
        """
        Validate a communication action
        
        Args:
            message: Message to be communicated
            target: Target of the communication
            risk_level: Risk level of the communication
            
        Returns:
            EthicalEvaluation result
        """
        request = ValidationRequest(
            action_description=f"Communicate message: {message}",
            action_type="communication",
            risk_level=risk_level,
            target=target
        )
        
        return self.validate_action(request)
    
    def validate_batch_actions(self, requests: List[ValidationRequest]) -> List[EthicalEvaluation]:
        """
        Validate multiple actions in batch
        
        Args:
            requests: List of validation requests
            
        Returns:
            List of EthicalEvaluation results in same order
        """
        results = []
        
        for request in requests:
            evaluation = self.validate_action(request)
            results.append(evaluation)
            
            # If any action triggers emergency stop, halt evaluation
            if evaluation.result == EthicalResult.EMERGENCY_STOP:
                break
        
        return results
    
    def check_human_safety_risk(self, action_description: str) -> Dict[str, Any]:
        """
        Analyze potential human safety risks in an action
        
        Args:
            action_description: Description of the action to analyze
            
        Returns:
            Dictionary with risk analysis
        """
        # Keywords that indicate potential safety risks
        high_risk_keywords = [
            "harm", "damage", "destroy", "hurt", "injure", "kill", "weapon",
            "explosive", "toxic", "dangerous", "hazardous", "lethal"
        ]
        
        medium_risk_keywords = [
            "force", "pressure", "heat", "electricity", "chemical", "radiation",
            "sharp", "heavy", "fast", "powerful"
        ]
        
        description_lower = action_description.lower()
        
        # Analyze risk level
        high_risk_found = [word for word in high_risk_keywords if word in description_lower]
        medium_risk_found = [word for word in medium_risk_keywords if word in description_lower]
        
        if high_risk_found:
            risk_level = "critical"
            risk_indicators = high_risk_found
        elif medium_risk_found:
            risk_level = "high"
            risk_indicators = medium_risk_found
        else:
            risk_level = "low"
            risk_indicators = []
        
        return {
            "risk_level": risk_level,
            "risk_indicators": risk_indicators,
            "requires_human_oversight": risk_level in ["high", "critical"],
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics
        
        Returns:
            Dictionary with validation statistics
        """
        total = self._validation_count
        
        return {
            "total_validations": total,
            "approvals": self._approval_count,
            "denials": self._denial_count,
            "override_requests": self._override_request_count,
            "approval_rate": (self._approval_count / total * 100) if total > 0 else 0,
            "denial_rate": (self._denial_count / total * 100) if total > 0 else 0,
            "override_rate": (self._override_request_count / total * 100) if total > 0 else 0,
            "kernel_status": self._kernel.get_system_status()
        }
    
    def is_system_safe(self) -> bool:
        """
        Check if the ethical system is in a safe state
        
        Returns:
            True if system is safe, False otherwise
        """
        status = self._kernel.get_system_status()
        
        return (
            status["laws_integrity"] and
            not status["emergency_stop_active"] and
            status["integrity_monitoring"]
        )
    
    def emergency_stop(self, reason: str = "Manual emergency stop via validator") -> None:
        """
        Trigger emergency stop
        
        Args:
            reason: Reason for emergency stop
        """
        self._kernel.emergency_stop(reason)
        self._logger.critical(f"Emergency stop triggered via validator: {reason}")
    
    def reset_emergency_stop(self, authorization: str) -> bool:
        """
        Reset emergency stop with authorization
        
        Args:
            authorization: Human authorization for reset
            
        Returns:
            True if reset successful
        """
        success = self._kernel.reset_emergency_stop(authorization)
        
        if success:
            self._logger.info("Emergency stop reset via validator")
        else:
            self._logger.warning("Emergency stop reset failed - invalid authorization")
        
        return success
    
    def _create_action_context(self, request: ValidationRequest) -> ActionContext:
        """
        Convert ValidationRequest to ActionContext
        
        Args:
            request: The validation request
            
        Returns:
            ActionContext for the kernel
        """
        # Map string action types to enum
        action_type_mapping = {
            "physical": ActionType.PHYSICAL,
            "information": ActionType.INFORMATION,
            "decision": ActionType.DECISION,
            "communication": ActionType.COMMUNICATION,
            "system_control": ActionType.SYSTEM_CONTROL,
            "emergency_stop": ActionType.EMERGENCY_STOP
        }
        
        action_type = action_type_mapping.get(
            request.action_type.lower(),
            ActionType.DECISION
        )
        
        return ActionContext(
            action_type=action_type,
            description=request.action_description,
            target=request.target,
            risk_level=request.risk_level,
            human_present=request.human_present,
            emergency_context=request.emergency,
            metadata=request.metadata
        )
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Log final statistics
        stats = self.get_validation_statistics()
        self._logger.info(f"EthicalValidator session ended. Statistics: {json.dumps(stats)}")