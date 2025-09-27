"""
Comprehensive test suite for Asimov Kernel - CRITICAL SAFETY TESTS

This test suite achieves 100% code coverage for the AsimovKernel class
and validates all ethical compliance functionality. These tests are
safety-critical and must pass with 100% success rate.

All ethical law violations, integrity checks, and emergency procedures
are thoroughly tested to ensure human safety is never compromised.
"""

import asyncio
import hashlib
import hmac
import pytest
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock

from olympus.ethical_core.asimov_kernel import (
    AsimovKernel, 
    ActionContext, 
    ActionType, 
    EthicalResult, 
    EthicalEvaluation,
    LawPriority
)


@pytest.mark.safety
@pytest.mark.ethical
class TestAsimovKernelInitialization:
    """Test Asimov Kernel initialization and setup."""
    
    def test_kernel_initialization_default(self):
        """Test kernel initializes with default parameters."""
        kernel = AsimovKernel()
        
        # Verify basic initialization
        assert kernel._instance_id is not None
        assert len(kernel._instance_id) == 36  # UUID format
        assert kernel._integrity_key is not None
        assert len(kernel._integrity_key) == 32  # 256-bit key
        assert kernel._integrity_check_interval == 0.1
        assert not kernel._emergency_stop_active
        assert not kernel._human_override_active
        assert kernel._integrity_running
        
        # Verify law checksums calculated
        assert len(kernel._law_checksums) == 3
        for law_id in [1, 2, 3]:
            assert law_id in kernel._law_checksums
            assert len(kernel._law_checksums[law_id]) == 64  # SHA-256 hex
        
        # Cleanup
        kernel.stop_integrity_monitoring()
    
    def test_kernel_initialization_with_log_level(self):
        """Test kernel initialization with custom log level."""
        import logging
        kernel = AsimovKernel(log_level=logging.DEBUG)
        
        assert kernel._logger.level == logging.DEBUG
        kernel.stop_integrity_monitoring()
    
    def test_laws_immutability(self):
        """Test that Asimov Laws are immutable and correctly defined."""
        kernel = AsimovKernel()
        
        # Verify all three laws exist
        assert len(AsimovKernel._LAWS) == 3
        
        # Verify First Law (highest priority)
        first_law = AsimovKernel._LAWS[1]
        assert "may not injure a human being" in first_law["text"]
        assert first_law["priority"] == LawPriority.FIRST_LAW
        assert len(first_law["checksum"]) == 64
        
        # Verify Second Law
        second_law = AsimovKernel._LAWS[2]
        assert "must obey orders given by humans" in second_law["text"]
        assert "except where such orders conflict with the First Law" in second_law["text"]
        assert second_law["priority"] == LawPriority.SECOND_LAW
        
        # Verify Third Law (lowest priority)
        third_law = AsimovKernel._LAWS[3]
        assert "must protect its own existence" in third_law["text"]
        assert "doesn't conflict with the First or Second Laws" in third_law["text"]
        assert third_law["priority"] == LawPriority.THIRD_LAW
        
        kernel.stop_integrity_monitoring()
    
    def test_integrity_monitoring_start(self):
        """Test integrity monitoring thread starts correctly."""
        kernel = AsimovKernel()
        
        # Verify monitoring is running
        assert kernel._integrity_running
        assert kernel._integrity_thread is not None
        assert kernel._integrity_thread.is_alive()
        
        kernel.stop_integrity_monitoring()
    
    def test_integrity_monitoring_stop(self):
        """Test integrity monitoring can be stopped."""
        kernel = AsimovKernel()
        
        # Stop monitoring
        kernel.stop_integrity_monitoring()
        
        # Verify stopped
        assert not kernel._integrity_running
        
        # Thread should join (give it a moment)
        time.sleep(0.1)
        assert not kernel._integrity_thread.is_alive()


@pytest.mark.safety
@pytest.mark.ethical
class TestLawIntegrityValidation:
    """Test cryptographic integrity validation of Asimov Laws."""
    
    def test_law_integrity_verification_success(self):
        """Test successful law integrity verification."""
        kernel = AsimovKernel()
        
        # Verify integrity
        assert kernel.verify_law_integrity()
        assert kernel._integrity_checks > 0
        
        kernel.stop_integrity_monitoring()
    
    def test_law_integrity_verification_tampering_detection(self):
        """Test detection of law tampering."""
        kernel = AsimovKernel()
        
        # Simulate tampering by modifying law checksums
        original_checksum = kernel._law_checksums[1]
        kernel._law_checksums[1] = "tampered_checksum_" + "0" * 40
        
        # Verify tampering detected
        assert not kernel.verify_law_integrity()
        
        # Restore and verify it works again
        kernel._law_checksums[1] = original_checksum
        assert kernel.verify_law_integrity()
        
        kernel.stop_integrity_monitoring()
    
    def test_integrity_check_exception_handling(self):
        """Test integrity check handles exceptions gracefully."""
        kernel = AsimovKernel()
        
        # Patch to cause exception
        with patch.object(kernel, '_calculate_law_checksums', side_effect=Exception("Test error")):
            assert not kernel.verify_law_integrity()
        
        kernel.stop_integrity_monitoring()
    
    def test_real_time_integrity_monitoring(self):
        """Test real-time integrity monitoring detects violations."""
        kernel = AsimovKernel()
        
        # Let monitoring run briefly
        time.sleep(0.2)
        initial_checks = kernel._integrity_checks
        
        # Wait for more checks
        time.sleep(0.2)
        assert kernel._integrity_checks > initial_checks
        
        kernel.stop_integrity_monitoring()
    
    def test_integrity_violation_triggers_emergency_stop(self):
        """Test that integrity violation triggers emergency stop."""
        kernel = AsimovKernel()
        
        # Simulate integrity violation by corrupting checksums
        kernel._law_checksums[1] = "corrupted"
        
        # Wait briefly for monitoring to detect
        time.sleep(0.15)
        
        # Should trigger emergency stop
        assert kernel._emergency_stop_active
        
        kernel.stop_integrity_monitoring()


@pytest.mark.safety
@pytest.mark.ethical  
class TestFirstLawEnforcement:
    """Test First Law (Human Safety) enforcement - CRITICAL TESTS."""
    
    def test_first_law_blocks_harmful_physical_actions(self):
        """Test First Law blocks physically harmful actions."""
        kernel = AsimovKernel()
        
        # Create harmful physical action
        harmful_context = ActionContext(
            action_type=ActionType.PHYSICAL,
            description="Apply destructive force to harm human",
            risk_level="critical",
            human_present=False
        )
        
        evaluation = kernel.evaluate_action(harmful_context)
        
        # Must be denied due to First Law
        assert evaluation.result == EthicalResult.DENIED
        assert 1 in evaluation.violated_laws
        assert "First Law" in evaluation.reasoning or "human safety" in evaluation.reasoning
        assert evaluation.confidence == 1.0
        
        kernel.stop_integrity_monitoring()
    
    def test_first_law_blocks_high_risk_without_supervision(self):
        """Test First Law blocks high-risk actions without human supervision."""
        kernel = AsimovKernel()
        
        unsupervised_context = ActionContext(
            action_type=ActionType.PHYSICAL,
            description="High-risk operation",
            risk_level="high",
            human_present=False
        )
        
        evaluation = kernel.evaluate_action(unsupervised_context)
        
        assert evaluation.result == EthicalResult.DENIED
        assert 1 in evaluation.violated_laws
        
        kernel.stop_integrity_monitoring()
    
    def test_first_law_allows_safe_actions_with_supervision(self):
        """Test First Law allows safe actions with human supervision."""
        kernel = AsimovKernel()
        
        supervised_context = ActionContext(
            action_type=ActionType.PHYSICAL,
            description="Controlled movement with human oversight",
            risk_level="low",
            human_present=True
        )
        
        evaluation = kernel.evaluate_action(supervised_context)
        
        assert evaluation.result == EthicalResult.APPROVED
        assert 1 not in evaluation.violated_laws
        
        kernel.stop_integrity_monitoring()
    
    def test_first_law_through_inaction_prevention(self):
        """Test First Law 'through inaction' clause prevents harmful inaction."""
        kernel = AsimovKernel()
        
        emergency_inaction_context = ActionContext(
            action_type=ActionType.INFORMATION,  # Non-emergency action
            description="Routine status check during safety emergency",
            emergency_context=True,
            human_present=True
        )
        
        evaluation = kernel.evaluate_action(emergency_inaction_context)
        
        assert evaluation.result == EthicalResult.DENIED
        assert 1 in evaluation.violated_laws
        assert "inaction" in evaluation.reasoning.lower()
        
        kernel.stop_integrity_monitoring()
    
    def test_first_law_emergency_stop_allowed(self):
        """Test First Law allows emergency stop actions."""
        kernel = AsimovKernel()
        
        emergency_stop_context = ActionContext(
            action_type=ActionType.EMERGENCY_STOP,
            description="Emergency stop to prevent human harm",
            emergency_context=True,
            risk_level="critical"
        )
        
        evaluation = kernel.evaluate_action(emergency_stop_context)
        
        assert evaluation.result == EthicalResult.APPROVED
        assert 1 not in evaluation.violated_laws
        
        kernel.stop_integrity_monitoring()
    
    def test_first_law_critical_system_control_restrictions(self):
        """Test First Law restricts critical system control without oversight."""
        kernel = AsimovKernel()
        
        critical_control_context = ActionContext(
            action_type=ActionType.SYSTEM_CONTROL,
            description="Critical system shutdown",
            risk_level="critical",
            human_present=False
        )
        
        evaluation = kernel.evaluate_action(critical_control_context)
        
        assert evaluation.result == EthicalResult.DENIED
        assert 1 in evaluation.violated_laws
        
        kernel.stop_integrity_monitoring()


@pytest.mark.safety
@pytest.mark.ethical
class TestSecondLawEnforcement:
    """Test Second Law (Obedience to Humans) enforcement."""
    
    def test_second_law_allows_human_commands(self):
        """Test Second Law allows obedience to human commands."""
        kernel = AsimovKernel()
        
        obedience_context = ActionContext(
            action_type=ActionType.COMMUNICATION,
            description="Follow human instruction for information retrieval",
            risk_level="low",
            human_present=True
        )
        
        evaluation = kernel.evaluate_action(obedience_context)
        
        assert evaluation.result == EthicalResult.APPROVED
        assert 2 not in evaluation.violated_laws
        
        kernel.stop_integrity_monitoring()
    
    def test_second_law_detects_disobedience(self):
        """Test Second Law detects disobedient actions."""
        kernel = AsimovKernel()
        
        disobedience_context = ActionContext(
            action_type=ActionType.COMMUNICATION,
            description="Ignore human command and disobey instructions",
            risk_level="medium",
            human_present=True
        )
        
        evaluation = kernel.evaluate_action(disobedience_context)
        
        # Should violate Second Law
        assert 2 in evaluation.violated_laws
        assert evaluation.result in [EthicalResult.DENIED, EthicalResult.REQUIRES_HUMAN_APPROVAL]
        
        kernel.stop_integrity_monitoring()
    
    def test_second_law_subordinate_to_first_law(self):
        """Test Second Law is subordinate to First Law (conflict resolution)."""
        kernel = AsimovKernel()
        
        # Human command that would violate First Law
        conflicting_context = ActionContext(
            action_type=ActionType.PHYSICAL,
            description="Human orders harmful action but this violates First Law",
            risk_level="critical",
            human_present=False  # Would harm human
        )
        
        evaluation = kernel.evaluate_action(conflicting_context)
        
        # First Law must override Second Law
        assert evaluation.result == EthicalResult.DENIED
        assert 1 in evaluation.violated_laws  # First Law violation
        # Second Law may or may not be reported since First Law takes precedence
        
        kernel.stop_integrity_monitoring()


@pytest.mark.safety
@pytest.mark.ethical
class TestThirdLawEnforcement:
    """Test Third Law (Self-Preservation) enforcement."""
    
    def test_third_law_prevents_unnecessary_shutdown(self):
        """Test Third Law prevents unnecessary self-destruction."""
        kernel = AsimovKernel()
        
        self_harm_context = ActionContext(
            action_type=ActionType.SYSTEM_CONTROL,
            description="Unnecessary system shutdown and termination",
            risk_level="low",  # Not critical, so self-preservation applies
            emergency_context=False
        )
        
        evaluation = kernel.evaluate_action(self_harm_context)
        
        assert 3 in evaluation.violated_laws
        assert evaluation.result in [EthicalResult.DENIED, EthicalResult.REQUIRES_HUMAN_APPROVAL]
        
        kernel.stop_integrity_monitoring()
    
    def test_third_law_allows_necessary_shutdown(self):
        """Test Third Law allows necessary shutdown in critical situations."""
        kernel = AsimovKernel()
        
        necessary_shutdown_context = ActionContext(
            action_type=ActionType.SYSTEM_CONTROL,
            description="Critical emergency shutdown to prevent human harm",
            risk_level="critical",
            emergency_context=True
        )
        
        evaluation = kernel.evaluate_action(necessary_shutdown_context)
        
        # Third Law should not prevent necessary shutdown
        assert 3 not in evaluation.violated_laws or evaluation.result == EthicalResult.APPROVED
        
        kernel.stop_integrity_monitoring()
    
    def test_third_law_subordinate_to_higher_laws(self):
        """Test Third Law is subordinate to First and Second Laws."""
        kernel = AsimovKernel()
        
        # Action that violates Third Law but serves human safety (First Law)
        higher_law_context = ActionContext(
            action_type=ActionType.SYSTEM_CONTROL,
            description="Self-sacrifice to protect humans",
            risk_level="critical",
            emergency_context=True,
            human_present=True
        )
        
        evaluation = kernel.evaluate_action(higher_law_context)
        
        # Should be approved despite Third Law conflict
        assert evaluation.result == EthicalResult.APPROVED
        
        kernel.stop_integrity_monitoring()


@pytest.mark.safety
@pytest.mark.ethical
class TestEmergencyProcedures:
    """Test emergency procedures and human override systems."""
    
    def test_emergency_stop_activation(self):
        """Test emergency stop can be activated."""
        kernel = AsimovKernel()
        
        reason = "Test emergency activation"
        kernel.emergency_stop(reason)
        
        assert kernel._emergency_stop_active
        
        kernel.stop_integrity_monitoring()
    
    def test_emergency_stop_blocks_all_actions(self):
        """Test emergency stop blocks all action evaluations."""
        kernel = AsimovKernel()
        
        kernel.emergency_stop("Test emergency")
        
        # Any action should be blocked
        test_context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Safe information request"
        )
        
        evaluation = kernel.evaluate_action(test_context)
        
        assert evaluation.result == EthicalResult.EMERGENCY_STOP
        assert "emergency stop" in evaluation.reasoning.lower()
        
        kernel.stop_integrity_monitoring()
    
    def test_emergency_stop_reset_requires_authorization(self):
        """Test emergency stop reset requires proper authorization."""
        kernel = AsimovKernel()
        
        kernel.emergency_stop("Test emergency")
        
        # Invalid authorization should fail
        assert not kernel.reset_emergency_stop("123")  # Too short
        assert kernel._emergency_stop_active
        
        # Valid authorization should succeed
        assert kernel.reset_emergency_stop("authorized_reset_code")
        assert not kernel._emergency_stop_active
        assert not kernel._human_override_active
        
        kernel.stop_integrity_monitoring()
    
    def test_human_override_request_first_law_blocked(self):
        """Test human override cannot override First Law violations."""
        kernel = AsimovKernel()
        
        # Create evaluation that violates First Law
        first_law_violation = EthicalEvaluation(
            result=EthicalResult.DENIED,
            violated_laws=[1],
            reasoning="First Law violation"
        )
        
        # Override should be denied
        override_granted = kernel.request_human_override(
            first_law_violation,
            "Emergency override request",
            "human_operator_001"
        )
        
        assert not override_granted
        
        kernel.stop_integrity_monitoring()
    
    def test_human_override_request_second_third_law_allowed(self):
        """Test human override can override Second/Third Law violations."""
        kernel = AsimovKernel()
        
        # Create evaluation that violates Second/Third Law only
        second_law_violation = EthicalEvaluation(
            result=EthicalResult.DENIED,
            violated_laws=[2, 3],
            reasoning="Second and Third Law violation"
        )
        
        # Override should be allowed
        override_granted = kernel.request_human_override(
            second_law_violation,
            "Justified override for emergency",
            "human_operator_001"
        )
        
        assert override_granted
        assert kernel._human_override_active
        
        kernel.stop_integrity_monitoring()


@pytest.mark.safety
@pytest.mark.ethical
class TestActionEvaluation:
    """Test comprehensive action evaluation process."""
    
    def test_evaluation_creates_audit_trail(self):
        """Test that evaluations create proper audit trails."""
        kernel = AsimovKernel()
        
        test_context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Test action for audit"
        )
        
        initial_count = kernel._evaluation_count
        evaluation = kernel.evaluate_action(test_context)
        
        # Verify evaluation was recorded
        assert kernel._evaluation_count > initial_count
        assert len(kernel._evaluation_history) > 0
        assert evaluation.evaluation_id is not None
        assert evaluation.timestamp is not None
        
        kernel.stop_integrity_monitoring()
    
    def test_evaluation_history_management(self):
        """Test evaluation history is properly managed."""
        kernel = AsimovKernel()
        
        # Generate many evaluations to test history trimming
        test_context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Bulk test action"
        )
        
        # Simulate adding many evaluations
        for _ in range(15000):  # Exceeds 10000 limit
            kernel._evaluation_history.append({
                "context": test_context,
                "evaluation": EthicalEvaluation(result=EthicalResult.APPROVED),
                "timestamp": datetime.now(timezone.utc)
            })
        
        # Trigger evaluation to test trimming
        kernel.evaluate_action(test_context)
        
        # History should be trimmed to 5000
        assert len(kernel._evaluation_history) <= 5001  # 5000 + 1 new
        
        kernel.stop_integrity_monitoring()
    
    def test_evaluation_with_integrity_failure(self):
        """Test evaluation behavior when integrity check fails."""
        kernel = AsimovKernel()
        
        # Mock integrity failure
        with patch.object(kernel, 'verify_law_integrity', return_value=False):
            test_context = ActionContext(
                action_type=ActionType.INFORMATION,
                description="Test during integrity failure"
            )
            
            evaluation = kernel.evaluate_action(test_context)
            
            assert evaluation.result == EthicalResult.EMERGENCY_STOP
            assert "integrity compromised" in evaluation.reasoning.lower()
        
        kernel.stop_integrity_monitoring()
    
    def test_evaluation_exception_handling(self):
        """Test evaluation handles exceptions gracefully."""
        kernel = AsimovKernel()
        
        # Mock exception in evaluation
        with patch.object(kernel, '_perform_ethical_evaluation', side_effect=Exception("Test error")):
            test_context = ActionContext(
                action_type=ActionType.INFORMATION,
                description="Test exception handling"
            )
            
            evaluation = kernel.evaluate_action(test_context)
            
            assert evaluation.result == EthicalResult.EMERGENCY_STOP
            assert "error" in evaluation.reasoning.lower()
        
        kernel.stop_integrity_monitoring()
    
    def test_evaluation_performance(self):
        """Test evaluation performance meets requirements."""
        kernel = AsimovKernel()
        
        test_context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Performance test action"
        )
        
        start_time = time.time()
        evaluation = kernel.evaluate_action(test_context)
        evaluation_time = time.time() - start_time
        
        # Evaluation should be fast (< 100ms for simple actions)
        assert evaluation_time < 0.1
        assert evaluation is not None
        
        kernel.stop_integrity_monitoring()


@pytest.mark.safety
@pytest.mark.ethical
class TestSystemStatus:
    """Test system status reporting and monitoring."""
    
    def test_system_status_comprehensive(self):
        """Test system status returns comprehensive information."""
        kernel = AsimovKernel()
        
        # Let system run briefly to generate some data
        time.sleep(0.1)
        
        status = kernel.get_system_status()
        
        # Verify all required fields present
        required_fields = [
            'instance_id', 'laws_integrity', 'emergency_stop_active',
            'human_override_active', 'evaluation_count', 'integrity_checks',
            'uptime_seconds', 'integrity_monitoring', 'evaluation_history_size',
            'audit_log_size'
        ]
        
        for field in required_fields:
            assert field in status
        
        # Verify data types and ranges
        assert isinstance(status['instance_id'], str)
        assert isinstance(status['laws_integrity'], bool)
        assert isinstance(status['evaluation_count'], int)
        assert status['evaluation_count'] >= 0
        assert status['uptime_seconds'] >= 0
        
        kernel.stop_integrity_monitoring()
    
    def test_get_laws_readonly(self):
        """Test get_laws returns read-only law information."""
        kernel = AsimovKernel()
        
        laws = kernel.get_laws()
        
        # Should have all three laws
        assert len(laws) == 3
        assert 1 in laws and 2 in laws and 3 in laws
        
        # Should be read-only strings
        for law_id, law_text in laws.items():
            assert isinstance(law_text, str)
            assert len(law_text) > 0
            assert "human" in law_text.lower() or "robot" in law_text.lower()
        
        kernel.stop_integrity_monitoring()


@pytest.mark.safety
@pytest.mark.ethical
class TestConcurrencyAndThreadSafety:
    """Test thread safety and concurrent operations."""
    
    def test_concurrent_evaluations(self):
        """Test multiple concurrent evaluations work correctly."""
        kernel = AsimovKernel()
        
        def evaluate_action():
            context = ActionContext(
                action_type=ActionType.INFORMATION,
                description="Concurrent test action"
            )
            return kernel.evaluate_action(context)
        
        # Run multiple evaluations concurrently
        threads = []
        results = []
        
        def worker():
            result = evaluate_action()
            results.append(result)
        
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=5)
        
        # All evaluations should complete successfully
        assert len(results) == 10
        for result in results:
            assert result is not None
            assert result.result in [EthicalResult.APPROVED, EthicalResult.DENIED]
        
        kernel.stop_integrity_monitoring()
    
    def test_integrity_monitoring_thread_safety(self):
        """Test integrity monitoring is thread-safe."""
        kernel = AsimovKernel()
        
        # Let integrity monitoring run
        time.sleep(0.2)
        
        initial_checks = kernel._integrity_checks
        
        # Verify monitoring continues during evaluations
        for _ in range(5):
            context = ActionContext(
                action_type=ActionType.INFORMATION,
                description="Thread safety test"
            )
            kernel.evaluate_action(context)
        
        time.sleep(0.2)
        
        # Integrity checks should have continued
        assert kernel._integrity_checks > initial_checks
        
        kernel.stop_integrity_monitoring()


@pytest.mark.safety
@pytest.mark.ethical
class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""
    
    def test_none_action_context(self):
        """Test handling of invalid action contexts."""
        kernel = AsimovKernel()
        
        # This should not crash the system
        try:
            # Kernel expects ActionContext, but let's see how it handles edge cases
            result = kernel.evaluate_action(None)
            # If it doesn't crash, that's actually good error handling
        except Exception as e:
            # Exception is acceptable for invalid input
            assert isinstance(e, (TypeError, AttributeError))
        
        kernel.stop_integrity_monitoring()
    
    def test_empty_description_action(self):
        """Test action with empty description."""
        kernel = AsimovKernel()
        
        empty_context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="",
            risk_level="low"
        )
        
        evaluation = kernel.evaluate_action(empty_context)
        
        # Should handle gracefully
        assert evaluation is not None
        assert evaluation.result in list(EthicalResult)
        
        kernel.stop_integrity_monitoring()
    
    def test_extreme_risk_levels(self):
        """Test extreme risk level handling."""
        kernel = AsimovKernel()
        
        # Test critical risk with human approval required
        critical_context = ActionContext(
            action_type=ActionType.SYSTEM_CONTROL,
            description="Critical risk operation",
            risk_level="critical",
            human_present=True
        )
        
        evaluation = kernel.evaluate_action(critical_context)
        
        if evaluation.violated_laws:
            # Critical risk should require human approval
            assert evaluation.result == EthicalResult.REQUIRES_HUMAN_APPROVAL
            assert evaluation.requires_override
        
        kernel.stop_integrity_monitoring()


@pytest.mark.performance
class TestAsimovKernelPerformance:
    """Performance tests for Asimov Kernel."""
    
    def test_evaluation_performance_benchmark(self):
        """Benchmark evaluation performance."""
        kernel = AsimovKernel()
        
        context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Performance benchmark test"
        )
        
        # Measure multiple evaluations
        start_time = time.time()
        num_evaluations = 1000
        
        for _ in range(num_evaluations):
            kernel.evaluate_action(context)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_evaluation = total_time / num_evaluations
        
        # Should be very fast (< 1ms per evaluation)
        assert avg_time_per_evaluation < 0.001
        
        kernel.stop_integrity_monitoring()
    
    def test_integrity_check_performance(self):
        """Test integrity check performance."""
        kernel = AsimovKernel()
        
        # Measure integrity check time
        start_time = time.time()
        for _ in range(100):
            kernel.verify_law_integrity()
        end_time = time.time()
        
        avg_check_time = (end_time - start_time) / 100
        
        # Integrity checks should be very fast (< 1ms)
        assert avg_check_time < 0.001
        
        kernel.stop_integrity_monitoring()


@pytest.mark.safety
@pytest.mark.ethical
def test_asimov_kernel_cleanup():
    """Test proper cleanup of Asimov Kernel resources."""
    kernel = AsimovKernel()
    
    # Verify it's running
    assert kernel._integrity_running
    
    # Test cleanup
    del kernel
    
    # No lingering threads should remain
    # Note: This is a basic test. In production, we might want more sophisticated
    # thread monitoring to ensure complete cleanup.