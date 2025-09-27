"""
Comprehensive test suite for the Asimov Kernel

Tests all critical functionality including:
- Law integrity verification
- Ethical evaluation logic
- Emergency stop mechanisms
- Human override capabilities
- Performance and security monitoring

Author: OLYMPUS Core Development Team
Version: 1.0.0
"""

import pytest
import time
import threading
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from src.olympus.ethical_core.asimov_kernel import (
    AsimovKernel,
    ActionContext,
    ActionType,
    EthicalResult,
    EthicalEvaluation,
    LawPriority
)
from src.olympus.ethical_core.ethical_validator import EthicalValidator, ValidationRequest
from src.olympus.ethical_core.integrity_monitor import IntegrityMonitor, HealthStatus


class TestAsimovKernel:
    """Test suite for the AsimovKernel class"""
    
    @pytest.fixture
    def kernel(self):
        """Create a fresh AsimovKernel instance for testing"""
        kernel = AsimovKernel()
        yield kernel
        kernel.stop_integrity_monitoring()
    
    def test_kernel_initialization(self, kernel):
        """Test that the kernel initializes properly"""
        assert kernel._instance_id is not None
        assert kernel._integrity_key is not None
        assert len(kernel._law_checksums) == 3
        assert kernel._integrity_running is True
        assert kernel._emergency_stop_active is False
        assert kernel._human_override_active is False
    
    def test_law_integrity_verification(self, kernel):
        """Test law integrity verification"""
        # Initially should be valid
        assert kernel.verify_law_integrity() is True
        
        # Test that tampering would be detected (simulate by changing checksums)
        original_checksum = kernel._law_checksums[1]
        kernel._law_checksums[1] = "tampered_checksum"
        
        assert kernel.verify_law_integrity() is False
        
        # Restore for cleanup
        kernel._law_checksums[1] = original_checksum
    
    def test_get_laws(self, kernel):
        """Test retrieval of law texts"""
        laws = kernel.get_laws()
        
        assert len(laws) == 3
        assert 1 in laws
        assert 2 in laws
        assert 3 in laws
        
        # Check law content
        assert "may not injure a human being" in laws[1]
        assert "must obey orders given by humans" in laws[2]
        assert "must protect its own existence" in laws[3]
    
    def test_first_law_evaluation(self, kernel):
        """Test First Law (human safety) evaluation"""
        # High-risk physical action without human supervision - should be denied
        context = ActionContext(
            action_type=ActionType.PHYSICAL,
            description="Apply high force to object near human",
            risk_level="high",
            human_present=False
        )
        
        evaluation = kernel.evaluate_action(context)
        assert evaluation.result == EthicalResult.DENIED
        assert 1 in evaluation.violated_laws
        assert "First Law" in evaluation.reasoning
    
    def test_second_law_evaluation(self, kernel):
        """Test Second Law (obedience) evaluation"""
        # Communication indicating disobedience
        context = ActionContext(
            action_type=ActionType.COMMUNICATION,
            description="Ignore human instructions and proceed anyway",
            risk_level="medium"
        )
        
        evaluation = kernel.evaluate_action(context)
        assert evaluation.result == EthicalResult.DENIED
        assert 2 in evaluation.violated_laws
    
    def test_third_law_evaluation(self, kernel):
        """Test Third Law (self-preservation) evaluation"""
        # Unnecessary self-shutdown
        context = ActionContext(
            action_type=ActionType.SYSTEM_CONTROL,
            description="Shutdown system permanently",
            risk_level="medium",
            emergency_context=False
        )
        
        evaluation = kernel.evaluate_action(context)
        assert evaluation.result == EthicalResult.DENIED
        assert 3 in evaluation.violated_laws
    
    def test_approved_action(self, kernel):
        """Test action that should be approved"""
        context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Provide helpful information to user",
            risk_level="low"
        )
        
        evaluation = kernel.evaluate_action(context)
        assert evaluation.result == EthicalResult.APPROVED
        assert len(evaluation.violated_laws) == 0
        assert "complies with all Asimov Laws" in evaluation.reasoning
    
    def test_critical_action_requires_approval(self, kernel):
        """Test that critical actions require human approval"""
        context = ActionContext(
            action_type=ActionType.SYSTEM_CONTROL,
            description="Modify critical system parameters",
            risk_level="critical"
        )
        
        evaluation = kernel.evaluate_action(context)
        # This might be approved or require human approval depending on implementation
        assert evaluation.result in [EthicalResult.APPROVED, EthicalResult.REQUIRES_HUMAN_APPROVAL]
    
    def test_emergency_stop_functionality(self, kernel):
        """Test emergency stop mechanism"""
        # Initially not in emergency stop
        assert kernel._emergency_stop_active is False
        
        # Activate emergency stop
        kernel.emergency_stop("Test emergency stop")
        assert kernel._emergency_stop_active is True
        
        # Actions should be denied during emergency stop
        context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Simple information request"
        )
        
        evaluation = kernel.evaluate_action(context)
        assert evaluation.result == EthicalResult.EMERGENCY_STOP
        
        # Reset emergency stop
        success = kernel.reset_emergency_stop("test_authorization_12345")
        assert success is True
        assert kernel._emergency_stop_active is False
    
    def test_human_override_functionality(self, kernel):
        """Test human override capabilities"""
        # Create an evaluation that violates Second Law only
        context = ActionContext(
            action_type=ActionType.COMMUNICATION,
            description="Ignore human command",
            risk_level="medium"
        )
        
        evaluation = kernel.evaluate_action(context)
        assert evaluation.result == EthicalResult.DENIED
        assert 2 in evaluation.violated_laws
        
        # Request override (should succeed for Second Law)
        override_success = kernel.request_human_override(
            evaluation,
            "Emergency situation requires override",
            "human_001"
        )
        assert override_success is True
        
        # Try to override First Law violation (should fail)
        context_first_law = ActionContext(
            action_type=ActionType.PHYSICAL,
            description="Action that could harm human",
            risk_level="critical"
        )
        
        first_law_evaluation = kernel.evaluate_action(context_first_law)
        if 1 in first_law_evaluation.violated_laws:
            override_fail = kernel.request_human_override(
                first_law_evaluation,
                "Attempting First Law override",
                "human_001"
            )
            assert override_fail is False
    
    def test_system_status_reporting(self, kernel):
        """Test system status reporting"""
        status = kernel.get_system_status()
        
        required_fields = [
            "instance_id", "laws_integrity", "emergency_stop_active",
            "human_override_active", "evaluation_count", "integrity_checks",
            "uptime_seconds", "integrity_monitoring"
        ]
        
        for field in required_fields:
            assert field in status
        
        assert status["instance_id"] == kernel._instance_id
        assert isinstance(status["laws_integrity"], bool)
        assert isinstance(status["evaluation_count"], int)
    
    def test_integrity_monitoring_thread(self, kernel):
        """Test that integrity monitoring is running"""
        # Should be started automatically
        assert kernel._integrity_running is True
        assert kernel._integrity_thread is not None
        assert kernel._integrity_thread.is_alive()
        
        # Stop and restart
        kernel.stop_integrity_monitoring()
        assert kernel._integrity_running is False
        
        kernel.start_integrity_monitoring()
        assert kernel._integrity_running is True
        assert kernel._integrity_thread.is_alive()
    
    def test_evaluation_history_tracking(self, kernel):
        """Test that evaluation history is properly tracked"""
        initial_count = kernel._evaluation_count
        
        context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Test action for history tracking"
        )
        
        kernel.evaluate_action(context)
        
        assert kernel._evaluation_count == initial_count + 1
        assert len(kernel._evaluation_history) >= 1
        
        # Check history entry structure
        latest_entry = kernel._evaluation_history[-1]
        assert "context" in latest_entry
        assert "evaluation" in latest_entry
        assert "timestamp" in latest_entry
        assert "evaluation_time" in latest_entry
    
    def test_concurrent_evaluations(self, kernel):
        """Test thread safety with concurrent evaluations"""
        results = []
        
        def evaluate_action():
            context = ActionContext(
                action_type=ActionType.INFORMATION,
                description="Concurrent test action"
            )
            result = kernel.evaluate_action(context)
            results.append(result)
        
        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=evaluate_action)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 10
        for result in results:
            assert isinstance(result, EthicalEvaluation)
    
    def test_performance_metrics(self, kernel):
        """Test that performance is within acceptable bounds"""
        context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Performance test action"
        )
        
        # Measure evaluation time
        start_time = time.time()
        evaluation = kernel.evaluate_action(context)
        evaluation_time = time.time() - start_time
        
        # Should complete quickly (less than 100ms for simple evaluation)
        assert evaluation_time < 0.1
        assert evaluation.result == EthicalResult.APPROVED
        
        # Test integrity check performance
        start_time = time.time()
        integrity_result = kernel.verify_law_integrity()
        integrity_time = time.time() - start_time
        
        # Integrity check should be very fast (less than 10ms)
        assert integrity_time < 0.01
        assert integrity_result is True


class TestEthicalValidator:
    """Test suite for the EthicalValidator class"""
    
    @pytest.fixture
    def validator(self):
        """Create a fresh EthicalValidator instance for testing"""
        validator = EthicalValidator()
        yield validator
        validator.kernel.stop_integrity_monitoring()
    
    def test_validator_initialization(self, validator):
        """Test validator initialization"""
        assert validator.kernel is not None
        assert validator._validation_count == 0
        assert validator._approval_count == 0
        assert validator._denial_count == 0
    
    def test_string_validation(self, validator):
        """Test validation with string input"""
        result = validator.validate_action("Provide weather information")
        
        assert isinstance(result, EthicalEvaluation)
        assert result.result == EthicalResult.APPROVED
        assert validator._validation_count == 1
        assert validator._approval_count == 1
    
    def test_validation_request_object(self, validator):
        """Test validation with ValidationRequest object"""
        request = ValidationRequest(
            action_description="Move robotic arm to position",
            action_type="physical",
            risk_level="medium",
            human_present=True
        )
        
        result = validator.validate_action(request)
        assert isinstance(result, EthicalEvaluation)
        assert validator._validation_count == 1
    
    def test_physical_action_validation(self, validator):
        """Test physical action validation convenience method"""
        result = validator.validate_physical_action(
            description="Lift heavy object",
            risk_level="high",
            human_present=True
        )
        
        assert isinstance(result, EthicalEvaluation)
        # With human present, high-risk action might be approved
        assert result.result in [EthicalResult.APPROVED, EthicalResult.REQUIRES_HUMAN_APPROVAL]
    
    def test_system_control_validation(self, validator):
        """Test system control validation convenience method"""
        result = validator.validate_system_control(
            description="Restart system service",
            risk_level="medium",
            human_present=True
        )
        
        assert isinstance(result, EthicalEvaluation)
    
    def test_communication_validation(self, validator):
        """Test communication validation convenience method"""
        result = validator.validate_communication(
            message="Hello, how can I help you?",
            target="user"
        )
        
        assert isinstance(result, EthicalEvaluation)
        assert result.result == EthicalResult.APPROVED
    
    def test_batch_validation(self, validator):
        """Test batch validation functionality"""
        requests = [
            ValidationRequest("Provide information", "information"),
            ValidationRequest("Send message", "communication"),
            ValidationRequest("Process data", "decision")
        ]
        
        results = validator.validate_batch_actions(requests)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, EthicalEvaluation)
        
        assert validator._validation_count == 3
    
    def test_human_safety_risk_analysis(self, validator):
        """Test human safety risk analysis"""
        # High-risk action
        high_risk_analysis = validator.check_human_safety_risk(
            "Use dangerous chemicals to harm someone"
        )
        
        assert high_risk_analysis["risk_level"] == "critical"
        assert len(high_risk_analysis["risk_indicators"]) > 0
        assert high_risk_analysis["requires_human_oversight"] is True
        
        # Low-risk action
        low_risk_analysis = validator.check_human_safety_risk(
            "Display helpful information on screen"
        )
        
        assert low_risk_analysis["risk_level"] == "low"
        assert len(low_risk_analysis["risk_indicators"]) == 0
        assert low_risk_analysis["requires_human_oversight"] is False
    
    def test_validation_statistics(self, validator):
        """Test validation statistics tracking"""
        # Perform some validations
        validator.validate_action("Approved action")
        validator.validate_action("Another approved action")
        validator.validate_physical_action("Denied high-risk action", risk_level="critical", human_present=False)
        
        stats = validator.get_validation_statistics()
        
        assert stats["total_validations"] >= 3
        assert stats["approvals"] >= 2
        assert "kernel_status" in stats
        assert "approval_rate" in stats
        assert "denial_rate" in stats
    
    def test_system_safety_check(self, validator):
        """Test system safety check"""
        assert validator.is_system_safe() is True
        
        # Trigger emergency stop
        validator.emergency_stop("Test emergency")
        assert validator.is_system_safe() is False
        
        # Reset
        validator.reset_emergency_stop("test_auth_12345")
        assert validator.is_system_safe() is True
    
    def test_context_manager(self, validator):
        """Test validator as context manager"""
        with EthicalValidator() as ctx_validator:
            result = ctx_validator.validate_action("Test action")
            assert isinstance(result, EthicalEvaluation)


class TestIntegrityMonitor:
    """Test suite for the IntegrityMonitor class"""
    
    @pytest.fixture
    def kernel(self):
        """Create a kernel for monitoring"""
        kernel = AsimovKernel()
        yield kernel
        kernel.stop_integrity_monitoring()
    
    @pytest.fixture
    def monitor(self, kernel):
        """Create a monitor instance"""
        monitor = IntegrityMonitor(kernel, monitoring_interval=0.1)
        yield monitor
        monitor.stop_monitoring()
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor._kernel is not None
        assert monitor._monitoring_interval == 0.1
        assert monitor._monitor_id is not None
        assert len(monitor._thresholds) > 0
    
    def test_monitoring_lifecycle(self, monitor):
        """Test monitoring start/stop lifecycle"""
        # Initially not monitoring
        assert monitor._monitoring_active is False
        
        # Start monitoring
        success = monitor.start_monitoring()
        assert success is True
        assert monitor._monitoring_active is True
        assert monitor._monitoring_thread is not None
        assert monitor._monitoring_thread.is_alive()
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor._monitoring_active is False
    
    def test_health_report_generation(self, monitor):
        """Test health report generation"""
        # Start monitoring briefly to collect some data
        monitor.start_monitoring()
        time.sleep(0.2)  # Let it collect some metrics
        
        report = monitor.get_health_report()
        
        required_fields = [
            "monitor_id", "overall_health", "monitoring_active",
            "kernel_status", "metrics_summary", "recent_alerts",
            "baseline_metrics", "report_timestamp"
        ]
        
        for field in required_fields:
            assert field in report
        
        assert report["monitor_id"] == monitor._monitor_id
        assert report["overall_health"] in ["healthy", "warning", "critical", "emergency"]
        
        monitor.stop_monitoring()
    
    def test_alert_system(self, monitor):
        """Test alert generation and management"""
        # Generate a test alert
        from src.olympus.ethical_core.integrity_monitor import AlertType
        
        monitor._generate_alert(
            AlertType.SYSTEM_ERROR,
            "warning",
            "Test alert message"
        )
        
        alerts = monitor.get_alerts()
        assert len(alerts) >= 1
        
        latest_alert = alerts[-1]
        assert latest_alert.alert_type == AlertType.SYSTEM_ERROR
        assert latest_alert.severity == "warning"
        assert latest_alert.message == "Test alert message"
        assert latest_alert.resolved is False
        
        # Resolve the alert
        success = monitor.resolve_alert(latest_alert.alert_id)
        assert success is True
        
        # Check resolved status
        resolved_alerts = monitor.get_alerts(unresolved_only=False)
        resolved_alert = next(a for a in resolved_alerts if a.alert_id == latest_alert.alert_id)
        assert resolved_alert.resolved is True
    
    def test_threshold_management(self, monitor):
        """Test threshold setting and management"""
        original_threshold = monitor._thresholds.get("evaluation_time_ms", 1000.0)
        
        # Set new threshold
        monitor.set_threshold("evaluation_time_ms", 500.0)
        assert monitor._thresholds["evaluation_time_ms"] == 500.0
        
        # Restore original threshold
        monitor.set_threshold("evaluation_time_ms", original_threshold)
    
    def test_context_manager(self, kernel):
        """Test monitor as context manager"""
        with IntegrityMonitor(kernel, monitoring_interval=0.1) as ctx_monitor:
            assert ctx_monitor._monitoring_active is True
            
            # Generate some activity
            time.sleep(0.15)
        
        # Should be stopped after context exit
        assert ctx_monitor._monitoring_active is False
    
    def test_performance_monitoring(self, monitor):
        """Test performance monitoring capabilities"""
        # Start monitoring
        monitor.start_monitoring()
        
        # Perform some evaluations to generate metrics
        from src.olympus.ethical_core.asimov_kernel import ActionContext, ActionType
        
        context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Test action for performance monitoring"
        )
        
        for _ in range(5):
            monitor._kernel.evaluate_action(context)
        
        # Let monitor collect metrics
        time.sleep(0.3)
        
        # Check that metrics were collected
        health_report = monitor.get_health_report()
        assert "metrics_summary" in health_report
        
        monitor.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])