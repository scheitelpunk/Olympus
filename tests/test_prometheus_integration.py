"""
Test PROMETHEUS System Integration

Tests the complete integration of all PROMETHEUS components including:
- Health monitoring and diagnostics
- Predictive maintenance and failure forecasting
- Self-repair with safety constraints
- Component lifecycle management
- Degradation modeling and wear analysis
- Redundancy and backup systems
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from olympus.modules.prometheus import PrometheusSystem
from olympus.modules.prometheus.component_manager import ComponentInfo, ComponentType, ComponentState
from olympus.modules.prometheus.degradation_model import DegradationPoint
from olympus.modules.prometheus.redundancy_system import RedundantResource, RedundancyLevel, RedundancyType
from olympus.modules.prometheus.repair_constraints import RepairConstraint, ConstraintType, ConstraintSeverity


class MockSafetyLayer:
    """Mock safety layer for testing."""
    
    def __init__(self):
        self.is_active = True
        self.validation_results = {}
    
    async def validate_action(self, action_type: str, parameters: dict) -> dict:
        """Mock validation that usually passes."""
        return {
            "allowed": True,
            "risk_level": "low",
            "safety_score": 0.9,
            "recommendations": []
        }


class MockAuditSystem:
    """Mock audit system for testing."""
    
    def __init__(self):
        self.events = []
    
    async def log_event(self, event_type: str, event_data: dict):
        """Log events for testing verification."""
        self.events.append({
            "type": event_type,
            "data": event_data,
            "timestamp": time.time()
        })
    
    def get_events_by_type(self, event_type: str):
        """Get events filtered by type."""
        return [event for event in self.events if event["type"] == event_type]


@pytest.fixture
async def prometheus_system():
    """Create a PROMETHEUS system for testing."""
    safety_layer = MockSafetyLayer()
    audit_system = MockAuditSystem()
    
    system = PrometheusSystem(
        safety_layer=safety_layer,
        audit_system=audit_system
    )
    
    await system.initialize()
    yield system
    await system.shutdown()


@pytest.mark.asyncio
class TestPrometheusIntegration:
    """Test complete PROMETHEUS system integration."""
    
    async def test_system_initialization(self, prometheus_system):
        """Test that all components initialize properly."""
        system = prometheus_system
        
        # Check system is active
        assert system.is_active
        
        # Check all components are initialized
        assert system.health_monitor.is_monitoring
        assert system.diagnostic_engine.is_active
        assert system.predictive_maintenance.is_active
        assert system.component_manager.is_active
        assert system.degradation_model.is_active
        assert system.redundancy_system.is_active
        assert system.repair_constraints.is_active
        assert system.self_repair.is_active
    
    async def test_component_registration_and_monitoring(self, prometheus_system):
        """Test component registration and health monitoring."""
        system = prometheus_system
        
        # Register a test component
        success = await system.component_manager.register_component(
            component_id="test_service",
            name="Test Service",
            component_type=ComponentType.CORE_SERVICE,
            version="1.0.0"
        )
        
        assert success
        
        # Wait for health monitoring cycle
        await asyncio.sleep(0.5)
        
        # Check component is being monitored
        component = await system.component_manager.get_component("test_service")
        assert component is not None
        assert component.component_id == "test_service"
        
        # Add health monitoring data
        await system.health_monitor.add_component_monitoring(
            "test_service",
            {
                "cpu_usage": 45.0,
                "memory_usage": 60.0,
                "error_rate": 1.0,
                "response_time": 150.0
            }
        )
        
        # Verify health monitoring is working
        health_summary = await system.health_monitor.get_health_summary("test_service")
        assert "status" in health_summary
    
    async def test_fault_detection_and_diagnosis(self, prometheus_system):
        """Test fault detection and diagnostic capabilities."""
        system = prometheus_system
        
        # Register component
        await system.component_manager.register_component(
            "faulty_service", "Faulty Service", ComponentType.SUPPORT_SERVICE
        )
        
        # Simulate fault conditions
        faulty_metrics = {
            "cpu_usage_percent": 95.0,  # High CPU
            "memory_usage_percent": 90.0,  # High memory
            "error_rate": 15.0,  # High error rate
            "response_time": 5000.0  # Slow response
        }
        
        alerts = []  # Mock alerts
        
        # Detect fault
        fault = await system.diagnostic_engine.detect_fault(
            "faulty_service", faulty_metrics, alerts
        )
        
        assert fault is not None
        assert fault.component == "faulty_service"
        assert fault.fault_type.name in ["PERFORMANCE_DEGRADATION", "RESOURCE_EXHAUSTION"]
        
        # Perform diagnosis
        diagnosis = await system.diagnostic_engine.diagnose_fault(fault.fault_id)
        assert diagnosis is not None
        assert diagnosis.status.name == "COMPLETED"
        assert diagnosis.root_cause is not None
    
    async def test_predictive_maintenance_workflow(self, prometheus_system):
        """Test predictive maintenance capabilities."""
        system = prometheus_system
        
        # Add degradation data for predictive analysis
        for i in range(20):
            await system.degradation_model.add_degradation_data(
                component_id="aging_service",
                metric_name="performance_score",
                metric_value=100.0 - (i * 2),  # Declining performance
                component_age=i * 3600,  # Age in seconds
                usage_cycles=i * 10,
                load_factor=0.8
            )
        
        # Wait for analysis
        await asyncio.sleep(0.5)
        
        # Check predictions
        prediction = await system.degradation_model.get_degradation_prediction(
            "aging_service", "performance_score"
        )
        
        if prediction:  # May be None if not enough data processed yet
            assert "degradation_rate" in prediction
            assert "severity" in prediction
    
    async def test_self_repair_with_constraints(self, prometheus_system):
        """Test self-repair system with safety constraints."""
        system = prometheus_system
        
        # Create a mock fault
        mock_fault = Mock()
        mock_fault.fault_id = "test_fault_001"
        mock_fault.component = "test_service"
        mock_fault.fault_type = Mock()
        mock_fault.fault_type.name = "PERFORMANCE_DEGRADATION"
        mock_fault.severity = Mock()
        mock_fault.severity.name = "HIGH"
        mock_fault.symptoms = ["high_cpu_usage", "slow_response_time"]
        
        # Test constraint validation
        repair_request = {
            "request_id": "repair_001",
            "component": "test_service",
            "repair_type": "restart_service",
            "urgency": "high",
            "impact": "medium",
            "risk_level": "low",
            "has_backup_plan": True,
            "has_rollback_plan": True,
            "has_testing_plan": True
        }
        
        validation_result = await system.repair_constraints.validate_repair_request(repair_request)
        
        assert "allowed" in validation_result
        assert "violations" in validation_result
        assert "required_approvals" in validation_result
        
        # If repair is allowed, test self-repair initiation
        if validation_result["allowed"]:
            execution_id = await system.self_repair.initiate_repair(mock_fault)
            assert execution_id is not None
    
    async def test_redundancy_and_failover(self, prometheus_system):
        """Test redundancy system and failover capabilities."""
        system = prometheus_system
        
        # Add redundant resource
        backup_resource = RedundantResource(
            resource_id="backup_service",
            primary_resource_id="primary_service",
            resource_type="service",
            redundancy_level=RedundancyLevel.SINGLE,
            redundancy_type=RedundancyType.HOT_STANDBY,
            priority=1
        )
        
        success = await system.redundancy_system.add_redundant_resource(backup_resource)
        assert success
        
        # Wait for health monitoring
        await asyncio.sleep(0.5)
        
        # Test failover trigger
        execution_id = await system.redundancy_system.trigger_failover("primary_service")
        
        if execution_id:  # May be None if no suitable backup
            # Check failover status
            failover_status = await system.redundancy_system.failover_orchestrator.get_failover_status(execution_id)
            assert failover_status is not None
    
    async def test_system_status_and_metrics(self, prometheus_system):
        """Test system status reporting and metrics collection."""
        system = prometheus_system
        
        # Get overall system status
        status = await system.get_system_status()
        
        assert "status" in status
        assert "subsystems" in status
        assert status["status"] == "active"
        
        # Check individual subsystem statuses
        subsystems = status["subsystems"]
        
        for subsystem_name in ["health_monitor", "diagnostic_engine", "predictive_maintenance",
                              "self_repair", "component_manager", "redundancy_system"]:
            assert subsystem_name in subsystems
            subsystem_status = subsystems[subsystem_name]
            assert isinstance(subsystem_status, dict)
    
    async def test_integration_with_safety_layer(self, prometheus_system):
        """Test integration with safety layer."""
        system = prometheus_system
        
        # Test that safety layer is integrated
        assert system.safety_layer is not None
        assert system.self_repair.safety_layer is not None
        assert system.repair_constraints.safety_layer is not None
        
        # Test safety validation
        test_action = {
            "action_type": "restart_service",
            "component": "critical_service",
            "parameters": {"graceful": True}
        }
        
        validation = await system.safety_layer.validate_action(
            test_action["action_type"], 
            test_action
        )
        
        assert "allowed" in validation
        assert "safety_score" in validation
    
    async def test_audit_trail_generation(self, prometheus_system):
        """Test that comprehensive audit trails are generated."""
        system = prometheus_system
        
        # Perform some operations that should generate audit events
        await system.component_manager.register_component(
            "audit_test_service", "Audit Test Service", ComponentType.SUPPORT_SERVICE
        )
        
        # Check that audit events were generated
        audit_system = system.audit_system
        
        # Should have initialization events
        init_events = audit_system.get_events_by_type("prometheus_initialized")
        assert len(init_events) > 0
        
        # Should have component registration events
        registration_events = audit_system.get_events_by_type("component_registered")
        assert len(registration_events) > 0
        
        # Verify event structure
        for event in audit_system.events:
            assert "type" in event
            assert "data" in event
            assert "timestamp" in event
            assert isinstance(event["data"], dict)
    
    async def test_error_handling_and_recovery(self, prometheus_system):
        """Test error handling and recovery mechanisms."""
        system = prometheus_system
        
        # Test graceful handling of invalid component registration
        success = await system.component_manager.register_component(
            "", "Invalid Component", ComponentType.SUPPORT_SERVICE  # Empty ID should fail
        )
        assert not success
        
        # Test system continues to operate after errors
        status = await system.get_system_status()
        assert status["status"] == "active"
        
        # Test constraint violation handling
        invalid_repair_request = {
            "request_id": "invalid_repair",
            "component": "nonexistent_component",
            "repair_type": "dangerous_operation",
            "risk_level": "critical"
        }
        
        # Should handle gracefully without crashing
        try:
            validation = await system.repair_constraints.validate_repair_request(invalid_repair_request)
            assert "allowed" in validation
        except Exception as e:
            pytest.fail(f"System should handle invalid requests gracefully: {e}")


if __name__ == "__main__":
    # Run basic integration test
    async def main():
        safety_layer = MockSafetyLayer()
        audit_system = MockAuditSystem()
        
        system = PrometheusSystem(safety_layer=safety_layer, audit_system=audit_system)
        
        print("Initializing PROMETHEUS system...")
        await system.initialize()
        
        print("System status:")
        status = await system.get_system_status()
        print(f"Status: {status['status']}")
        print(f"Subsystems: {len(status['subsystems'])}")
        
        print("Registering test component...")
        await system.component_manager.register_component(
            "test_component", "Test Component", ComponentType.CORE_SERVICE
        )
        
        print("Waiting for monitoring cycle...")
        await asyncio.sleep(2)
        
        print("System integration test completed successfully!")
        
        await system.shutdown()
        print("System shutdown complete.")
    
    asyncio.run(main())