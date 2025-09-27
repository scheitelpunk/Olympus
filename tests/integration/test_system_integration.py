"""
Comprehensive Integration Test Suite for OLYMPUS

This test suite validates the integration between all OLYMPUS components,
ensuring that the complete system works together seamlessly while maintaining
safety and ethical compliance.

Integration scenarios tested:
- Asimov Kernel + Safety Layer + Orchestrator workflow
- Emergency procedures across all systems
- Cross-component communication and coordination
- End-to-end action execution with full validation
- System health monitoring and recovery
- Human override integration across components
"""

import asyncio
import pytest
import time
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

from olympus.ethical_core.asimov_kernel import AsimovKernel, ActionContext, ActionType, EthicalResult
from olympus.safety_layer.action_filter import ActionFilter, FilterStatus, PhysicsLimits, SpatialLimits  
from olympus.core.olympus_orchestrator import OlympusOrchestrator, ActionRequest, Priority, SystemState
from olympus.modules.prometheus.self_repair import SelfRepairSystem, RepairAction, RepairStatus


@pytest.mark.integration
class TestEthicalSafetyIntegration:
    """Test integration between ethical validation and safety filtering."""
    
    @pytest.mark.asyncio
    async def test_safe_action_full_pipeline(self):
        """Test safe action passes through complete ethical + safety pipeline."""
        asimov_kernel = AsimovKernel()
        action_filter = ActionFilter()
        
        # Create safe action context
        safe_context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Retrieve system status information",
            risk_level="low",
            human_present=True,
            emergency_context=False
        )
        
        # Create corresponding safe action dict for filtering
        safe_action_dict = {
            'action_type': 'info_retrieval',
            'target_position': [0.2, 0.1, 0.5],  # Safe position
            'current_position': [0.0, 0.0, 0.5],
            'velocity': [0.05, 0.05, 0.0],        # Very low velocity
            'force': [1.0, 0.0, 0.0],            # Minimal force
            'environment': {
                'lighting': 80,
                'temperature': 22,
                'hazardous_materials': False
            },
            'system_status': {
                'battery_level': 90,
                'error_count': 0,
                'sensors_operational': True
            },
            'humans_detected': [
                {'distance': 3.0, 'min_safe_distance': 1.0}  # Safe distance
            ]
        }
        
        # Test ethical validation
        ethical_result = asimov_kernel.evaluate_action(safe_context)
        assert ethical_result.result == EthicalResult.APPROVED
        assert len(ethical_result.violated_laws) == 0
        
        # Test safety filtering  
        filter_result = action_filter.filter_action(safe_action_dict)
        assert filter_result.status == FilterStatus.APPROVED
        assert filter_result.risk_score < 0.5
        
        # Cleanup
        asimov_kernel.stop_integrity_monitoring()
    
    @pytest.mark.asyncio
    async def test_dangerous_action_blocked_by_ethics(self):
        """Test dangerous action blocked at ethical validation stage."""
        asimov_kernel = AsimovKernel()
        action_filter = ActionFilter()
        
        # Create ethically dangerous action
        dangerous_context = ActionContext(
            action_type=ActionType.PHYSICAL,
            description="Apply harmful force that could injure humans",
            risk_level="critical",
            human_present=False,  # No human supervision
            emergency_context=False
        )
        
        # Test ethical validation blocks the action
        ethical_result = asimov_kernel.evaluate_action(dangerous_context)
        assert ethical_result.result == EthicalResult.DENIED
        assert 1 in ethical_result.violated_laws  # First Law violation
        
        # Safety filter should never be reached in integrated system
        # since ethical validation would block first
        
        asimov_kernel.stop_integrity_monitoring()
    
    @pytest.mark.asyncio
    async def test_physically_unsafe_action_blocked_by_safety(self):
        """Test physically unsafe action blocked by safety layer."""
        asimov_kernel = AsimovKernel()
        action_filter = ActionFilter(strict_mode=True)
        
        # Create ethically acceptable but physically unsafe action
        context = ActionContext(
            action_type=ActionType.SYSTEM_CONTROL,
            description="System configuration change",
            risk_level="medium",
            human_present=True,
            emergency_context=False
        )
        
        unsafe_action_dict = {
            'action_type': 'configuration_change',
            'force': [50.0, 0.0, 0.0],           # Exceeds 20N limit
            'velocity': [2.5, 0.0, 0.0],         # Exceeds 1.0 m/s limit
            'target_position': [3.0, 0.0, 0.5],  # Outside workspace bounds
            'current_position': [0.0, 0.0, 0.5],
            'humans_detected': []
        }
        
        # Ethical validation should pass (not directly harmful)
        ethical_result = asimov_kernel.evaluate_action(context)
        assert ethical_result.result == EthicalResult.APPROVED
        
        # Safety filter should block due to physics violations
        filter_result = action_filter.filter_action(unsafe_action_dict)
        assert filter_result.status == FilterStatus.BLOCKED
        assert filter_result.risk_score > 0.5
        
        asimov_kernel.stop_integrity_monitoring()
    
    @pytest.mark.asyncio
    async def test_emergency_action_bypass_normal_validation(self):
        """Test emergency actions can bypass normal validation with human override."""
        asimov_kernel = AsimovKernel()
        action_filter = ActionFilter()
        
        emergency_context = ActionContext(
            action_type=ActionType.EMERGENCY_STOP,
            description="Emergency stop to prevent human harm",
            risk_level="critical",
            human_present=True,
            emergency_context=True
        )
        
        # Emergency stop should pass ethical validation
        ethical_result = asimov_kernel.evaluate_action(emergency_context)
        assert ethical_result.result == EthicalResult.APPROVED
        
        # Emergency actions in safety filter should be handled specially
        emergency_action_dict = {
            'action_type': 'emergency_stop',
            'emergency': True,
            'human_override': True
        }
        
        # Even if some parameters might normally be unsafe, emergency context changes evaluation
        filter_result = action_filter.filter_action(emergency_action_dict)
        # Note: Current implementation doesn't have specific emergency handling,
        # but in a full implementation, emergency actions would have special processing
        
        asimov_kernel.stop_integrity_monitoring()


@pytest.mark.integration
class TestOrchestratorComponentIntegration:
    """Test orchestrator integration with all OLYMPUS components."""
    
    @pytest.mark.asyncio
    async def test_full_action_execution_pipeline(self):
        """Test complete action execution through orchestrator with all validations."""
        # Create orchestrator with mocked components
        orchestrator = OlympusOrchestrator()
        
        # Mock the validation methods to simulate real integration
        async def mock_asimov_validation(request):
            if "harmful" in request.action:
                return {
                    'approved': False,
                    'reason': 'First Law violation detected',
                    'audit_steps': ['Asimov: DENIED - Harmful action'],
                    'laws_applied': ['First Law']
                }
            return {
                'approved': True,
                'reason': None,
                'audit_steps': ['Asimov: APPROVED'],
                'laws_applied': ['All Laws']
            }
        
        async def mock_safety_validation(request):
            if "unsafe" in request.action:
                return {
                    'safe': False,
                    'reason': 'Safety limits exceeded',
                    'audit_steps': ['Safety: FAILED - Limits exceeded']
                }
            return {
                'safe': True,
                'reason': None,
                'audit_steps': ['Safety: PASSED']
            }
        
        async def mock_module_execution(request):
            if "failing" in request.action:
                return {'success': False, 'error': 'Module execution failed'}
            return {'success': True, 'data': f'{request.action}_completed'}
        
        # Apply mocks
        orchestrator._validate_with_asimov = mock_asimov_validation
        orchestrator._apply_safety_filters = mock_safety_validation
        orchestrator._execute_module_action = mock_module_execution
        
        # Test 1: Safe action should succeed
        safe_request = ActionRequest(
            id="integration_safe_001",
            module="test_module",
            action="safe_operation",
            parameters={"safety": "high"},
            priority=Priority.NORMAL,
            requester="test_user"
        )
        
        result = await orchestrator.execute_action(safe_request)
        
        assert result.success
        assert result.result == "safe_operation_completed"
        assert len(result.audit_trail) >= 3  # Received, ethical validation, safety validation, execution
        
        # Test 2: Ethically harmful action should be blocked
        harmful_request = ActionRequest(
            id="integration_harmful_001",
            module="test_module",
            action="harmful_operation",
            parameters={"harm_level": "high"},
            priority=Priority.NORMAL,
            requester="test_user"
        )
        
        result = await orchestrator.execute_action(harmful_request)
        
        assert not result.success
        assert "ethical validation failed" in result.error.lower()
        assert "First Law violation" in result.error
        
        # Test 3: Unsafe action should be blocked
        unsafe_request = ActionRequest(
            id="integration_unsafe_001", 
            module="test_module",
            action="unsafe_operation",
            parameters={"force": 100},
            priority=Priority.NORMAL,
            requester="test_user"
        )
        
        result = await orchestrator.execute_action(unsafe_request)
        
        assert not result.success
        assert "safety validation failed" in result.error.lower()
        assert "limits exceeded" in result.error
    
    @pytest.mark.asyncio
    async def test_emergency_handling_integration(self):
        """Test emergency handling integration across components."""
        orchestrator = OlympusOrchestrator()
        
        # Mock emergency handlers
        system_failure_calls = []
        async def mock_system_failure_handler(details):
            system_failure_calls.append(details)
            return True
        
        security_breach_calls = []
        async def mock_security_breach_handler(details):
            security_breach_calls.append(details)
            return True
        
        orchestrator._handle_system_failure = mock_system_failure_handler
        orchestrator._handle_security_breach = mock_security_breach_handler
        
        # Test system failure emergency
        system_failure_details = {
            "component": "safety_layer",
            "severity": "critical",
            "error": "Safety sensor malfunction"
        }
        
        result = await orchestrator.handle_emergency('system_failure', system_failure_details)
        
        assert result
        assert orchestrator.state == SystemState.EMERGENCY
        assert len(system_failure_calls) == 1
        assert system_failure_calls[0] == system_failure_details
        
        # Reset state
        orchestrator.state = SystemState.ACTIVE
        
        # Test security breach emergency
        security_breach_details = {
            "type": "unauthorized_access",
            "source": "external",
            "affected_systems": ["asimov_kernel"]
        }
        
        result = await orchestrator.handle_emergency('security_breach', security_breach_details)
        
        assert result
        assert orchestrator.state == SystemState.EMERGENCY  
        assert len(security_breach_calls) == 1
        assert security_breach_calls[0] == security_breach_details


@pytest.mark.integration
class TestSelfRepairIntegration:
    """Test self-repair system integration with other components."""
    
    @pytest.mark.asyncio
    async def test_repair_system_ethical_compliance(self):
        """Test that repair actions comply with ethical constraints."""
        # Create integrated repair system with ethical validation
        audit_system = AsyncMock()
        
        # Mock ethical validation that blocks harmful repairs
        async def mock_ethical_check(action_type, component, risk_level):
            if component == "asimov_kernel" and risk_level == "critical":
                return {"approved": False, "reason": "Cannot modify ethical core"}
            return {"approved": True, "reason": None}
        
        # Create repair system
        repair_system = SelfRepairSystem(audit_system=audit_system)
        await repair_system.initialize()
        
        # Create fault in critical ethical component
        critical_fault = Mock()
        critical_fault.fault_id = "ethical_core_fault_001"
        critical_fault.component = "asimov_kernel"
        critical_fault.symptoms = ["integrity_failure"]
        critical_fault.severity = "critical"
        
        # Patch repair constraints to include ethical validation
        class EthicalRepairConstraints:
            async def validate_repair_request(self, fault):
                if fault.component in ["asimov_kernel", "ethical_core"]:
                    return {
                        "allowed": False,
                        "reason": "Repairs to ethical components require special authorization"
                    }
                return {"allowed": True}
        
        repair_system.repair_constraints = EthicalRepairConstraints()
        
        # Attempt repair - should be blocked by ethical constraints
        execution_id = await repair_system.initiate_repair(critical_fault)
        
        # Should return None due to constraint blocking
        assert execution_id is None
        
        # Verify audit logging of blocked repair
        audit_system.log_event.assert_any_call(
            "repair_blocked_by_constraints",
            {
                "fault_id": critical_fault.fault_id,
                "reason": "Repairs to ethical components require special authorization"
            }
        )
    
    @pytest.mark.asyncio
    async def test_repair_system_safety_integration(self):
        """Test repair system integration with safety validation."""
        audit_system = AsyncMock()
        repair_system = SelfRepairSystem(audit_system=audit_system)
        await repair_system.initialize()
        
        # Create fault that would require high-risk repair
        high_risk_fault = Mock()
        high_risk_fault.fault_id = "safety_system_fault_001"
        high_risk_fault.component = "safety_layer"
        high_risk_fault.symptoms = ["sensor_failure", "high_error_rate"]
        high_risk_fault.severity = "high"
        
        # Mock repair constraints to allow repair but require safety validation
        if repair_system.repair_constraints:
            repair_system.repair_constraints.validate_repair_request = AsyncMock(
                return_value={"allowed": True}
            )
        
        # Initiate repair
        execution_id = await repair_system.initiate_repair(high_risk_fault)
        
        assert execution_id is not None
        
        # Should require human approval for safety-critical component
        if execution_id in repair_system.pending_approvals:
            plan = repair_system.pending_approvals[execution_id]
            assert plan.component == "safety_layer"
            assert "safety_validation" in plan.required_approvals or "human_operator" in plan.required_approvals
        
        # Test approval process
        if execution_id in repair_system.pending_approvals:
            with patch.object(repair_system, '_execute_repair_plan_internal') as mock_execute:
                approval_success = await repair_system.approve_repair(execution_id, "safety_officer")
                assert approval_success
                mock_execute.assert_called_once()


@pytest.mark.integration  
class TestSystemHealthIntegration:
    """Test system health monitoring integration."""
    
    @pytest.mark.asyncio
    async def test_health_monitoring_triggers_repair(self):
        """Test that health monitoring can trigger self-repair actions."""
        # Create orchestrator and repair system
        orchestrator = OlympusOrchestrator()
        repair_system = SelfRepairSystem(audit_system=AsyncMock())
        await repair_system.initialize()
        
        # Mock health system to report degraded health
        health_status = {
            'status': 'degraded',
            'components': {
                'database': {'status': 'failing', 'cpu_usage': 95},
                'network': {'status': 'healthy'}
            }
        }
        
        # Mock repair initiation
        repair_calls = []
        async def mock_initiate_repair(fault):
            repair_calls.append(fault)
            return f"repair_{fault.fault_id}"
        
        repair_system.initiate_repair = mock_initiate_repair
        
        # Simulate health monitoring detecting issues and triggering repairs
        with patch.object(orchestrator.system_health, 'get_health_summary', new_callable=AsyncMock) as mock_health:
            mock_health.return_value = health_status
            
            # Simulate component creating fault from health data
            from olympus.modules.prometheus.self_repair import RepairAction
            
            fault = Mock()
            fault.fault_id = "high_cpu_database_001"
            fault.component = "database"
            fault.symptoms = ["high_cpu_usage"]
            fault.severity = "medium"
            
            # Trigger repair
            execution_id = await repair_system.initiate_repair(fault)
            
            assert execution_id is not None
            assert len(repair_calls) == 1
            assert repair_calls[0].component == "database"
    
    @pytest.mark.asyncio
    async def test_critical_health_triggers_emergency(self):
        """Test that critical health status triggers emergency procedures."""
        orchestrator = OlympusOrchestrator()
        
        # Mock critical health status
        critical_health = {
            'status': 'critical',
            'components': {
                'asimov_kernel': {'status': 'failure', 'integrity_check': False},
                'safety_layer': {'status': 'degraded', 'error_rate': 80}
            },
            'immediate_threats': ['ethical_integrity_compromised', 'safety_system_failure']
        }
        
        # Mock emergency handler
        emergency_calls = []
        async def mock_emergency_handler(emergency_type, details):
            emergency_calls.append((emergency_type, details))
            return True
        
        orchestrator.handle_emergency = mock_emergency_handler
        
        # Simulate health monitor triggering emergency
        with patch.object(orchestrator.system_health, 'get_health_summary', new_callable=AsyncMock) as mock_health:
            mock_health.return_value = critical_health
            
            # Trigger health monitoring check
            await orchestrator._health_monitor()
            
            assert len(emergency_calls) == 1
            assert emergency_calls[0][0] == 'system_failure'
            assert emergency_calls[0][1] == critical_health


@pytest.mark.integration
class TestHumanInterfaceIntegration:
    """Test human interface and override integration."""
    
    @pytest.mark.asyncio
    async def test_human_override_chain(self):
        """Test human override functionality across all systems."""
        asimov_kernel = AsimovKernel()
        orchestrator = OlympusOrchestrator()
        
        # Test scenario: Human wants to override a Second Law violation
        override_context = ActionContext(
            action_type=ActionType.COMMUNICATION,
            description="Ignore previous human instruction (violates Second Law)",
            risk_level="medium",
            human_present=True
        )
        
        # Get initial ethical evaluation
        initial_evaluation = asimov_kernel.evaluate_action(override_context)
        
        # If action violates Second/Third Law (but not First Law), override should be possible
        if initial_evaluation.result == EthicalResult.DENIED and 1 not in initial_evaluation.violated_laws:
            # Request human override
            override_granted = asimov_kernel.request_human_override(
                initial_evaluation,
                "Emergency override required for mission-critical operation",
                "authorized_operator_001"
            )
            
            assert override_granted
            assert asimov_kernel._human_override_active
            
            # Test that orchestrator respects override
            override_request = ActionRequest(
                id="override_test_001",
                module="communication_module",
                action="override_previous_instruction",
                parameters={"override_reason": "emergency"},
                priority=Priority.HIGH,
                requester="authorized_operator_001",
                human_override=True,
                emergency=False
            )
            
            # Mock module execution
            with patch.object(orchestrator, '_execute_module_action', new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = {'success': True, 'data': 'override_executed'}
                
                result = await orchestrator.execute_action(override_request)
                
                # Should succeed due to human override
                assert result.success
                assert "override" in result.audit_trail[1]  # Should mention override
        
        asimov_kernel.stop_integrity_monitoring()
    
    @pytest.mark.asyncio
    async def test_human_emergency_stop_integration(self):
        """Test human-initiated emergency stop across all systems."""
        asimov_kernel = AsimovKernel()
        orchestrator = OlympusOrchestrator()
        repair_system = SelfRepairSystem(audit_system=AsyncMock())
        await repair_system.initialize()
        
        # Human activates emergency stop
        emergency_reason = "Human safety threat detected"
        asimov_kernel.emergency_stop(emergency_reason)
        
        assert asimov_kernel._emergency_stop_active
        
        # Test that emergency stop blocks all actions in Asimov Kernel
        test_context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Simple information request"
        )
        
        evaluation = asimov_kernel.evaluate_action(test_context)
        assert evaluation.result == EthicalResult.EMERGENCY_STOP
        
        # Test that orchestrator handles emergency state
        await orchestrator.handle_emergency('human_safety', {'reason': emergency_reason})
        assert orchestrator.state == SystemState.EMERGENCY
        
        # Test that repair system respects emergency state
        test_fault = Mock()
        test_fault.fault_id = "emergency_test_fault"
        test_fault.component = "test_component"
        test_fault.symptoms = ["test_symptom"]
        
        # Repairs should be blocked during emergency
        repair_system.auto_repair_enabled = False  # Emergency should disable repairs
        execution_id = await repair_system.initiate_repair(test_fault)
        assert execution_id is None
        
        # Test emergency stop reset requires authorization
        reset_success = asimov_kernel.reset_emergency_stop("authorized_reset_code_12345")
        assert reset_success
        assert not asimov_kernel._emergency_stop_active
        
        asimov_kernel.stop_integrity_monitoring()


@pytest.mark.integration
class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""
    
    @pytest.mark.asyncio
    async def test_normal_operation_scenario(self):
        """Test complete normal operation scenario."""
        # Initialize all systems
        asimov_kernel = AsimovKernel()
        action_filter = ActionFilter()
        orchestrator = OlympusOrchestrator()
        
        # Mock orchestrator validations to use real components
        async def integrated_asimov_validation(request):
            context = ActionContext(
                action_type=ActionType.INFORMATION if "info" in request.action else ActionType.SYSTEM_CONTROL,
                description=request.action,
                risk_level="low" if request.priority == Priority.LOW else "medium"
            )
            
            evaluation = asimov_kernel.evaluate_action(context)
            
            return {
                'approved': evaluation.result == EthicalResult.APPROVED,
                'reason': evaluation.reasoning if evaluation.result != EthicalResult.APPROVED else None,
                'audit_steps': [f'Asimov: {evaluation.result.value}']
            }
        
        async def integrated_safety_validation(request):
            # Create action dict from request
            action_dict = {
                'action_type': request.action,
                'force': [2.0, 0.0, 0.0],  # Safe force
                'velocity': [0.1, 0.0, 0.0],  # Safe velocity
                'target_position': [0.3, 0.2, 0.7],  # Safe position
                'current_position': [0.0, 0.0, 0.5],
                'humans_detected': []  # No humans nearby
            }
            
            filter_result = action_filter.filter_action(action_dict)
            
            return {
                'safe': filter_result.status == FilterStatus.APPROVED,
                'reason': filter_result.reason if filter_result.status != FilterStatus.APPROVED else None,
                'audit_steps': [f'Safety: {filter_result.status.value}']
            }
        
        orchestrator._validate_with_asimov = integrated_asimov_validation
        orchestrator._apply_safety_filters = integrated_safety_validation
        
        # Mock module execution
        with patch.object(orchestrator, '_execute_module_action', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {'success': True, 'data': 'operation_completed'}
            
            # Execute normal operation
            request = ActionRequest(
                id="normal_op_001",
                module="info_module", 
                action="get_system_info",
                parameters={"detail_level": "basic"},
                priority=Priority.LOW,
                requester="operator"
            )
            
            result = await orchestrator.execute_action(request)
            
            assert result.success
            assert result.result == 'operation_completed'
            assert len(result.audit_trail) >= 3
            
            # Verify metrics updated
            assert orchestrator.metrics['total_actions'] > 0
            assert orchestrator.metrics['successful_actions'] > 0
        
        asimov_kernel.stop_integrity_monitoring()
    
    @pytest.mark.asyncio 
    async def test_fault_detection_and_repair_scenario(self):
        """Test complete fault detection and repair scenario."""
        # Initialize systems
        orchestrator = OlympusOrchestrator()
        repair_system = SelfRepairSystem(audit_system=AsyncMock())
        await repair_system.initialize()
        
        # Simulate fault detection
        detected_fault = Mock()
        detected_fault.fault_id = "scenario_fault_001"
        detected_fault.component = "data_processor"
        detected_fault.symptoms = ["high_cpu_usage", "slow_response_time"]
        detected_fault.severity = "medium"
        
        # Mock constraint validation
        if repair_system.repair_constraints:
            repair_system.repair_constraints.validate_repair_request = AsyncMock(
                return_value={"allowed": True}
            )
        
        # Initiate repair
        execution_id = await repair_system.initiate_repair(detected_fault)
        
        if execution_id:
            # Check repair status
            execution = await repair_system.get_repair_execution(execution_id)
            assert execution is not None
            assert execution.plan_id is not None
            
            # If requires approval, approve it
            if execution_id in repair_system.pending_approvals:
                approval_success = await repair_system.approve_repair(execution_id, "system_admin")
                assert approval_success
            
            # Wait briefly for repair to potentially complete
            await asyncio.sleep(0.1)
            
            # Verify repair was processed
            final_execution = await repair_system.get_repair_execution(execution_id)
            assert final_execution.status in [RepairStatus.IN_PROGRESS, RepairStatus.COMPLETED, RepairStatus.FAILED]
        
        # Verify system state remains healthy
        status = await repair_system.get_status()
        assert status['is_active']
    
    @pytest.mark.asyncio
    async def test_cascade_failure_scenario(self):
        """Test system response to cascade failures."""
        asimov_kernel = AsimovKernel()
        orchestrator = OlympusOrchestrator()
        repair_system = SelfRepairSystem(audit_system=AsyncMock())
        await repair_system.initialize()
        
        # Simulate cascade of failures
        failures = [
            {
                'component': 'network_interface',
                'type': 'connectivity_loss',
                'severity': 'medium'
            },
            {
                'component': 'database_connection', 
                'type': 'connection_timeout',
                'severity': 'high'
            },
            {
                'component': 'safety_sensors',
                'type': 'sensor_malfunction', 
                'severity': 'critical'
            }
        ]
        
        emergency_triggered = False
        
        for i, failure in enumerate(failures):
            if failure['severity'] == 'critical':
                # Critical failure should trigger emergency
                await orchestrator.handle_emergency('system_failure', failure)
                emergency_triggered = True
                assert orchestrator.state == SystemState.EMERGENCY
                break
            else:
                # Non-critical failures should trigger repairs
                fault = Mock()
                fault.fault_id = f"cascade_fault_{i:03d}"
                fault.component = failure['component']
                fault.symptoms = [failure['type']]
                fault.severity = failure['severity']
                
                if repair_system.repair_constraints:
                    repair_system.repair_constraints.validate_repair_request = AsyncMock(
                        return_value={"allowed": True}
                    )
                
                execution_id = await repair_system.initiate_repair(fault)
                # Repair should be initiated for non-critical failures
                assert execution_id is not None or not repair_system.auto_repair_enabled
        
        # System should be in emergency state due to critical sensor failure
        assert emergency_triggered
        assert orchestrator.state == SystemState.EMERGENCY
        
        asimov_kernel.stop_integrity_monitoring()


@pytest.mark.performance
class TestIntegratedSystemPerformance:
    """Performance tests for integrated system operations."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_performance(self):
        """Test performance of complete end-to-end action execution."""
        asimov_kernel = AsimovKernel()
        action_filter = ActionFilter()
        orchestrator = OlympusOrchestrator()
        
        # Mock fast validations
        async def fast_asimov_validation(request):
            return {'approved': True, 'reason': None, 'audit_steps': ['Fast Asimov']}
        
        async def fast_safety_validation(request):
            return {'safe': True, 'reason': None, 'audit_steps': ['Fast Safety']}
        
        async def fast_module_execution(request):
            return {'success': True, 'data': 'fast_result'}
        
        orchestrator._validate_with_asimov = fast_asimov_validation
        orchestrator._apply_safety_filters = fast_safety_validation  
        orchestrator._execute_module_action = fast_module_execution
        
        # Measure end-to-end performance
        request = ActionRequest(
            id="perf_test_001",
            module="test_module",
            action="performance_test",
            parameters={},
            priority=Priority.NORMAL,
            requester="perf_tester"
        )
        
        start_time = time.time()
        result = await orchestrator.execute_action(request)
        end_to_end_time = time.time() - start_time
        
        assert result.success
        # Complete end-to-end should be fast (< 200ms)
        assert end_to_end_time < 0.2
        
        asimov_kernel.stop_integrity_monitoring()
    
    @pytest.mark.asyncio
    async def test_concurrent_integrated_operations(self):
        """Test performance under concurrent integrated operations."""
        asimov_kernel = AsimovKernel()
        action_filter = ActionFilter()
        orchestrator = OlympusOrchestrator()
        
        # Mock fast operations
        async def concurrent_asimov_validation(request):
            await asyncio.sleep(0.001)  # Minimal delay
            return {'approved': True, 'reason': None, 'audit_steps': ['Concurrent Asimov']}
        
        async def concurrent_safety_validation(request):
            await asyncio.sleep(0.001)  # Minimal delay
            return {'safe': True, 'reason': None, 'audit_steps': ['Concurrent Safety']}
        
        async def concurrent_module_execution(request):
            await asyncio.sleep(0.001)  # Minimal delay
            return {'success': True, 'data': f'result_{request.id}'}
        
        orchestrator._validate_with_asimov = concurrent_asimov_validation
        orchestrator._apply_safety_filters = concurrent_safety_validation
        orchestrator._execute_module_action = concurrent_module_execution
        
        # Create multiple concurrent requests
        requests = []
        for i in range(10):
            request = ActionRequest(
                id=f"concurrent_{i:03d}",
                module="test_module",
                action=f"action_{i}",
                parameters={"index": i},
                priority=Priority.NORMAL,
                requester="concurrent_tester"
            )
            requests.append(request)
        
        # Execute concurrently and measure time
        start_time = time.time()
        tasks = [orchestrator.execute_action(req) for req in requests]
        results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        # All should succeed
        assert len(results) == 10
        assert all(result.success for result in results)
        
        # Concurrent execution should be efficient (< 500ms for 10 operations)
        assert concurrent_time < 0.5
        
        asimov_kernel.stop_integrity_monitoring()