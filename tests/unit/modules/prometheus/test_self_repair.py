"""
Comprehensive test suite for PROMETHEUS Self-Repair System

PROMETHEUS provides autonomous system correction and recovery capabilities.
This test suite validates all self-repair functionality including fault
detection, repair planning, action execution, and safety compliance.

Critical functionality tested:
- Repair plan creation and validation
- Action executor safety and effectiveness  
- Human approval workflows for high-risk repairs
- Rollback capabilities and failure recovery
- Integration with safety and ethical frameworks
- Performance under various failure scenarios
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock, patch
from collections import deque

from olympus.modules.prometheus.self_repair import (
    SelfRepairSystem,
    RepairPlanner,
    RepairActionExecutor, 
    RepairPlan,
    RepairExecution,
    RepairAction,
    RepairStatus,
    RepairRisk
)


@pytest.mark.unit
class TestRepairActionExecutor:
    """Test individual repair action execution."""
    
    def test_executor_initialization(self):
        """Test repair action executor initializes correctly."""
        audit_system = AsyncMock()
        component_manager = AsyncMock()
        
        executor = RepairActionExecutor(audit_system, component_manager)
        
        assert executor.audit_system == audit_system
        assert executor.component_manager == component_manager
        assert executor.logger is not None
    
    @pytest.mark.asyncio
    async def test_restart_service_action(self):
        """Test service restart action execution."""
        executor = RepairActionExecutor(audit_system=AsyncMock())
        
        success, message = await executor.execute_action(
            RepairAction.RESTART_SERVICE,
            "test_service", 
            {"restart_type": "graceful"}
        )
        
        assert success
        assert "restarted" in message.lower()
        assert "test_service" in message
    
    @pytest.mark.asyncio
    async def test_restart_service_force_mode(self):
        """Test force restart mode."""
        executor = RepairActionExecutor()
        
        success, message = await executor.execute_action(
            RepairAction.RESTART_SERVICE,
            "stuck_service",
            {"restart_type": "force"}
        )
        
        assert success
        assert "force restarted" in message.lower()
    
    @pytest.mark.asyncio
    async def test_invalid_restart_type(self):
        """Test handling of invalid restart type."""
        executor = RepairActionExecutor()
        
        success, message = await executor.execute_action(
            RepairAction.RESTART_SERVICE,
            "test_service",
            {"restart_type": "invalid_type"}
        )
        
        assert not success
        assert "unknown restart type" in message.lower()
    
    @pytest.mark.asyncio
    async def test_clear_cache_action(self):
        """Test cache clearing action."""
        executor = RepairActionExecutor()
        
        success, message = await executor.execute_action(
            RepairAction.CLEAR_CACHE,
            "web_service",
            {"cache_types": ["memory", "disk"]}
        )
        
        assert success
        assert "cleared" in message.lower()
        assert "memory cache" in message
        assert "disk cache" in message
    
    @pytest.mark.asyncio
    async def test_free_resources_action(self):
        """Test resource freeing action."""
        executor = RepairActionExecutor()
        
        success, message = await executor.execute_action(
            RepairAction.FREE_RESOURCES,
            "memory_intensive_service", 
            {"resource_types": ["memory"]}
        )
        
        assert success
        assert "freed memory" in message.lower()
    
    @pytest.mark.asyncio
    async def test_optimize_configuration_action(self):
        """Test configuration optimization."""
        executor = RepairActionExecutor()
        
        success, message = await executor.execute_action(
            RepairAction.OPTIMIZE_CONFIGURATION,
            "database_service",
            {"optimization_type": "performance"}
        )
        
        assert success
        assert "optimized" in message.lower()
        assert "performance" in message.lower()
    
    @pytest.mark.asyncio
    async def test_scale_resources_action(self):
        """Test resource scaling action."""
        executor = RepairActionExecutor()
        
        success, message = await executor.execute_action(
            RepairAction.SCALE_RESOURCES,
            "overloaded_service",
            {"scaling_factor": 2.0, "resource_type": "compute"}
        )
        
        assert success
        assert "scaled" in message.lower()
        assert "2.0" in message
    
    @pytest.mark.asyncio
    async def test_unknown_action_handling(self):
        """Test handling of unknown repair actions."""
        executor = RepairActionExecutor()
        
        # Create a mock unknown action (this would normally fail at enum level)
        with patch('olympus.modules.prometheus.self_repair.RepairAction') as mock_action:
            mock_action.UNKNOWN_ACTION = "unknown_action"
            
            success, message = await executor.execute_action(
                mock_action.UNKNOWN_ACTION,
                "test_component",
                {}
            )
            
            assert not success
            assert "unknown repair action" in message.lower()
    
    @pytest.mark.asyncio
    async def test_action_audit_logging(self):
        """Test that actions are properly logged to audit system."""
        audit_system = AsyncMock()
        executor = RepairActionExecutor(audit_system=audit_system)
        
        await executor.execute_action(
            RepairAction.CLEAR_CACHE,
            "test_service",
            {"cache_types": ["memory"]}
        )
        
        # Verify audit logging calls
        assert audit_system.log_event.call_count >= 2  # Start and completion
        
        # Check audit log content
        calls = audit_system.log_event.call_args_list
        start_call = calls[0]
        assert start_call[0][0] == "repair_action_started"
        
        completion_call = calls[1]
        assert completion_call[0][0] == "repair_action_completed"
    
    @pytest.mark.asyncio
    async def test_action_exception_handling(self):
        """Test graceful handling of action execution exceptions."""
        audit_system = AsyncMock()
        executor = RepairActionExecutor(audit_system=audit_system)
        
        # Mock the internal method to raise an exception
        with patch.object(executor, '_clear_cache', side_effect=Exception("Simulated failure")):
            success, message = await executor.execute_action(
                RepairAction.CLEAR_CACHE,
                "failing_service",
                {}
            )
            
            assert not success
            assert "simulated failure" in message.lower()
            
            # Verify error was logged
            audit_system.log_event.assert_any_call(
                "repair_action_failed",
                {
                    "action": "clear_cache", 
                    "component": "failing_service",
                    "error": "Simulated failure"
                }
            )


@pytest.mark.unit
class TestRepairPlanner:
    """Test repair plan creation and strategy selection."""
    
    def test_planner_initialization(self):
        """Test repair planner initializes with strategies."""
        planner = RepairPlanner()
        
        assert planner.repair_strategies is not None
        assert len(planner.repair_strategies) > 0
        
        # Verify key strategies are present
        expected_strategies = [
            'high_cpu_usage',
            'high_memory_usage', 
            'disk_full',
            'service_unresponsive',
            'performance_degradation'
        ]
        
        for strategy in expected_strategies:
            assert strategy in planner.repair_strategies
            assert len(planner.repair_strategies[strategy]) > 0
    
    @pytest.mark.asyncio
    async def test_repair_plan_creation(self, mock_fault):
        """Test creation of comprehensive repair plans."""
        planner = RepairPlanner(audit_system=AsyncMock())
        
        plan = await planner.create_repair_plan(mock_fault)
        
        # Verify plan structure
        assert plan.plan_id is not None
        assert plan.fault_id == mock_fault.fault_id
        assert plan.component == mock_fault.component
        assert plan.primary_action in RepairAction
        assert isinstance(plan.backup_actions, list)
        assert plan.estimated_duration > 0
        assert plan.risk_level in RepairRisk
        assert isinstance(plan.required_approvals, list)
        assert isinstance(plan.safety_checks, list)
        assert plan.rollback_plan is not None
    
    @pytest.mark.asyncio
    async def test_fault_to_strategy_mapping(self, mock_fault):
        """Test mapping of fault symptoms to repair strategies."""
        planner = RepairPlanner()
        
        # Test high CPU usage mapping
        mock_fault.symptoms = ["high_cpu_usage"]
        strategy = planner._map_fault_to_strategy(mock_fault)
        assert strategy == 'high_cpu_usage'
        
        # Test high memory usage mapping
        mock_fault.symptoms = ["high_memory_usage"]
        strategy = planner._map_fault_to_strategy(mock_fault)
        assert strategy == 'high_memory_usage'
        
        # Test disk full mapping
        mock_fault.symptoms = ["high_disk_usage"]
        strategy = planner._map_fault_to_strategy(mock_fault)
        assert strategy == 'disk_full'
        
        # Test default mapping for unknown symptoms
        mock_fault.symptoms = ["unknown_symptom"]
        strategy = planner._map_fault_to_strategy(mock_fault)
        assert strategy == 'performance_degradation'  # Default
    
    @pytest.mark.asyncio
    async def test_risk_assessment(self):
        """Test repair action risk assessment."""
        planner = RepairPlanner()
        
        # Create test fault for critical component
        critical_fault = Mock()
        critical_fault.component = "safety_layer"
        
        # Test low-risk action on critical component becomes higher risk
        risk = await planner._assess_repair_risk(critical_fault, RepairAction.CLEAR_CACHE)
        assert risk in [RepairRisk.LOW, RepairRisk.MEDIUM]  # Escalated from MINIMAL
        
        # Test high-risk action on critical component becomes critical
        risk = await planner._assess_repair_risk(critical_fault, RepairAction.REPLACE_COMPONENT)
        assert risk == RepairRisk.CRITICAL
        
        # Test normal component risk assessment
        normal_fault = Mock()
        normal_fault.component = "normal_service"
        
        risk = await planner._assess_repair_risk(normal_fault, RepairAction.CLEAR_CACHE)
        assert risk == RepairRisk.MINIMAL
    
    @pytest.mark.asyncio
    async def test_required_approvals_determination(self):
        """Test determination of required approvals based on risk."""
        planner = RepairPlanner()
        
        # Test high risk requires human operator approval
        approvals = await planner._determine_required_approvals(RepairRisk.HIGH, Mock(component="test"))
        assert "human_operator" in approvals
        
        # Test critical risk requires multiple approvals
        approvals = await planner._determine_required_approvals(RepairRisk.CRITICAL, Mock(component="test"))
        assert "human_operator" in approvals
        assert "safety_officer" in approvals
        assert "system_administrator" in approvals
        
        # Test critical component requires additional approvals
        critical_component_fault = Mock()
        critical_component_fault.component = "asimov_kernel"
        
        approvals = await planner._determine_required_approvals(RepairRisk.MEDIUM, critical_component_fault)
        assert "safety_validation" in approvals
        assert "ethical_review" in approvals
    
    @pytest.mark.asyncio
    async def test_safety_checks_creation(self):
        """Test creation of safety checks for repairs."""
        planner = RepairPlanner()
        
        # Test restart service safety checks
        checks = await planner._create_safety_checks(Mock(), RepairAction.RESTART_SERVICE)
        
        required_checks = [
            "system_stability_verified",
            "no_critical_processes_affected",
            "graceful_shutdown_possible",
            "no_active_transactions",
            "backup_services_available"
        ]
        
        for check in required_checks:
            assert check in checks
        
        # Test component replacement safety checks  
        checks = await planner._create_safety_checks(Mock(), RepairAction.REPLACE_COMPONENT)
        
        replacement_checks = [
            "replacement_component_verified",
            "rollback_plan_tested", 
            "system_backup_completed"
        ]
        
        for check in replacement_checks:
            assert check in checks
    
    @pytest.mark.asyncio
    async def test_repair_duration_estimation(self):
        """Test repair duration estimation."""
        planner = RepairPlanner()
        
        # Test primary action duration
        duration = await planner._estimate_repair_duration(
            RepairAction.RESTART_SERVICE,
            [RepairAction.CLEAR_CACHE]
        )
        
        assert duration > 0
        assert duration >= 300  # At least primary action duration (restart = 300s)
        
        # Test complex repair with multiple backup actions
        duration = await planner._estimate_repair_duration(
            RepairAction.REPLACE_COMPONENT,
            [RepairAction.RESTART_SERVICE, RepairAction.ACTIVATE_REDUNDANCY]
        )
        
        assert duration >= 1800  # At least component replacement duration
    
    @pytest.mark.asyncio
    async def test_success_criteria_definition(self):
        """Test definition of repair success criteria."""
        planner = RepairPlanner()
        
        fault = Mock()
        fault.symptoms = ["high_cpu_usage", "slow_response_time"]
        
        criteria = await planner._define_success_criteria(fault)
        
        # Base criteria should always be present
        base_criteria = [
            "fault_symptoms_resolved",
            "system_performance_restored", 
            "no_new_errors_generated"
        ]
        
        for criterion in base_criteria:
            assert criterion in criteria
        
        # Symptom-specific criteria should be added
        assert "cpu_usage_below_threshold" in criteria
        assert "response_time_improved" in criteria


@pytest.mark.unit
class TestSelfRepairSystem:
    """Test complete self-repair system functionality."""
    
    @pytest.mark.asyncio
    async def test_system_initialization(self):
        """Test self-repair system initialization."""
        audit_system = AsyncMock()
        repair_system = SelfRepairSystem(audit_system=audit_system)
        
        await repair_system.initialize()
        
        assert repair_system.is_active
        assert repair_system.auto_repair_enabled
        
        # Verify audit logging
        audit_system.log_event.assert_called_with(
            "self_repair_system_initialized",
            {
                "auto_repair_enabled": True,
                "max_concurrent_repairs": 3
            }
        )
    
    @pytest.mark.asyncio
    async def test_system_shutdown(self):
        """Test self-repair system shutdown."""
        audit_system = AsyncMock()
        repair_system = SelfRepairSystem(audit_system=audit_system)
        
        await repair_system.initialize()
        
        # Add mock active repair
        repair_system.active_repairs["test_repair"] = Mock()
        
        await repair_system.shutdown()
        
        assert not repair_system.is_active
        assert len(repair_system.active_repairs) == 0  # Should be cancelled
    
    @pytest.mark.asyncio
    async def test_repair_initiation_auto_approved(self, mock_fault):
        """Test repair initiation for auto-approved repairs."""
        repair_system = SelfRepairSystem(audit_system=AsyncMock())
        await repair_system.initialize()
        
        # Mock repair planner to create low-risk plan
        with patch.object(repair_system.repair_planner, 'create_repair_plan') as mock_create_plan:
            mock_plan = Mock()
            mock_plan.plan_id = "test_plan_001"
            mock_plan.fault_id = mock_fault.fault_id
            mock_plan.required_approvals = []  # No approvals required
            mock_create_plan.return_value = mock_plan
            
            execution_id = await repair_system.initiate_repair(mock_fault)
            
            assert execution_id is not None
            assert execution_id in repair_system.active_repairs
    
    @pytest.mark.asyncio
    async def test_repair_initiation_requires_approval(self, mock_fault):
        """Test repair initiation for high-risk repairs requiring approval."""
        repair_system = SelfRepairSystem(audit_system=AsyncMock())
        await repair_system.initialize()
        
        # Mock repair planner to create high-risk plan
        with patch.object(repair_system.repair_planner, 'create_repair_plan') as mock_create_plan:
            mock_plan = Mock()
            mock_plan.plan_id = "test_plan_002"
            mock_plan.fault_id = mock_fault.fault_id
            mock_plan.required_approvals = ["human_operator"]  # Requires approval
            mock_plan.risk_level = RepairRisk.HIGH
            mock_create_plan.return_value = mock_plan
            
            execution_id = await repair_system.initiate_repair(mock_fault)
            
            assert execution_id is not None
            assert execution_id in repair_system.pending_approvals
            
            # Verify execution status
            execution = repair_system.active_repairs[execution_id]
            assert execution.status == RepairStatus.AWAITING_APPROVAL
    
    @pytest.mark.asyncio
    async def test_repair_approval_process(self, mock_fault):
        """Test human approval process for repairs."""
        repair_system = SelfRepairSystem(audit_system=AsyncMock())
        await repair_system.initialize()
        
        # Create pending approval
        mock_plan = Mock()
        mock_plan.plan_id = "approval_test_001"
        mock_plan.fault_id = mock_fault.fault_id
        
        execution = Mock()
        execution.human_approvals = []
        execution.status = RepairStatus.AWAITING_APPROVAL
        
        execution_id = "exec_approval_001"
        repair_system.pending_approvals[execution_id] = mock_plan
        repair_system.active_repairs[execution_id] = execution
        
        # Mock repair execution
        with patch.object(repair_system, '_execute_repair_plan_internal') as mock_execute:
            approval_result = await repair_system.approve_repair(execution_id, "human_operator_001")
            
            assert approval_result
            assert execution_id not in repair_system.pending_approvals
            assert "human_operator_001" in execution.human_approvals
            assert execution.status == RepairStatus.APPROVED
            
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_repair_approval_invalid_execution_id(self):
        """Test approval with invalid execution ID."""
        repair_system = SelfRepairSystem()
        await repair_system.initialize()
        
        result = await repair_system.approve_repair("invalid_id", "approver")
        assert not result
    
    @pytest.mark.asyncio
    async def test_safety_checks_execution(self):
        """Test safety checks are executed before repairs."""
        repair_system = SelfRepairSystem(audit_system=AsyncMock())
        
        mock_plan = Mock()
        mock_plan.safety_checks = [
            "system_stability_verified",
            "no_critical_processes_affected"
        ]
        
        mock_execution = Mock()
        mock_execution.safety_validations = []
        
        # Test safety checks pass
        with patch.object(repair_system, '_execute_safety_check', return_value=True):
            result = await repair_system._perform_safety_checks(mock_plan, mock_execution)
            
            assert result
            assert len(mock_execution.safety_validations) == 2
        
        # Test safety check failure
        mock_execution.safety_validations = []
        with patch.object(repair_system, '_execute_safety_check', return_value=False):
            result = await repair_system._perform_safety_checks(mock_plan, mock_execution)
            
            assert not result
            assert len(mock_execution.safety_validations) == 0
    
    @pytest.mark.asyncio
    async def test_repair_action_execution(self):
        """Test repair action execution through action executor."""
        repair_system = SelfRepairSystem()
        
        mock_execution = Mock()
        mock_execution.current_action = None
        mock_execution.actions_completed = []
        mock_execution.actions_failed = []
        mock_execution.error_messages = []
        
        # Mock successful action execution
        with patch.object(repair_system.action_executor, 'execute_action') as mock_execute:
            mock_execute.return_value = (True, "Action completed successfully")
            
            result = await repair_system._execute_action(
                RepairAction.RESTART_SERVICE,
                "test_component", 
                mock_execution
            )
            
            assert result
            assert RepairAction.RESTART_SERVICE in mock_execution.actions_completed
            assert len(mock_execution.actions_failed) == 0
            
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_repair_verification(self):
        """Test repair success verification."""
        repair_system = SelfRepairSystem()
        
        mock_plan = Mock()
        mock_plan.success_criteria = [
            "fault_symptoms_resolved",
            "system_performance_restored"
        ]
        
        mock_execution = Mock()
        
        # Test successful verification
        with patch.object(repair_system, '_check_success_criterion', return_value=True):
            result = await repair_system._verify_repair_success(mock_plan, mock_execution)
            assert result
        
        # Test failed verification
        with patch.object(repair_system, '_check_success_criterion', return_value=False):
            result = await repair_system._verify_repair_success(mock_plan, mock_execution)
            assert not result
    
    @pytest.mark.asyncio
    async def test_repair_completion_statistics(self):
        """Test repair completion updates statistics."""
        repair_system = SelfRepairSystem()
        
        mock_execution = Mock()
        mock_execution.execution_id = "stats_test_001" 
        mock_execution.start_time = time.time() - 10  # 10 seconds ago
        mock_execution.actions_completed = [RepairAction.RESTART_SERVICE]
        
        initial_attempted = repair_system.total_repairs_attempted
        initial_successful = repair_system.total_repairs_successful
        
        await repair_system._complete_repair(mock_execution)
        
        assert repair_system.total_repairs_attempted == initial_attempted + 1
        assert repair_system.total_repairs_successful == initial_successful + 1
        assert repair_system.repair_success_rate == 1.0  # 100% success
        assert mock_execution.status == RepairStatus.COMPLETED
        assert mock_execution.execution_id not in repair_system.active_repairs
    
    @pytest.mark.asyncio
    async def test_repair_failure_handling(self):
        """Test repair failure handling and statistics."""
        repair_system = SelfRepairSystem()
        
        mock_execution = Mock()
        mock_execution.execution_id = "failure_test_001"
        mock_execution.start_time = time.time() - 5
        mock_execution.error_messages = []
        
        initial_attempted = repair_system.total_repairs_attempted
        
        await repair_system._fail_repair(mock_execution, "Test failure reason")
        
        assert repair_system.total_repairs_attempted == initial_attempted + 1
        assert repair_system.total_repairs_successful == 0  # No change
        assert mock_execution.status == RepairStatus.FAILED
        assert "Test failure reason" in mock_execution.error_messages
    
    @pytest.mark.asyncio
    async def test_existing_repair_detection(self):
        """Test detection of existing repairs for the same fault."""
        repair_system = SelfRepairSystem()
        
        # Add existing repair
        existing_execution = Mock()
        existing_execution.plan_id = "repair_fault_123_timestamp"
        repair_system.active_repairs["existing_001"] = existing_execution
        
        # Test finding existing repair
        result = repair_system._find_existing_repair("fault_123")
        assert result == "existing_001"
        
        # Test no existing repair found
        result = repair_system._find_existing_repair("fault_456")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_auto_repair_enable_disable(self):
        """Test enabling and disabling auto-repair functionality."""
        audit_system = AsyncMock()
        repair_system = SelfRepairSystem(audit_system=audit_system)
        
        # Test disable
        result = await repair_system.disable_auto_repair()
        assert result
        assert not repair_system.auto_repair_enabled
        
        # Test enable
        result = await repair_system.enable_auto_repair() 
        assert result
        assert repair_system.auto_repair_enabled
        
        # Verify audit logging
        assert audit_system.log_event.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_system_status_reporting(self):
        """Test system status reporting."""
        repair_system = SelfRepairSystem()
        await repair_system.initialize()
        
        # Add some test data
        repair_system.active_repairs["test_001"] = Mock()
        repair_system.pending_approvals["test_002"] = Mock()
        repair_system.total_repairs_attempted = 10
        repair_system.total_repairs_successful = 8
        repair_system.repair_success_rate = 0.8
        
        status = await repair_system.get_status()
        
        required_fields = [
            'is_active', 'auto_repair_enabled', 'active_repairs',
            'pending_approvals', 'repair_queue_size', 'total_repairs_attempted',
            'total_repairs_successful', 'repair_success_rate', 'max_concurrent_repairs'
        ]
        
        for field in required_fields:
            assert field in status
        
        assert status['active_repairs'] == 1
        assert status['pending_approvals'] == 1
        assert status['total_repairs_attempted'] == 10
        assert status['repair_success_rate'] == 0.8


@pytest.mark.integration
class TestRepairSystemIntegration:
    """Integration tests for repair system components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_repair_workflow(self, mock_fault):
        """Test complete end-to-end repair workflow."""
        audit_system = AsyncMock()
        repair_system = SelfRepairSystem(audit_system=audit_system)
        await repair_system.initialize()
        
        # Mock constraint validation to allow repair
        if repair_system.repair_constraints:
            with patch.object(repair_system.repair_constraints, 'validate_repair_request') as mock_constraints:
                mock_constraints.return_value = {"allowed": True}
        
        # Execute repair workflow
        execution_id = await repair_system.initiate_repair(mock_fault)
        
        assert execution_id is not None
        
        # Wait for repair to potentially complete (it runs asynchronously)
        await asyncio.sleep(0.1)
        
        # Verify repair was initiated
        if execution_id in repair_system.active_repairs:
            execution = repair_system.active_repairs[execution_id] 
            assert execution.plan_id is not None
        elif execution_id in repair_system.pending_approvals:
            # High-risk repair requiring approval
            plan = repair_system.pending_approvals[execution_id]
            assert plan.fault_id == mock_fault.fault_id
    
    @pytest.mark.asyncio
    async def test_repair_plan_to_execution_integration(self, mock_fault):
        """Test integration between repair planning and execution."""
        repair_system = SelfRepairSystem(audit_system=AsyncMock())
        await repair_system.initialize()
        
        # Create repair plan
        plan = await repair_system.repair_planner.create_repair_plan(mock_fault)
        
        # Execute plan
        execution = RepairExecution(
            execution_id="integration_test_001",
            plan_id=plan.plan_id,
            status=RepairStatus.APPROVED
        )
        
        # Test plan execution
        with patch.object(repair_system, '_perform_safety_checks', return_value=True):
            with patch.object(repair_system, '_execute_action', return_value=True):
                with patch.object(repair_system, '_verify_repair_success', return_value=True):
                    
                    repair_system.active_repairs[execution.execution_id] = execution
                    
                    # Execute plan internally
                    await repair_system._execute_repair_plan_internal(plan, execution)
                    
                    # Verify execution completed successfully
                    assert execution.status == RepairStatus.COMPLETED


@pytest.mark.performance
class TestRepairSystemPerformance:
    """Performance tests for repair system operations."""
    
    @pytest.mark.asyncio
    async def test_repair_planning_performance(self, mock_fault):
        """Benchmark repair plan creation performance."""
        planner = RepairPlanner()
        
        start_time = time.time()
        plan = await planner.create_repair_plan(mock_fault)
        planning_time = time.time() - start_time
        
        assert plan is not None
        # Planning should be fast (< 100ms)
        assert planning_time < 0.1
    
    @pytest.mark.asyncio
    async def test_action_execution_performance(self):
        """Benchmark repair action execution performance."""
        executor = RepairActionExecutor()
        
        start_time = time.time()
        success, message = await executor.execute_action(
            RepairAction.CLEAR_CACHE,
            "test_component",
            {"cache_types": ["memory"]}
        )
        execution_time = time.time() - start_time
        
        assert success
        # Simple actions should complete quickly
        assert execution_time < 1.0
    
    @pytest.mark.asyncio
    async def test_concurrent_repair_handling(self, mock_fault):
        """Test handling multiple concurrent repairs."""
        repair_system = SelfRepairSystem()
        await repair_system.initialize()
        
        # Create multiple fault instances
        faults = []
        for i in range(5):
            fault = Mock()
            fault.fault_id = f"concurrent_fault_{i:03d}"
            fault.component = f"component_{i}"
            fault.symptoms = ["high_cpu_usage"]
            faults.append(fault)
        
        # Initiate repairs concurrently
        start_time = time.time()
        tasks = [repair_system.initiate_repair(fault) for fault in faults]
        execution_ids = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        # All repairs should be initiated
        successful_initiations = sum(1 for eid in execution_ids if eid is not None)
        assert successful_initiations >= 3  # At least some should succeed
        
        # Concurrent initiation should be reasonably fast
        assert concurrent_time < 2.0