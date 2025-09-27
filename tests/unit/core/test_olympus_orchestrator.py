"""
Comprehensive test suite for OLYMPUS Orchestrator

The OlympusOrchestrator is the central nervous system that coordinates all
OLYMPUS operations. This test suite validates system coordination, ethical
integration, emergency handling, and overall system integrity.

Critical functionality tested:
- System initialization and shutdown
- Action execution with ethical validation 
- Emergency procedures and human overrides
- Module coordination and health monitoring
- Performance and reliability under load
"""

import asyncio
import pytest
import time
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from concurrent.futures import ThreadPoolExecutor

from olympus.core.olympus_orchestrator import (
    OlympusOrchestrator,
    ActionRequest,
    ActionResult,
    Priority,
    SystemState
)


@pytest.mark.unit
class TestOrchestratorInitialization:
    """Test orchestrator initialization and configuration."""
    
    def test_orchestrator_creation(self):
        """Test orchestrator can be created with default config."""
        orchestrator = OlympusOrchestrator()
        
        assert orchestrator.state == SystemState.INITIALIZING
        assert orchestrator.startup_time is not None
        assert orchestrator.action_queue is not None
        assert isinstance(orchestrator.executor, ThreadPoolExecutor)
        assert orchestrator.metrics['total_actions'] == 0
    
    @pytest.mark.asyncio
    async def test_system_initialization_success(self, temp_dir):
        """Test successful system initialization."""
        config_path = temp_dir / "test_config.yaml"
        config_path.write_text("""
logging:
  level: ERROR
system:
  name: "test_olympus"
  max_workers: 2
""")
        
        orchestrator = OlympusOrchestrator(str(config_path))
        
        # Mock all component initializations to succeed
        with patch.object(orchestrator.config_manager, 'initialize', new_callable=AsyncMock) as mock_config, \
             patch.object(orchestrator.identity_manager, 'initialize', new_callable=AsyncMock) as mock_identity, \
             patch.object(orchestrator.consciousness_kernel, 'initialize', new_callable=AsyncMock) as mock_consciousness, \
             patch.object(orchestrator.module_manager, 'initialize', new_callable=AsyncMock) as mock_modules, \
             patch.object(orchestrator.system_health, 'initialize', new_callable=AsyncMock) as mock_health:
            
            # Mock health check
            with patch.object(orchestrator, 'get_system_health', new_callable=AsyncMock) as mock_health_check:
                mock_health_check.return_value = {'status': 'healthy'}
                
                success = await orchestrator.initialize_system()
                
                assert success
                assert orchestrator.state == SystemState.ACTIVE
                
                # Verify all components were initialized
                mock_config.assert_called_once()
                mock_identity.assert_called_once()  
                mock_consciousness.assert_called_once()
                mock_modules.assert_called_once()
                mock_health.assert_called_once()
        
        await orchestrator.shutdown(graceful=True)
    
    @pytest.mark.asyncio
    async def test_system_initialization_component_failure(self):
        """Test system initialization handles component failures."""
        orchestrator = OlympusOrchestrator()
        
        # Mock a component initialization failure
        with patch.object(orchestrator.config_manager, 'initialize', new_callable=AsyncMock) as mock_config:
            mock_config.side_effect = Exception("Initialization failed")
            
            success = await orchestrator.initialize_system()
            
            assert not success
            assert orchestrator.state == SystemState.EMERGENCY
    
    @pytest.mark.asyncio  
    async def test_system_initialization_degraded_health(self, temp_dir):
        """Test system handles degraded health during initialization."""
        config_path = temp_dir / "test_config.yaml"
        config_path.write_text("logging:\n  level: ERROR")
        
        orchestrator = OlympusOrchestrator(str(config_path))
        
        # Mock successful component init but critical health
        with patch.object(orchestrator.config_manager, 'initialize', new_callable=AsyncMock), \
             patch.object(orchestrator.identity_manager, 'initialize', new_callable=AsyncMock), \
             patch.object(orchestrator.consciousness_kernel, 'initialize', new_callable=AsyncMock), \
             patch.object(orchestrator.module_manager, 'initialize', new_callable=AsyncMock), \
             patch.object(orchestrator.system_health, 'initialize', new_callable=AsyncMock):
            
            with patch.object(orchestrator, 'get_system_health', new_callable=AsyncMock) as mock_health:
                mock_health.return_value = {'status': 'critical'}
                
                success = await orchestrator.initialize_system()
                
                assert success  # Still succeeds but state is degraded
                assert orchestrator.state == SystemState.DEGRADED
        
        await orchestrator.shutdown(graceful=True)


@pytest.mark.unit
class TestActionExecution:
    """Test action execution with ethical validation."""
    
    @pytest.mark.asyncio
    async def test_safe_action_execution(self):
        """Test execution of ethically approved actions."""
        orchestrator = OlympusOrchestrator()
        
        request = ActionRequest(
            id="test_action_001",
            module="test_module",
            action="safe_action",
            parameters={"param1": "value1"},
            priority=Priority.NORMAL,
            requester="test_user"
        )
        
        # Mock ethical validation to approve
        with patch.object(orchestrator, '_validate_with_asimov', new_callable=AsyncMock) as mock_asimov:
            mock_asimov.return_value = {
                'approved': True,
                'reason': None,
                'audit_steps': ['Asimov validation: APPROVED']
            }
            
            # Mock safety validation to pass
            with patch.object(orchestrator, '_apply_safety_filters', new_callable=AsyncMock) as mock_safety:
                mock_safety.return_value = {
                    'safe': True,
                    'reason': None,
                    'audit_steps': ['Safety filter: PASSED']
                }
                
                # Mock module execution to succeed
                with patch.object(orchestrator, '_execute_module_action', new_callable=AsyncMock) as mock_execute:
                    mock_execute.return_value = {
                        'success': True,
                        'data': 'action_completed'
                    }
                    
                    result = await orchestrator.execute_action(request)
                    
                    assert result.success
                    assert result.request_id == "test_action_001"
                    assert result.result == 'action_completed'
                    assert result.error is None
                    assert len(result.audit_trail) > 0
                    
                    # Verify validations were called
                    mock_asimov.assert_called_once_with(request)
                    mock_safety.assert_called_once_with(request)
                    mock_execute.assert_called_once_with(request)
    
    @pytest.mark.asyncio
    async def test_ethical_violation_blocks_action(self):
        """Test that ethical violations block action execution."""
        orchestrator = OlympusOrchestrator()
        
        request = ActionRequest(
            id="test_action_002",
            module="test_module", 
            action="harmful_action",
            parameters={"harm_level": "high"},
            priority=Priority.HIGH,
            requester="test_user"
        )
        
        # Mock ethical validation to deny
        with patch.object(orchestrator, '_validate_with_asimov', new_callable=AsyncMock) as mock_asimov:
            mock_asimov.return_value = {
                'approved': False,
                'reason': 'First Law violation: potential harm to humans',
                'audit_steps': ['Asimov validation: DENIED - First Law violation']
            }
            
            result = await orchestrator.execute_action(request)
            
            assert not result.success
            assert result.request_id == "test_action_002"
            assert "ethical validation failed" in result.error.lower()
            assert "First Law violation" in result.error
            assert len(result.audit_trail) > 0
    
    @pytest.mark.asyncio
    async def test_safety_violation_blocks_action(self):
        """Test that safety violations block action execution."""
        orchestrator = OlympusOrchestrator()
        
        request = ActionRequest(
            id="test_action_003",
            module="test_module",
            action="unsafe_action", 
            parameters={"force": 100},
            priority=Priority.NORMAL,
            requester="test_user"
        )
        
        # Mock ethical validation to pass
        with patch.object(orchestrator, '_validate_with_asimov', new_callable=AsyncMock) as mock_asimov:
            mock_asimov.return_value = {
                'approved': True,
                'reason': None,
                'audit_steps': ['Asimov validation: APPROVED']
            }
            
            # Mock safety validation to fail
            with patch.object(orchestrator, '_apply_safety_filters', new_callable=AsyncMock) as mock_safety:
                mock_safety.return_value = {
                    'safe': False,
                    'reason': 'Force limit exceeded',
                    'audit_steps': ['Safety filter: FAILED - Force limit exceeded']
                }
                
                result = await orchestrator.execute_action(request)
                
                assert not result.success
                assert "safety validation failed" in result.error.lower()
                assert "Force limit exceeded" in result.error
    
    @pytest.mark.asyncio
    async def test_emergency_action_with_override(self):
        """Test emergency actions with human override bypass validation."""
        orchestrator = OlympusOrchestrator()
        
        request = ActionRequest(
            id="emergency_001",
            module="safety_module",
            action="emergency_stop",
            parameters={"immediate": True},
            priority=Priority.CRITICAL,
            requester="human_operator",
            emergency=True,
            human_override=True
        )
        
        # Mock module execution
        with patch.object(orchestrator, '_execute_module_action', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {
                'success': True,
                'data': 'emergency_stop_activated'
            }
            
            result = await orchestrator.execute_action(request)
            
            assert result.success
            assert "EMERGENCY: Human override activated" in result.audit_trail[1]
            assert orchestrator.metrics['emergency_activations'] > 0
    
    @pytest.mark.asyncio
    async def test_action_execution_exception_handling(self):
        """Test action execution handles exceptions gracefully."""
        orchestrator = OlympusOrchestrator()
        
        request = ActionRequest(
            id="error_action_001",
            module="faulty_module",
            action="crash_action",
            parameters={},
            priority=Priority.NORMAL,
            requester="test_user"
        )
        
        # Mock validation to pass but execution to fail
        with patch.object(orchestrator, '_validate_with_asimov', new_callable=AsyncMock) as mock_asimov:
            mock_asimov.return_value = {
                'approved': True,
                'reason': None,
                'audit_steps': ['Asimov validation: APPROVED']
            }
            
            with patch.object(orchestrator, '_apply_safety_filters', new_callable=AsyncMock) as mock_safety:
                mock_safety.return_value = {
                    'safe': True,
                    'reason': None,
                    'audit_steps': ['Safety filter: PASSED']
                }
                
                with patch.object(orchestrator, '_execute_module_action', side_effect=Exception("Module crashed")):
                    result = await orchestrator.execute_action(request)
                    
                    assert not result.success
                    assert "Module crashed" in result.error
                    assert result.execution_time > 0


@pytest.mark.unit
class TestSystemStatus:
    """Test system status reporting and monitoring."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_system_status(self):
        """Test comprehensive system status reporting."""
        orchestrator = OlympusOrchestrator()
        
        # Mock all status methods
        with patch.object(orchestrator.identity_manager, 'get_current_identity', new_callable=AsyncMock) as mock_identity:
            mock_identity.return_value = {"id": "olympus_001", "role": "coordinator"}
            
            with patch.object(orchestrator.module_manager, 'get_all_module_status', new_callable=AsyncMock) as mock_modules:
                mock_modules.return_value = {
                    "prometheus": {"status": "active", "health": "good"},
                    "atlas": {"status": "active", "health": "good"}
                }
                
                with patch.object(orchestrator.system_health, 'get_comprehensive_health', new_callable=AsyncMock) as mock_health:
                    mock_health.return_value = {"overall": "healthy", "components": ["all_green"]}
                    
                    with patch.object(orchestrator.consciousness_kernel, 'get_consciousness_state', new_callable=AsyncMock) as mock_consciousness:
                        mock_consciousness.return_value = {"awareness_level": 0.8, "active": True}
                        
                        status = await orchestrator.get_system_status()
                        
                        # Verify all required status components
                        assert 'system' in status
                        assert 'modules' in status
                        assert 'health' in status
                        assert 'consciousness' in status
                        assert 'performance' in status
                        assert 'active_actions' in status
                        assert 'queue_size' in status
                        
                        # Verify system details
                        assert status['system']['state'] == orchestrator.state.value
                        assert 'uptime' in status['system']
                        assert 'identity' in status['system']
    
    @pytest.mark.asyncio
    async def test_system_health_summary(self):
        """Test system health summary."""
        orchestrator = OlympusOrchestrator()
        
        with patch.object(orchestrator.system_health, 'get_health_summary', new_callable=AsyncMock) as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "components": {"all": "operational"},
                "last_check": time.time()
            }
            
            health = await orchestrator.get_system_health()
            
            assert health['status'] == 'healthy'
            assert 'components' in health
            mock_health.assert_called_once()


@pytest.mark.unit
class TestEmergencyHandling:
    """Test emergency procedures and crisis management."""
    
    @pytest.mark.asyncio
    async def test_emergency_activation(self):
        """Test emergency activation changes system state."""
        orchestrator = OlympusOrchestrator()
        
        emergency_details = {
            "type": "system_failure",
            "severity": "critical",
            "affected_components": ["power_system", "safety_sensors"]
        }
        
        # Mock emergency handler
        mock_handler = AsyncMock(return_value=True)
        orchestrator.emergency_handlers['system_failure'] = mock_handler
        
        result = await orchestrator.handle_emergency('system_failure', emergency_details)
        
        assert result
        assert orchestrator.state == SystemState.EMERGENCY
        assert orchestrator.metrics['emergency_activations'] > 0
        mock_handler.assert_called_once_with(emergency_details)
    
    @pytest.mark.asyncio
    async def test_emergency_handler_failure_triggers_shutdown(self):
        """Test that emergency handler failure triggers shutdown."""
        orchestrator = OlympusOrchestrator()
        
        # Mock failing emergency handler
        mock_handler = AsyncMock(side_effect=Exception("Handler crashed"))
        orchestrator.emergency_handlers['security_breach'] = mock_handler
        
        with patch.object(orchestrator, '_emergency_shutdown', new_callable=AsyncMock) as mock_shutdown:
            result = await orchestrator.handle_emergency('security_breach', {})
            
            assert not result
            mock_shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_unknown_emergency_type_default_handling(self):
        """Test unknown emergency types use default handling."""
        orchestrator = OlympusOrchestrator()
        
        with patch.object(orchestrator, '_emergency_shutdown', new_callable=AsyncMock) as mock_shutdown:
            result = await orchestrator.handle_emergency('unknown_emergency', {})
            
            assert not result
            mock_shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_specific_emergency_handlers(self):
        """Test specific emergency handler registration and execution."""
        orchestrator = OlympusOrchestrator()
        
        # Test all registered emergency types
        emergency_types = [
            'system_failure',
            'security_breach', 
            'ethical_violation',
            'hardware_failure',
            'human_safety'
        ]
        
        for emergency_type in emergency_types:
            assert emergency_type in orchestrator.emergency_handlers
            
            # Test handler execution
            with patch.object(orchestrator.emergency_handlers[emergency_type], '__call__', new_callable=AsyncMock) as mock_handler:
                mock_handler.return_value = True
                
                result = await orchestrator.handle_emergency(emergency_type, {"test": "data"})
                
                assert result
                mock_handler.assert_called_once_with({"test": "data"})


@pytest.mark.unit  
class TestSystemShutdown:
    """Test system shutdown procedures."""
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful system shutdown."""
        orchestrator = OlympusOrchestrator()
        orchestrator.state = SystemState.ACTIVE
        
        # Add mock active action
        mock_request = ActionRequest(
            id="active_001",
            module="test",
            action="long_running",
            parameters={},
            priority=Priority.NORMAL,
            requester="test"
        )
        orchestrator.active_actions["active_001"] = mock_request
        
        # Mock component shutdowns
        with patch.object(orchestrator.system_health, 'shutdown', new_callable=AsyncMock) as mock_health_shutdown, \
             patch.object(orchestrator.module_manager, 'shutdown', new_callable=AsyncMock) as mock_module_shutdown, \
             patch.object(orchestrator.consciousness_kernel, 'shutdown', new_callable=AsyncMock) as mock_consciousness_shutdown, \
             patch.object(orchestrator.identity_manager, 'shutdown', new_callable=AsyncMock) as mock_identity_shutdown, \
             patch.object(orchestrator.config_manager, 'shutdown', new_callable=AsyncMock) as mock_config_shutdown:
            
            await orchestrator.shutdown(graceful=True)
            
            assert orchestrator.state == SystemState.SHUTDOWN
            assert orchestrator._shutdown_event.is_set()
            
            # Verify components were shut down in reverse order
            mock_health_shutdown.assert_called_once()
            mock_module_shutdown.assert_called_once()
            mock_consciousness_shutdown.assert_called_once()
            mock_identity_shutdown.assert_called_once()
            mock_config_shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_immediate_shutdown(self):
        """Test immediate system shutdown."""
        orchestrator = OlympusOrchestrator()
        orchestrator.state = SystemState.ACTIVE
        
        # Add active actions that should be abandoned
        orchestrator.active_actions["active_001"] = Mock()
        orchestrator.active_actions["active_002"] = Mock()
        
        await orchestrator.shutdown(graceful=False)
        
        assert orchestrator.state == SystemState.SHUTDOWN
        assert orchestrator._shutdown_event.is_set()


@pytest.mark.unit
class TestConcurrencyAndPerformance:
    """Test concurrent operations and performance."""
    
    @pytest.mark.asyncio
    async def test_concurrent_action_execution(self):
        """Test multiple concurrent actions execute properly."""
        orchestrator = OlympusOrchestrator()
        
        # Create multiple action requests
        requests = []
        for i in range(5):
            request = ActionRequest(
                id=f"concurrent_{i:03d}",
                module="test_module",
                action=f"action_{i}",
                parameters={"index": i},
                priority=Priority.NORMAL,
                requester="test_user"
            )
            requests.append(request)
        
        # Mock validations and execution
        with patch.object(orchestrator, '_validate_with_asimov', new_callable=AsyncMock) as mock_asimov:
            mock_asimov.return_value = {
                'approved': True,
                'reason': None,
                'audit_steps': ['Asimov validation: APPROVED']
            }
            
            with patch.object(orchestrator, '_apply_safety_filters', new_callable=AsyncMock) as mock_safety:
                mock_safety.return_value = {
                    'safe': True,
                    'reason': None, 
                    'audit_steps': ['Safety filter: PASSED']
                }
                
                with patch.object(orchestrator, '_execute_module_action', new_callable=AsyncMock) as mock_execute:
                    mock_execute.return_value = {'success': True, 'data': 'completed'}
                    
                    # Execute all requests concurrently
                    tasks = [orchestrator.execute_action(req) for req in requests]
                    results = await asyncio.gather(*tasks)
                    
                    # All should succeed
                    assert len(results) == 5
                    for i, result in enumerate(results):
                        assert result.success
                        assert result.request_id == f"concurrent_{i:03d}"
    
    def test_metrics_tracking(self):
        """Test performance metrics are tracked correctly."""
        orchestrator = OlympusOrchestrator()
        
        # Verify initial metrics
        assert orchestrator.metrics['total_actions'] == 0
        assert orchestrator.metrics['successful_actions'] == 0
        assert orchestrator.metrics['failed_actions'] == 0
        assert orchestrator.metrics['emergency_activations'] == 0
        assert orchestrator.metrics['average_response_time'] == 0.0
        
        # Simulate completed actions
        mock_result_success = ActionResult(
            request_id="test_001",
            success=True,
            execution_time=0.5
        )
        
        mock_result_failure = ActionResult(
            request_id="test_002", 
            success=False,
            execution_time=0.3,
            error="Test error"
        )
        
        # Test metrics update
        asyncio.run(orchestrator._complete_action("test_001", mock_result_success))
        asyncio.run(orchestrator._complete_action("test_002", mock_result_failure))
        
        assert orchestrator.metrics['total_actions'] == 2
        assert orchestrator.metrics['successful_actions'] == 1
        assert orchestrator.metrics['failed_actions'] == 1
        assert orchestrator.metrics['average_response_time'] > 0
    
    @pytest.mark.asyncio
    async def test_action_queue_management(self):
        """Test action queue operates correctly."""
        orchestrator = OlympusOrchestrator()
        
        # Queue should be empty initially  
        assert orchestrator.action_queue.qsize() == 0
        
        # Test queue operations
        test_item = {"test": "data"}
        await orchestrator.action_queue.put(test_item)
        
        assert orchestrator.action_queue.qsize() == 1
        
        retrieved_item = await orchestrator.action_queue.get()
        assert retrieved_item == test_item
        assert orchestrator.action_queue.qsize() == 0


@pytest.mark.integration
class TestOrchestratorIntegration:
    """Integration tests for orchestrator with other components."""
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self):
        """Test integration with system health monitoring."""
        orchestrator = OlympusOrchestrator()
        
        # Mock critical health status
        with patch.object(orchestrator.system_health, 'get_health_summary', new_callable=AsyncMock) as mock_health:
            mock_health.return_value = {'status': 'critical'}
            
            with patch.object(orchestrator, 'handle_emergency', new_callable=AsyncMock) as mock_emergency:
                # Simulate health monitoring detecting critical status
                await orchestrator._health_monitor()
                
                mock_emergency.assert_called_once_with('system_failure', {'status': 'critical'})
    
    @pytest.mark.asyncio
    async def test_consciousness_monitoring_integration(self):
        """Test integration with consciousness kernel monitoring."""
        orchestrator = OlympusOrchestrator()
        
        with patch.object(orchestrator.consciousness_kernel, 'update_consciousness_state', new_callable=AsyncMock) as mock_update:
            # Run one iteration of consciousness monitoring
            orchestrator._shutdown_event.set()  # Stop after one iteration
            
            await orchestrator._consciousness_monitor()
            
            mock_update.assert_called_once()


@pytest.mark.performance
class TestOrchestratorPerformance:
    """Performance benchmarks for orchestrator operations."""
    
    @pytest.mark.asyncio
    async def test_action_processing_performance(self):
        """Benchmark action processing performance."""
        orchestrator = OlympusOrchestrator()
        
        request = ActionRequest(
            id="perf_test_001",
            module="test_module",
            action="simple_action", 
            parameters={},
            priority=Priority.NORMAL,
            requester="perf_tester"
        )
        
        # Mock fast validations and execution
        with patch.object(orchestrator, '_validate_with_asimov', new_callable=AsyncMock) as mock_asimov:
            mock_asimov.return_value = {
                'approved': True,
                'reason': None,
                'audit_steps': ['Fast validation']
            }
            
            with patch.object(orchestrator, '_apply_safety_filters', new_callable=AsyncMock) as mock_safety:
                mock_safety.return_value = {
                    'safe': True,
                    'reason': None,
                    'audit_steps': ['Fast safety check']
                }
                
                with patch.object(orchestrator, '_execute_module_action', new_callable=AsyncMock) as mock_execute:
                    mock_execute.return_value = {'success': True, 'data': 'fast_result'}
                    
                    # Measure processing time
                    start_time = time.time()
                    result = await orchestrator.execute_action(request)
                    processing_time = time.time() - start_time
                    
                    assert result.success
                    # Should process quickly (< 100ms for simple actions)
                    assert processing_time < 0.1
    
    @pytest.mark.asyncio
    async def test_status_reporting_performance(self):
        """Benchmark status reporting performance."""
        orchestrator = OlympusOrchestrator()
        
        # Mock all status methods for speed
        with patch.object(orchestrator.identity_manager, 'get_current_identity', new_callable=AsyncMock) as mock_identity, \
             patch.object(orchestrator.module_manager, 'get_all_module_status', new_callable=AsyncMock) as mock_modules, \
             patch.object(orchestrator.system_health, 'get_comprehensive_health', new_callable=AsyncMock) as mock_health, \
             patch.object(orchestrator.consciousness_kernel, 'get_consciousness_state', new_callable=AsyncMock) as mock_consciousness:
            
            mock_identity.return_value = {"id": "olympus_001"}
            mock_modules.return_value = {"test": "status"}
            mock_health.return_value = {"health": "good"}
            mock_consciousness.return_value = {"state": "active"}
            
            # Measure status reporting time
            start_time = time.time()
            status = await orchestrator.get_system_status()
            reporting_time = time.time() - start_time
            
            assert status is not None
            # Status reporting should be very fast (< 50ms)
            assert reporting_time < 0.05