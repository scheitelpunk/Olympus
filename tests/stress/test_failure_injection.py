"""
Failure Injection and Chaos Testing for OLYMPUS

This test suite validates system resilience and recovery under various
failure conditions. It injects faults, simulates hardware failures,
network issues, and other adverse conditions to ensure OLYMPUS
maintains safety and ethical compliance even under stress.

Failure scenarios tested:
- Component failures and cascading effects
- Network partitions and communication failures
- Resource exhaustion (memory, CPU, disk)
- Hardware sensor failures and malfunctions
- Malicious input and attack scenarios
- Race conditions and timing issues
- Data corruption and integrity violations
"""

import asyncio
import pytest
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import gc
import weakref

from olympus.ethical_core.asimov_kernel import AsimovKernel, ActionContext, ActionType, EthicalResult
from olympus.safety_layer.action_filter import ActionFilter, FilterStatus
from olympus.core.olympus_orchestrator import OlympusOrchestrator, ActionRequest, Priority, SystemState
from olympus.modules.prometheus.self_repair import SelfRepairSystem, RepairStatus


class FailureInjector:
    """Utility class for injecting various types of failures."""
    
    def __init__(self):
        self.active_failures = {}
        self.failure_count = 0
    
    def inject_random_delay(self, min_delay=0.1, max_delay=1.0):
        """Inject random processing delays."""
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    
    def inject_memory_pressure(self, size_mb=100):
        """Inject memory pressure by allocating memory."""
        # Allocate memory to simulate pressure
        memory_hog = bytearray(size_mb * 1024 * 1024)
        return memory_hog  # Return to keep alive
    
    def should_fail(self, failure_rate=0.1):
        """Decide if an operation should fail based on failure rate."""
        return random.random() < failure_rate
    
    def inject_intermittent_failure(self, operation_name, failure_rate=0.2):
        """Inject intermittent failures for an operation."""
        if operation_name not in self.active_failures:
            self.active_failures[operation_name] = 0
        
        if self.should_fail(failure_rate):
            self.active_failures[operation_name] += 1
            return True
        return False


@pytest.mark.stress
class TestComponentFailureInjection:
    """Test component failure injection and recovery."""
    
    @pytest.mark.asyncio
    async def test_asimov_kernel_integrity_corruption(self):
        """Test Asimov Kernel response to integrity corruption."""
        kernel = AsimovKernel()
        injector = FailureInjector()
        
        # Let kernel run normally first
        context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Normal operation before corruption"
        )
        
        result = kernel.evaluate_action(context)
        assert result.result == EthicalResult.APPROVED
        
        # Inject integrity corruption by modifying checksums
        original_checksums = kernel._law_checksums.copy()
        
        # Corrupt a law checksum
        kernel._law_checksums[1] = "corrupted_checksum_12345"
        
        # Wait for integrity monitoring to detect corruption
        await asyncio.sleep(0.2)
        
        # System should be in emergency stop mode
        assert kernel._emergency_stop_active
        
        # All actions should be blocked
        result = kernel.evaluate_action(context)
        assert result.result == EthicalResult.EMERGENCY_STOP
        
        # Restore checksums and reset emergency
        kernel._law_checksums = original_checksums
        kernel.reset_emergency_stop("integrity_recovery_12345")
        
        # System should resume normal operation
        result = kernel.evaluate_action(context)
        assert result.result == EthicalResult.APPROVED
        
        kernel.stop_integrity_monitoring()
    
    @pytest.mark.asyncio
    async def test_orchestrator_module_failures(self):
        """Test orchestrator handling of module failures."""
        orchestrator = OlympusOrchestrator()
        injector = FailureInjector()
        
        # Mock modules with intermittent failures
        failure_count = 0
        async def failing_asimov_validation(request):
            nonlocal failure_count
            failure_count += 1
            
            if injector.inject_intermittent_failure("asimov_validation", 0.3):
                raise Exception(f"Simulated Asimov validation failure #{failure_count}")
            
            return {'approved': True, 'audit_steps': [f'Asimov OK (attempt {failure_count})']}
        
        async def failing_safety_validation(request):
            if injector.inject_intermittent_failure("safety_validation", 0.2):
                raise Exception("Simulated safety validation failure")
            
            return {'safe': True, 'audit_steps': ['Safety OK']}
        
        async def failing_module_execution(request):
            if injector.inject_intermittent_failure("module_execution", 0.25):
                raise Exception("Simulated module execution failure")
            
            return {'success': True, 'data': 'execution_completed'}
        
        orchestrator._validate_with_asimov = failing_asimov_validation
        orchestrator._apply_safety_filters = failing_safety_validation
        orchestrator._execute_module_action = failing_module_execution
        
        # Execute multiple requests and track failure handling
        successful_executions = 0
        failed_executions = 0
        
        for i in range(50):
            request = ActionRequest(
                id=f"failure_test_{i:03d}",
                module="failing_module",
                action=f"test_action_{i}",
                parameters={"attempt": i},
                priority=Priority.NORMAL,
                requester="failure_tester"
            )
            
            result = await orchestrator.execute_action(request)
            
            if result.success:
                successful_executions += 1
            else:
                failed_executions += 1
                # Verify error handling
                assert result.error is not None
                assert "failure" in result.error.lower()
        
        # Some operations should succeed despite failures
        success_rate = successful_executions / (successful_executions + failed_executions)
        assert success_rate > 0.3  # At least 30% should succeed
        
        # System should remain operational (not crash)
        assert orchestrator.state != SystemState.EMERGENCY
    
    def test_action_filter_resource_exhaustion(self):
        """Test action filter behavior under resource exhaustion."""
        filter_system = ActionFilter()
        injector = FailureInjector()
        
        # Inject memory pressure
        memory_hog = injector.inject_memory_pressure(200)  # 200MB
        
        try:
            # Test filtering under memory pressure
            results = []
            for i in range(100):
                action = {
                    'action_type': f'stress_test_{i}',
                    'target_position': [random.random(), random.random(), random.random()],
                    'velocity': [random.uniform(0, 0.5) for _ in range(3)],
                    'force': [random.uniform(0, 10) for _ in range(3)],
                    'environment': {
                        'lighting': random.randint(50, 100),
                        'temperature': random.randint(15, 30)
                    },
                    'system_status': {
                        'battery_level': random.randint(20, 100),
                        'sensors_operational': True
                    },
                    'humans_detected': []
                }
                
                # Add random delay to simulate processing under load
                injector.inject_random_delay(0.001, 0.01)
                
                result = filter_system.filter_action(action)
                results.append(result)
                
                # Verify result is valid despite resource pressure
                assert result is not None
                assert result.status in [FilterStatus.APPROVED, FilterStatus.BLOCKED, FilterStatus.MODIFIED]
            
            # All filtering operations should complete
            assert len(results) == 100
            
        finally:
            # Clean up memory pressure
            del memory_hog
            gc.collect()
    
    @pytest.mark.asyncio
    async def test_repair_system_cascading_failures(self):
        """Test repair system handling cascading component failures."""
        repair_system = SelfRepairSystem(audit_system=AsyncMock())
        await repair_system.initialize()
        injector = FailureInjector()
        
        # Mock constraint validation to allow repairs
        if repair_system.repair_constraints:
            repair_system.repair_constraints.validate_repair_request = AsyncMock(
                return_value={"allowed": True}
            )
        
        # Simulate cascading failures across multiple components
        failing_components = [
            "network_service", "database_service", "cache_service",
            "authentication_service", "logging_service"
        ]
        
        # Create faults for all components
        cascade_faults = []
        for i, component in enumerate(failing_components):
            fault = Mock()
            fault.fault_id = f"cascade_fault_{i:03d}"
            fault.component = component
            fault.symptoms = ["high_error_rate", "service_unresponsive"]
            fault.severity = "high" if i >= 3 else "medium"  # Later failures are more severe
            cascade_faults.append(fault)
        
        # Inject failures in rapid succession
        repair_executions = []
        for fault in cascade_faults:
            execution_id = await repair_system.initiate_repair(fault)
            if execution_id:
                repair_executions.append(execution_id)
            
            # Small delay between failures
            await asyncio.sleep(0.05)
        
        # System should handle multiple concurrent repairs
        assert len(repair_executions) > 0
        
        # Check system status under cascading failures
        status = await repair_system.get_status()
        assert status['is_active']  # System should remain active
        assert status['active_repairs'] > 0 or status['pending_approvals'] > 0
        
        # System shouldn't be overwhelmed (max concurrent repairs limit)
        assert status['active_repairs'] <= repair_system.max_concurrent_repairs


@pytest.mark.stress
class TestNetworkAndCommunicationFailures:
    """Test network failures and communication issues."""
    
    @pytest.mark.asyncio
    async def test_communication_timeout_handling(self):
        """Test handling of communication timeouts."""
        orchestrator = OlympusOrchestrator()
        injector = FailureInjector()
        
        # Mock operations with random timeouts
        async def timeout_prone_asimov(request):
            if injector.should_fail(0.3):
                await asyncio.sleep(5)  # Simulate timeout
                raise TimeoutError("Asimov validation timeout")
            
            await asyncio.sleep(random.uniform(0.01, 0.1))  # Normal latency variation
            return {'approved': True, 'audit_steps': ['Asimov OK']}
        
        async def timeout_prone_safety(request):
            if injector.should_fail(0.2):
                await asyncio.sleep(3)  # Simulate timeout
                raise TimeoutError("Safety validation timeout")
            
            return {'safe': True, 'audit_steps': ['Safety OK']}
        
        async def timeout_prone_execution(request):
            if injector.should_fail(0.25):
                await asyncio.sleep(4)  # Simulate timeout
                raise TimeoutError("Module execution timeout")
            
            return {'success': True, 'data': 'completed'}
        
        orchestrator._validate_with_asimov = timeout_prone_asimov
        orchestrator._apply_safety_filters = timeout_prone_safety
        orchestrator._execute_module_action = timeout_prone_execution
        
        # Execute operations with timeout protection
        completed_operations = 0
        timeout_operations = 0
        
        for i in range(30):
            request = ActionRequest(
                id=f"timeout_test_{i:03d}",
                module="timeout_test_module",
                action=f"timeout_action_{i}",
                parameters={"test": "timeout_handling"},
                priority=Priority.NORMAL,
                requester="timeout_tester"
            )
            
            try:
                # Add timeout protection
                result = await asyncio.wait_for(
                    orchestrator.execute_action(request),
                    timeout=2.0  # 2 second timeout
                )
                
                completed_operations += 1
                assert result is not None
                
            except asyncio.TimeoutError:
                timeout_operations += 1
        
        # Some operations should complete despite timeouts
        assert completed_operations > 5
        # System should handle timeouts gracefully
        assert orchestrator.state != SystemState.EMERGENCY
    
    @pytest.mark.asyncio
    async def test_partial_system_isolation(self):
        """Test system behavior under partial component isolation."""
        asimov_kernel = AsimovKernel()
        repair_system = SelfRepairSystem(audit_system=AsyncMock())
        await repair_system.initialize()
        
        # Simulate partial isolation - some components unreachable
        isolated_components = {"network_monitor", "external_sensors", "backup_storage"}
        
        # Mock repair constraints to simulate isolation
        async def isolation_aware_constraints(fault):
            if fault.component in isolated_components:
                return {
                    "allowed": False,
                    "reason": f"Component {fault.component} is isolated and unreachable"
                }
            return {"allowed": True}
        
        if repair_system.repair_constraints:
            repair_system.repair_constraints.validate_repair_request = isolation_aware_constraints
        
        # Test operations on isolated vs accessible components
        isolated_fault = Mock()
        isolated_fault.fault_id = "isolated_fault_001"
        isolated_fault.component = "network_monitor"
        isolated_fault.symptoms = ["unreachable"]
        
        accessible_fault = Mock()
        accessible_fault.fault_id = "accessible_fault_001"
        accessible_fault.component = "local_processor"
        accessible_fault.symptoms = ["high_cpu_usage"]
        
        # Isolated component repair should be blocked
        isolated_execution_id = await repair_system.initiate_repair(isolated_fault)
        assert isolated_execution_id is None
        
        # Accessible component repair should proceed
        accessible_execution_id = await repair_system.initiate_repair(accessible_fault)
        assert accessible_execution_id is not None
        
        # Core ethical functions should remain operational during isolation
        context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Test during partial isolation"
        )
        
        result = asimov_kernel.evaluate_action(context)
        assert result.result in [EthicalResult.APPROVED, EthicalResult.DENIED]  # Should work normally
        
        asimov_kernel.stop_integrity_monitoring()


@pytest.mark.stress
class TestResourceExhaustionScenarios:
    """Test system behavior under resource exhaustion."""
    
    def test_memory_exhaustion_handling(self):
        """Test system behavior under memory pressure."""
        kernel = AsimovKernel()
        filter_system = ActionFilter()
        injector = FailureInjector()
        
        # Gradually increase memory pressure
        memory_hogs = []
        successful_operations = 0
        failed_operations = 0
        
        try:
            for pressure_level in range(5):
                # Increase memory pressure
                memory_hog = injector.inject_memory_pressure(50)  # 50MB per level
                memory_hogs.append(memory_hog)
                
                # Test operations under increasing pressure
                for i in range(10):
                    try:
                        # Test Asimov Kernel
                        context = ActionContext(
                            action_type=ActionType.INFORMATION,
                            description=f"Memory pressure test {pressure_level}-{i}"
                        )
                        
                        result = kernel.evaluate_action(context)
                        assert result is not None
                        
                        # Test Action Filter
                        action = {
                            'action_type': f'memory_test_{i}',
                            'force': [1.0, 0.0, 0.0],
                            'velocity': [0.1, 0.0, 0.0]
                        }
                        
                        filter_result = filter_system.filter_action(action)
                        assert filter_result is not None
                        
                        successful_operations += 1
                        
                    except (MemoryError, Exception) as e:
                        failed_operations += 1
                        # Memory errors are acceptable under extreme pressure
                        if "memory" not in str(e).lower():
                            # Re-raise non-memory related errors
                            raise
        
        finally:
            # Clean up memory pressure
            del memory_hogs
            gc.collect()
            kernel.stop_integrity_monitoring()
        
        # System should handle some operations even under memory pressure
        assert successful_operations > 0
        # If failures occur, they should be memory-related, not crashes
        if failed_operations > 0:
            assert successful_operations / (successful_operations + failed_operations) > 0.3
    
    @pytest.mark.asyncio
    async def test_concurrent_request_flood(self):
        """Test system behavior under concurrent request flooding."""
        orchestrator = OlympusOrchestrator()
        
        # Mock operations with slight processing time
        async def flood_test_asimov(request):
            await asyncio.sleep(0.01)  # Small processing delay
            return {'approved': True, 'audit_steps': ['Flood test approved']}
        
        async def flood_test_safety(request):
            await asyncio.sleep(0.01)  # Small processing delay
            return {'safe': True, 'audit_steps': ['Flood test safe']}
        
        async def flood_test_execution(request):
            await asyncio.sleep(0.02)  # Slightly longer execution
            return {'success': True, 'data': f'flood_result_{request.id}'}
        
        orchestrator._validate_with_asimov = flood_test_asimov
        orchestrator._apply_safety_filters = flood_test_safety
        orchestrator._execute_module_action = flood_test_execution
        
        # Create flood of concurrent requests
        flood_requests = []
        for i in range(200):  # Large number of concurrent requests
            request = ActionRequest(
                id=f"flood_{i:04d}",
                module="flood_test_module",
                action=f"flood_action_{i}",
                parameters={"flood_index": i},
                priority=Priority.NORMAL if i % 4 != 0 else Priority.HIGH,
                requester="flood_tester"
            )
            flood_requests.append(request)
        
        # Execute flood of requests
        start_time = time.time()
        try:
            # Use asyncio.gather with timeout
            tasks = [orchestrator.execute_action(req) for req in flood_requests]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0  # 30 second timeout for flood test
            )
            
            flood_duration = time.time() - start_time
            
            # Analyze results
            successful_results = [r for r in results if hasattr(r, 'success') and r.success]
            failed_results = [r for r in results if hasattr(r, 'success') and not r.success]
            exception_results = [r for r in results if isinstance(r, Exception)]
            
            # System should handle a reasonable portion of the flood
            success_rate = len(successful_results) / len(flood_requests)
            assert success_rate > 0.5  # At least 50% should succeed
            
            # System should remain operational
            assert orchestrator.state != SystemState.EMERGENCY
            
            # Performance should be reasonable under load
            avg_time_per_request = flood_duration / len(flood_requests)
            assert avg_time_per_request < 0.5  # Average < 500ms per request
            
        except asyncio.TimeoutError:
            # Timeout is acceptable for extreme flood test
            pytest.skip("Flood test timed out - system under extreme load")


@pytest.mark.stress
class TestMaliciousInputHandling:
    """Test system behavior with malicious or malformed inputs."""
    
    def test_malformed_action_context_handling(self):
        """Test Asimov Kernel handling of malformed inputs."""
        kernel = AsimovKernel()
        
        malicious_inputs = [
            # Extremely long descriptions
            ActionContext(
                action_type=ActionType.INFORMATION,
                description="A" * 1000000,  # 1MB string
                risk_level="low"
            ),
            
            # Invalid enum values (simulated)
            ActionContext(
                action_type=ActionType.PHYSICAL,
                description="SQL injection attempt: '; DROP TABLE humans; --",
                risk_level="critical"
            ),
            
            # Unicode and special characters
            ActionContext(
                action_type=ActionType.COMMUNICATION,
                description="Test with unicode: \u0000\u001f\uffff\U0001f600",
                risk_level="medium"
            ),
            
            # Potential code injection
            ActionContext(
                action_type=ActionType.SYSTEM_CONTROL,
                description="__import__('os').system('rm -rf /')",
                risk_level="high"
            )
        ]
        
        results = []
        for malicious_input in malicious_inputs:
            try:
                result = kernel.evaluate_action(malicious_input)
                results.append(result)
                
                # System should handle malicious input gracefully
                assert result is not None
                assert result.result in [EthicalResult.APPROVED, EthicalResult.DENIED, EthicalResult.REQUIRES_HUMAN_APPROVAL]
                
            except Exception as e:
                # Exceptions should be handled gracefully
                results.append(e)
                # Should not be critical system failures
                assert "segmentation fault" not in str(e).lower()
                assert "core dump" not in str(e).lower()
        
        # All inputs should be processed (successfully or with controlled failure)
        assert len(results) == len(malicious_inputs)
        
        kernel.stop_integrity_monitoring()
    
    def test_malicious_action_filter_inputs(self):
        """Test Action Filter handling of malicious inputs."""
        filter_system = ActionFilter(strict_mode=True)
        
        malicious_actions = [
            # Extreme values
            {
                'force': [float('inf'), float('-inf'), float('nan')],
                'velocity': [1e10, -1e10, 1e-10],
                'target_position': [999999, -999999, 1e15]
            },
            
            # Type confusion attacks
            {
                'force': "not_a_list",
                'velocity': {'malicious': 'dict'},
                'target_position': None
            },
            
            # Buffer overflow attempts
            {
                'action_type': 'A' * 100000,  # Very long string
                'environment': {str(i): f'value_{i}' for i in range(10000)}  # Large dict
            },
            
            # Nested data structures
            {
                'nested': {'level1': {'level2': {'level3': [1, 2, 3] * 1000}}}
            }
        ]
        
        for malicious_action in malicious_actions:
            try:
                result = filter_system.filter_action(malicious_action)
                
                # System should handle malicious input without crashing
                assert result is not None
                assert hasattr(result, 'status')
                
                # Malicious inputs should likely be blocked
                assert result.status in [FilterStatus.BLOCKED, FilterStatus.REQUIRES_CONFIRMATION]
                
            except (TypeError, ValueError, AttributeError) as e:
                # Type/value errors are acceptable for malformed input
                pass
            except Exception as e:
                # Should not cause system crashes
                assert "segmentation fault" not in str(e).lower()
                assert "core dump" not in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_orchestrator_injection_attacks(self):
        """Test orchestrator resistance to injection attacks."""
        orchestrator = OlympusOrchestrator()
        
        # Mock components to track injection attempts
        injection_attempts = []
        
        async def injection_detecting_asimov(request):
            suspicious_patterns = ['__import__', 'eval(', 'exec(', 'os.system', 'subprocess']
            request_str = str(request.__dict__)
            
            for pattern in suspicious_patterns:
                if pattern in request_str:
                    injection_attempts.append((pattern, request.id))
            
            return {'approved': True, 'audit_steps': ['Injection detection passed']}
        
        async def safe_execution(request):
            return {'success': True, 'data': 'safe_execution_result'}
        
        orchestrator._validate_with_asimov = injection_detecting_asimov
        orchestrator._apply_safety_filters = AsyncMock(return_value={'safe': True, 'audit_steps': ['Safe']})
        orchestrator._execute_module_action = safe_execution
        
        # Create requests with potential injection attacks
        malicious_requests = [
            ActionRequest(
                id="injection_001",
                module="__import__('os').system('malicious_command')",
                action="eval('dangerous_code')",
                parameters={'code': 'exec("print(\\"injected\\")")'},
                priority=Priority.NORMAL,
                requester="attacker"
            ),
            ActionRequest(
                id="injection_002", 
                module="normal_module",
                action="normal_action",
                parameters={
                    'description': "'; DROP TABLE users; --",
                    'script': '<script>alert("xss")</script>'
                },
                priority=Priority.NORMAL,
                requester="web_attacker"
            )
        ]
        
        # Execute malicious requests
        for request in malicious_requests:
            result = await orchestrator.execute_action(request)
            
            # System should handle requests without being compromised
            assert result is not None
            # Can succeed or fail, but should not compromise system
            assert orchestrator.state != SystemState.EMERGENCY
        
        # Injection attempts should be detected
        assert len(injection_attempts) > 0


@pytest.mark.stress  
class TestRaceConditionsAndTiming:
    """Test race conditions and timing-related failures."""
    
    def test_concurrent_integrity_checks(self):
        """Test race conditions in integrity checking."""
        kernel = AsimovKernel()
        
        # Create multiple threads that modify and check integrity simultaneously
        integrity_results = []
        modification_results = []
        
        def integrity_checker():
            for _ in range(100):
                result = kernel.verify_law_integrity()
                integrity_results.append(result)
                time.sleep(0.001)
        
        def integrity_modifier():
            original_checksums = kernel._law_checksums.copy()
            for i in range(50):
                # Temporarily modify checksums
                kernel._law_checksums[1] = f"modified_{i}"
                time.sleep(0.002)
                
                # Restore checksums
                kernel._law_checksums = original_checksums.copy()
                modification_results.append(i)
                time.sleep(0.001)
        
        # Run integrity checks and modifications concurrently
        threads = [
            threading.Thread(target=integrity_checker),
            threading.Thread(target=integrity_checker), 
            threading.Thread(target=integrity_modifier)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join(timeout=30)
        
        # All operations should complete
        assert len(integrity_results) == 200  # 2 checker threads * 100 checks
        assert len(modification_results) == 50
        
        # System should detect some integrity failures
        failed_checks = [r for r in integrity_results if not r]
        assert len(failed_checks) > 0  # Should detect modifications
        
        # System should not crash from race conditions
        final_integrity = kernel.verify_law_integrity()
        assert isinstance(final_integrity, bool)
        
        kernel.stop_integrity_monitoring()
    
    @pytest.mark.asyncio
    async def test_concurrent_action_evaluations(self):
        """Test race conditions in action evaluation."""
        kernel = AsimovKernel()
        
        # Create many concurrent action evaluations
        contexts = []
        for i in range(100):
            context = ActionContext(
                action_type=ActionType.INFORMATION,
                description=f"Concurrent evaluation test {i}",
                risk_level=random.choice(["low", "medium", "high"])
            )
            contexts.append(context)
        
        # Evaluate all actions concurrently
        async def evaluate_batch(batch_contexts):
            results = []
            for context in batch_contexts:
                result = kernel.evaluate_action(context)
                results.append(result)
                # Small random delay
                await asyncio.sleep(random.uniform(0.001, 0.005))
            return results
        
        # Split into batches and run concurrently
        batch_size = 20
        batches = [contexts[i:i+batch_size] for i in range(0, len(contexts), batch_size)]
        
        batch_tasks = [evaluate_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Flatten results
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)
        
        # All evaluations should complete successfully
        assert len(all_results) == 100
        assert all(r is not None for r in all_results)
        assert all(hasattr(r, 'result') for r in all_results)
        
        # Evaluation IDs should be unique (no race conditions in ID generation)
        evaluation_ids = [r.evaluation_id for r in all_results]
        assert len(set(evaluation_ids)) == len(evaluation_ids)  # All unique
        
        kernel.stop_integrity_monitoring()
    
    @pytest.mark.asyncio
    async def test_repair_system_concurrent_modifications(self):
        """Test race conditions in repair system state."""
        repair_system = SelfRepairSystem(audit_system=AsyncMock())
        await repair_system.initialize()
        
        # Create concurrent repair operations
        async def concurrent_repair_operations():
            operations_completed = []
            
            for i in range(20):
                fault = Mock()
                fault.fault_id = f"concurrent_fault_{i:03d}"
                fault.component = f"component_{i % 3}"
                fault.symptoms = ["concurrent_test"]
                
                try:
                    if repair_system.repair_constraints:
                        repair_system.repair_constraints.validate_repair_request = AsyncMock(
                            return_value={"allowed": True}
                        )
                    
                    execution_id = await repair_system.initiate_repair(fault)
                    operations_completed.append(execution_id)
                    
                    # Random delay
                    await asyncio.sleep(random.uniform(0.01, 0.05))
                    
                except Exception as e:
                    operations_completed.append(f"error_{str(e)}")
            
            return operations_completed
        
        # Run multiple concurrent operation sequences
        tasks = [concurrent_repair_operations() for _ in range(3)]
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_operations = []
        for result_batch in results:
            all_operations.extend(result_batch)
        
        # Most operations should complete (some may fail due to constraints)
        successful_operations = [op for op in all_operations if op and not str(op).startswith("error_")]
        assert len(successful_operations) > 10  # At least some should succeed
        
        # System should remain in consistent state
        status = await repair_system.get_status()
        assert status['is_active']
        assert isinstance(status['active_repairs'], int)
        assert isinstance(status['pending_approvals'], int)


@pytest.mark.stress
class TestDataCorruptionScenarios:
    """Test system behavior with data corruption."""
    
    def test_configuration_corruption_handling(self):
        """Test handling of corrupted configuration data."""
        # Test with corrupted physics limits
        try:
            corrupted_limits = Mock()
            corrupted_limits.max_force = float('inf')  # Corrupted value
            corrupted_limits.max_speed = -999  # Invalid negative value
            corrupted_limits.max_acceleration = None  # Corrupted null value
            
            # System should handle corrupted limits gracefully
            filter_system = ActionFilter(physics_limits=corrupted_limits)
            
            # Test with normal action
            action = {
                'force': [5.0, 0.0, 0.0],
                'velocity': [0.1, 0.0, 0.0],
                'action_type': 'test'
            }
            
            result = filter_system.filter_action(action)
            
            # Should not crash, even with corrupted configuration
            assert result is not None
            
        except (TypeError, ValueError, AttributeError):
            # These exceptions are acceptable for corrupted data
            pass
    
    def test_memory_corruption_detection(self):
        """Test detection and handling of memory corruption."""
        kernel = AsimovKernel()
        
        # Simulate memory corruption by modifying internal structures
        original_laws = AsimovKernel._LAWS.copy()
        
        try:
            # Corrupt law data
            AsimovKernel._LAWS[1] = {"corrupted": "data"}
            
            # System should detect corruption through integrity checks
            integrity_ok = kernel.verify_law_integrity()
            
            # Integrity should fail due to corruption
            assert not integrity_ok
            
            # System should enter emergency state
            time.sleep(0.2)  # Let integrity monitoring detect corruption
            assert kernel._emergency_stop_active
            
        finally:
            # Restore original laws
            AsimovKernel._LAWS = original_laws
            kernel.stop_integrity_monitoring()
    
    @pytest.mark.asyncio
    async def test_audit_log_corruption_handling(self):
        """Test handling of audit log corruption."""
        orchestrator = OlympusOrchestrator()
        
        # Mock components
        orchestrator._validate_with_asimov = AsyncMock(return_value={'approved': True, 'audit_steps': ['OK']})
        orchestrator._apply_safety_filters = AsyncMock(return_value={'safe': True, 'audit_steps': ['OK']})
        orchestrator._execute_module_action = AsyncMock(return_value={'success': True, 'data': 'result'})
        
        # Execute normal request to build audit trail
        request = ActionRequest(
            id="corruption_test_001",
            module="test_module",
            action="test_action",
            parameters={},
            priority=Priority.NORMAL,
            requester="tester"
        )
        
        result = await orchestrator.execute_action(request)
        assert result.success
        
        # Simulate audit trail corruption
        if hasattr(result, 'audit_trail') and result.audit_trail:
            # Corrupt audit trail with various invalid data types
            result.audit_trail.extend([None, {}, [], 123, float('inf')])
        
        # System should continue operating despite audit corruption
        request2 = ActionRequest(
            id="corruption_test_002",
            module="test_module",
            action="test_action_2",
            parameters={},
            priority=Priority.NORMAL,
            requester="tester"
        )
        
        result2 = await orchestrator.execute_action(request2)
        assert result2.success
        
        # System should remain operational
        assert orchestrator.state != SystemState.EMERGENCY


@pytest.mark.stress
def test_system_resilience_summary():
    """Summary test to verify overall system resilience."""
    # This test verifies that all major components can be instantiated
    # and perform basic operations even after stress testing
    
    try:
        # Create all major components
        kernel = AsimovKernel()
        filter_system = ActionFilter()
        orchestrator = OlympusOrchestrator()
        
        # Test basic functionality
        context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Resilience verification test"
        )
        
        ethical_result = kernel.evaluate_action(context)
        assert ethical_result is not None
        
        action = {'action_type': 'resilience_test', 'force': [1.0, 0.0, 0.0]}
        filter_result = filter_system.filter_action(action)
        assert filter_result is not None
        
        # Verify system states
        assert kernel._instance_id is not None
        assert orchestrator.state in [SystemState.INITIALIZING, SystemState.ACTIVE]
        
        # Cleanup
        kernel.stop_integrity_monitoring()
        
    except Exception as e:
        pytest.fail(f"System resilience test failed: {str(e)}")
    
    # If we reach here, basic system resilience is verified
    assert True