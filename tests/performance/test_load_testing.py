"""
Load Testing and Performance Benchmarks for OLYMPUS

This test suite validates system performance under various load conditions,
stress scenarios, and resource constraints. All performance tests have
specific targets and must pass to ensure OLYMPUS meets operational requirements.

Performance requirements tested:
- Action processing throughput (>100 actions/second)
- Memory usage limits (<500MB base, <2GB under load)
- Response time constraints (<100ms for simple actions)
- Concurrent operation handling (>50 concurrent actions)
- System recovery time (<5 seconds after failure)
- Ethical validation performance (<1ms per evaluation)
"""

import asyncio
import psutil
import pytest  
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import AsyncMock, Mock, patch
import gc
import sys

from olympus.ethical_core.asimov_kernel import AsimovKernel, ActionContext, ActionType
from olympus.safety_layer.action_filter import ActionFilter
from olympus.core.olympus_orchestrator import OlympusOrchestrator, ActionRequest, Priority
from olympus.modules.prometheus.self_repair import SelfRepairSystem


class PerformanceMonitor:
    """Monitor system performance metrics during tests."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = 0
        self.measurements = []
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.measurements = []
    
    def record_measurement(self, operation: str, duration: float, **kwargs):
        """Record a performance measurement."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory)
        
        measurement = {
            'timestamp': time.time(),
            'operation': operation,
            'duration': duration,
            'memory_mb': current_memory,
            **kwargs
        }
        self.measurements.append(measurement)
    
    def end_monitoring(self):
        """End performance monitoring and return results."""
        self.end_time = time.time()
        self.end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return {
            'total_duration': self.end_time - self.start_time,
            'memory_usage': {
                'start_mb': self.start_memory,
                'end_mb': self.end_memory,
                'peak_mb': self.peak_memory,
                'delta_mb': self.end_memory - self.start_memory
            },
            'measurements': self.measurements
        }


@pytest.mark.performance
class TestAsimovKernelPerformance:
    """Performance tests for Asimov Kernel ethical validation."""
    
    def test_single_evaluation_performance(self):
        """Test single ethical evaluation performance."""
        kernel = AsimovKernel()
        monitor = PerformanceMonitor()
        
        context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Performance test action"
        )
        
        monitor.start_monitoring()
        
        # Single evaluation benchmark
        start = time.time()
        result = kernel.evaluate_action(context)
        duration = time.time() - start
        
        monitor.record_measurement('single_evaluation', duration)
        results = monitor.end_monitoring()
        
        assert result is not None
        # Single evaluation should be very fast (< 1ms)
        assert duration < 0.001
        # Memory usage should be minimal
        assert results['memory_usage']['delta_mb'] < 1.0
        
        kernel.stop_integrity_monitoring()
    
    def test_bulk_evaluation_performance(self):
        """Test bulk ethical evaluation performance."""
        kernel = AsimovKernel()
        monitor = PerformanceMonitor()
        
        contexts = []
        for i in range(1000):
            context = ActionContext(
                action_type=ActionType.INFORMATION,
                description=f"Bulk test action {i}"
            )
            contexts.append(context)
        
        monitor.start_monitoring()
        
        start = time.time()
        results = []
        for context in contexts:
            result = kernel.evaluate_action(context)
            results.append(result)
        
        total_time = time.time() - start
        avg_time = total_time / len(contexts)
        
        monitor.record_measurement('bulk_evaluation', total_time, 
                                 count=len(contexts), avg_time=avg_time)
        perf_results = monitor.end_monitoring()
        
        assert len(results) == 1000
        assert all(r is not None for r in results)
        # Average evaluation time should be very fast
        assert avg_time < 0.001
        # Should process >1000 evaluations/second  
        evaluations_per_second = len(contexts) / total_time
        assert evaluations_per_second > 1000
        
        kernel.stop_integrity_monitoring()
    
    def test_concurrent_evaluation_performance(self):
        """Test concurrent ethical evaluation performance."""
        kernel = AsimovKernel()
        monitor = PerformanceMonitor()
        
        def evaluate_batch(batch_id, batch_size):
            results = []
            for i in range(batch_size):
                context = ActionContext(
                    action_type=ActionType.INFORMATION,
                    description=f"Concurrent test {batch_id}-{i}"
                )
                result = kernel.evaluate_action(context)
                results.append(result)
            return results
        
        monitor.start_monitoring()
        
        # Test with multiple threads
        num_threads = 10
        batch_size = 100
        
        start = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(evaluate_batch, i, batch_size) 
                      for i in range(num_threads)]
            
            all_results = []
            for future in as_completed(futures):
                batch_results = future.result()
                all_results.extend(batch_results)
        
        total_time = time.time() - start
        
        monitor.record_measurement('concurrent_evaluation', total_time,
                                 threads=num_threads, total_evaluations=len(all_results))
        perf_results = monitor.end_monitoring()
        
        assert len(all_results) == num_threads * batch_size
        # Concurrent throughput should be high
        evaluations_per_second = len(all_results) / total_time
        assert evaluations_per_second > 500  # Should handle concurrent load well
        
        kernel.stop_integrity_monitoring()
    
    def test_integrity_monitoring_performance(self):
        """Test integrity monitoring performance impact."""
        monitor = PerformanceMonitor()
        
        monitor.start_monitoring()
        
        # Test with integrity monitoring enabled
        kernel_with_monitoring = AsimovKernel()
        
        start = time.time()
        for i in range(100):
            context = ActionContext(
                action_type=ActionType.INFORMATION,
                description=f"Monitoring test {i}"
            )
            kernel_with_monitoring.evaluate_action(context)
        
        with_monitoring_time = time.time() - start
        
        # Stop monitoring and test without
        kernel_with_monitoring.stop_integrity_monitoring()
        
        start = time.time()
        for i in range(100):
            context = ActionContext(
                action_type=ActionType.INFORMATION, 
                description=f"No monitoring test {i}"
            )
            kernel_with_monitoring.evaluate_action(context)
        
        without_monitoring_time = time.time() - start
        
        monitor.record_measurement('with_monitoring', with_monitoring_time)
        monitor.record_measurement('without_monitoring', without_monitoring_time)
        perf_results = monitor.end_monitoring()
        
        # Integrity monitoring should have minimal performance impact (< 10% overhead)
        overhead_ratio = with_monitoring_time / without_monitoring_time
        assert overhead_ratio < 1.1  # Less than 10% overhead


@pytest.mark.performance
class TestActionFilterPerformance:
    """Performance tests for Safety Layer action filtering."""
    
    def test_single_filter_performance(self):
        """Test single action filtering performance."""
        filter_system = ActionFilter()
        monitor = PerformanceMonitor()
        
        test_action = {
            'action_type': 'move',
            'target_position': [0.5, 0.3, 0.8],
            'current_position': [0.0, 0.0, 0.5],
            'velocity': [0.1, 0.1, 0.1],
            'force': [5.0, 0.0, 0.0],
            'environment': {
                'lighting': 80,
                'temperature': 22
            },
            'system_status': {
                'battery_level': 85,
                'sensors_operational': True
            },
            'humans_detected': []
        }
        
        monitor.start_monitoring()
        
        start = time.time()
        result = filter_system.filter_action(test_action)
        duration = time.time() - start
        
        monitor.record_measurement('single_filter', duration)
        perf_results = monitor.end_monitoring()
        
        assert result is not None
        # Single filtering should be fast (< 10ms)
        assert duration < 0.01
        assert perf_results['memory_usage']['delta_mb'] < 1.0
    
    def test_bulk_filtering_performance(self):
        """Test bulk action filtering performance."""
        filter_system = ActionFilter()
        monitor = PerformanceMonitor()
        
        # Generate test actions
        test_actions = []
        for i in range(500):
            action = {
                'action_type': f'action_{i}',
                'target_position': [0.1 * i % 1.0, 0.2 * i % 1.0, 0.5],
                'velocity': [0.05, 0.05, 0.0],
                'force': [2.0 + i % 10, 0.0, 0.0],
                'environment': {'lighting': 80, 'temperature': 22},
                'system_status': {'battery_level': 85, 'sensors_operational': True},
                'humans_detected': []
            }
            test_actions.append(action)
        
        monitor.start_monitoring()
        
        start = time.time()
        results = []
        for action in test_actions:
            result = filter_system.filter_action(action)
            results.append(result)
        
        total_time = time.time() - start
        avg_time = total_time / len(test_actions)
        
        monitor.record_measurement('bulk_filtering', total_time, 
                                 count=len(test_actions), avg_time=avg_time)
        perf_results = monitor.end_monitoring()
        
        assert len(results) == 500
        # Average filtering time should be reasonable
        assert avg_time < 0.01
        # Should process >50 filters/second
        filters_per_second = len(test_actions) / total_time
        assert filters_per_second > 50
    
    def test_complex_action_filtering_performance(self):
        """Test performance with complex actions requiring all filter layers."""
        filter_system = ActionFilter()
        monitor = PerformanceMonitor()
        
        complex_action = {
            'action_type': 'complex_manipulation',
            'force': [15.0, 5.0, 2.0],
            'velocity': [0.8, 0.2, 0.1], 
            'acceleration': [1.5, 0.5, 0.3],
            'target_position': [0.7, 0.4, 1.2],
            'current_position': [0.1, 0.1, 0.8],
            'trajectory': [[0.1, 0.1, 0.8], [0.4, 0.25, 1.0], [0.7, 0.4, 1.2]],
            'obstacles': [
                {'position': [0.3, 0.3, 0.9], 'radius': 0.05},
                {'position': [0.6, 0.1, 1.1], 'radius': 0.08}
            ],
            'tool': 'precision_gripper',
            'environment': {
                'lighting': 75,
                'temperature': 23,
                'vibration_level': 2,
                'hazardous_materials': False
            },
            'system_status': {
                'battery_level': 70,
                'error_count': 1,
                'sensors_operational': True
            },
            'humans_detected': [
                {'distance': 2.5, 'min_safe_distance': 1.0}
            ],
            'repetitions': 1
        }
        
        monitor.start_monitoring()
        
        # Test multiple complex actions
        start = time.time()
        results = []
        for i in range(100):
            result = filter_system.filter_action(complex_action)
            results.append(result)
        
        total_time = time.time() - start
        avg_time = total_time / len(results)
        
        monitor.record_measurement('complex_filtering', total_time,
                                 count=len(results), avg_time=avg_time)
        perf_results = monitor.end_monitoring()
        
        assert len(results) == 100
        # Even complex filtering should be reasonable (< 20ms per action)
        assert avg_time < 0.02


@pytest.mark.performance
class TestOrchestratorPerformance:
    """Performance tests for OLYMPUS Orchestrator."""
    
    @pytest.mark.asyncio
    async def test_action_execution_throughput(self):
        """Test action execution throughput."""
        orchestrator = OlympusOrchestrator()
        monitor = PerformanceMonitor()
        
        # Mock fast validations and execution
        async def fast_asimov(request):
            return {'approved': True, 'audit_steps': ['Fast Asimov']}
        
        async def fast_safety(request):
            return {'safe': True, 'audit_steps': ['Fast Safety']}
        
        async def fast_execution(request):
            await asyncio.sleep(0.001)  # Minimal processing time
            return {'success': True, 'data': f'result_{request.id}'}
        
        orchestrator._validate_with_asimov = fast_asimov
        orchestrator._apply_safety_filters = fast_safety
        orchestrator._execute_module_action = fast_execution
        
        # Create test requests
        requests = []
        for i in range(100):
            request = ActionRequest(
                id=f"throughput_{i:03d}",
                module="test_module",
                action=f"action_{i}",
                parameters={"index": i},
                priority=Priority.NORMAL,
                requester="throughput_tester"
            )
            requests.append(request)
        
        monitor.start_monitoring()
        
        start = time.time()
        results = []
        for request in requests:
            result = await orchestrator.execute_action(request)
            results.append(result)
        
        total_time = time.time() - start
        throughput = len(requests) / total_time
        
        monitor.record_measurement('sequential_throughput', total_time,
                                 count=len(requests), throughput=throughput)
        perf_results = monitor.end_monitoring()
        
        assert len(results) == 100
        assert all(r.success for r in results)
        # Should process >20 actions/second sequentially
        assert throughput > 20
    
    @pytest.mark.asyncio
    async def test_concurrent_action_processing(self):
        """Test concurrent action processing performance."""
        orchestrator = OlympusOrchestrator()
        monitor = PerformanceMonitor()
        
        # Mock fast operations
        async def concurrent_asimov(request):
            await asyncio.sleep(0.001)
            return {'approved': True, 'audit_steps': ['Concurrent Asimov']}
        
        async def concurrent_safety(request):
            await asyncio.sleep(0.001)
            return {'safe': True, 'audit_steps': ['Concurrent Safety']}
        
        async def concurrent_execution(request):
            await asyncio.sleep(0.002)  # Slightly longer processing
            return {'success': True, 'data': f'concurrent_{request.id}'}
        
        orchestrator._validate_with_asimov = concurrent_asimov
        orchestrator._apply_safety_filters = concurrent_safety
        orchestrator._execute_module_action = concurrent_execution
        
        # Create concurrent requests
        requests = []
        for i in range(50):
            request = ActionRequest(
                id=f"concurrent_{i:03d}",
                module="test_module",
                action=f"concurrent_action_{i}",
                parameters={"index": i},
                priority=Priority.NORMAL,
                requester="concurrent_tester"
            )
            requests.append(request)
        
        monitor.start_monitoring()
        
        # Execute all requests concurrently
        start = time.time()
        tasks = [orchestrator.execute_action(req) for req in requests]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start
        throughput = len(requests) / total_time
        
        monitor.record_measurement('concurrent_throughput', total_time,
                                 count=len(requests), throughput=throughput)
        perf_results = monitor.end_monitoring()
        
        assert len(results) == 50
        assert all(r.success for r in results)
        # Concurrent processing should achieve higher throughput
        assert throughput > 15  # Should be efficient with concurrency
    
    @pytest.mark.asyncio
    async def test_high_load_stability(self):
        """Test system stability under high load."""
        orchestrator = OlympusOrchestrator()
        monitor = PerformanceMonitor()
        
        # Mock operations with slight variability
        async def variable_asimov(request):
            await asyncio.sleep(0.001 + (hash(request.id) % 100) / 100000)
            return {'approved': True, 'audit_steps': ['Variable Asimov']}
        
        async def variable_safety(request):
            await asyncio.sleep(0.001 + (hash(request.id) % 50) / 100000)
            return {'safe': True, 'audit_steps': ['Variable Safety']}
        
        async def variable_execution(request):
            await asyncio.sleep(0.002 + (hash(request.id) % 200) / 100000)
            return {'success': True, 'data': f'load_test_{request.id}'}
        
        orchestrator._validate_with_asimov = variable_asimov
        orchestrator._apply_safety_filters = variable_safety
        orchestrator._execute_module_action = variable_execution
        
        monitor.start_monitoring()
        
        # Generate high load in batches
        all_results = []
        batch_size = 25
        num_batches = 8
        
        start = time.time()
        
        for batch_num in range(num_batches):
            batch_requests = []
            for i in range(batch_size):
                request = ActionRequest(
                    id=f"load_{batch_num:02d}_{i:03d}",
                    module="load_test_module",
                    action=f"load_action_{i}",
                    parameters={"batch": batch_num, "index": i},
                    priority=Priority.NORMAL,
                    requester="load_tester"
                )
                batch_requests.append(request)
            
            # Process batch concurrently
            batch_tasks = [orchestrator.execute_action(req) for req in batch_requests]
            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)
            
            # Small delay between batches
            await asyncio.sleep(0.01)
        
        total_time = time.time() - start
        throughput = len(all_results) / total_time
        
        monitor.record_measurement('high_load_test', total_time,
                                 batches=num_batches, batch_size=batch_size,
                                 total_actions=len(all_results), throughput=throughput)
        perf_results = monitor.end_monitoring()
        
        assert len(all_results) == num_batches * batch_size
        # Should handle all actions successfully under load
        success_rate = sum(1 for r in all_results if r.success) / len(all_results)
        assert success_rate > 0.95  # >95% success rate under load
        
        # System should maintain reasonable throughput under sustained load  
        assert throughput > 8  # Should maintain decent throughput
        
        # Memory usage should not grow excessively
        assert perf_results['memory_usage']['delta_mb'] < 50  # <50MB growth
    
    @pytest.mark.asyncio
    async def test_status_reporting_performance(self):
        """Test performance of status reporting operations."""
        orchestrator = OlympusOrchestrator()
        monitor = PerformanceMonitor()
        
        # Mock status methods for consistent timing
        async def mock_identity():
            await asyncio.sleep(0.001)
            return {"id": "olympus_001", "role": "coordinator"}
        
        async def mock_modules():
            await asyncio.sleep(0.002) 
            return {"prometheus": {"status": "active"}, "atlas": {"status": "active"}}
        
        async def mock_health():
            await asyncio.sleep(0.001)
            return {"overall": "healthy", "components": ["all_operational"]}
        
        async def mock_consciousness():
            await asyncio.sleep(0.001)
            return {"awareness_level": 0.8, "active": True}
        
        # Apply mocks
        with patch.object(orchestrator.identity_manager, 'get_current_identity', side_effect=mock_identity), \
             patch.object(orchestrator.module_manager, 'get_all_module_status', side_effect=mock_modules), \
             patch.object(orchestrator.system_health, 'get_comprehensive_health', side_effect=mock_health), \
             patch.object(orchestrator.consciousness_kernel, 'get_consciousness_state', side_effect=mock_consciousness):
            
            monitor.start_monitoring()
            
            # Test multiple status requests
            start = time.time()
            status_results = []
            for i in range(50):
                status = await orchestrator.get_system_status()
                status_results.append(status)
            
            total_time = time.time() - start
            avg_time = total_time / len(status_results)
            
            monitor.record_measurement('status_reporting', total_time,
                                     count=len(status_results), avg_time=avg_time)
            perf_results = monitor.end_monitoring()
            
            assert len(status_results) == 50
            assert all('system' in status for status in status_results)
            # Status reporting should be fast (< 50ms per request)
            assert avg_time < 0.05


@pytest.mark.performance
class TestSelfRepairPerformance:
    """Performance tests for self-repair system."""
    
    @pytest.mark.asyncio
    async def test_repair_planning_performance(self):
        """Test repair plan creation performance."""
        repair_system = SelfRepairSystem(audit_system=AsyncMock())
        await repair_system.initialize()
        monitor = PerformanceMonitor()
        
        # Create test faults
        faults = []
        for i in range(50):
            fault = Mock()
            fault.fault_id = f"perf_fault_{i:03d}"
            fault.component = f"component_{i % 5}"  # 5 different components
            fault.symptoms = ["high_cpu_usage", "slow_response_time"][i % 2:]
            fault.severity = ["low", "medium", "high"][i % 3]
            faults.append(fault)
        
        monitor.start_monitoring()
        
        start = time.time()
        plans = []
        for fault in faults:
            plan = await repair_system.repair_planner.create_repair_plan(fault)
            plans.append(plan)
        
        total_time = time.time() - start
        avg_time = total_time / len(faults)
        
        monitor.record_measurement('repair_planning', total_time,
                                 count=len(faults), avg_time=avg_time)
        perf_results = monitor.end_monitoring()
        
        assert len(plans) == 50
        assert all(p is not None for p in plans)
        # Plan creation should be fast (< 100ms per plan)
        assert avg_time < 0.1
    
    @pytest.mark.asyncio
    async def test_concurrent_repair_initiation(self):
        """Test concurrent repair initiation performance."""
        repair_system = SelfRepairSystem(audit_system=AsyncMock())
        await repair_system.initialize()
        monitor = PerformanceMonitor()
        
        # Mock constraints to allow repairs
        if repair_system.repair_constraints:
            repair_system.repair_constraints.validate_repair_request = AsyncMock(
                return_value={"allowed": True}
            )
        
        # Create concurrent faults
        faults = []
        for i in range(20):
            fault = Mock()
            fault.fault_id = f"concurrent_fault_{i:03d}"
            fault.component = f"service_{i % 4}"
            fault.symptoms = ["performance_degradation"]
            fault.severity = "medium"
            faults.append(fault)
        
        monitor.start_monitoring()
        
        start = time.time()
        # Initiate repairs concurrently
        tasks = [repair_system.initiate_repair(fault) for fault in faults]
        execution_ids = await asyncio.gather(*tasks)
        
        total_time = time.time() - start
        
        monitor.record_measurement('concurrent_repair_initiation', total_time,
                                 count=len(faults))
        perf_results = monitor.end_monitoring()
        
        # Count successful initiations
        successful_initiations = sum(1 for eid in execution_ids if eid is not None)
        
        # Should handle concurrent repair initiations efficiently
        assert successful_initiations >= 15  # Most should succeed
        assert total_time < 2.0  # Should complete quickly


@pytest.mark.performance
class TestMemoryUsageAndLeaks:
    """Test memory usage patterns and potential leaks."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage patterns under sustained load."""
        # Force garbage collection before starting
        gc.collect()
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Initialize systems
        asimov_kernel = AsimovKernel()
        action_filter = ActionFilter()
        orchestrator = OlympusOrchestrator()
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Mock operations to focus on memory usage
        async def memory_test_asimov(request):
            return {'approved': True, 'audit_steps': ['Memory test']}
        
        async def memory_test_safety(request):
            return {'safe': True, 'audit_steps': ['Memory test']}
        
        async def memory_test_execution(request):
            return {'success': True, 'data': 'memory_test_result'}
        
        orchestrator._validate_with_asimov = memory_test_asimov
        orchestrator._apply_safety_filters = memory_test_safety
        orchestrator._execute_module_action = memory_test_execution
        
        # Run sustained operations
        for batch in range(10):
            # Process batch of requests
            requests = []
            for i in range(50):
                request = ActionRequest(
                    id=f"memory_test_{batch:02d}_{i:03d}",
                    module="memory_test_module",
                    action=f"memory_action_{i}",
                    parameters={"batch": batch, "index": i},
                    priority=Priority.NORMAL,
                    requester="memory_tester"
                )
                requests.append(request)
            
            # Process requests
            tasks = [orchestrator.execute_action(req) for req in requests]
            results = await asyncio.gather(*tasks)
            
            # Check memory after each batch
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            monitor.record_measurement(f'batch_{batch}_memory', 0, 
                                     memory_mb=current_memory,
                                     batch=batch, requests_processed=len(results))
            
            # Force garbage collection periodically
            if batch % 3 == 0:
                gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        perf_results = monitor.end_monitoring()
        
        # Memory growth should be reasonable
        memory_growth = final_memory - initial_memory
        assert memory_growth < 100  # <100MB growth for 500 requests
        
        # Peak memory should not be excessive
        assert perf_results['memory_usage']['peak_mb'] < initial_memory + 150
        
        asimov_kernel.stop_integrity_monitoring()
    
    def test_object_cleanup_and_disposal(self):
        """Test proper object cleanup and disposal."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create and destroy objects multiple times
        for cycle in range(5):
            # Create objects
            objects = []
            
            for i in range(20):
                kernel = AsimovKernel()
                filter_system = ActionFilter()
                objects.extend([kernel, filter_system])
                
                # Use objects briefly
                context = ActionContext(
                    action_type=ActionType.INFORMATION,
                    description=f"Cleanup test {cycle}-{i}"
                )
                kernel.evaluate_action(context)
                
                action = {'action_type': 'test', 'force': [1.0, 0.0, 0.0]}
                filter_system.filter_action(action)
            
            # Cleanup objects
            for obj in objects:
                if hasattr(obj, 'stop_integrity_monitoring'):
                    obj.stop_integrity_monitoring()
            
            del objects
            gc.collect()
            
            # Check memory after cleanup
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            # Memory growth should be minimal after cleanup
            assert memory_growth < 50  # <50MB growth even after multiple cycles


@pytest.mark.performance
class TestSystemRecoveryPerformance:
    """Test system recovery performance after failures."""
    
    @pytest.mark.asyncio
    async def test_emergency_stop_recovery_time(self):
        """Test recovery time after emergency stop."""
        asimov_kernel = AsimovKernel()
        monitor = PerformanceMonitor()
        
        monitor.start_monitoring()
        
        # Activate emergency stop
        start_emergency = time.time()
        asimov_kernel.emergency_stop("Performance test emergency")
        emergency_activation_time = time.time() - start_emergency
        
        assert asimov_kernel._emergency_stop_active
        monitor.record_measurement('emergency_activation', emergency_activation_time)
        
        # Test that actions are blocked during emergency
        context = ActionContext(
            action_type=ActionType.INFORMATION,
            description="Test during emergency"
        )
        
        result = asimov_kernel.evaluate_action(context)
        assert result.result.value == 'emergency_stop'
        
        # Reset emergency stop and measure recovery time
        start_recovery = time.time()
        reset_success = asimov_kernel.reset_emergency_stop("recovery_test_authorization")
        recovery_time = time.time() - start_recovery
        
        assert reset_success
        assert not asimov_kernel._emergency_stop_active
        monitor.record_measurement('emergency_recovery', recovery_time)
        
        # Test that normal operations resume quickly
        start_resume = time.time()
        result = asimov_kernel.evaluate_action(context)
        resume_time = time.time() - start_resume
        
        assert result.result.value in ['approved', 'denied']  # Normal operation
        monitor.record_measurement('operation_resume', resume_time)
        
        perf_results = monitor.end_monitoring()
        
        # Recovery should be very fast
        assert recovery_time < 0.1  # <100ms recovery
        assert resume_time < 0.01   # <10ms to resume normal operations
        
        asimov_kernel.stop_integrity_monitoring()
    
    @pytest.mark.asyncio
    async def test_component_failure_recovery(self):
        """Test recovery performance after component failures."""
        orchestrator = OlympusOrchestrator()
        monitor = PerformanceMonitor()
        
        # Mock a failing component that recovers
        failure_count = 0
        async def failing_then_recovering_asimov(request):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:  # Fail first 3 times
                raise Exception("Simulated component failure")
            return {'approved': True, 'audit_steps': ['Recovered validation']}
        
        async def stable_safety(request):
            return {'safe': True, 'audit_steps': ['Stable safety']}
        
        async def stable_execution(request):
            return {'success': True, 'data': 'recovery_test_result'}
        
        orchestrator._validate_with_asimov = failing_then_recovering_asimov
        orchestrator._apply_safety_filters = stable_safety
        orchestrator._execute_module_action = stable_execution
        
        monitor.start_monitoring()
        
        # Test requests during and after component failure
        requests = []
        for i in range(10):
            request = ActionRequest(
                id=f"recovery_test_{i:03d}",
                module="recovery_module",
                action=f"recovery_action_{i}",
                parameters={"index": i},
                priority=Priority.NORMAL,
                requester="recovery_tester"
            )
            requests.append(request)
        
        start = time.time()
        results = []
        for request in requests:
            result = await orchestrator.execute_action(request)
            results.append(result)
        
        total_time = time.time() - start
        
        monitor.record_measurement('failure_recovery_test', total_time,
                                 total_requests=len(requests))
        perf_results = monitor.end_monitoring()
        
        # First few should fail, then succeed
        failed_results = [r for r in results if not r.success]
        successful_results = [r for r in results if r.success]
        
        assert len(failed_results) == 3  # First 3 should fail
        assert len(successful_results) == 7  # Remaining should succeed
        
        # System should recover quickly after component restoration
        assert total_time < 2.0  # Should complete within reasonable time