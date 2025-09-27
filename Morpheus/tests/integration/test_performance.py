"""Performance benchmarking tests for MORPHEUS system.

Tests system performance under various load conditions including:
- Processing throughput and latency measurements
- Memory usage optimization
- Concurrent processing capabilities
- Real-time performance validation
- Scalability testing
"""

import pytest
import numpy as np
import time
import threading
import psutil
import os
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import tempfile
import yaml
from pathlib import Path

from morpheus.dream_sim.dream_orchestrator import DreamOrchestrator, DreamConfig
from morpheus.integration.material_bridge import MaterialBridge
from morpheus.perception.tactile_processor import TactileProcessor, TactileProcessorConfig
from morpheus.core.types import SensoryExperience, ContactPoint, Vector3D, TactileSignature


class TestPerformanceBenchmarks:
    """Performance benchmarking and stress testing."""
    
    @pytest.fixture
    def temp_gasm_directory(self):
        """Create minimal GASM directory for performance testing."""
        sample_config = {
            'materials': {
                'steel': {
                    'color': [0.7, 0.7, 0.7, 1.0],
                    'friction': 0.8,
                    'restitution': 0.2,
                    'density': 7850,
                    'young_modulus': 200e9,
                    'poisson_ratio': 0.3
                },
                'aluminum': {
                    'color': [0.9, 0.9, 0.9, 1.0],
                    'friction': 0.6,
                    'restitution': 0.4,
                    'density': 2700,
                    'young_modulus': 70e9,
                    'poisson_ratio': 0.33
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            gasm_path = Path(temp_dir)
            config_dir = gasm_path / "assets" / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            config_file = config_dir / "simulation_params.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(sample_config, f)
            
            yield gasm_path
    
    @pytest.fixture
    def material_bridge(self, temp_gasm_directory):
        """Create material bridge for performance testing."""
        return MaterialBridge(temp_gasm_directory)
    
    @pytest.fixture
    def tactile_processor(self, material_bridge):
        """Create tactile processor optimized for performance testing."""
        config = TactileProcessorConfig(
            sensitivity=0.01,
            max_contact_points=20,  # Reasonable limit for performance
            enable_vibration_analysis=True,
            enable_texture_classification=True,
            sampling_rate=1000
        )
        return TactileProcessor(config, material_bridge)
    
    @pytest.fixture
    def mock_database(self):
        """Create high-performance mock database."""
        db = Mock()
        
        # Fast mock operations
        db.store_experience.return_value = 1
        db.get_recent_experiences.return_value = []
        db.store_dream_session.return_value = "session-1"
        db.store_learned_strategy.return_value = True
        
        return db
    
    @pytest.fixture
    def dream_orchestrator(self, mock_database, material_bridge):
        """Create dream orchestrator for performance testing."""
        config = DreamConfig(
            parallel_dreams=4,  # Use multiple cores
            max_iterations=20,
            min_improvement=0.05,
            neural_learning_rate=0.001
        )
        return DreamOrchestrator(mock_database, material_bridge, config)
    
    def generate_contact_pattern(self, num_contacts: int, force_range: tuple = (0.5, 5.0)):
        """Generate realistic contact pattern for testing."""
        contacts = []
        
        for i in range(num_contacts):
            contact = {
                'position': np.random.uniform(-0.02, 0.02, 3).tolist(),
                'normal': [0, 0, 1],
                'force': np.random.uniform(*force_range),
                'object_a': 1,
                'object_b': 2,
                'link_a': i,
                'link_b': 0
            }
            contacts.append(contact)
        
        return contacts
    
    def test_tactile_processing_throughput(self, tactile_processor):
        """Benchmark tactile processing throughput."""
        num_samples = 1000
        materials = ['steel', 'aluminum']
        
        # Prepare test data
        test_contacts = [
            self.generate_contact_pattern(np.random.randint(1, 6))
            for _ in range(num_samples)
        ]
        
        # Benchmark processing
        start_time = time.time()
        successful_processes = 0
        processing_times = []
        
        for i, contacts in enumerate(test_contacts):
            iteration_start = time.time()
            
            signature = tactile_processor.process_contacts(
                contact_points=contacts,
                material_name=materials[i % len(materials)],
                timestamp=time.time()
            )
            
            iteration_time = time.time() - iteration_start
            processing_times.append(iteration_time)
            
            if signature is not None:
                successful_processes += 1
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        throughput = successful_processes / total_time  # samples per second
        avg_latency = np.mean(processing_times) * 1000  # milliseconds
        p95_latency = np.percentile(processing_times, 95) * 1000
        p99_latency = np.percentile(processing_times, 99) * 1000
        success_rate = successful_processes / num_samples
        
        # Performance assertions
        assert throughput > 200  # At least 200 samples/second
        assert avg_latency < 5.0  # Average latency < 5ms
        assert p95_latency < 10.0  # 95th percentile < 10ms
        assert p99_latency < 20.0  # 99th percentile < 20ms
        assert success_rate > 0.98  # >98% success rate
        
        print(f"Tactile Processing Performance:")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  Average Latency: {avg_latency:.2f}ms")
        print(f"  P95 Latency: {p95_latency:.2f}ms")
        print(f"  P99 Latency: {p99_latency:.2f}ms")
        print(f"  Success Rate: {success_rate:.2%}")
    
    def test_material_bridge_caching_performance(self, material_bridge):
        """Benchmark material bridge caching performance."""
        materials = ['steel', 'aluminum']
        num_computations = 5000
        
        # Clear cache first
        material_bridge.clear_cache()
        
        # Test without cache (cold)
        start_time = time.time()
        for i in range(num_computations // 2):
            material = materials[i % len(materials)]
            material_bridge.compute_tactile_signature(
                material_name=material,
                contact_force=1.0 + (i % 10) * 0.1,  # Vary parameters
                contact_velocity=0.1 * (i % 5)
            )
        cold_time = time.time() - start_time
        
        # Test with cache (warm)
        start_time = time.time()
        for i in range(num_computations // 2):
            material = materials[i % len(materials)]
            material_bridge.compute_tactile_signature(
                material_name=material,
                contact_force=1.0,  # Same parameters (should hit cache)
                contact_velocity=0.1
            )
        warm_time = time.time() - start_time
        
        # Calculate metrics
        cold_throughput = (num_computations // 2) / cold_time
        warm_throughput = (num_computations // 2) / warm_time
        speedup = warm_throughput / cold_throughput
        
        # Get cache statistics
        cache_stats = material_bridge.get_cache_stats()
        
        # Performance assertions
        assert speedup > 5.0  # Cache should provide significant speedup
        assert warm_throughput > 1000  # Should be very fast with cache
        assert cache_stats['tactile_signature']['hits'] > 0
        
        print(f"Caching Performance:")
        print(f"  Cold Throughput: {cold_throughput:.1f} ops/sec")
        print(f"  Warm Throughput: {warm_throughput:.1f} ops/sec")
        print(f"  Cache Speedup: {speedup:.1f}x")
        print(f"  Cache Hit Rate: {cache_stats['tactile_signature']['hits']}")
    
    def test_concurrent_processing_scalability(self, tactile_processor):
        """Test concurrent processing scalability."""
        num_threads_list = [1, 2, 4, 8]
        samples_per_thread = 100
        
        results = {}
        
        for num_threads in num_threads_list:
            # Test concurrent processing
            start_time = time.time()
            results_queue = []
            threads = []
            
            def worker_function(worker_id):
                worker_results = []
                for i in range(samples_per_thread):
                    contacts = self.generate_contact_pattern(3)
                    
                    signature = tactile_processor.process_contacts(
                        contact_points=contacts,
                        material_name='steel',
                        timestamp=time.time()
                    )
                    
                    worker_results.append(signature is not None)
                results_queue.append(worker_results)
            
            # Start threads
            for i in range(num_threads):
                thread = threading.Thread(target=worker_function, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            # Calculate metrics
            total_samples = num_threads * samples_per_thread
            successful_samples = sum(sum(worker_results) for worker_results in results_queue)
            throughput = successful_samples / total_time
            success_rate = successful_samples / total_samples
            
            results[num_threads] = {
                'throughput': throughput,
                'success_rate': success_rate,
                'total_time': total_time
            }
        
        # Analyze scalability
        baseline_throughput = results[1]['throughput']
        
        print(f"Concurrent Processing Scalability:")
        for num_threads in num_threads_list:
            result = results[num_threads]
            speedup = result['throughput'] / baseline_throughput
            print(f"  {num_threads} threads: {result['throughput']:.1f} ops/sec "
                  f"(speedup: {speedup:.1f}x, success: {result['success_rate']:.2%})")
        
        # Performance assertions
        assert results[2]['throughput'] > baseline_throughput * 1.5  # 2x threads should be >1.5x faster
        assert results[4]['throughput'] > baseline_throughput * 2.0  # 4x threads should be >2x faster
        
        # All configurations should maintain high success rates
        for result in results.values():
            assert result['success_rate'] > 0.95
    
    def test_memory_usage_optimization(self, tactile_processor):
        """Test memory usage under sustained load."""
        process = psutil.Process(os.getpid())
        
        # Baseline memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Sustained processing load
        num_iterations = 2000
        memory_measurements = []
        
        for i in range(num_iterations):
            # Generate contacts
            contacts = self.generate_contact_pattern(np.random.randint(1, 8))
            
            # Process
            signature = tactile_processor.process_contacts(
                contact_points=contacts,
                material_name='steel',
                timestamp=time.time() + i * 0.001
            )
            
            # Measure memory every 100 iterations
            if i % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory - initial_memory)
        
        # Analyze memory usage
        max_memory_increase = max(memory_measurements)
        final_memory_increase = memory_measurements[-1]
        memory_growth_rate = (final_memory_increase - memory_measurements[0]) / len(memory_measurements)
        
        # Performance assertions
        assert max_memory_increase < 100  # Max memory increase < 100MB
        assert final_memory_increase < 50  # Final increase < 50MB
        assert abs(memory_growth_rate) < 1.0  # Minimal memory growth per iteration
        
        print(f"Memory Usage:")
        print(f"  Initial Memory: {initial_memory:.1f}MB")
        print(f"  Max Increase: {max_memory_increase:.1f}MB")
        print(f"  Final Increase: {final_memory_increase:.1f}MB")
        print(f"  Growth Rate: {memory_growth_rate:.3f}MB per 100 iterations")
    
    def test_dream_orchestrator_performance(self, dream_orchestrator, mock_database):
        """Benchmark dream orchestrator performance."""
        # Create large dataset of experiences
        num_experiences = 1000
        experiences = []
        
        for i in range(num_experiences):
            exp = {
                'id': i,
                'primary_material': ['steel', 'aluminum'][i % 2],
                'action_type': ['grasp', 'push', 'slide'][i % 3],
                'forces': [2.0 + (i % 10) * 0.1, 1.0, 0.5],
                'action_params': {
                    'force': 10.0 + (i % 20) * 0.5,
                    'duration': 1.0 + (i % 10) * 0.1
                },
                'success': i % 3 != 0,  # 2/3 success rate
                'reward': np.random.uniform(0.1, 1.0),
                'timestamp': time.time() + i,
                'fused_embedding': np.random.randn(128).tolist()
            }
            experiences.append(exp)
        
        mock_database.get_recent_experiences.return_value = experiences
        
        # Benchmark dream session
        start_time = time.time()
        session = dream_orchestrator.dream(duration_seconds=10.0)
        total_time = time.time() - start_time
        
        # Calculate metrics
        experiences_per_second = session.experiences_processed / total_time
        variations_per_second = session.variations_generated / total_time
        
        # Performance assertions
        assert session.experiences_processed > 100  # Should process many experiences
        assert experiences_per_second > 50  # At least 50 experiences/sec
        assert total_time < 15.0  # Should complete within reasonable time
        
        # Check neural convergence
        assert 0 <= session.neural_convergence <= 1
        
        print(f"Dream Orchestrator Performance:")
        print(f"  Experiences Processed: {session.experiences_processed}")
        print(f"  Processing Rate: {experiences_per_second:.1f} exp/sec")
        print(f"  Variations Generated: {session.variations_generated}")
        print(f"  Variation Rate: {variations_per_second:.1f} var/sec")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Neural Convergence: {session.neural_convergence:.3f}")
    
    def test_real_time_processing_simulation(self, tactile_processor):
        """Test real-time processing capabilities."""
        # Simulate real-time tactile stream at different frequencies
        frequencies = [100, 500, 1000]  # Hz
        
        for freq in frequencies:
            dt = 1.0 / freq
            duration = 1.0  # 1 second test
            expected_samples = int(duration * freq)
            
            print(f"Testing {freq}Hz real-time processing...")
            
            start_time = time.time()
            processed_count = 0
            late_count = 0
            
            for i in range(expected_samples):
                iteration_start = time.time()
                
                # Generate contacts
                contacts = self.generate_contact_pattern(3)
                
                # Process
                signature = tactile_processor.process_contacts(
                    contact_points=contacts,
                    material_name='steel',
                    timestamp=start_time + i * dt
                )
                
                if signature is not None:
                    processed_count += 1
                
                processing_time = time.time() - iteration_start
                
                # Check if processing exceeded time budget
                if processing_time > dt:
                    late_count += 1
                
                # Maintain timing
                elapsed = time.time() - start_time
                expected_elapsed = (i + 1) * dt
                if elapsed < expected_elapsed:
                    time.sleep(expected_elapsed - elapsed)
            
            # Calculate metrics
            total_elapsed = time.time() - start_time
            actual_frequency = processed_count / total_elapsed
            on_time_rate = (expected_samples - late_count) / expected_samples
            processing_success_rate = processed_count / expected_samples
            
            print(f"  Target Frequency: {freq}Hz")
            print(f"  Actual Frequency: {actual_frequency:.1f}Hz")
            print(f"  On-Time Rate: {on_time_rate:.2%}")
            print(f"  Success Rate: {processing_success_rate:.2%}")
            
            # Performance assertions (more lenient for higher frequencies)
            if freq <= 500:
                assert on_time_rate > 0.95  # 95% on-time for ≤500Hz
                assert processing_success_rate > 0.98  # 98% success
            else:
                assert on_time_rate > 0.80  # 80% on-time for 1000Hz
                assert processing_success_rate > 0.95  # 95% success
    
    def test_batch_processing_efficiency(self, material_bridge):
        """Test batch processing efficiency."""
        batch_sizes = [1, 10, 50, 100, 500]
        materials = ['steel', 'aluminum']
        
        results = {}
        
        for batch_size in batch_sizes:
            # Prepare batch
            batch_requests = []
            for i in range(batch_size):
                request = {
                    'material': materials[i % len(materials)],
                    'contact_force': 1.0 + (i % 10) * 0.1,
                    'contact_velocity': 0.1 * (i % 5)
                }
                batch_requests.append(request)
            
            # Process batch
            start_time = time.time()
            
            batch_results = []
            for request in batch_requests:
                signature = material_bridge.compute_tactile_signature(
                    material_name=request['material'],
                    contact_force=request['contact_force'],
                    contact_velocity=request['contact_velocity']
                )
                batch_results.append(signature)
            
            processing_time = time.time() - start_time
            
            # Calculate metrics
            throughput = batch_size / processing_time
            time_per_item = processing_time / batch_size * 1000  # ms
            
            results[batch_size] = {
                'throughput': throughput,
                'time_per_item': time_per_item,
                'total_time': processing_time
            }
        
        print(f"Batch Processing Efficiency:")
        for batch_size in batch_sizes:
            result = results[batch_size]
            print(f"  Batch Size {batch_size:3d}: {result['throughput']:6.1f} ops/sec, "
                  f"{result['time_per_item']:5.2f}ms per item")
        
        # Should see improved efficiency with larger batches (due to caching)
        assert results[100]['throughput'] > results[1]['throughput'] * 5
        assert results[500]['throughput'] > results[10]['throughput'] * 2
    
    def test_stress_test_high_load(self, tactile_processor, material_bridge):
        """Stress test with high sustained load."""
        # High load parameters
        duration_seconds = 30
        target_frequency = 200  # 200Hz
        num_concurrent_threads = 4
        
        print(f"Starting stress test: {duration_seconds}s at {target_frequency}Hz with {num_concurrent_threads} threads...")
        
        # Shared metrics
        results = {
            'processed_count': 0,
            'error_count': 0,
            'start_time': time.time()
        }
        results_lock = threading.Lock()
        
        def stress_worker(worker_id):
            """Worker function for stress testing."""
            worker_processed = 0
            worker_errors = 0
            
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                try:
                    # Generate work
                    contacts = self.generate_contact_pattern(np.random.randint(1, 6))
                    
                    # Process tactile
                    signature = tactile_processor.process_contacts(
                        contact_points=contacts,
                        material_name=np.random.choice(['steel', 'aluminum']),
                        timestamp=time.time()
                    )
                    
                    # Compute material properties
                    material_bridge.compute_tactile_signature(
                        material_name=np.random.choice(['steel', 'aluminum']),
                        contact_force=np.random.uniform(0.5, 5.0)
                    )
                    
                    worker_processed += 1
                    
                    # Control frequency
                    time.sleep(1.0 / target_frequency / num_concurrent_threads)
                    
                except Exception as e:
                    worker_errors += 1
                    print(f"Worker {worker_id} error: {e}")
            
            # Update shared results
            with results_lock:
                results['processed_count'] += worker_processed
                results['error_count'] += worker_errors
        
        # Start stress workers
        threads = []
        for i in range(num_concurrent_threads):
            thread = threading.Thread(target=stress_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Monitor system resources during test
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = initial_memory
        
        # Wait for completion while monitoring
        for thread in threads:
            thread.join()
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
        
        # Calculate final metrics
        total_time = time.time() - results['start_time']
        actual_frequency = results['processed_count'] / total_time
        error_rate = results['error_count'] / (results['processed_count'] + results['error_count'])
        memory_increase = peak_memory - initial_memory
        
        print(f"Stress Test Results:")
        print(f"  Target Frequency: {target_frequency}Hz")
        print(f"  Actual Frequency: {actual_frequency:.1f}Hz")
        print(f"  Error Rate: {error_rate:.2%}")
        print(f"  Memory Increase: {memory_increase:.1f}MB")
        print(f"  Total Processed: {results['processed_count']}")
        
        # Stress test assertions
        assert actual_frequency > target_frequency * 0.8  # Achieve 80% of target
        assert error_rate < 0.02  # Less than 2% error rate
        assert memory_increase < 200  # Less than 200MB memory increase
        assert results['processed_count'] > duration_seconds * target_frequency * 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])  # -s to see print output