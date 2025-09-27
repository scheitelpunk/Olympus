"""Integration tests for MORPHEUS system.

Tests the complete system integration including:
- Multi-modal perception pipeline
- Component interaction and data flow
- End-to-end processing workflows
- Performance benchmarks
- Error handling across components
"""

import pytest
import numpy as np
import time
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from morpheus.dream_sim.dream_orchestrator import DreamOrchestrator, DreamConfig
from morpheus.integration.material_bridge import MaterialBridge
from morpheus.perception.tactile_processor import TactileProcessor, TactileProcessorConfig
from morpheus.storage.postgres_storage import PostgreSQLStorage
from morpheus.core.types import (
    SensoryExperience, ContactPoint, Vector3D, TactileSignature,
    ActionType, MaterialType
)


class TestMORPHEUSIntegration:
    """Integration tests for the complete MORPHEUS system."""
    
    @pytest.fixture
    def sample_gasm_config(self):
        """Create sample GASM configuration for testing."""
        return {
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
                },
                'rubber': {
                    'color': [0.2, 0.2, 0.2, 1.0],
                    'friction': 1.2,
                    'restitution': 0.9,
                    'density': 1200,
                    'young_modulus': 1e6,
                    'poisson_ratio': 0.47
                }
            }
        }
    
    @pytest.fixture
    def temp_gasm_directory(self, sample_gasm_config):
        """Create temporary GASM directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gasm_path = Path(temp_dir)
            
            # Create directory structure
            config_dir = gasm_path / "assets" / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Write config file
            config_file = config_dir / "simulation_params.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(sample_gasm_config, f)
            
            yield gasm_path
    
    @pytest.fixture
    def material_bridge(self, temp_gasm_directory):
        """Create material bridge instance."""
        return MaterialBridge(temp_gasm_directory)
    
    @pytest.fixture
    def tactile_processor(self, material_bridge):
        """Create tactile processor instance."""
        config = TactileProcessorConfig(
            sensitivity=0.01,
            max_contact_points=20,
            enable_vibration_analysis=True,
            enable_texture_classification=True
        )
        return TactileProcessor(config, material_bridge)
    
    @pytest.fixture
    def mock_database(self):
        """Create mock database for testing."""
        db = Mock()
        
        # Mock experience storage
        db.store_experience.return_value = 123
        db.get_recent_experiences.return_value = []
        db.store_dream_session.return_value = "session-123"
        db.store_learned_strategy.return_value = True
        
        return db
    
    @pytest.fixture
    def dream_orchestrator(self, mock_database, material_bridge):
        """Create dream orchestrator instance."""
        config = DreamConfig(
            parallel_dreams=2,
            max_iterations=5,
            min_improvement=0.05
        )
        return DreamOrchestrator(mock_database, material_bridge, config)
    
    @pytest.fixture
    def sample_contact_points(self):
        """Generate sample contact points for testing."""
        contact_points = []
        
        # Create a pattern of contact points (simulating finger grasp)
        positions = [
            [0.02, 0.0, 0.0],    # Thumb
            [-0.01, 0.015, 0.0], # Index finger
            [-0.01, 0.005, 0.0], # Middle finger
            [-0.01, -0.005, 0.0], # Ring finger
            [-0.01, -0.015, 0.0]  # Pinky
        ]
        
        forces = [2.5, 2.0, 1.8, 1.5, 1.2]  # Decreasing forces
        
        for i, (pos, force) in enumerate(zip(positions, forces)):
            contact_point = {
                'position': pos,
                'normal': [0, 0, 1],
                'force': force,
                'object_a': 1,
                'object_b': 2,
                'link_a': i,
                'link_b': 0
            }
            contact_points.append(contact_point)
        
        return contact_points
    
    def test_material_bridge_tactile_integration(self, material_bridge, tactile_processor, sample_contact_points):
        """Test integration between material bridge and tactile processor."""
        # Process contacts for steel material
        signature = tactile_processor.process_contacts(
            contact_points=sample_contact_points,
            material_name='steel'
        )
        
        assert signature is not None
        assert isinstance(signature, TactileSignature)
        assert signature.material == 'steel'
        assert signature.total_force > 0
        assert signature.contact_area > 0
        assert signature.pressure > 0
        assert len(signature.contact_points) > 0
        
        # Verify material properties influence
        steel_hardness = signature.hardness
        
        # Process same contacts for rubber
        rubber_signature = tactile_processor.process_contacts(
            contact_points=sample_contact_points,
            material_name='rubber'
        )
        
        assert rubber_signature is not None
        assert rubber_signature.material == 'rubber'
        
        # Rubber should be softer than steel
        assert rubber_signature.hardness < steel_hardness
        
        # Different materials should have different texture descriptors
        assert signature.texture_descriptor != rubber_signature.texture_descriptor or \
               abs(signature.hardness - rubber_signature.hardness) > 0.1
    
    def test_sensory_experience_creation(self, tactile_processor, sample_contact_points):
        """Test creation of complete sensory experience."""
        # Process tactile data
        tactile_signature = tactile_processor.process_contacts(
            contact_points=sample_contact_points,
            material_name='aluminum'
        )
        
        assert tactile_signature is not None
        
        # Create complete sensory experience
        experience = SensoryExperience(
            tactile=tactile_signature,
            primary_material='aluminum',
            action_type=ActionType.GRASP,
            action_parameters={
                'grip_force': 15.0,
                'approach_speed': 0.1,
                'grasp_duration': 2.0
            },
            success=True,
            reward=0.75,
            tags=['integration_test', 'grasp', 'aluminum'],
            notes='Integration test for grasp action'
        )
        
        # Compute embeddings
        experience.compute_embeddings()
        
        assert experience.tactile_embedding is not None
        assert len(experience.tactile_embedding) > 0
        assert np.all(np.isfinite(experience.tactile_embedding))
        
        # Verify experience data consistency
        assert experience.primary_material == tactile_signature.material
        assert experience.action_type == ActionType.GRASP
        assert experience.success is True
    
    def test_dream_orchestrator_integration(self, dream_orchestrator, mock_database):
        """Test dream orchestrator integration with experiences."""
        # Create sample experiences for dreaming
        sample_experiences = []
        
        for i in range(5):
            exp = {
                'id': i + 1,
                'primary_material': ['steel', 'aluminum', 'rubber'][i % 3],
                'action_type': ['grasp', 'push', 'slide'][i % 3],
                'forces': [2.0 + i * 0.5, 1.0, 0.5],
                'action_params': {
                    'force': 10.0 + i * 2.0,
                    'duration': 1.0 + i * 0.2
                },
                'success': i % 2 == 0,  # Alternating success/failure
                'reward': 0.1 + i * 0.2,
                'timestamp': time.time() + i,
                'fused_embedding': np.random.randn(128).tolist()
            }
            sample_experiences.append(exp)
        
        # Mock database to return sample experiences
        mock_database.get_recent_experiences.return_value = sample_experiences
        
        # Run dream session
        session = dream_orchestrator.dream(duration_seconds=2.0)
        
        # Verify dream session results
        assert session is not None
        assert session.experiences_processed > 0
        assert session.end_time > session.start_time
        
        # Verify database interactions
        mock_database.get_recent_experiences.assert_called_once()
        mock_database.store_dream_session.assert_called_once()
        
        # Check session metrics
        assert hasattr(session, 'compute_metrics')
        assert 'session_duration' in session.compute_metrics
        assert session.compute_metrics['session_duration'] > 0
    
    def test_multi_material_processing_pipeline(self, tactile_processor, material_bridge):
        """Test processing pipeline with multiple materials."""
        materials = ['steel', 'aluminum', 'rubber']
        contact_forces = [3.0, 2.0, 1.0]  # Different contact forces
        
        results = []
        
        for material, force in zip(materials, contact_forces):
            # Create contact points with material-specific properties
            contact_points = [{
                'position': [0, 0, 0],
                'normal': [0, 0, 1],
                'force': force,
                'object_a': 1,
                'object_b': 2
            }]
            
            # Process tactile signature
            signature = tactile_processor.process_contacts(
                contact_points=contact_points,
                material_name=material
            )
            
            results.append({
                'material': material,
                'signature': signature,
                'force': force
            })
        
        # Verify all materials were processed
        assert len(results) == 3
        
        # Check material-specific differences
        steel_result = next(r for r in results if r['material'] == 'steel')
        rubber_result = next(r for r in results if r['material'] == 'rubber')
        
        # Steel should be harder than rubber
        assert steel_result['signature'].hardness > rubber_result['signature'].hardness
        
        # Steel should feel cooler than rubber
        assert steel_result['signature'].temperature_feel < rubber_result['signature'].temperature_feel
        
        # Different deformation characteristics
        assert abs(steel_result['signature'].deformation - rubber_result['signature'].deformation) > 0.1
    
    def test_performance_benchmarking(self, tactile_processor, sample_contact_points):
        """Test system performance with realistic workloads."""
        # Benchmark tactile processing
        num_iterations = 50
        start_time = time.time()
        
        signatures = []
        for i in range(num_iterations):
            # Vary contact points slightly
            varied_contacts = []
            for cp in sample_contact_points:
                varied_cp = cp.copy()
                varied_cp['force'] += np.random.normal(0, 0.1)  # Add noise
                varied_contacts.append(varied_cp)
            
            signature = tactile_processor.process_contacts(
                contact_points=varied_contacts,
                material_name='steel',
                timestamp=time.time() + i * 0.01
            )
            
            if signature:
                signatures.append(signature)
        
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert len(signatures) > 0
        assert processing_time < 10.0  # Should complete within 10 seconds
        
        avg_time_per_process = processing_time / num_iterations
        assert avg_time_per_process < 0.2  # Less than 200ms per processing
        
        # Check processing statistics
        stats = tactile_processor.get_processing_stats()
        assert stats['processing_count'] >= num_iterations
        assert stats['average_processing_time'] > 0
    
    def test_error_handling_integration(self, tactile_processor, dream_orchestrator, mock_database):
        """Test error handling across integrated components."""
        # Test tactile processor with invalid data
        invalid_contacts = [
            {'position': 'invalid'},  # Invalid position
            {'force': -1.0},          # Invalid force (negative)
            {}                        # Empty contact
        ]
        
        # Should handle gracefully without crashing
        signature = tactile_processor.process_contacts(
            contact_points=invalid_contacts,
            material_name='steel'
        )
        
        # May return None or valid signature depending on how many contacts are valid
        assert signature is None or isinstance(signature, TactileSignature)
        
        # Test dream orchestrator with database errors
        mock_database.get_recent_experiences.side_effect = Exception("Database connection failed")
        
        # Should handle database errors gracefully
        try:
            session = dream_orchestrator.dream(duration_seconds=1.0)
            # If no exception, should return empty session
            assert session.experiences_processed == 0
        except Exception as e:
            # Expected behavior - propagate database errors
            assert "Database connection failed" in str(e)
    
    def test_memory_usage_optimization(self, tactile_processor, sample_contact_points):
        """Test memory usage during intensive processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many tactile signatures
        signatures = []
        for i in range(200):
            signature = tactile_processor.process_contacts(
                contact_points=sample_contact_points,
                material_name='steel',
                timestamp=time.time() + i * 0.001
            )
            
            if signature:
                signatures.append(signature)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 200 signatures)
        assert memory_increase < 100
        
        # Clean up
        signatures.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def test_concurrent_processing(self, tactile_processor, sample_contact_points):
        """Test concurrent processing capabilities."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def process_contacts_worker(worker_id):
            """Worker function for concurrent processing."""
            try:
                # Process contacts with slight variations
                modified_contacts = []
                for cp in sample_contact_points:
                    modified_cp = cp.copy()
                    modified_cp['force'] += worker_id * 0.1  # Vary by worker
                    modified_contacts.append(modified_cp)
                
                signature = tactile_processor.process_contacts(
                    contact_points=modified_contacts,
                    material_name='steel'
                )
                
                results_queue.put({
                    'worker_id': worker_id,
                    'signature': signature,
                    'success': signature is not None
                })
                
            except Exception as e:
                results_queue.put({
                    'worker_id': worker_id,
                    'error': str(e),
                    'success': False
                })
        
        # Start concurrent workers
        num_workers = 5
        threads = []
        
        for i in range(num_workers):
            thread = threading.Thread(target=process_contacts_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all workers to complete
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verify results
        assert len(results) == num_workers
        
        # All workers should succeed
        successful_results = [r for r in results if r.get('success', False)]
        assert len(successful_results) == num_workers
        
        # Each worker should have unique signature
        signatures = [r['signature'] for r in successful_results]
        assert all(sig is not None for sig in signatures)
    
    def test_data_consistency_across_components(self, material_bridge, tactile_processor):
        """Test data consistency across integrated components."""
        material_name = 'aluminum'
        
        # Get material properties from bridge
        material_props = material_bridge.get_material(material_name)
        assert material_props is not None
        
        # Compute material signatures
        tactile_prediction = material_bridge.compute_tactile_signature(
            material_name=material_name,
            contact_force=2.0,
            contact_velocity=0.1
        )
        
        # Process actual contacts through tactile processor
        contact_points = [{
            'position': [0, 0, 0],
            'normal': [0, 0, 1],
            'force': 2.0,
            'object_a': 1,
            'object_b': 2
        }]
        
        actual_signature = tactile_processor.process_contacts(
            contact_points=contact_points,
            material_name=material_name
        )
        
        # Compare predictions with actual processing
        assert actual_signature is not None
        assert actual_signature.material == material_name
        
        # Material properties should influence both predictions and processing
        # Hardness should be correlated
        predicted_hardness = tactile_prediction['hardness']
        actual_hardness = actual_signature.hardness
        
        # Should be reasonably close (allowing for processing differences)
        hardness_diff = abs(predicted_hardness - actual_hardness)
        assert hardness_diff < 0.5  # Within 50% of prediction
    
    def test_real_time_processing_simulation(self, tactile_processor, sample_contact_points):
        """Test real-time processing simulation."""
        # Simulate real-time tactile stream
        processing_times = []
        signatures = []
        
        # Process contacts at 100Hz simulation rate
        dt = 0.01  # 10ms intervals
        duration = 1.0  # 1 second simulation
        
        start_time = time.time()
        simulation_time = 0
        
        while simulation_time < duration:
            iteration_start = time.time()
            
            # Add some noise to simulate sensor variations
            noisy_contacts = []
            for cp in sample_contact_points:
                noisy_cp = cp.copy()
                noisy_cp['force'] += np.random.normal(0, 0.05)  # 5% noise
                noisy_contacts.append(noisy_cp)
            
            signature = tactile_processor.process_contacts(
                contact_points=noisy_contacts,
                material_name='steel',
                timestamp=start_time + simulation_time
            )
            
            if signature:
                signatures.append(signature)
            
            processing_time = time.time() - iteration_start
            processing_times.append(processing_time)
            
            simulation_time += dt
            
            # Sleep to maintain real-time rate (if processing is faster than dt)
            sleep_time = max(0, dt - processing_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Analyze real-time performance
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        
        # Real-time constraints: processing should be faster than sampling rate
        assert avg_processing_time < dt  # Average processing faster than 100Hz
        assert max_processing_time < dt * 2  # Max processing within 2x tolerance
        
        # Should have processed most samples
        assert len(signatures) >= int(duration / dt * 0.8)  # At least 80% success rate
        
        # Check vibration analysis with continuous stream
        if len(signatures) > 10:
            last_signature = signatures[-1]
            # Should have meaningful vibration spectrum after processing stream
            assert len(last_signature.vibration_spectrum) == 32
            assert np.sum(last_signature.vibration_spectrum) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])