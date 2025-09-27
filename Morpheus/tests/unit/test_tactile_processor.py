"""Comprehensive test suite for Tactile Processor.

Tests the complete tactile processing functionality including:
- Contact point analysis and processing
- Vibration analysis and frequency spectrum computation
- Texture classification and material integration
- PyBullet integration and simulation mode
- Performance metrics and calibration
- Error handling and edge cases
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from morpheus.perception.tactile_processor import (
    TactileProcessor, TactileProcessorConfig, ContactAnalyzer,
    VibrationAnalyzer, TextureClassifier
)
from morpheus.core.types import ContactPoint, TactileSignature, Vector3D
from morpheus.integration.material_bridge import MaterialBridge


class TestTactileProcessorConfig:
    """Test tactile processor configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TactileProcessorConfig()
        
        assert config.sensitivity == 0.01
        assert config.sampling_rate == 1000
        assert config.vibration_window == 0.1
        assert config.max_contact_points == 50
        assert config.embedding_dim == 64
        assert config.use_materials == True
        assert config.force_threshold == 0.001
        assert config.contact_area_threshold == 1e-6
        assert config.enable_vibration_analysis == True
        assert config.enable_texture_classification == True
        assert config.spatial_resolution == 0.001
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TactileProcessorConfig(
            sensitivity=0.05,
            sampling_rate=500,
            max_contact_points=100
        )
        
        assert config.sensitivity == 0.05
        assert config.sampling_rate == 500
        assert config.max_contact_points == 100
        # Other values should remain default
        assert config.vibration_window == 0.1


class TestContactAnalyzer:
    """Test contact point analysis functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create contact analyzer instance."""
        config = TactileProcessorConfig()
        return ContactAnalyzer(config)
    
    @pytest.fixture
    def sample_contacts(self):
        """Create sample contact points."""
        contacts = []
        positions = [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0.01, 0.01, 0]]
        forces = [1.0, 1.5, 0.8, 1.2]
        
        for i, (pos, force) in enumerate(zip(positions, forces)):
            contact = ContactPoint(
                position=Vector3D.from_array(np.array(pos)),
                normal=Vector3D.from_array(np.array([0, 0, 1])),
                force_magnitude=force,
                object_a=1,
                object_b=2,
                link_a=0,
                link_b=0
            )
            contacts.append(contact)
        
        return contacts
    
    def test_empty_contacts(self, analyzer):
        """Test analysis with no contacts."""
        result = analyzer.analyze_contact_distribution([])
        
        expected_keys = ['centroid', 'spread', 'cluster_count', 'area_estimate', 'contact_density']
        for key in expected_keys:
            assert key in result
        
        assert result['centroid'] == [0, 0, 0]
        assert result['spread'] == 0
        assert result['cluster_count'] == 0
        assert result['area_estimate'] == 0
    
    def test_single_contact(self, analyzer):
        """Test analysis with single contact."""
        contact = ContactPoint(
            position=Vector3D.from_array(np.array([1, 2, 3])),
            normal=Vector3D.from_array(np.array([0, 0, 1])),
            force_magnitude=5.0,
            object_a=1, object_b=2, link_a=0, link_b=0
        )
        
        result = analyzer.analyze_contact_distribution([contact])
        
        assert result['centroid'] == [1, 2, 3]
        assert result['spread'] == 0
        assert result['cluster_count'] == 1
        assert result['area_estimate'] > 0
    
    def test_multiple_contacts(self, analyzer, sample_contacts):
        """Test analysis with multiple contacts."""
        result = analyzer.analyze_contact_distribution(sample_contacts)
        
        assert isinstance(result['centroid'], list)
        assert len(result['centroid']) == 3
        assert result['spread'] > 0
        assert result['cluster_count'] >= 1
        assert result['area_estimate'] > 0
        assert result['contact_density'] > 0
    
    def test_weighted_centroid(self, analyzer):
        """Test that centroid is properly weighted by force."""
        # Two contacts with very different forces
        contacts = [
            ContactPoint(
                position=Vector3D.from_array(np.array([0, 0, 0])),
                normal=Vector3D.from_array(np.array([0, 0, 1])),
                force_magnitude=1.0,
                object_a=1, object_b=2, link_a=0, link_b=0
            ),
            ContactPoint(
                position=Vector3D.from_array(np.array([1, 0, 0])),
                normal=Vector3D.from_array(np.array([0, 0, 1])),
                force_magnitude=9.0,  # Much stronger
                object_a=1, object_b=2, link_a=0, link_b=0
            )
        ]
        
        result = analyzer.analyze_contact_distribution(contacts)
        
        # Centroid should be closer to the stronger contact
        assert result['centroid'][0] > 0.5  # Closer to x=1 than x=0
    
    def test_cluster_estimation(self, analyzer):
        """Test cluster count estimation."""
        # Create two distinct clusters
        cluster1 = [
            ContactPoint(
                position=Vector3D.from_array(np.array([0, 0, 0])),
                normal=Vector3D.from_array(np.array([0, 0, 1])),
                force_magnitude=1.0,
                object_a=1, object_b=2, link_a=0, link_b=0
            ),
            ContactPoint(
                position=Vector3D.from_array(np.array([0.001, 0, 0])),
                normal=Vector3D.from_array(np.array([0, 0, 1])),
                force_magnitude=1.0,
                object_a=1, object_b=2, link_a=0, link_b=0
            )
        ]
        
        cluster2 = [
            ContactPoint(
                position=Vector3D.from_array(np.array([1, 0, 0])),
                normal=Vector3D.from_array(np.array([0, 0, 1])),
                force_magnitude=1.0,
                object_a=1, object_b=2, link_a=0, link_b=0
            )
        ]
        
        all_contacts = cluster1 + cluster2
        result = analyzer.analyze_contact_distribution(all_contacts)
        
        # Should detect 2 clusters
        assert result['cluster_count'] == 2
    
    def test_area_estimation(self, analyzer, sample_contacts):
        """Test contact area estimation."""
        result = analyzer.analyze_contact_distribution(sample_contacts)
        
        # Area should be reasonable for the given contact pattern
        assert result['area_estimate'] > 0
        assert result['area_estimate'] < 1.0  # Shouldn't be huge


class TestVibrationAnalyzer:
    """Test vibration analysis functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create vibration analyzer instance."""
        config = TactileProcessorConfig(
            sampling_rate=100,  # Lower rate for testing
            vibration_window=0.1
        )
        return VibrationAnalyzer(config)
    
    def test_empty_history(self, analyzer):
        """Test analysis with empty force history."""
        spectrum = analyzer.analyze_vibration()
        
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape == (32,)
        assert np.all(spectrum == 0)
    
    def test_add_force_samples(self, analyzer):
        """Test adding force samples."""
        timestamps = [0.0, 0.01, 0.02, 0.03]
        forces = [1.0, 1.5, 0.8, 1.2]
        
        for t, f in zip(timestamps, forces):
            analyzer.add_force_sample(f, t)
        
        assert len(analyzer.force_history) == 4
        assert len(analyzer.time_history) == 4
    
    def test_vibration_analysis_sine_wave(self, analyzer):
        """Test vibration analysis with sine wave input."""
        # Generate sine wave
        duration = 0.1
        sampling_rate = 100
        freq = 10  # Hz
        
        t = np.linspace(0, duration, int(sampling_rate * duration))
        force_signal = 1 + 0.5 * np.sin(2 * np.pi * freq * t)
        
        for time_val, force_val in zip(t, force_signal):
            analyzer.add_force_sample(force_val, time_val)
        
        spectrum = analyzer.analyze_vibration()
        
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape == (32,)
        assert np.sum(spectrum) > 0  # Should detect frequency content
    
    def test_vibration_features(self, analyzer):
        """Test vibration feature extraction."""
        # Add some sample data
        for i in range(20):
            analyzer.add_force_sample(1.0 + 0.1 * np.sin(i), i * 0.01)
        
        features = analyzer.get_vibration_features()
        
        required_keys = [
            'spectral_centroid', 'spectral_rolloff', 'spectral_spread',
            'dominant_frequency_bin', 'total_energy'
        ]
        
        for key in required_keys:
            assert key in features
            assert isinstance(features[key], (int, float))
            assert np.isfinite(features[key])
    
    def test_vibration_features_empty(self, analyzer):
        """Test vibration features with no data."""
        features = analyzer.get_vibration_features()
        
        # Should return zeros without crashing
        for key, value in features.items():
            if key != 'dominant_frequency_bin':
                assert value == 0
    
    def test_frequency_binning(self, analyzer):
        """Test frequency spectrum binning."""
        # Add impulse (should have broad frequency content)
        analyzer.add_force_sample(10.0, 0.0)  # Strong impulse
        for i in range(1, 20):
            analyzer.add_force_sample(1.0, i * 0.01)  # Background
        
        spectrum = analyzer.analyze_vibration()
        
        # Should have energy distributed across bins
        assert np.sum(spectrum) > 0
        assert np.max(spectrum) <= 1.0  # Should be normalized


class TestTextureClassifier:
    """Test texture classification functionality."""
    
    @pytest.fixture
    def classifier(self):
        """Create texture classifier instance."""
        config = TactileProcessorConfig()
        return TextureClassifier(config)
    
    def test_smooth_classification(self, classifier):
        """Test classification of smooth texture."""
        vibration_features = {
            'spectral_rolloff': 0.2,
            'total_energy': 0.05,
            'spectral_spread': 0.1
        }
        
        contact_pattern = {
            'contact_density': 5.0,
            'cluster_count': 1
        }
        
        texture = classifier.classify_texture(
            vibration_features, 0.1, contact_pattern
        )
        
        assert texture == 'smooth'
    
    def test_rough_classification(self, classifier):
        """Test classification of rough texture."""
        vibration_features = {
            'spectral_rolloff': 0.8,
            'total_energy': 0.9,
            'spectral_spread': 0.7
        }
        
        contact_pattern = {
            'contact_density': 20.0,
            'cluster_count': 3
        }
        
        texture = classifier.classify_texture(
            vibration_features, 0.9, contact_pattern
        )
        
        assert texture == 'rough'
    
    def test_textured_classification(self, classifier):
        """Test classification of textured surface."""
        vibration_features = {
            'spectral_rolloff': 0.5,
            'total_energy': 0.4,
            'spectral_spread': 0.4
        }
        
        contact_pattern = {
            'contact_density': 15.0,
            'cluster_count': 2
        }
        
        texture = classifier.classify_texture(
            vibration_features, 0.5, contact_pattern
        )
        
        assert texture in ['textured', 'smooth', 'rough']  # Valid classification
    
    def test_texture_confidence(self, classifier):
        """Test texture classification confidence."""
        vibration_features = {
            'spectral_rolloff': 0.2,
            'total_energy': 0.05
        }
        
        confidence = classifier.get_texture_confidence(
            'smooth', vibration_features, 0.1
        )
        
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1


class TestTactileProcessor:
    """Test the complete tactile processor."""
    
    @pytest.fixture
    def config(self):
        """Create processor configuration."""
        return TactileProcessorConfig(
            sensitivity=0.01,
            max_contact_points=10,  # Smaller for testing
            enable_vibration_analysis=True,
            enable_texture_classification=True
        )
    
    @pytest.fixture
    def mock_material_bridge(self):
        """Create mock material bridge."""
        bridge = Mock()
        
        # Mock tactile signature
        mock_signature = {
            'hardness': 0.7,
            'deformation_mm': 0.5,
            'texture_roughness': 0.6,
            'thermal_feel': 0.3,
            'grip_quality': 0.8,
            'stiffness': 0.7
        }
        
        bridge.compute_tactile_signature.return_value = mock_signature
        bridge.get_material.return_value = Mock(
            friction=0.8,
            restitution=0.2,
            young_modulus=200e9,
            poisson_ratio=0.3
        )
        
        return bridge
    
    @pytest.fixture
    def processor(self, config, mock_material_bridge):
        """Create tactile processor instance."""
        return TactileProcessor(config, mock_material_bridge)
    
    def test_initialization(self, processor):
        """Test processor initialization."""
        assert isinstance(processor, TactileProcessor)
        assert hasattr(processor, 'config')
        assert hasattr(processor, 'material_bridge')
        assert hasattr(processor, 'contact_analyzer')
        assert hasattr(processor, 'vibration_analyzer')
        assert hasattr(processor, 'texture_classifier')
        assert processor.processing_count == 0
    
    @patch('morpheus.perception.tactile_processor.PYBULLET_AVAILABLE', False)
    def test_process_contacts_manual(self, processor):
        """Test processing manual contact points."""
        contact_points = [
            {
                'position': [0, 0, 0],
                'normal': [0, 0, 1],
                'force': 1.5,
                'object_a': 1,
                'object_b': 2
            },
            {
                'position': [0.01, 0, 0],
                'normal': [0, 0, 1],
                'force': 2.0,
                'object_a': 1,
                'object_b': 2
            }
        ]
        
        signature = processor.process_contacts(
            contact_points=contact_points,
            material_name='steel'
        )
        
        assert isinstance(signature, TactileSignature)
        assert signature.material == 'steel'
        assert len(signature.contact_points) <= 2
        assert signature.total_force > 0
        assert signature.contact_area > 0
        assert signature.pressure > 0
    
    def test_process_contacts_filtering(self, processor):
        """Test contact filtering by sensitivity."""
        # Include contacts below and above threshold
        contact_points = [
            {
                'position': [0, 0, 0],
                'normal': [0, 0, 1],
                'force': 0.005,  # Below sensitivity
                'object_a': 1,
                'object_b': 2
            },
            {
                'position': [0.01, 0, 0],
                'normal': [0, 0, 1],
                'force': 1.0,  # Above sensitivity
                'object_a': 1,
                'object_b': 2
            }
        ]
        
        signature = processor.process_contacts(
            contact_points=contact_points,
            material_name='steel'
        )
        
        assert signature is not None
        assert len(signature.contact_points) == 1  # Only strong contact
    
    def test_process_contacts_max_limit(self, processor):
        """Test limiting maximum contact points."""
        # Create more contacts than the limit
        contact_points = []
        for i in range(20):  # More than max_contact_points=10
            contact_points.append({
                'position': [i * 0.01, 0, 0],
                'normal': [0, 0, 1],
                'force': 1.0 + i * 0.1,  # Increasing force
                'object_a': 1,
                'object_b': 2
            })
        
        signature = processor.process_contacts(
            contact_points=contact_points,
            material_name='steel'
        )
        
        assert signature is not None
        assert len(signature.contact_points) <= processor.config.max_contact_points
        
        # Should keep the strongest contacts
        forces = [cp.force_magnitude for cp in signature.contact_points]
        assert max(forces) >= 1.9  # Should include some of the stronger ones
    
    def test_process_contacts_no_contacts(self, processor):
        """Test processing with no valid contacts."""
        contact_points = [
            {
                'position': [0, 0, 0],
                'normal': [0, 0, 1],
                'force': 0.005,  # Below sensitivity
                'object_a': 1,
                'object_b': 2
            }
        ]
        
        signature = processor.process_contacts(
            contact_points=contact_points,
            material_name='steel'
        )
        
        assert signature is None
    
    @patch('morpheus.perception.tactile_processor.PYBULLET_AVAILABLE', True)
    @patch('morpheus.perception.tactile_processor.p')
    def test_process_contacts_pybullet(self, mock_pybullet, processor):
        """Test processing PyBullet contact points."""
        # Mock PyBullet contact data
        mock_contact = [
            0, 1, 2, 0, 0,  # contact flags, bodyA, bodyB, linkA, linkB
            [0, 0, 0],      # positionOnA
            [0, 0, 0],      # positionOnB  
            [0, 0, 1],      # contactNormalOnB
            0.01,           # distance
            1.5,            # normalForce
            [0, 0, 0]       # friction force (optional)
        ]
        
        mock_pybullet.getContactPoints.return_value = [mock_contact]
        
        signature = processor.process_contacts(
            body_id=1,
            material_name='steel'
        )
        
        assert signature is not None
        mock_pybullet.getContactPoints.assert_called_once_with(bodyA=1)
    
    def test_material_integration(self, processor, mock_material_bridge):
        """Test integration with material bridge."""
        contact_points = [{
            'position': [0, 0, 0],
            'normal': [0, 0, 1],
            'force': 1.0,
            'object_a': 1,
            'object_b': 2
        }]
        
        signature = processor.process_contacts(
            contact_points=contact_points,
            material_name='steel'
        )
        
        assert signature is not None
        
        # Should have called material bridge
        mock_material_bridge.compute_tactile_signature.assert_called_once()
        call_args = mock_material_bridge.compute_tactile_signature.call_args
        assert call_args[1]['material_name'] == 'steel'
    
    def test_vibration_analysis_integration(self, processor):
        """Test vibration analysis integration."""
        contact_points = [{
            'position': [0, 0, 0],
            'normal': [0, 0, 1],
            'force': 1.0,
            'object_a': 1,
            'object_b': 2
        }]
        
        # Process multiple times to build vibration history
        signatures = []
        for i in range(5):
            signature = processor.process_contacts(
                contact_points=contact_points,
                material_name='steel',
                timestamp=i * 0.1
            )
            if signature:
                signatures.append(signature)
        
        assert len(signatures) > 0
        
        # Last signature should have vibration data
        last_signature = signatures[-1]
        assert len(last_signature.vibration_spectrum) == 32
    
    def test_texture_classification_integration(self, processor):
        """Test texture classification integration."""
        contact_points = [{
            'position': [0, 0, 0],
            'normal': [0, 0, 1],
            'force': 1.0,
            'object_a': 1,
            'object_b': 2
        }]
        
        signature = processor.process_contacts(
            contact_points=contact_points,
            material_name='steel'
        )
        
        assert signature is not None
        assert signature.texture_descriptor in ['smooth', 'textured', 'rough', 'sticky']
    
    def test_processing_statistics(self, processor):
        """Test processing statistics tracking."""
        initial_stats = processor.get_processing_stats()
        assert initial_stats['processing_count'] == 0
        
        contact_points = [{
            'position': [0, 0, 0],
            'normal': [0, 0, 1],
            'force': 1.0,
            'object_a': 1,
            'object_b': 2
        }]
        
        processor.process_contacts(
            contact_points=contact_points,
            material_name='steel'
        )
        
        final_stats = processor.get_processing_stats()
        assert final_stats['processing_count'] == 1
        assert final_stats['total_processing_time'] > 0
        assert final_stats['average_processing_time'] > 0
    
    def test_reset_processor(self, processor):
        """Test processor reset functionality."""
        # Add some data
        contact_points = [{
            'position': [0, 0, 0],
            'normal': [0, 0, 1],
            'force': 1.0,
            'object_a': 1,
            'object_b': 2
        }]
        
        processor.process_contacts(
            contact_points=contact_points,
            material_name='steel'
        )
        
        # Reset
        processor.reset()
        
        # Buffers should be cleared
        assert len(processor.vibration_analyzer.force_history) == 0
        assert len(processor.vibration_analyzer.time_history) == 0
        assert len(processor.contact_analyzer.contact_history) == 0
    
    def test_calibrate_sensitivity(self, processor):
        """Test sensitivity calibration."""
        # Create test data
        test_contacts = [
            {'position': [0, 0, 0], 'normal': [0, 0, 1], 'force': 0.005},  # Weak
            {'position': [0, 0, 0], 'normal': [0, 0, 1], 'force': 0.1},    # Strong
            {'position': [0, 0, 0], 'normal': [0, 0, 1], 'force': 0.001},  # Very weak
            {'position': [0, 0, 0], 'normal': [0, 0, 1], 'force': 1.0}     # Very strong
        ]
        
        expected_detections = [False, True, False, True]
        
        original_sensitivity = processor.config.sensitivity
        
        optimal_threshold = processor.calibrate_sensitivity(test_contacts, expected_detections)
        
        assert isinstance(optimal_threshold, float)
        assert optimal_threshold > 0
        assert processor.config.sensitivity == optimal_threshold
    
    def test_error_handling_invalid_contacts(self, processor):
        """Test error handling with invalid contact data."""
        invalid_contacts = [
            {'position': 'invalid'},  # Invalid position
            {'force': 'not_a_number'},  # Invalid force
            {}  # Empty contact
        ]
        
        signature = processor.process_contacts(
            contact_points=invalid_contacts,
            material_name='steel'
        )
        
        # Should handle gracefully
        assert signature is None or isinstance(signature, TactileSignature)
    
    def test_concurrent_processing(self, processor):
        """Test concurrent processing of contacts."""
        import threading
        
        contact_points = [{
            'position': [0, 0, 0],
            'normal': [0, 0, 1],
            'force': 1.0,
            'object_a': 1,
            'object_b': 2
        }]
        
        results = []
        
        def process_contacts():
            signature = processor.process_contacts(
                contact_points=contact_points,
                material_name='steel'
            )
            results.append(signature)
        
        # Start multiple threads
        threads = [threading.Thread(target=process_contacts) for _ in range(5)]
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 5
        # All should succeed or all should fail (consistent behavior)
        success_count = sum(1 for r in results if r is not None)
        assert success_count == 0 or success_count == 5
    
    def test_string_representations(self, processor):
        """Test string representations."""
        str_repr = str(processor)
        repr_repr = repr(processor)
        
        assert isinstance(str_repr, str)
        assert isinstance(repr_repr, str)
        assert 'TactileProcessor' in str_repr
        assert 'TactileProcessor' in repr_repr
    
    def test_memory_efficiency(self, processor):
        """Test memory usage during intensive processing."""
        contact_points = [{
            'position': [0, 0, 0],
            'normal': [0, 0, 1],
            'force': 1.0,
            'object_a': 1,
            'object_b': 2
        }]
        
        # Process many contacts
        for i in range(100):
            processor.process_contacts(
                contact_points=contact_points,
                material_name='steel',
                timestamp=i * 0.01
            )
        
        # Should not accumulate unlimited data
        stats = processor.get_processing_stats()
        assert stats['vibration_buffer_size'] <= processor.config.sampling_rate * processor.config.vibration_window
    
    def test_performance_benchmarking(self, processor):
        """Test processing performance."""
        contact_points = [{
            'position': [0, 0, 0],
            'normal': [0, 0, 1],
            'force': 1.0,
            'object_a': 1,
            'object_b': 2
        }]
        
        import time
        
        start_time = time.time()
        
        for i in range(50):
            processor.process_contacts(
                contact_points=contact_points,
                material_name='steel'
            )
        
        processing_time = time.time() - start_time
        
        # Should process reasonably quickly
        assert processing_time < 5.0  # 5 seconds for 50 processes
        
        stats = processor.get_processing_stats()
        assert stats['average_processing_time'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])