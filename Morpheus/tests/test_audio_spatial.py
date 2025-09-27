#!/usr/bin/env python3
"""
Test suite for Audio Spatial Processor functionality.

Tests 3D spatial audio processing, material inference,
and multi-source audio handling.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from morpheus.perception import AudioProcessor, AudioSource, AudioSignature

class TestAudioProcessor:
    """Test cases for Audio Processor."""
    
    @pytest.fixture
    def audio_config(self):
        """Standard audio processor configuration."""
        return {
            'max_sources': 5,
            'frequency_range': [20, 20000],
            'speed_of_sound': 343.0,
            'doppler_enabled': True,
            'sampling_rate': 44100,
            'hrtf_enabled': True,
            'material_inference': True
        }
        
    @pytest.fixture
    def audio_processor(self, audio_config):
        """Audio processor instance for testing."""
        return AudioProcessor(audio_config)
        
    def test_initialization(self, audio_processor, audio_config):
        """Test audio processor initialization."""
        assert audio_processor.max_sources == audio_config['max_sources']
        assert audio_processor.frequency_range == audio_config['frequency_range']
        assert audio_processor.speed_of_sound == audio_config['speed_of_sound']
        assert audio_processor.doppler_enabled == audio_config['doppler_enabled']
        assert len(audio_processor.active_sources) == 0
        assert len(audio_processor.material_acoustics) > 0
        
    def test_hrtf_initialization(self, audio_processor):
        """Test HRTF database initialization."""
        assert len(audio_processor.hrtf_azimuths) > 0
        assert len(audio_processor.hrtf_elevations) > 0
        assert len(audio_processor.hrtf_filters) > 0
        
        # Test HRTF filter structure
        first_filter = next(iter(audio_processor.hrtf_filters.values()))
        assert 'left_gain' in first_filter
        assert 'right_gain' in first_filter
        assert 'itd_samples' in first_filter
        
    def test_listener_update(self, audio_processor):
        """Test listener position and orientation update."""
        position = np.array([1.0, 2.0, 3.0])
        velocity = np.array([0.1, 0.2, 0.0])
        orientation = np.array([1.0, 0.0, 0.0])
        
        audio_processor.update_listener(position, velocity, orientation)
        
        np.testing.assert_array_equal(audio_processor.listener_position, position)
        np.testing.assert_array_equal(audio_processor.listener_velocity, velocity)
        np.testing.assert_array_almost_equal(
            audio_processor.listener_orientation,
            orientation / np.linalg.norm(orientation)
        )
        
    def test_audio_source_management(self, audio_processor):
        """Test adding and removing audio sources."""
        # Add source
        audio_processor.add_audio_source(
            'source1',
            position=np.array([1.0, 0.0, 0.0]),
            frequency=1000.0,
            amplitude=0.5,
            material='steel'
        )
        
        assert len(audio_processor.active_sources) == 1
        assert 'source1' in audio_processor.active_sources
        
        source = audio_processor.active_sources['source1']
        assert source.frequency == 1000.0
        assert source.amplitude == 0.5
        assert source.material == 'steel'
        
        # Remove source
        audio_processor.remove_audio_source('source1')
        assert len(audio_processor.active_sources) == 0
        
    def test_doppler_shift_computation(self, audio_processor):
        """Test Doppler frequency shift calculation."""
        frequency = 1000.0
        
        # Test approaching source
        relative_velocity = -10.0  # Approaching
        doppler_shift = audio_processor._compute_doppler_shift(frequency, relative_velocity)
        assert doppler_shift > 0  # Frequency should increase
        
        # Test receding source
        relative_velocity = 10.0  # Receding
        doppler_shift = audio_processor._compute_doppler_shift(frequency, relative_velocity)
        assert doppler_shift < 0  # Frequency should decrease
        
        # Test stationary source
        relative_velocity = 0.0
        doppler_shift = audio_processor._compute_doppler_shift(frequency, relative_velocity)
        assert abs(doppler_shift) < 1e-6  # Should be near zero
        
    def test_distance_attenuation(self, audio_processor):
        """Test distance-based amplitude attenuation."""
        # Test near distance
        near_attenuation = audio_processor._compute_distance_attenuation(1.0)
        
        # Test far distance
        far_attenuation = audio_processor._compute_distance_attenuation(10.0)
        
        # Far distance should have more attenuation
        assert far_attenuation < near_attenuation
        assert near_attenuation > 0
        assert far_attenuation > 0
        
    def test_cartesian_to_spherical(self, audio_processor):
        """Test coordinate system conversion."""
        # Test cardinal directions
        directions = [
            ([1, 0, 0], (0, 0)),        # Positive X
            ([0, 1, 0], (np.pi/2, 0)),  # Positive Y  
            ([0, 0, 1], (0, np.pi/2)),  # Positive Z
            ([-1, 0, 0], (np.pi, 0))    # Negative X
        ]
        
        for direction, expected in directions:
            direction = np.array(direction)
            azimuth, elevation = audio_processor._cartesian_to_spherical(direction)
            
            assert abs(azimuth - expected[0]) < 1e-6
            assert abs(elevation - expected[1]) < 1e-6
            
    def test_material_acoustics(self, audio_processor):
        """Test material acoustic properties."""
        # Test predefined materials
        materials = ['steel', 'rubber', 'wood', 'plastic']
        
        for material in materials:
            assert material in audio_processor.material_acoustics
            props = audio_processor.material_acoustics[material]
            
            assert 'reflection_coeff' in props
            assert 'absorption_coeff' in props
            assert 'resonant_freq' in props
            assert 'decay_rate' in props
            
            assert 0 <= props['reflection_coeff'] <= 1
            assert 0 <= props['absorption_coeff'] <= 1
            assert props['resonant_freq'] > 0
            assert props['decay_rate'] > 0
            
    def test_single_source_processing(self, audio_processor):
        """Test processing of a single audio source."""
        # Add a source
        source = AudioSource(
            id='test_source',
            position=np.array([2.0, 1.0, 0.5]),
            velocity=np.array([0.1, 0.0, 0.0]),
            frequency=2000.0,
            amplitude=0.8,
            material='steel'
        )
        
        processed = audio_processor._process_single_source(source)
        
        assert 'source' in processed
        assert 'distance' in processed
        assert 'direction' in processed
        assert 'doppler_shift' in processed
        assert 'attenuation' in processed
        assert 'material_effects' in processed
        assert 'effective_amplitude' in processed
        assert 'effective_frequency' in processed
        
        assert processed['distance'] > 0
        assert processed['effective_amplitude'] > 0
        assert processed['effective_frequency'] > 0
        
    def test_spatial_audio_processing(self, audio_processor):
        """Test complete spatial audio processing."""
        # Add multiple sources
        audio_processor.add_audio_source(
            'source1', np.array([1.0, 0.0, 0.0]), 1000.0, 0.5, material='steel'
        )
        audio_processor.add_audio_source(
            'source2', np.array([0.0, 1.0, 0.0]), 2000.0, 0.3, material='rubber'
        )
        
        # Process audio
        robot_position = np.array([0.0, 0.0, 0.0])
        robot_velocity = np.array([0.0, 0.0, 0.0])
        
        signature = audio_processor.process_spatial_audio(robot_position, robot_velocity)
        
        assert signature is not None
        assert isinstance(signature, AudioSignature)
        assert len(signature.sources) == 2
        assert signature.dominant_frequency > 0
        assert len(signature.frequency_spectrum) == 32
        assert signature.spatial_map.shape == (8, 8)
        
    def test_audio_signature_embedding(self, audio_processor):
        """Test audio signature to embedding conversion."""
        # Create a mock signature
        signature = AudioSignature(
            timestamp=time.time(),
            sources=[AudioSource('test', np.array([1, 0, 0]), np.array([0, 0, 0]), 1000, 0.5)],
            listener_position=np.zeros(3),
            listener_velocity=np.zeros(3),
            dominant_frequency=1000.0,
            frequency_spectrum=np.random.rand(32),
            spatial_map=np.random.rand(8, 8),
            doppler_shifts=[10.0],
            reverb_characteristics={'rt60': 0.5, 'clarity': 0.8, 'warmth': 0.6, 'spaciousness': 0.7},
            material_predictions={'steel': 0.8, 'rubber': 0.2}
        )
        
        embedding = signature.to_embedding()
        
        assert embedding.shape == (32,)
        assert np.all(np.isfinite(embedding))
        assert np.all(embedding >= 0)  # Most values should be positive due to normalization
        
    def test_frequency_spectrum_computation(self, audio_processor):
        """Test frequency spectrum computation."""
        frequencies = [100, 500, 1000, 5000, 10000]
        amplitudes = [0.8, 0.6, 1.0, 0.4, 0.2]
        
        spectrum = audio_processor._compute_frequency_spectrum(frequencies, amplitudes)
        
        assert spectrum.shape == (32,)
        assert np.all(spectrum >= 0)
        assert np.all(spectrum <= 1)  # Should be normalized
        assert np.max(spectrum) <= 1.0
        
    def test_spatial_map_computation(self, audio_processor):
        """Test spatial audio map computation."""
        # Create mock processed sources
        processed_sources = [
            {
                'source': AudioSource('s1', np.array([1, 0, 0]), np.zeros(3), 1000, 0.5),
                'effective_amplitude': 0.8
            },
            {
                'source': AudioSource('s2', np.array([-0.5, 1, 0]), np.zeros(3), 2000, 0.3),
                'effective_amplitude': 0.4
            }
        ]
        
        listener_pos = np.zeros(3)
        spatial_map = audio_processor._compute_spatial_map(processed_sources, listener_pos)
        
        assert spatial_map.shape == (8, 8)
        assert np.all(spatial_map >= 0)
        
    def test_material_prediction(self, audio_processor):
        """Test material prediction from acoustic characteristics."""
        # Create processed sources with known materials
        processed_sources = [
            {
                'source': AudioSource('s1', np.zeros(3), np.zeros(3), 8000, 0.5, material='steel'),
                'material_effects': {'reflection': 0.9, 'absorption': 0.9},
                'effective_amplitude': 0.5
            },
            {
                'source': AudioSource('s2', np.zeros(3), np.zeros(3), 500, 0.3, material='rubber'),
                'material_effects': {'reflection': 0.2, 'absorption': 0.2},
                'effective_amplitude': 0.3
            }
        ]
        
        predictions = audio_processor._predict_materials(processed_sources)
        
        assert isinstance(predictions, dict)
        assert 'steel' in predictions
        assert 'rubber' in predictions
        assert all(0 <= v <= 1 for v in predictions.values())
        
    def test_source_cleanup(self, audio_processor):
        """Test cleanup of inactive audio sources."""
        # Add sources with old timestamps
        old_time = time.time() - 10.0
        
        source = AudioSource('old_source', np.zeros(3), np.zeros(3), 1000, 0.5)
        source.last_update = old_time
        audio_processor.active_sources['old_source'] = source
        
        # Add recent source
        audio_processor.add_audio_source('new_source', np.zeros(3), 1000, 0.5)
        
        assert len(audio_processor.active_sources) == 2
        
        # Cleanup with short max_age
        audio_processor.cleanup_inactive_sources(max_age=5.0)
        
        assert len(audio_processor.active_sources) == 1
        assert 'new_source' in audio_processor.active_sources
        assert 'old_source' not in audio_processor.active_sources
        
    def test_performance_metrics(self, audio_processor):
        """Test performance metrics collection."""
        # Process some audio to generate metrics
        audio_processor.add_audio_source('test', np.zeros(3), 1000, 0.5)
        audio_processor.process_spatial_audio(np.zeros(3))
        
        metrics = audio_processor.get_performance_metrics()
        
        assert 'processed_frames' in metrics
        assert 'active_sources' in metrics
        assert 'average_processing_time' in metrics
        assert 'processing_fps' in metrics
        
        assert metrics['processed_frames'] > 0
        assert metrics['active_sources'] >= 0
        
    def test_room_acoustics_setup(self, audio_processor):
        """Test room acoustics configuration."""
        room_size = (5.0, 4.0, 3.0)
        wall_materials = ['concrete', 'wood']
        
        # Should not raise exception
        audio_processor.set_room_acoustics(room_size, wall_materials, humidity=0.5, temperature=22.0)
        
        # Speed of sound should be updated based on temperature
        expected_speed = 331.3 * np.sqrt(1 + 22.0 / 273.15)
        assert abs(audio_processor.speed_of_sound - expected_speed) < 1.0
        
    def test_audio_stream_processing(self, audio_processor):
        """Test real audio stream processing."""
        # Generate synthetic audio data
        sample_rate = 44100
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        
        # Create audio with specific frequencies
        t = np.linspace(0, duration, samples)
        audio_data = (
            0.5 * np.sin(2 * np.pi * 1000 * t) +  # 1kHz tone
            0.3 * np.sin(2 * np.pi * 2000 * t)    # 2kHz tone
        )
        
        analysis = audio_processor.process_audio_stream(audio_data, sample_rate)
        
        assert 'dominant_frequencies' in analysis
        assert 'estimated_materials' in analysis
        assert 'spatial_activity' in analysis
        
        # Should detect the dominant frequencies
        dominant_freqs = analysis['dominant_frequencies']
        assert len(dominant_freqs) > 0
        
        # Check if frequencies are in expected range
        for freq in dominant_freqs[:2]:  # Check first 2
            assert 900 <= freq <= 2100  # Allow some tolerance

class TestAudioSource:
    """Test cases for Audio Source data structure."""
    
    def test_audio_source_creation(self):
        """Test audio source creation and properties."""
        position = np.array([1.0, 2.0, 3.0])
        velocity = np.array([0.1, 0.2, 0.0])
        
        source = AudioSource(
            id='test_source',
            position=position,
            velocity=velocity,
            frequency=1500.0,
            amplitude=0.75,
            material='wood'
        )
        
        assert source.id == 'test_source'
        np.testing.assert_array_equal(source.position, position)
        np.testing.assert_array_equal(source.velocity, velocity)
        assert source.frequency == 1500.0
        assert source.amplitude == 0.75
        assert source.material == 'wood'
        assert source.is_active is True

if __name__ == '__main__':
    pytest.main([__file__, '-v'])