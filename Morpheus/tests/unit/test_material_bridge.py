"""Comprehensive test suite for Material Bridge.

Tests the complete material integration functionality including:
- GASM-Robotics configuration loading
- Material property computation
- Tactile and audio signature generation
- Material interaction modeling
- Caching and performance optimization
- Error handling and edge cases
"""

import pytest
import numpy as np
import yaml
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from typing import Dict, Any

from morpheus.integration.material_bridge import MaterialBridge
from morpheus.core.types import MaterialProperties, MaterialType, MaterialInteraction


class TestMaterialBridge:
    """Test the complete material bridge functionality."""
    
    @pytest.fixture
    def sample_gasm_config(self):
        """Create sample GASM configuration."""
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
                },
                'plastic': {
                    'color': [0.1, 0.5, 0.8, 1.0],
                    'friction': 0.4,
                    'restitution': 0.6,
                    'density': 1200,
                    'young_modulus': 2e9,
                    'poisson_ratio': 0.35
                },
                'glass': {
                    'color': [0.9, 0.9, 1.0, 0.8],
                    'friction': 0.2,
                    'restitution': 0.1,
                    'density': 2500,
                    'young_modulus': 70e9,
                    'poisson_ratio': 0.22
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
    
    def test_initialization_success(self, material_bridge):
        """Test successful initialization."""
        assert isinstance(material_bridge, MaterialBridge)
        assert len(material_bridge.materials) > 0
        assert 'default' in material_bridge.materials
        assert material_bridge.interaction_cache == {}
    
    def test_initialization_missing_config(self):
        """Test initialization with missing config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError):
                MaterialBridge(temp_dir)
    
    def test_initialization_invalid_config(self, temp_gasm_directory):
        """Test initialization with invalid config."""
        # Overwrite with invalid YAML
        config_file = temp_gasm_directory / "assets" / "configs" / "simulation_params.yaml"
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(ValueError):
            MaterialBridge(temp_gasm_directory)
    
    def test_material_loading(self, material_bridge, sample_gasm_config):
        """Test proper material loading."""
        expected_materials = list(sample_gasm_config['materials'].keys()) + ['default']
        
        for material_name in expected_materials:
            assert material_name in material_bridge.materials
            material = material_bridge.materials[material_name]
            assert isinstance(material, MaterialProperties)
            assert material.name == material_name
    
    def test_material_type_mapping(self, material_bridge):
        """Test material type mapping."""
        steel = material_bridge.get_material('steel')
        aluminum = material_bridge.get_material('aluminum')
        rubber = material_bridge.get_material('rubber')
        plastic = material_bridge.get_material('plastic')
        glass = material_bridge.get_material('glass')
        
        assert steel.material_type == MaterialType.METAL
        assert aluminum.material_type == MaterialType.METAL
        assert rubber.material_type == MaterialType.RUBBER
        assert plastic.material_type == MaterialType.PLASTIC
        assert glass.material_type == MaterialType.GLASS
    
    def test_get_material_existing(self, material_bridge):
        """Test getting existing material."""
        steel = material_bridge.get_material('steel')
        
        assert isinstance(steel, MaterialProperties)
        assert steel.name == 'steel'
        assert steel.friction == 0.8
        assert steel.density == 7850
    
    def test_get_material_nonexistent(self, material_bridge):
        """Test getting non-existent material returns default."""
        material = material_bridge.get_material('nonexistent')
        
        assert isinstance(material, MaterialProperties)
        assert material.name == 'default'
    
    def test_list_materials(self, material_bridge):
        """Test listing available materials."""
        materials = material_bridge.list_materials()
        
        assert isinstance(materials, list)
        assert 'steel' in materials
        assert 'aluminum' in materials
        assert 'default' in materials
    
    def test_get_material_type(self, material_bridge):
        """Test getting material type."""
        assert material_bridge.get_material_type('steel') == MaterialType.METAL
        assert material_bridge.get_material_type('rubber') == MaterialType.RUBBER
        assert material_bridge.get_material_type('nonexistent') == MaterialType.UNKNOWN
    
    def test_tactile_signature_basic(self, material_bridge):
        """Test basic tactile signature computation."""
        signature = material_bridge.compute_tactile_signature('steel')
        
        assert isinstance(signature, dict)
        required_keys = [
            'hardness', 'deformation_mm', 'texture_roughness', 'texture_descriptor',
            'thermal_feel', 'vibration_damping', 'grip_quality', 'elasticity',
            'weight_density', 'pressure', 'stiffness'
        ]
        
        for key in required_keys:
            assert key in signature
            assert isinstance(signature[key], (int, float, str))
    
    def test_tactile_signature_parameters(self, material_bridge):
        """Test tactile signature with different parameters."""
        # Test with varying parameters
        sig_low_force = material_bridge.compute_tactile_signature(
            'steel', contact_force=0.1
        )
        sig_high_force = material_bridge.compute_tactile_signature(
            'steel', contact_force=10.0
        )
        
        # Higher force should result in higher pressure and deformation
        assert sig_high_force['pressure'] > sig_low_force['pressure']
        assert sig_high_force['deformation_mm'] > sig_low_force['deformation_mm']
    
    def test_tactile_signature_velocity_effects(self, material_bridge):
        """Test tactile signature velocity effects."""
        sig_static = material_bridge.compute_tactile_signature(
            'steel', contact_velocity=0.0
        )
        sig_sliding = material_bridge.compute_tactile_signature(
            'steel', contact_velocity=1.0
        )
        
        # Sliding should affect texture perception
        assert isinstance(sig_static['texture_descriptor'], str)
        assert isinstance(sig_sliding['texture_descriptor'], str)
    
    def test_tactile_signature_materials(self, material_bridge):
        """Test tactile signatures for different materials."""
        steel_sig = material_bridge.compute_tactile_signature('steel')
        rubber_sig = material_bridge.compute_tactile_signature('rubber')
        
        # Steel should be harder and less elastic than rubber
        assert steel_sig['hardness'] > rubber_sig['hardness']
        assert steel_sig['elasticity'] < rubber_sig['elasticity']
        assert steel_sig['thermal_feel'] < rubber_sig['thermal_feel']
    
    def test_tactile_signature_caching(self, material_bridge):
        """Test tactile signature caching."""
        # Clear cache first
        material_bridge.clear_cache()
        
        # First call
        sig1 = material_bridge.compute_tactile_signature('steel', contact_force=1.0)
        
        # Second call with same parameters should use cache
        sig2 = material_bridge.compute_tactile_signature('steel', contact_force=1.0)
        
        assert sig1 == sig2
        
        # Check cache stats
        cache_stats = material_bridge.get_cache_stats()
        assert cache_stats['tactile_signature']['hits'] > 0
    
    def test_audio_signature_basic(self, material_bridge):
        """Test basic audio signature computation."""
        signature = material_bridge.compute_audio_signature('steel')
        
        assert isinstance(signature, dict)
        required_keys = [
            'fundamental_freq', 'amplitude', 'decay_rate', 'harmonics',
            'brightness', 'sound_speed', 'spectral_centroid'
        ]
        
        for key in required_keys:
            assert key in signature
    
    def test_audio_signature_impact_velocity(self, material_bridge):
        """Test audio signature with different impact velocities."""
        sig_slow = material_bridge.compute_audio_signature('steel', impact_velocity=0.1)
        sig_fast = material_bridge.compute_audio_signature('steel', impact_velocity=5.0)
        
        # Higher velocity should result in higher amplitude
        assert sig_fast['amplitude'] > sig_slow['amplitude']
    
    def test_audio_signature_object_size(self, material_bridge):
        """Test audio signature with different object sizes."""
        sig_small = material_bridge.compute_audio_signature('steel', object_size=0.01)
        sig_large = material_bridge.compute_audio_signature('steel', object_size=1.0)
        
        # Smaller objects should have higher fundamental frequency
        assert sig_small['fundamental_freq'] > sig_large['fundamental_freq']
    
    def test_audio_signature_materials(self, material_bridge):
        """Test audio signatures for different materials."""
        metal_sig = material_bridge.compute_audio_signature('steel')
        rubber_sig = material_bridge.compute_audio_signature('rubber')
        glass_sig = material_bridge.compute_audio_signature('glass')
        
        # Metal should be brighter than rubber
        assert metal_sig['brightness'] > rubber_sig['brightness']
        
        # Glass should have rich harmonics
        assert len(glass_sig['harmonics']) > 1
        
        # Rubber should have limited harmonics
        assert len(rubber_sig['harmonics']) <= len(metal_sig['harmonics'])
    
    def test_material_interaction_basic(self, material_bridge):
        """Test basic material interaction computation."""
        interaction = material_bridge.compute_interaction('steel', 'aluminum')
        
        assert isinstance(interaction, MaterialInteraction)
        assert interaction.material1 == 'steel'
        assert interaction.material2 == 'aluminum'
        assert 0 <= interaction.combined_friction <= 2.0
        assert 0 <= interaction.combined_restitution <= 1.0
        assert interaction.effective_modulus > 0
    
    def test_material_interaction_same_materials(self, material_bridge):
        """Test interaction between same materials."""
        interaction = material_bridge.compute_interaction('steel', 'steel')
        
        steel_props = material_bridge.get_material('steel')
        
        # Combined properties should match single material
        assert abs(interaction.combined_friction - steel_props.friction) < 0.1
        assert abs(interaction.combined_restitution - steel_props.restitution) < 0.1
    
    def test_material_interaction_predictions(self, material_bridge):
        """Test material interaction predictions."""
        steel_rubber = material_bridge.compute_interaction('steel', 'rubber')
        steel_glass = material_bridge.compute_interaction('steel', 'glass')
        
        # Steel-rubber should have good grip
        assert steel_rubber.grip_prediction or steel_rubber.combined_friction > 0.5
        
        # Steel-glass might not bounce much
        assert isinstance(steel_glass.bounce_prediction, bool)
    
    def test_material_interaction_caching(self, material_bridge):
        """Test material interaction caching."""
        material_bridge.clear_cache()
        
        # First computation
        interaction1 = material_bridge.compute_interaction('steel', 'aluminum')
        
        # Second computation should use cache
        interaction2 = material_bridge.compute_interaction('steel', 'aluminum')
        
        assert interaction1.combined_friction == interaction2.combined_friction
        
        cache_stats = material_bridge.get_cache_stats()
        assert cache_stats['interaction']['hits'] > 0
    
    def test_predict_sensory_outcome_basic(self, material_bridge):
        """Test basic sensory outcome prediction."""
        scenario = {
            'material': 'steel',
            'force': 5.0,
            'velocity': 1.0,
            'impact_velocity': 2.0
        }
        
        outcome = material_bridge.predict_sensory_outcome(scenario)
        
        assert isinstance(outcome, dict)
        assert 'material' in outcome
        assert 'material_type' in outcome
        assert 'tactile' in outcome
        assert 'audio' in outcome
        assert 'confidence' in outcome
        assert 0 <= outcome['confidence'] <= 1
    
    def test_predict_sensory_outcome_missing_material(self, material_bridge):
        """Test sensory prediction with missing material."""
        scenario = {'force': 5.0}  # No material specified
        
        outcome = material_bridge.predict_sensory_outcome(scenario)
        
        assert outcome['material'] == 'default'
        assert outcome['confidence'] < 1.0
    
    def test_predict_sensory_outcome_confidence(self, material_bridge):
        """Test confidence calculation in sensory prediction."""
        # Complete scenario should have high confidence
        complete_scenario = {
            'material': 'steel',
            'force': 5.0,
            'velocity': 1.0,
            'impact_velocity': 2.0
        }
        
        # Incomplete scenario should have lower confidence
        incomplete_scenario = {
            'material': 'steel'
        }
        
        complete_outcome = material_bridge.predict_sensory_outcome(complete_scenario)
        incomplete_outcome = material_bridge.predict_sensory_outcome(incomplete_scenario)
        
        assert complete_outcome['confidence'] > incomplete_outcome['confidence']
    
    def test_generate_vibration_spectrum_basic(self, material_bridge):
        """Test basic vibration spectrum generation."""
        force_history = [1.0, 1.5, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7, 1.4, 1.0]
        
        spectrum = material_bridge.generate_vibration_spectrum('steel', force_history)
        
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape == (32,)
        assert np.all(spectrum >= 0)
        assert np.all(spectrum <= 1)  # Should be normalized
    
    def test_generate_vibration_spectrum_empty(self, material_bridge):
        """Test vibration spectrum with empty force history."""
        spectrum = material_bridge.generate_vibration_spectrum('steel', [])
        
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape == (32,)
        assert np.all(spectrum == 0)
    
    def test_generate_vibration_spectrum_materials(self, material_bridge):
        """Test vibration spectra for different materials."""
        force_history = [1.0, 1.5, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7, 1.4, 1.0]
        
        steel_spectrum = material_bridge.generate_vibration_spectrum('steel', force_history)
        rubber_spectrum = material_bridge.generate_vibration_spectrum('rubber', force_history)
        
        # Different materials should produce different spectra
        assert not np.allclose(steel_spectrum, rubber_spectrum)
    
    def test_default_signatures(self, material_bridge):
        """Test default signature methods."""
        default_tactile = material_bridge._default_tactile_signature()
        default_audio = material_bridge._default_audio_signature()
        
        assert isinstance(default_tactile, dict)
        assert isinstance(default_audio, dict)
        assert 'hardness' in default_tactile
        assert 'fundamental_freq' in default_audio
    
    def test_cache_management(self, material_bridge):
        """Test cache management functions."""
        # Generate some cache entries
        material_bridge.compute_tactile_signature('steel')
        material_bridge.compute_audio_signature('aluminum')
        material_bridge.compute_interaction('steel', 'aluminum')
        
        # Get cache stats
        stats_before = material_bridge.get_cache_stats()
        assert stats_before['tactile_signature']['size'] > 0
        
        # Clear cache
        material_bridge.clear_cache()
        
        # Cache should be empty
        stats_after = material_bridge.get_cache_stats()
        assert stats_after['tactile_signature']['size'] == 0
        assert stats_after['audio_signature']['size'] == 0
        assert stats_after['interaction']['size'] == 0
    
    def test_string_representations(self, material_bridge):
        """Test string representations."""
        str_repr = str(material_bridge)
        repr_repr = repr(material_bridge)
        
        assert isinstance(str_repr, str)
        assert isinstance(repr_repr, str)
        assert 'MaterialBridge' in str_repr
        assert 'MaterialBridge' in repr_repr
        assert 'materials=' in str_repr
    
    def test_material_properties_validation(self, temp_gasm_directory):
        """Test validation of material properties during loading."""
        # Create config with missing required property
        invalid_config = {
            'materials': {
                'invalid_material': {
                    'color': [1, 0, 0, 1],
                    'friction': 0.5
                    # Missing density, young_modulus, etc.
                }
            }
        }
        
        config_file = temp_gasm_directory / "assets" / "configs" / "simulation_params.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Should still create bridge but skip invalid materials
        bridge = MaterialBridge(temp_gasm_directory)
        assert 'invalid_material' not in bridge.materials
        assert 'default' in bridge.materials
    
    def test_thermal_conductivity_fallback(self, material_bridge):
        """Test thermal conductivity fallback behavior."""
        # Materials without explicit thermal conductivity should use defaults
        steel = material_bridge.get_material('steel')
        aluminum = material_bridge.get_material('aluminum')
        
        # Should have thermal conductivity attribute (with default value)
        assert hasattr(steel, 'thermal_conductivity')
        assert hasattr(aluminum, 'thermal_conductivity')
    
    def test_edge_cases_tactile(self, material_bridge):
        """Test edge cases for tactile signature computation."""
        # Zero force
        sig_zero = material_bridge.compute_tactile_signature('steel', contact_force=0.0)
        assert sig_zero['pressure'] == 0.0
        
        # Very high force
        sig_high = material_bridge.compute_tactile_signature('steel', contact_force=1000.0)
        assert sig_high['pressure'] > 0
        
        # Very small contact area
        sig_small_area = material_bridge.compute_tactile_signature(
            'steel', contact_area=1e-10
        )
        assert np.isfinite(sig_small_area['pressure'])
    
    def test_edge_cases_audio(self, material_bridge):
        """Test edge cases for audio signature computation."""
        # Zero impact velocity
        sig_zero = material_bridge.compute_audio_signature('steel', impact_velocity=0.0)
        assert sig_zero['amplitude'] >= 0
        
        # Very high velocity
        sig_high = material_bridge.compute_audio_signature('steel', impact_velocity=100.0)
        assert sig_high['amplitude'] > 0
        
        # Very small object
        sig_small = material_bridge.compute_audio_signature('steel', object_size=1e-6)
        assert np.isfinite(sig_small['fundamental_freq'])
    
    def test_performance_large_dataset(self, material_bridge):
        """Test performance with large number of computations."""
        import time
        
        # Time multiple computations
        start_time = time.time()
        
        for i in range(100):
            material_bridge.compute_tactile_signature('steel', contact_force=i*0.1)
        
        computation_time = time.time() - start_time
        
        # Should complete reasonably quickly (adjust threshold as needed)
        assert computation_time < 5.0  # 5 seconds for 100 computations
        
        # Check that caching is working
        cache_stats = material_bridge.get_cache_stats()
        assert cache_stats['tactile_signature']['hits'] > 0
    
    def test_concurrent_access(self, material_bridge):
        """Test concurrent access to material bridge."""
        import threading
        
        results = []
        
        def compute_signature():
            sig = material_bridge.compute_tactile_signature('steel')
            results.append(sig)
        
        # Start multiple threads
        threads = [threading.Thread(target=compute_signature) for _ in range(10)]
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 10
        
        # All results should be the same (deterministic computation)
        for result in results[1:]:
            assert result == results[0]


class TestMaterialBridgeErrors:
    """Test error handling in material bridge."""
    
    def test_invalid_yaml_config(self, temp_gasm_directory):
        """Test handling of invalid YAML configuration."""
        config_file = temp_gasm_directory / "assets" / "configs" / "simulation_params.yaml"
        
        # Write invalid YAML
        with open(config_file, 'w') as f:
            f.write("materials:\n  steel:\n    invalid: yaml: [incomplete")
        
        with pytest.raises(ValueError, match="Failed to load GASM config"):
            MaterialBridge(temp_gasm_directory)
    
    def test_missing_materials_section(self, temp_gasm_directory):
        """Test handling of config without materials section."""
        config_file = temp_gasm_directory / "assets" / "configs" / "simulation_params.yaml"
        
        # Write config without materials
        with open(config_file, 'w') as f:
            yaml.dump({'other_section': {}}, f)
        
        bridge = MaterialBridge(temp_gasm_directory)
        
        # Should only have default material
        assert len(bridge.materials) == 1
        assert 'default' in bridge.materials
    
    def test_unknown_material_fallback(self, material_bridge):
        """Test fallback behavior for unknown materials."""
        sig = material_bridge.compute_tactile_signature('nonexistent_material')
        
        # Should return default signature without error
        assert isinstance(sig, dict)
        assert 'hardness' in sig


if __name__ == '__main__':
    pytest.main([__file__, '-v'])