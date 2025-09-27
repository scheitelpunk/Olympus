"""GASM-Robotics integration tests for MORPHEUS.

Tests the complete integration with GASM-Robotics including:
- Material property loading and validation
- PyBullet simulation integration
- Physics engine compatibility
- Robot control and sensor integration
- Performance with realistic scenarios
"""

import pytest
import numpy as np
import tempfile
import yaml
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from morpheus.integration.material_bridge import MaterialBridge
from morpheus.integration.gasm_bridge import GASMBridge
from morpheus.perception.tactile_processor import TactileProcessor, TactileProcessorConfig
from morpheus.core.types import (
    MaterialProperties, MaterialType, ContactPoint, Vector3D,
    TactileSignature, SensoryExperience
)


class TestGASMIntegration:
    """Test integration with GASM-Robotics system."""
    
    @pytest.fixture
    def comprehensive_gasm_config(self):
        """Create comprehensive GASM configuration with all material types."""
        return {
            'materials': {
                # Metals
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
                'copper': {
                    'color': [0.9, 0.6, 0.4, 1.0],
                    'friction': 0.7,
                    'restitution': 0.3,
                    'density': 8960,
                    'young_modulus': 120e9,
                    'poisson_ratio': 0.34
                },
                
                # Plastics
                'abs_plastic': {
                    'color': [0.2, 0.3, 0.8, 1.0],
                    'friction': 0.4,
                    'restitution': 0.6,
                    'density': 1200,
                    'young_modulus': 2.5e9,
                    'poisson_ratio': 0.35
                },
                'nylon': {
                    'color': [1.0, 1.0, 0.9, 1.0],
                    'friction': 0.3,
                    'restitution': 0.7,
                    'density': 1140,
                    'young_modulus': 3e9,
                    'poisson_ratio': 0.4
                },
                
                # Elastomers
                'rubber': {
                    'color': [0.2, 0.2, 0.2, 1.0],
                    'friction': 1.2,
                    'restitution': 0.9,
                    'density': 1200,
                    'young_modulus': 1e6,
                    'poisson_ratio': 0.47
                },
                'silicone': {
                    'color': [0.9, 0.9, 1.0, 0.8],
                    'friction': 1.0,
                    'restitution': 0.8,
                    'density': 1000,
                    'young_modulus': 0.5e6,
                    'poisson_ratio': 0.49
                },
                
                # Glass and ceramics
                'glass': {
                    'color': [0.9, 0.9, 1.0, 0.3],
                    'friction': 0.2,
                    'restitution': 0.1,
                    'density': 2500,
                    'young_modulus': 70e9,
                    'poisson_ratio': 0.22
                },
                'ceramic': {
                    'color': [0.8, 0.7, 0.6, 1.0],
                    'friction': 0.5,
                    'restitution': 0.05,
                    'density': 3800,
                    'young_modulus': 380e9,
                    'poisson_ratio': 0.27
                },
                
                # Composites
                'carbon_fiber': {
                    'color': [0.1, 0.1, 0.1, 1.0],
                    'friction': 0.3,
                    'restitution': 0.4,
                    'density': 1600,
                    'young_modulus': 150e9,
                    'poisson_ratio': 0.28
                },
                'fiberglass': {
                    'color': [0.9, 0.9, 0.8, 1.0],
                    'friction': 0.4,
                    'restitution': 0.3,
                    'density': 1800,
                    'young_modulus': 45e9,
                    'poisson_ratio': 0.3
                }
            },
            
            # Simulation parameters
            'simulation': {
                'time_step': 1.0/240.0,  # 240Hz
                'gravity': [0, 0, -9.81],
                'solver_iterations': 10,
                'constraint_solver_type': 'CONSTRAINT_SOLVER_LCP_DANTZIG',
                'use_real_time_simulation': False,
                'enable_gui': False
            },
            
            # Robot configuration
            'robot': {
                'name': 'franka_panda',
                'urdf_path': 'assets/urdf/panda.urdf',
                'end_effector_link': 'panda_hand',
                'joint_limits': {
                    'position': [[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
                               [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525],
                               [-2.8973, 2.8973]],
                    'velocity': [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100],
                    'effort': [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
                },
                'tactile_sensors': {
                    'finger_tip_left': {'link_id': 10, 'sensor_type': 'force'},
                    'finger_tip_right': {'link_id': 11, 'sensor_type': 'force'},
                    'palm': {'link_id': 8, 'sensor_type': 'pressure'}
                }
            },
            
            # Objects for manipulation
            'objects': {
                'cube_steel': {
                    'geometry': 'box',
                    'size': [0.05, 0.05, 0.05],
                    'material': 'steel',
                    'mass': 0.98  # Calculated from density and volume
                },
                'sphere_rubber': {
                    'geometry': 'sphere',
                    'radius': 0.03,
                    'material': 'rubber',
                    'mass': 0.136
                },
                'cylinder_aluminum': {
                    'geometry': 'cylinder',
                    'radius': 0.025,
                    'height': 0.1,
                    'material': 'aluminum',
                    'mass': 0.53
                }
            }
        }
    
    @pytest.fixture
    def temp_gasm_directory(self, comprehensive_gasm_config):
        """Create temporary GASM directory structure with comprehensive config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gasm_path = Path(temp_dir)
            
            # Create directory structure
            config_dir = gasm_path / "assets" / "configs"
            urdf_dir = gasm_path / "assets" / "urdf"
            meshes_dir = gasm_path / "assets" / "meshes"
            
            config_dir.mkdir(parents=True, exist_ok=True)
            urdf_dir.mkdir(parents=True, exist_ok=True)
            meshes_dir.mkdir(parents=True, exist_ok=True)
            
            # Write main config file
            config_file = config_dir / "simulation_params.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(comprehensive_gasm_config, f)
            
            # Create sample URDF files
            sample_urdf = """<?xml version="1.0"?>
<robot name="sample_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
    </inertial>
  </link>
</robot>"""
            
            urdf_files = ['panda.urdf', 'cube.urdf', 'sphere.urdf', 'cylinder.urdf']
            for urdf_file in urdf_files:
                (urdf_dir / urdf_file).write_text(sample_urdf)
            
            yield gasm_path
    
    @pytest.fixture
    def material_bridge(self, temp_gasm_directory):
        """Create material bridge with comprehensive materials."""
        return MaterialBridge(temp_gasm_directory)
    
    @pytest.fixture
    def gasm_bridge(self, temp_gasm_directory):
        """Create GASM bridge instance."""
        return GASMBridge(temp_gasm_directory)
    
    @pytest.fixture
    def tactile_processor(self, material_bridge):
        """Create tactile processor for integration testing."""
        config = TactileProcessorConfig(
            sensitivity=0.001,  # High sensitivity for testing
            max_contact_points=50,
            enable_vibration_analysis=True,
            enable_texture_classification=True,
            sampling_rate=1000
        )
        return TactileProcessor(config, material_bridge)
    
    def test_comprehensive_material_loading(self, material_bridge, comprehensive_gasm_config):
        """Test loading of comprehensive material database."""
        expected_materials = list(comprehensive_gasm_config['materials'].keys()) + ['default']
        
        # Verify all materials are loaded
        loaded_materials = material_bridge.list_materials()
        
        for material in expected_materials:
            assert material in loaded_materials
        
        # Test material type classification
        material_type_tests = [
            ('steel', MaterialType.METAL),
            ('aluminum', MaterialType.METAL),
            ('copper', MaterialType.METAL),
            ('abs_plastic', MaterialType.PLASTIC),
            ('nylon', MaterialType.PLASTIC),
            ('rubber', MaterialType.RUBBER),
            ('silicone', MaterialType.RUBBER),
            ('glass', MaterialType.GLASS),
            ('ceramic', MaterialType.COMPOSITE),  # High modulus composite
            ('carbon_fiber', MaterialType.COMPOSITE),
            ('fiberglass', MaterialType.COMPOSITE)
        ]
        
        for material_name, expected_type in material_type_tests:
            material = material_bridge.get_material(material_name)
            assert material is not None
            assert material.material_type == expected_type
    
    def test_material_property_validation(self, material_bridge):
        """Test validation of loaded material properties."""
        test_materials = ['steel', 'aluminum', 'rubber', 'glass', 'carbon_fiber']
        
        for material_name in test_materials:
            material = material_bridge.get_material(material_name)
            assert material is not None
            
            # Basic property validation
            assert 0 <= material.friction <= 2.0
            assert 0 <= material.restitution <= 1.0
            assert material.density > 0
            assert material.young_modulus > 0
            assert 0 <= material.poisson_ratio < 0.5
            
            # Color validation
            assert len(material.color) == 4  # RGBA
            assert all(0 <= c <= 1 for c in material.color)
            
            # Derived properties validation
            assert material.thermal_conductivity > 0
            assert material.surface_roughness >= 0
            assert material.hardness_shore >= 0
    
    def test_tactile_signatures_across_materials(self, material_bridge):
        """Test tactile signature generation across different materials."""
        materials_to_test = ['steel', 'aluminum', 'rubber', 'glass', 'nylon', 'ceramic']
        test_conditions = [
            {'contact_force': 1.0, 'contact_velocity': 0.0, 'contact_area': 0.001},
            {'contact_force': 5.0, 'contact_velocity': 0.1, 'contact_area': 0.005},
            {'contact_force': 10.0, 'contact_velocity': 1.0, 'contact_area': 0.01}
        ]
        
        results = {}
        
        for material in materials_to_test:
            material_results = []
            
            for conditions in test_conditions:
                signature = material_bridge.compute_tactile_signature(
                    material_name=material,
                    **conditions
                )
                
                assert isinstance(signature, dict)
                assert 'hardness' in signature
                assert 'texture_descriptor' in signature
                assert 'thermal_feel' in signature
                
                material_results.append(signature)
            
            results[material] = material_results
        
        # Comparative analysis
        # Steel should be harder than rubber
        steel_hardness = results['steel'][0]['hardness']
        rubber_hardness = results['rubber'][0]['hardness']
        assert steel_hardness > rubber_hardness
        
        # Glass should have lower friction feel than rubber
        glass_grip = results['glass'][0]['grip_quality']
        rubber_grip = results['rubber'][0]['grip_quality']
        assert glass_grip < rubber_grip
        
        # Ceramic should be very hard
        ceramic_hardness = results['ceramic'][0]['hardness']
        assert ceramic_hardness > 0.8  # Should be very hard
    
    def test_audio_signatures_material_dependency(self, material_bridge):
        """Test audio signature generation with material dependency."""
        materials = ['steel', 'glass', 'plastic', 'rubber']
        impact_conditions = [
            {'impact_velocity': 0.5, 'object_size': 0.05},
            {'impact_velocity': 2.0, 'object_size': 0.1},
            {'impact_velocity': 5.0, 'object_size': 0.02}
        ]
        
        audio_results = {}
        
        for material in materials:
            material_audio = []
            
            for conditions in impact_conditions:
                audio_sig = material_bridge.compute_audio_signature(
                    material_name=material,
                    **conditions
                )
                
                assert isinstance(audio_sig, dict)
                assert 'fundamental_freq' in audio_sig
                assert 'amplitude' in audio_sig
                assert 'brightness' in audio_sig
                assert 'harmonics' in audio_sig
                
                # Validate frequency range
                assert 20 <= audio_sig['fundamental_freq'] <= 20000
                assert audio_sig['amplitude'] >= 0
                assert 0 <= audio_sig['brightness'] <= 1
                
                material_audio.append(audio_sig)
            
            audio_results[material] = material_audio
        
        # Material-specific audio characteristics
        # Glass should be brighter than rubber
        glass_brightness = audio_results['glass'][0]['brightness']
        rubber_brightness = audio_results['rubber'][0]['brightness']
        assert glass_brightness > rubber_brightness
        
        # Steel should have rich harmonics
        steel_harmonics = len(audio_results['steel'][0]['harmonics'])
        rubber_harmonics = len(audio_results['rubber'][0]['harmonics'])
        assert steel_harmonics >= rubber_harmonics
    
    def test_material_interaction_modeling(self, material_bridge):
        """Test material interaction modeling between different materials."""
        interaction_pairs = [
            ('steel', 'rubber'),    # Hard-soft interaction
            ('glass', 'silicone'),  # Brittle-elastic interaction
            ('aluminum', 'nylon'),  # Metal-polymer interaction
            ('ceramic', 'steel'),   # Hard-hard interaction
            ('rubber', 'rubber')    # Same material interaction
        ]
        
        for mat1, mat2 in interaction_pairs:
            interaction = material_bridge.compute_interaction(
                mat1, mat2,
                contact_force=5.0,
                relative_velocity=0.1
            )
            
            assert interaction.material1 == mat1
            assert interaction.material2 == mat2
            assert 0 <= interaction.combined_friction <= 2.0
            assert 0 <= interaction.combined_restitution <= 1.0
            assert interaction.effective_modulus > 0
            assert interaction.contact_stiffness > 0
            assert isinstance(interaction.grip_prediction, bool)
            assert isinstance(interaction.bounce_prediction, bool)
        
        # Test specific interaction behaviors
        steel_rubber = material_bridge.compute_interaction('steel', 'rubber')
        glass_glass = material_bridge.compute_interaction('glass', 'glass')
        
        # Steel-rubber should have good grip potential
        assert steel_rubber.combined_friction > 0.5
        
        # Glass-glass should be slippery
        assert glass_glass.combined_friction < steel_rubber.combined_friction
    
    @patch('morpheus.perception.tactile_processor.PYBULLET_AVAILABLE', True)
    @patch('morpheus.perception.tactile_processor.p')
    def test_pybullet_integration_simulation(self, mock_pybullet, tactile_processor):
        """Test PyBullet integration simulation."""
        # Mock PyBullet physics simulation
        mock_contact_data = [
            # Contact data format: [bodyA, bodyB, linkA, linkB, posA, posB, normal, distance, force, ...]
            [1, 2, 0, 0, [0.02, 0, 0], [0.02, 0, 0], [0, 0, 1], 0.001, 2.5, [0, 0, 0]],
            [1, 2, 1, 0, [-0.01, 0.015, 0], [-0.01, 0.015, 0], [0, 0, 1], 0.001, 2.0, [0, 0, 0]],
            [1, 2, 2, 0, [-0.01, -0.015, 0], [-0.01, -0.015, 0], [0, 0, 1], 0.001, 1.5, [0, 0, 0]]
        ]
        
        mock_pybullet.getContactPoints.return_value = mock_contact_data
        
        # Test PyBullet contact processing
        signature = tactile_processor.process_contacts(
            body_id=1,
            material_name='steel'
        )
        
        assert signature is not None
        assert isinstance(signature, TactileSignature)
        assert len(signature.contact_points) == 3  # Should process all 3 contacts
        assert signature.total_force > 0
        
        mock_pybullet.getContactPoints.assert_called_once_with(bodyA=1)
    
    def test_robot_sensor_integration(self, gasm_bridge, comprehensive_gasm_config):
        """Test robot sensor integration configuration."""
        # Verify robot configuration loading
        robot_config = comprehensive_gasm_config['robot']
        
        # Test tactile sensor configuration
        tactile_sensors = robot_config['tactile_sensors']
        
        expected_sensors = ['finger_tip_left', 'finger_tip_right', 'palm']
        for sensor_name in expected_sensors:
            assert sensor_name in tactile_sensors
            sensor_config = tactile_sensors[sensor_name]
            assert 'link_id' in sensor_config
            assert 'sensor_type' in sensor_config
            assert isinstance(sensor_config['link_id'], int)
    
    def test_simulation_performance_benchmarking(self, tactile_processor, material_bridge):
        """Test simulation performance with realistic scenarios."""
        # Simulate high-frequency tactile processing (1000Hz)
        num_samples = 1000
        materials = ['steel', 'aluminum', 'rubber']
        
        processing_times = []
        successful_processes = 0
        
        start_time = time.time()
        
        for i in range(num_samples):
            # Rotate through materials
            material = materials[i % len(materials)]
            
            # Generate realistic contact pattern
            num_contacts = np.random.randint(1, 6)  # 1-5 contacts
            contact_points = []
            
            for j in range(num_contacts):
                contact_points.append({
                    'position': np.random.normal(0, 0.01, 3).tolist(),  # Small variations
                    'normal': [0, 0, 1],
                    'force': np.random.uniform(0.5, 5.0),  # Realistic force range
                    'object_a': 1,
                    'object_b': 2
                })
            
            iteration_start = time.time()
            
            signature = tactile_processor.process_contacts(
                contact_points=contact_points,
                material_name=material,
                timestamp=start_time + i * 0.001  # 1ms intervals
            )
            
            processing_time = time.time() - iteration_start
            processing_times.append(processing_time)
            
            if signature is not None:
                successful_processes += 1
        
        total_time = time.time() - start_time
        
        # Performance metrics
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        success_rate = successful_processes / num_samples
        
        # Performance assertions
        assert avg_processing_time < 0.001  # Average < 1ms (1000Hz capability)
        assert max_processing_time < 0.005  # Max < 5ms
        assert success_rate > 0.95  # >95% success rate
        assert total_time < 10.0  # Total simulation time reasonable
    
    def test_multi_object_scene_simulation(self, material_bridge, comprehensive_gasm_config):
        """Test multi-object scene with different materials."""
        objects_config = comprehensive_gasm_config['objects']
        
        # Simulate scene with multiple objects
        scene_objects = []
        
        for obj_name, obj_config in objects_config.items():
            material_name = obj_config['material']
            
            # Get material properties
            material = material_bridge.get_material(material_name)
            assert material is not None
            
            # Compute expected tactile properties for each object
            tactile_sig = material_bridge.compute_tactile_signature(
                material_name=material_name,
                contact_force=2.0,
                contact_area=0.001
            )
            
            audio_sig = material_bridge.compute_audio_signature(
                material_name=material_name,
                impact_velocity=1.0,
                object_size=0.05
            )
            
            scene_objects.append({
                'name': obj_name,
                'material': material_name,
                'material_props': material,
                'tactile_signature': tactile_sig,
                'audio_signature': audio_sig
            })
        
        # Verify scene diversity
        assert len(scene_objects) >= 3
        
        # Check material diversity in scene
        materials_in_scene = set(obj['material'] for obj in scene_objects)
        assert len(materials_in_scene) >= 3  # At least 3 different materials
        
        # Verify distinctive properties
        hardness_values = [obj['tactile_signature']['hardness'] for obj in scene_objects]
        assert max(hardness_values) - min(hardness_values) > 0.3  # Significant hardness variation
    
    def test_vibration_analysis_material_dependency(self, material_bridge):
        """Test vibration analysis with material-specific characteristics."""
        materials = ['steel', 'aluminum', 'rubber', 'glass']
        
        # Generate force history patterns
        force_patterns = {
            'impulse': [0, 0, 10, 0, 0, 0, 0, 0, 0, 0],  # Sharp impulse
            'oscillation': [1 + 0.5 * np.sin(i * 0.5) for i in range(20)],  # Oscillating
            'decay': [5 * np.exp(-i * 0.3) for i in range(15)]  # Exponential decay
        }
        
        results = {}
        
        for material in materials:
            material_results = {}
            
            for pattern_name, force_history in force_patterns.items():
                spectrum = material_bridge.generate_vibration_spectrum(
                    material_name=material,
                    force_history=force_history,
                    sampling_rate=1000.0
                )
                
                assert isinstance(spectrum, np.ndarray)
                assert spectrum.shape == (32,)
                assert np.all(spectrum >= 0)
                assert np.all(spectrum <= 1)
                
                material_results[pattern_name] = spectrum
            
            results[material] = material_results
        
        # Material-specific vibration characteristics
        # Steel should have different frequency response than rubber
        steel_impulse = results['steel']['impulse']
        rubber_impulse = results['rubber']['impulse']
        
        # Should have different spectral characteristics
        spectral_difference = np.sum(np.abs(steel_impulse - rubber_impulse))
        assert spectral_difference > 0.1  # Significant difference
    
    def test_error_handling_invalid_gasm_config(self, temp_gasm_directory):
        """Test error handling with invalid GASM configurations."""
        # Test missing materials section
        config_file = temp_gasm_directory / "assets" / "configs" / "simulation_params.yaml"
        
        invalid_configs = [
            {},  # Empty config
            {'materials': {}},  # Empty materials
            {'materials': {'invalid_mat': {'friction': 0.5}}},  # Missing required properties
            {'materials': {'steel': {'friction': 'invalid'}}},  # Invalid property type
        ]
        
        for invalid_config in invalid_configs:
            # Overwrite config file
            with open(config_file, 'w') as f:
                yaml.dump(invalid_config, f)
            
            try:
                bridge = MaterialBridge(temp_gasm_directory)
                # Should create bridge but may skip invalid materials
                materials = bridge.list_materials()
                # Should always have at least default material
                assert 'default' in materials
            except ValueError:
                # Expected for severely invalid configs
                pass
    
    def test_memory_efficiency_large_scene(self, material_bridge):
        """Test memory efficiency with large multi-material scenes."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate large scene with many material computations
        materials = material_bridge.list_materials()
        num_computations = 1000
        
        signatures = []
        
        for i in range(num_computations):
            material = materials[i % len(materials)]
            
            # Compute both tactile and audio signatures
            tactile_sig = material_bridge.compute_tactile_signature(
                material_name=material,
                contact_force=np.random.uniform(0.5, 5.0),
                contact_velocity=np.random.uniform(0, 2.0)
            )
            
            audio_sig = material_bridge.compute_audio_signature(
                material_name=material,
                impact_velocity=np.random.uniform(0.1, 3.0),
                object_size=np.random.uniform(0.01, 0.1)
            )
            
            signatures.append((tactile_sig, audio_sig))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase excessively due to caching
        assert memory_increase < 50  # Less than 50MB increase
        
        # Verify caching is working
        cache_stats = material_bridge.get_cache_stats()
        assert cache_stats['tactile_signature']['hits'] > 0
        assert cache_stats['audio_signature']['hits'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])