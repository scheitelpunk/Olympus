"""
Basic integration tests for MORPHEUS core components.

Tests the fundamental functionality without requiring external dependencies
like pydantic or database connections.
"""

import sys
import unittest
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestCoreTypes(unittest.TestCase):
    """Test core type definitions."""
    
    def test_vector3d(self):
        """Test Vector3D functionality."""
        from morpheus.core.types import Vector3D
        
        # Test creation
        vec = Vector3D(3.0, 4.0, 0.0)
        self.assertEqual(vec.x, 3.0)
        self.assertEqual(vec.y, 4.0)
        self.assertEqual(vec.z, 0.0)
        
        # Test magnitude
        self.assertAlmostEqual(vec.magnitude(), 5.0, places=5)
        
        # Test normalization
        normalized = vec.normalize()
        self.assertAlmostEqual(normalized.magnitude(), 1.0, places=5)
        
        # Test array conversion
        arr = vec.to_array()
        self.assertEqual(len(arr), 3)
        self.assertEqual(arr[0], 3.0)
    
    def test_material_properties(self):
        """Test MaterialProperties functionality."""
        from morpheus.core.types import MaterialProperties, MaterialType
        
        # Test creation with defaults
        mat = MaterialProperties(name="test_material")
        self.assertEqual(mat.name, "test_material")
        self.assertEqual(mat.material_type, MaterialType.UNKNOWN)
        
        # Test with specific properties
        steel = MaterialProperties(
            name="steel",
            friction=0.8,
            young_modulus=200e9,
            density=7850
        )
        
        # Test post-init calculations
        self.assertIsNotNone(steel.thermal_conductivity)
        self.assertIsNotNone(steel.surface_roughness)
        self.assertIsNotNone(steel.hardness_shore)
        
        # Test material type determination
        material_type = steel.get_material_type()
        self.assertEqual(material_type, MaterialType.METAL)
    
    def test_contact_point(self):
        """Test ContactPoint functionality."""
        from morpheus.core.types import ContactPoint, Vector3D
        
        pos = Vector3D(1.0, 2.0, 3.0)
        normal = Vector3D(0.0, 0.0, 1.0)
        
        contact = ContactPoint(
            position=pos,
            normal=normal,
            force_magnitude=5.0,
            object_a=1,
            object_b=2
        )
        
        self.assertEqual(contact.force_magnitude, 5.0)
        self.assertEqual(contact.position, pos)
        self.assertEqual(contact.normal, normal)
        
        # Test dictionary conversion
        contact_dict = contact.to_dict()
        self.assertIn('position', contact_dict)
        self.assertIn('force_magnitude', contact_dict)
        self.assertEqual(contact_dict['force_magnitude'], 5.0)
    
    def test_tactile_signature(self):
        """Test TactileSignature functionality."""
        import numpy as np
        from morpheus.core.types import TactileSignature, ContactPoint, Vector3D
        
        # Create sample contact points
        contacts = [
            ContactPoint(
                position=Vector3D(0.0, 0.0, 0.0),
                normal=Vector3D(0.0, 0.0, 1.0),
                force_magnitude=2.0,
                object_a=1,
                object_b=0
            ),
            ContactPoint(
                position=Vector3D(0.01, 0.0, 0.0),
                normal=Vector3D(0.0, 0.0, 1.0), 
                force_magnitude=3.0,
                object_a=1,
                object_b=0
            )
        ]
        
        vibration = np.random.random(32)
        
        signature = TactileSignature(
            timestamp=1234567890.0,
            material="steel",
            contact_points=contacts,
            total_force=5.0,
            contact_area=0.001,
            pressure=5000.0,
            texture_descriptor="smooth",
            hardness=0.8,
            temperature_feel=25.0,
            vibration_spectrum=vibration,
            grip_quality=0.7,
            deformation=0.5,
            stiffness=0.9
        )
        
        # Test basic properties
        self.assertEqual(signature.material, "steel")
        self.assertEqual(signature.total_force, 5.0)
        self.assertEqual(len(signature.contact_points), 2)
        
        # Test embedding generation
        embedding = signature.to_embedding()
        self.assertEqual(len(embedding), 64)  # Default embedding dimension
        self.assertTrue(np.all(np.isfinite(embedding)))
        
        # Test dictionary conversion
        sig_dict = signature.to_dict()
        self.assertIn('material', sig_dict)
        self.assertIn('contact_points', sig_dict)
        self.assertEqual(sig_dict['material'], "steel")
    
    def test_sensory_experience(self):
        """Test SensoryExperience functionality."""
        from morpheus.core.types import SensoryExperience, ActionType
        
        experience = SensoryExperience(
            primary_material="steel",
            action_type=ActionType.TOUCH,
            success=True,
            reward=1.0
        )
        
        # Test UUID generation
        self.assertIsNotNone(experience.experience_id)
        self.assertIsNotNone(experience.session_id)
        
        # Test properties
        self.assertEqual(experience.primary_material, "steel")
        self.assertEqual(experience.action_type, ActionType.TOUCH)
        self.assertTrue(experience.success)
        
        # Test dictionary conversion
        exp_dict = experience.to_dict()
        self.assertIn('experience_id', exp_dict)
        self.assertIn('primary_material', exp_dict)
        self.assertEqual(exp_dict['action_type'], 'TOUCH')


class TestMaterialBridge(unittest.TestCase):
    """Test material bridge with mock GASM data."""
    
    def test_material_bridge_fallback(self):
        """Test material bridge with non-existent GASM path."""
        from morpheus.integration.material_bridge import MaterialBridge
        
        # Test with non-existent path (should raise FileNotFoundError)
        with self.assertRaises(FileNotFoundError):
            MaterialBridge("/non/existent/path")


class TestTactileProcessor(unittest.TestCase):
    """Test tactile processor with mock data."""
    
    def test_tactile_processor_config(self):
        """Test tactile processor configuration."""
        from morpheus.perception.tactile_processor import TactileProcessorConfig
        
        config = TactileProcessorConfig()
        self.assertEqual(config.sensitivity, 0.01)
        self.assertEqual(config.sampling_rate, 1000)
        self.assertTrue(config.use_materials)
        
        # Test custom configuration
        custom_config = TactileProcessorConfig(
            sensitivity=0.05,
            max_contact_points=20
        )
        self.assertEqual(custom_config.sensitivity, 0.05)
        self.assertEqual(custom_config.max_contact_points, 20)


def run_integration_test():
    """Run integration test and return results."""
    
    print("=== MORPHEUS Core Integration Test ===")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestCoreTypes))
    suite.addTest(unittest.makeSuite(TestMaterialBridge))
    suite.addTest(unittest.makeSuite(TestTactileProcessor))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\\n✅ All core integration tests PASSED!")
        print("🚀 MORPHEUS core components are ready for use")
    else:
        print("\\n❌ Some tests FAILED!")
        print("🔧 Please check the implementation")
    
    return success


if __name__ == '__main__':
    success = run_integration_test()
    sys.exit(0 if success else 1)