#!/usr/bin/env python3
"""
MORPHEUS Tactile Processing Demo

Demonstrates the advanced tactile processing capabilities of MORPHEUS,
including contact analysis, vibration processing, and texture classification.
"""

import sys
import numpy as np
import time
import logging
from pathlib import Path

# Add morpheus to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from morpheus.integration.material_bridge import MaterialBridge
from morpheus.perception.tactile_processor import TactileProcessor, TactileProcessorConfig
from morpheus.core.types import Vector3D

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simulated_contacts(material: str, scenario: str) -> list:
    """Create simulated contact data for demonstration."""
    
    contacts = []
    
    if scenario == "single_point":
        # Single contact point
        contacts = [{
            'position': [0.0, 0.0, 0.0],
            'normal': [0.0, 0.0, 1.0], 
            'force': 5.0,
            'object_a': 1,
            'object_b': 0
        }]
        
    elif scenario == "multi_point":
        # Multiple contact points in a pattern
        for i in range(5):
            angle = i * 2 * np.pi / 5
            contacts.append({
                'position': [0.01 * np.cos(angle), 0.01 * np.sin(angle), 0.0],
                'normal': [0.0, 0.0, 1.0],
                'force': 2.0 + np.random.random() * 3.0,
                'object_a': 1,
                'object_b': 0
            })
            
    elif scenario == "sliding":
        # Sliding contact with varying force
        for i in range(10):
            x = i * 0.002  # 2mm increments
            force = 3.0 + 2.0 * np.sin(i * 0.5)  # Varying force
            contacts.append({
                'position': [x, 0.0, 0.0],
                'normal': [0.0, 0.0, 1.0],
                'force': force,
                'object_a': 1,
                'object_b': 0
            })
            
    elif scenario == "pressure_wave":
        # Pressure wave simulation
        for i in range(8):
            for j in range(8):
                x = (i - 3.5) * 0.001
                y = (j - 3.5) * 0.001
                distance = np.sqrt(x**2 + y**2)
                force = 1.0 + 4.0 * np.exp(-distance * 100) * np.cos(distance * 50)
                
                if force > 0.1:  # Only add contacts with significant force
                    contacts.append({
                        'position': [x, y, 0.0],
                        'normal': [0.0, 0.0, 1.0],
                        'force': max(force, 0.1),
                        'object_a': 1,
                        'object_b': 0
                    })
    
    return contacts


def demonstrate_material_processing():
    """Demonstrate tactile processing with different materials."""
    
    print("\\n=== Material-Aware Tactile Processing ===")
    
    # Initialize components
    gasm_path = Path(__file__).parent.parent / "GASM-Robotics"
    
    if not gasm_path.exists():
        print("⚠️  GASM-Robotics not found. Using default materials.")
        return
        
    bridge = MaterialBridge(gasm_path)
    config = TactileProcessorConfig(
        sensitivity=0.01,
        sampling_rate=1000,
        enable_vibration_analysis=True,
        enable_texture_classification=True
    )
    processor = TactileProcessor(config, bridge)
    
    # Test different materials with same contact pattern
    materials = ['steel', 'rubber', 'plastic']
    scenario = "multi_point"
    
    for material in materials:
        print(f"\\n🤚 Processing {material.upper()} contact...")
        
        contacts = create_simulated_contacts(material, scenario)
        
        signature = processor.process_contacts(
            contact_points=contacts,
            material_name=material,
            timestamp=time.time()
        )
        
        if signature:
            print(f"  Contact Points: {len(signature.contact_points)}")
            print(f"  Total Force: {signature.total_force:.2f} N")
            print(f"  Pressure: {signature.pressure:.0f} Pa")
            print(f"  Texture: {signature.texture_descriptor}")
            print(f"  Hardness: {signature.hardness:.2f}")
            print(f"  Temperature Feel: {signature.temperature_feel:.1f}°C")
            print(f"  Grip Quality: {signature.grip_quality:.2f}")
            print(f"  Stiffness: {signature.stiffness:.2f}")
            
            # Show vibration characteristics
            vib_energy = np.sum(signature.vibration_spectrum**2)
            dominant_freq = np.argmax(signature.vibration_spectrum)
            print(f"  Vibration Energy: {vib_energy:.3f}")
            print(f"  Dominant Frequency Bin: {dominant_freq}")
        else:
            print("  No tactile signature generated")


def demonstrate_contact_scenarios():
    """Demonstrate different contact scenarios."""
    
    print("\\n=== Contact Scenario Analysis ===")
    
    # Setup with default configuration  
    gasm_path = Path(__file__).parent.parent / "GASM-Robotics"
    
    if gasm_path.exists():
        bridge = MaterialBridge(gasm_path)
    else:
        print("⚠️  Using simulation mode (GASM-Robotics not found)")
        # Create a minimal mock bridge for demo
        class MockBridge:
            def get_material(self, name):
                return None
            def compute_tactile_signature(self, **kwargs):
                return {
                    'hardness': 0.5,
                    'texture_roughness': 0.5,
                    'thermal_feel': 0.5,
                    'grip_quality': 0.5,
                    'stiffness': 0.5,
                    'deformation_mm': 1.0,
                    'texture_descriptor': 'smooth'
                }
        bridge = MockBridge()
    
    config = TactileProcessorConfig(enable_texture_classification=True)
    processor = TactileProcessor(config, bridge)
    
    scenarios = [
        ("single_point", "Single Point Contact", "steel"),
        ("multi_point", "Multi-Point Contact", "rubber"), 
        ("sliding", "Sliding Contact", "plastic"),
        ("pressure_wave", "Pressure Wave", "steel")
    ]
    
    for scenario_name, description, material in scenarios:
        print(f"\\n🎯 {description} ({material}):")
        
        contacts = create_simulated_contacts(material, scenario_name)
        print(f"  Generated {len(contacts)} contact points")
        
        signature = processor.process_contacts(
            contact_points=contacts,
            material_name=material,
            timestamp=time.time()
        )
        
        if signature:
            print(f"  ✅ Contact Area: {signature.contact_area*1e6:.2f} mm²")
            print(f"  ✅ Average Pressure: {signature.pressure:.0f} Pa") 
            print(f"  ✅ Texture: {signature.texture_descriptor}")
            print(f"  ✅ Total Force: {signature.total_force:.2f} N")
            
            # Convert to embedding
            embedding = signature.to_embedding()
            print(f"  ✅ Embedding: {len(embedding)} dimensions")
            print(f"  ✅ Embedding norm: {np.linalg.norm(embedding):.3f}")
        else:
            print("  ❌ No signature generated")


def demonstrate_vibration_analysis():
    """Demonstrate vibration analysis capabilities."""
    
    print("\\n=== Vibration Analysis ===")
    
    # Setup
    config = TactileProcessorConfig(
        sampling_rate=1000,
        vibration_window=0.2,
        enable_vibration_analysis=True
    )
    
    # Create mock bridge for this demo
    class MockBridge:
        def get_material(self, name):
            return None
        def compute_tactile_signature(self, **kwargs):
            return {
                'hardness': 0.5, 'texture_roughness': 0.3, 'thermal_feel': 0.5,
                'grip_quality': 0.5, 'stiffness': 0.5, 'deformation_mm': 1.0,
                'texture_descriptor': 'smooth'
            }
            
    processor = TactileProcessor(config, MockBridge())
    
    print("\\n🌊 Simulating vibration patterns...")
    
    # Simulate a series of contacts with varying forces
    base_time = time.time()
    
    # Pattern 1: Steady contact
    print("\\n📊 Pattern 1: Steady Contact")
    for i in range(50):
        contacts = [{
            'position': [0.0, 0.0, 0.0],
            'normal': [0.0, 0.0, 1.0],
            'force': 5.0 + 0.5 * np.random.randn(),  # Small noise
            'object_a': 1, 'object_b': 0
        }]
        
        signature = processor.process_contacts(
            contact_points=contacts,
            material_name="steel",
            timestamp=base_time + i * 0.001
        )
    
    if hasattr(processor.vibration_analyzer, 'get_vibration_features'):
        features = processor.vibration_analyzer.get_vibration_features()
        print(f"  Spectral Centroid: {features.get('spectral_centroid', 0):.3f}")
        print(f"  Total Energy: {features.get('total_energy', 0):.3f}")
    
    # Reset processor for next pattern
    processor.reset()
    
    # Pattern 2: Oscillating contact
    print("\\n📊 Pattern 2: Oscillating Contact")
    for i in range(50):
        force = 5.0 + 3.0 * np.sin(i * 0.5)  # 5Hz oscillation
        contacts = [{
            'position': [0.0, 0.0, 0.0],
            'normal': [0.0, 0.0, 1.0],
            'force': max(force, 0.1),
            'object_a': 1, 'object_b': 0
        }]
        
        signature = processor.process_contacts(
            contact_points=contacts,
            material_name="rubber",
            timestamp=base_time + i * 0.001
        )
    
    if hasattr(processor.vibration_analyzer, 'get_vibration_features'):
        features = processor.vibration_analyzer.get_vibration_features()
        print(f"  Spectral Centroid: {features.get('spectral_centroid', 0):.3f}")
        print(f"  Total Energy: {features.get('total_energy', 0):.3f}")


def demonstrate_processor_statistics():
    """Show processor performance statistics."""
    
    print("\\n=== Processor Statistics ===")
    
    config = TactileProcessorConfig()
    
    # Mock bridge
    class MockBridge:
        def get_material(self, name): return None
        def compute_tactile_signature(self, **kwargs):
            return {'hardness': 0.5, 'texture_roughness': 0.5, 'thermal_feel': 0.5,
                   'grip_quality': 0.5, 'stiffness': 0.5, 'deformation_mm': 1.0,
                   'texture_descriptor': 'smooth'}
    
    processor = TactileProcessor(config, MockBridge())
    
    # Process some contacts to generate statistics
    for i in range(20):
        contacts = create_simulated_contacts("steel", "single_point")
        processor.process_contacts(contact_points=contacts, material_name="steel")
    
    stats = processor.get_processing_stats()
    
    print(f"📈 Processing Statistics:")
    print(f"  Total Processes: {stats['processing_count']}")
    print(f"  Average Time: {stats['average_processing_time']*1000:.2f} ms")
    print(f"  Total Time: {stats['total_processing_time']:.3f} seconds")
    print(f"  Vibration Buffer: {stats['vibration_buffer_size']} samples")
    
    print(f"\\n⚙️  Processor Configuration:")
    print(f"  Sensitivity: {config.sensitivity:.3f} N")
    print(f"  Sampling Rate: {config.sampling_rate} Hz")
    print(f"  Max Contact Points: {config.max_contact_points}")
    print(f"  Embedding Dimension: {config.embedding_dim}")
    print(f"  Vibration Analysis: {config.enable_vibration_analysis}")
    print(f"  Texture Classification: {config.enable_texture_classification}")


def main():
    """Run the complete tactile processing demonstration."""
    
    print("=== MORPHEUS Tactile Processing Demo ===")
    print("🤖 Advanced tactile perception with material awareness\\n")
    
    try:
        demonstrate_material_processing()
        demonstrate_contact_scenarios()
        demonstrate_vibration_analysis()  
        demonstrate_processor_statistics()
        
        print("\\n✅ Tactile processing demo completed successfully!")
        print("🎯 Key capabilities demonstrated:")
        print("   • Material-aware tactile processing")
        print("   • Multi-point contact analysis") 
        print("   • Vibration pattern recognition")
        print("   • Texture classification")
        print("   • Real-time performance metrics")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        logger.exception("Demo error")


if __name__ == "__main__":
    main()