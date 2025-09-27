#!/usr/bin/env python3
"""
Basic MORPHEUS Material Bridge Demo

Demonstrates the material bridge functionality with GASM-Robotics integration.
Shows how to load materials, compute tactile and audio signatures, and predict
material interactions.
"""

import sys
import logging
from pathlib import Path

# Add morpheus to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from morpheus.integration.material_bridge import MaterialBridge
from morpheus.core.types import MaterialType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run basic material bridge demonstration."""
    
    print("=== MORPHEUS Material Bridge Demo ===\\n")
    
    # Initialize material bridge (adjust path as needed)
    gasm_path = Path(__file__).parent.parent / "GASM-Robotics"
    
    if not gasm_path.exists():
        print(f"❌ GASM-Robotics not found at {gasm_path}")
        print("Please ensure GASM-Robotics is available at the expected location.")
        return
    
    try:
        bridge = MaterialBridge(gasm_path)
        print(f"✅ Material bridge initialized successfully")
        print(f"📁 GASM path: {bridge.gasm_path}")
        print(f"🔧 Loaded materials: {bridge.list_materials()}\\n")
        
    except Exception as e:
        print(f"❌ Failed to initialize material bridge: {e}")
        return
    
    # Demonstrate material properties
    print("=== Material Properties ===")
    
    for material_name in ['steel', 'rubber', 'plastic']:
        material = bridge.get_material(material_name)
        if material:
            print(f"\\n📦 {material_name.upper()}:")
            print(f"  Type: {material.material_type.name}")
            print(f"  Friction: {material.friction:.2f}")
            print(f"  Restitution: {material.restitution:.2f}")
            print(f"  Density: {material.density:,.0f} kg/m³")
            print(f"  Young's Modulus: {material.young_modulus:.2e} Pa")
            print(f"  Thermal Conductivity: {material.thermal_conductivity:.1f} W/mK")
        else:
            print(f"⚠️  Material '{material_name}' not found")
    
    # Demonstrate tactile signatures
    print("\\n=== Tactile Signatures ===")
    
    test_scenarios = [
        {'material': 'steel', 'force': 5.0, 'velocity': 0.1},
        {'material': 'rubber', 'force': 5.0, 'velocity': 0.1},
        {'material': 'plastic', 'force': 5.0, 'velocity': 0.1},
    ]
    
    for scenario in test_scenarios:
        material_name = scenario['material']
        tactile = bridge.compute_tactile_signature(
            material_name=material_name,
            contact_force=scenario['force'],
            contact_velocity=scenario['velocity']
        )
        
        print(f"\\n🤚 {material_name.upper()} Tactile (Force: {scenario['force']}N):")
        print(f"  Hardness: {tactile['hardness']:.2f}")
        print(f"  Texture: {tactile['texture_descriptor']}")
        print(f"  Thermal Feel: {tactile['thermal_feel']:.2f} (0=cool, 1=warm)")
        print(f"  Grip Quality: {tactile['grip_quality']:.2f}")
        print(f"  Deformation: {tactile['deformation_mm']:.3f} mm")
    
    # Demonstrate audio signatures
    print("\\n=== Audio Signatures ===")
    
    for scenario in test_scenarios:
        material_name = scenario['material']
        audio = bridge.compute_audio_signature(
            material_name=material_name,
            impact_velocity=1.0
        )
        
        print(f"\\n🔊 {material_name.upper()} Audio (Impact: 1.0 m/s):")
        print(f"  Fundamental Freq: {audio['fundamental_freq']:.0f} Hz")
        print(f"  Amplitude: {audio['amplitude']:.1f}")
        print(f"  Brightness: {audio['brightness']:.2f}")
        print(f"  Harmonics: {len(audio['harmonics'])} components")
    
    # Demonstrate material interactions
    print("\\n=== Material Interactions ===")
    
    interactions = [
        ('steel', 'steel', 'Metal-on-metal contact'),
        ('rubber', 'steel', 'Rubber grip on metal'), 
        ('plastic', 'plastic', 'Plastic-on-plastic sliding'),
    ]
    
    for mat1, mat2, description in interactions:
        interaction = bridge.compute_interaction(mat1, mat2, contact_force=10.0)
        
        print(f"\\n⚙️  {description.upper()}:")
        print(f"  Materials: {mat1} + {mat2}")
        print(f"  Combined Friction: {interaction.combined_friction:.2f}")
        print(f"  Combined Restitution: {interaction.combined_restitution:.2f}")
        print(f"  Expected Sound: {interaction.expected_sound_frequency:.0f} Hz")
        print(f"  Good Grip: {'Yes' if interaction.grip_prediction else 'No'}")
        print(f"  Will Bounce: {'Yes' if interaction.bounce_prediction else 'No'}")
    
    # Demonstrate predictive capabilities
    print("\\n=== Sensory Predictions ===")
    
    scenarios = [
        {
            'material': 'steel',
            'force': 15.0,
            'velocity': 0.5,
            'impact_velocity': 2.0,
            'description': 'Strong steel contact with sliding'
        },
        {
            'material': 'rubber', 
            'force': 3.0,
            'velocity': 0.0,
            'impact_velocity': 0.5,
            'description': 'Gentle rubber compression'
        }
    ]
    
    for scenario in scenarios:
        print(f"\\n🎯 {scenario['description'].upper()}:")
        
        prediction = bridge.predict_sensory_outcome(scenario)
        
        print(f"  Material: {prediction['material']} ({prediction.get('material_type', 'unknown')})")
        print(f"  Confidence: {prediction['confidence']:.1%}")
        
        if 'tactile' in prediction:
            tactile = prediction['tactile']
            print(f"  Predicted Tactile:")
            print(f"    - Hardness: {tactile.get('hardness', 0):.2f}")
            print(f"    - Texture: {tactile.get('texture_descriptor', 'unknown')}")
            print(f"    - Temperature: {tactile.get('thermal_feel', 0.5)*40:.1f}°C")
        
        if 'audio' in prediction:
            audio = prediction['audio']
            print(f"  Predicted Audio:")
            print(f"    - Frequency: {audio.get('fundamental_freq', 0):.0f} Hz")
            print(f"    - Amplitude: {audio.get('amplitude', 0):.1f}")
    
    # Performance statistics
    print("\\n=== Performance Statistics ===")
    cache_stats = bridge.get_cache_stats()
    
    print(f"Cache Performance:")
    for cache_name, stats in cache_stats.items():
        hit_rate = stats['hits'] / (stats['hits'] + stats['misses']) if (stats['hits'] + stats['misses']) > 0 else 0
        print(f"  {cache_name}: {hit_rate:.1%} hit rate ({stats['size']} entries)")
    
    print(f"\\n✅ Demo completed successfully!")
    print(f"🔗 Integration with GASM-Robotics verified")
    print(f"🤖 Material-aware perception ready for robotic systems")


if __name__ == "__main__":
    main()