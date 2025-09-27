#!/usr/bin/env python3
"""
Basic Perception Demo for MORPHEUS System
Demonstrates multi-modal perception capabilities with material-aware processing.

This example shows:
1. Material-based tactile processing
2. Spatial audio analysis
3. Sensory fusion
4. Database storage
5. Basic prediction capabilities
"""

import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add MORPHEUS to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morpheus.core.orchestrator import MorpheusOrchestrator
from morpheus.core.config import ConfigurationManager
from morpheus.utils.demo_helpers import create_synthetic_observation, print_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BasicPerceptionDemo:
    """
    Demonstrates basic MORPHEUS perception capabilities.
    """
    
    def __init__(self):
        """Initialize demo with configuration."""
        self.config_manager = ConfigurationManager("configs/default_config.yaml")
        self.morpheus = None
        
        # Demo materials to test
        self.test_materials = ['steel', 'rubber', 'plastic', 'glass', 'aluminum']
        
        # Demo scenarios
        self.test_scenarios = [
            {'action_type': 'touch', 'force_range': [1, 5]},
            {'action_type': 'grip', 'force_range': [5, 15]}, 
            {'action_type': 'tap', 'force_range': [0.5, 2]},
            {'action_type': 'slide', 'force_range': [2, 8]}
        ]
        
    def setup_system(self):
        """Initialize MORPHEUS system."""
        print("=== Initializing MORPHEUS System ===")
        
        try:
            # Database configuration
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'morpheus',
                'user': 'morpheus_user',
                'password': 'morpheus_pass'
            }
            
            # Initialize MORPHEUS
            config_path = "configs/default_config.yaml"
            gasm_path = "GASM-Robotics"
            
            self.morpheus = MorpheusOrchestrator(
                config_path=config_path,
                gasm_roboting_path=gasm_path,
                db_config=db_config
            )
            
            print(f"✓ MORPHEUS initialized successfully")
            print(f"✓ Session ID: {self.morpheus.session_id}")
            print(f"✓ Available materials: {list(self.morpheus.material_bridge.materials.keys())}")
            
        except Exception as e:
            print(f"✗ Failed to initialize MORPHEUS: {e}")
            print("Please ensure:")
            print("  1. PostgreSQL is running (docker-compose up postgres)")
            print("  2. GASM-Robotics directory exists")
            print("  3. Configuration files are present")
            sys.exit(1)
            
    def demonstrate_material_perception(self):
        """Demonstrate material-aware tactile perception."""
        print("\n=== Material Perception Demonstration ===")
        
        for material in self.test_materials:
            print(f"\n--- Testing {material.upper()} ---")
            
            # Create realistic observation for material
            observation = create_synthetic_observation(
                material=material,
                action_type='touch',
                contact_force=np.random.uniform(2, 8),
                robot_position=np.random.uniform(-0.1, 0.1, 3) + [0, 0, 0.5],
                success=True
            )
            
            # Process through MORPHEUS
            result = self.morpheus.perceive(observation)
            
            # Display results
            print_results(f"Material: {material}", result)
            
            # Show material-specific insights
            if 'tactile' in result and result['tactile']:
                tactile_data = result['tactile']
                print(f"  Tactile Analysis:")
                print(f"    • Hardness: {tactile_data.get('hardness', 0):.2f}")
                print(f"    • Texture: {tactile_data.get('texture', 'unknown')}")
                print(f"    • Temperature: {tactile_data.get('temperature', 20):.1f}°C")
                print(f"    • Grip Quality: {tactile_data.get('grip_quality', 0):.2f}")
                
    def demonstrate_force_analysis(self):
        """Demonstrate force-based tactile analysis."""
        print("\n=== Force Analysis Demonstration ===")
        
        material = 'steel'  # Use consistent material
        force_levels = [0.5, 2.0, 5.0, 10.0, 20.0]  # Different force levels
        
        print(f"Testing {material} with varying contact forces:")
        
        results = []
        for force in force_levels:
            observation = create_synthetic_observation(
                material=material,
                action_type='press',
                contact_force=force,
                robot_position=[0, 0, 0.5],
                success=force >= 1.0  # Minimum force threshold
            )
            
            result = self.morpheus.perceive(observation)
            results.append((force, result))
            
            print(f"  Force: {force:4.1f}N → Success: {result.get('success', False)}")
            
        # Analyze force-response relationship
        self._analyze_force_response(results)
        
    def demonstrate_multi_modal_fusion(self):
        """Demonstrate multi-modal sensory fusion."""
        print("\n=== Multi-Modal Fusion Demonstration ===")
        
        # Create observation with multiple modalities
        observation = create_synthetic_observation(
            material='rubber',
            action_type='impact',
            contact_force=8.0,
            robot_position=[0.2, 0.1, 0.6],
            robot_velocity=[0.5, 0, 0],  # Moving robot for audio
            impact_velocity=1.2,  # For audio generation
            success=True
        )
        
        # Add visual features (simulated from GASM)
        observation['visual_features'] = np.random.randn(128).tolist()
        
        result = self.morpheus.perceive(observation)
        
        print("Multi-modal fusion results:")
        print(f"  Experience ID: {result['experience_id']}")
        print(f"  Fused embedding shape: {len(result['fused_embedding'])}")
        
        # Show individual modality contributions
        if 'tactile' in result and result['tactile']:
            print(f"  Tactile: Contact detected, force={observation['contact_force']:.1f}N")
            
        if 'audio' in result and result['audio']:
            print(f"  Audio: Sound detected from impact")
            
        if 'visual_features' in observation:
            print(f"  Visual: Features processed ({len(observation['visual_features'])} dims)")
            
    def demonstrate_prediction_capability(self):
        """Demonstrate predictive capabilities."""
        print("\n=== Prediction Demonstration ===")
        
        # Create observation with action for prediction
        observation = create_synthetic_observation(
            material='plastic',
            action_type='grip',
            contact_force=6.0,
            robot_position=[0, 0, 0.5],
            success=True
        )
        
        # Add action parameters for prediction
        observation['action'] = {
            'position': [0.1, 0, 0],  # Move right 10cm
            'orientation': [0, 0, 0.1],  # Slight rotation
            'gripper': 0.8  # Close gripper 80%
        }
        
        result = self.morpheus.perceive(observation)
        
        if 'prediction' in result and result['prediction']:
            prediction = result['prediction']
            print("Prediction results:")
            print(f"  Confidence: {prediction['confidence']:.2f}")
            print(f"  Predicted state dims: {len(prediction['predicted_state'])}")
            print(f"  Uncertainty: {np.mean(prediction['uncertainty']):.3f}")
            
    def demonstrate_scenario_comparison(self):
        """Compare different interaction scenarios."""
        print("\n=== Scenario Comparison Demonstration ===")
        
        scenarios = [
            {'material': 'steel', 'action': 'grip', 'force': 10},
            {'material': 'rubber', 'action': 'grip', 'force': 10}, 
            {'material': 'glass', 'action': 'grip', 'force': 10},
            {'material': 'steel', 'action': 'tap', 'force': 2},
            {'material': 'rubber', 'action': 'tap', 'force': 2},
        ]
        
        print("Comparing scenarios:")
        print(f"{'Material':<8} {'Action':<6} {'Force':<6} {'Success':<8} {'Grip Quality':<12}")
        print("-" * 50)
        
        for scenario in scenarios:
            observation = create_synthetic_observation(
                material=scenario['material'],
                action_type=scenario['action'],
                contact_force=scenario['force'],
                robot_position=[0, 0, 0.5],
                success=True
            )
            
            result = self.morpheus.perceive(observation)
            
            # Extract grip quality if available
            grip_quality = 0.0
            if 'tactile' in result and result['tactile']:
                grip_quality = result['tactile'].get('grip_quality', 0.0)
                
            print(f"{scenario['material']:<8} {scenario['action']:<6} "
                  f"{scenario['force']:<6} {str(result.get('success', False)):<8} "
                  f"{grip_quality:<12.2f}")
                  
    def demonstrate_material_interaction_prediction(self):
        """Demonstrate material interaction prediction."""
        print("\n=== Material Interaction Prediction ===")
        
        interaction_pairs = [
            ('steel', 'steel'),
            ('rubber', 'steel'), 
            ('plastic', 'rubber'),
            ('glass', 'aluminum')
        ]
        
        print("Material interaction predictions:")
        print(f"{'Material 1':<10} {'Material 2':<10} {'Friction':<8} {'Bounce':<8} {'Sound Hz':<10}")
        print("-" * 55)
        
        for mat1, mat2 in interaction_pairs:
            prediction = self.morpheus.predict_material_interaction(
                mat1, mat2,
                scenario={'force': 5.0, 'velocity': 0.1, 'impact_velocity': 0.5}
            )
            
            interaction = prediction['interaction']
            
            print(f"{mat1:<10} {mat2:<10} "
                  f"{interaction['combined_friction']:<8.2f} "
                  f"{str(interaction['bounce_prediction']):<8} "
                  f"{interaction['expected_sound_freq']:<10.0f}")
                  
    def generate_session_summary(self):
        """Generate and display session summary."""
        print("\n=== Session Summary ===")
        
        summary = self.morpheus.get_session_summary()
        
        print(f"Session ID: {summary['session_id']}")
        print(f"Total Perceptions: {summary['perception_count']}")
        print(f"Dream Sessions: {summary['dream_count']}")
        print(f"Materials Explored: {', '.join(summary['materials_explored'])}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Database Experiences: {summary['database_experiences']}")
        print(f"Learned Strategies: {summary['learned_strategies']}")
        
    def cleanup(self):
        """Clean up resources."""
        if self.morpheus:
            self.morpheus.cleanup()
            print("\n✓ System cleanup completed")
            
    def _analyze_force_response(self, results):
        """Analyze force-response relationship."""
        forces = [r[0] for r in results]
        successes = [r[1].get('success', False) for r in results]
        
        # Find success threshold
        success_threshold = None
        for force, success in zip(forces, successes):
            if success and success_threshold is None:
                success_threshold = force
                break
                
        if success_threshold:
            print(f"  Success threshold: ~{success_threshold:.1f}N")
        else:
            print("  No clear success threshold found")
            

def main():
    """Main demo function."""
    print("MORPHEUS Basic Perception Demo")
    print("=" * 50)
    
    demo = BasicPerceptionDemo()
    
    try:
        # Setup
        demo.setup_system()
        
        # Run demonstrations
        demo.demonstrate_material_perception()
        demo.demonstrate_force_analysis()
        demo.demonstrate_multi_modal_fusion()
        demo.demonstrate_prediction_capability()
        demo.demonstrate_scenario_comparison()
        demo.demonstrate_material_interaction_prediction()
        
        # Summary
        demo.generate_session_summary()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        logger.exception("Demo execution failed")
    finally:
        # Always cleanup
        demo.cleanup()
        
    print("\nDemo completed!")


if __name__ == "__main__":
    main()