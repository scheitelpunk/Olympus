#!/usr/bin/env python3
"""Basic demonstration of MORPHEUS perception capabilities.

This example shows how MORPHEUS processes multi-modal sensory data
with material-based tactile and audio processing. It demonstrates:

- Material property-based perception
- Tactile signature computation  
- Audio signature processing
- Basic prediction capabilities
- Session tracking and metrics

Usage:
    python -m morpheus.examples.basic_perception
    
Requirements:
    - PostgreSQL database running
    - GASM-Robotics materials configuration
    - MORPHEUS configuration file
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Add MORPHEUS to path if needed
if __name__ == "__main__":
    morpheus_root = Path(__file__).parent.parent.parent
    if str(morpheus_root) not in sys.path:
        sys.path.insert(0, str(morpheus_root))

from morpheus.core.orchestrator import MorpheusOrchestrator, create_morpheus_system
from morpheus.core.config import ConfigManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BasicPerceptionDemo:
    """Demonstrates basic MORPHEUS perception capabilities."""
    
    def __init__(self, config_path: str = None, gasm_path: str = None):
        """Initialize demo with configuration.
        
        Args:
            config_path: Optional path to MORPHEUS config file
            gasm_path: Optional path to GASM-Robotics directory
        """
        self.config_path = config_path or self._find_default_config()
        self.gasm_path = gasm_path or self._find_gasm_path()
        self.morpheus = None
        
    def _find_default_config(self) -> str:
        """Find default configuration file."""
        possible_paths = [
            "configs/default_config.yaml",
            "../configs/default_config.yaml", 
            "../../configs/default_config.yaml",
            "/mnt/c/dev/coding/Morpheus/configs/default_config.yaml"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        logger.warning("No config file found, using defaults")
        return None
    
    def _find_gasm_path(self) -> str:
        """Find GASM-Robotics path."""
        possible_paths = [
            "../GASM-Robotics",
            "../../GASM-Robotics", 
            "/mnt/c/dev/coding/Morpheus/GASM-Robotics",
            "../../../GASM-Robotics"
        ]
        
        for path in possible_paths:
            gasm_path = Path(path)
            config_file = gasm_path / "assets/configs/simulation_params.yaml"
            if config_file.exists():
                return str(gasm_path)
        
        logger.warning("GASM-Robotics path not found")
        return None
    
    def initialize_system(self) -> bool:
        """Initialize MORPHEUS system.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("=== Initializing MORPHEUS System ===")
            
            # Database configuration (with fallback)
            db_config = {
                'host': os.getenv('DATABASE_HOST', 'localhost'),
                'port': int(os.getenv('DATABASE_PORT', 5432)),
                'database': os.getenv('DATABASE_NAME', 'morpheus'),
                'user': os.getenv('DATABASE_USER', 'morpheus_user'),
                'password': os.getenv('DATABASE_PASSWORD', 'morpheus_pass')
            }
            
            logger.info(f"Config path: {self.config_path}")
            logger.info(f"GASM path: {self.gasm_path}")
            logger.info(f"Database: {db_config['host']}:{db_config['port']}/{db_config['database']}")
            
            # Create MORPHEUS system
            self.morpheus = create_morpheus_system(
                config_path=self.config_path,
                gasm_path=self.gasm_path,
                database_config=db_config
            )
            
            logger.info("MORPHEUS system initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MORPHEUS: {e}")
            return False
    
    def test_material_perception(self, materials: List[str] = None) -> Dict[str, Any]:
        """Test perception with different materials.
        
        Args:
            materials: List of materials to test (uses defaults if None)
            
        Returns:
            Dictionary with test results
        """
        if materials is None:
            materials = ['steel', 'rubber', 'plastic', 'glass', 'wood']
        
        logger.info("=== Testing Material Perception ===")
        results = {}
        
        for material in materials:
            logger.info(f"\n--- Testing {material.upper()} ---")
            
            try:
                # Create realistic observation for material
                observation = self._create_material_observation(material)
                
                # Process perception
                result = self.morpheus.perceive(observation)
                
                # Extract and display key results
                perception_summary = self._summarize_perception(result, material)
                results[material] = perception_summary
                
                # Display results
                self._display_material_results(material, perception_summary)
                
            except Exception as e:
                logger.error(f"Failed to process {material}: {e}")
                results[material] = {'error': str(e)}
        
        return results
    
    def _create_material_observation(self, material: str) -> Dict[str, Any]:
        """Create realistic observation for material testing.
        
        Args:
            material: Material name
            
        Returns:
            Observation dictionary
        """
        # Material-specific parameters
        material_params = {
            'steel': {'force': 8.0, 'velocity': 0.05, 'duration': 1.0},
            'rubber': {'force': 3.0, 'velocity': 0.2, 'duration': 1.5},
            'plastic': {'force': 5.0, 'velocity': 0.1, 'duration': 1.2},
            'glass': {'force': 2.0, 'velocity': 0.02, 'duration': 0.8},
            'wood': {'force': 6.0, 'velocity': 0.08, 'duration': 1.0}
        }
        
        params = material_params.get(material, {'force': 5.0, 'velocity': 0.1, 'duration': 1.0})
        
        return {
            'material': material,
            'body_id': 1,  # Dummy PyBullet body ID
            'robot_position': np.random.randn(3) * 0.05,  # Small position variation
            'robot_velocity': [params['velocity'], 0, 0],
            'contact_force': params['force'],
            'forces': [params['force'], 0, 0],
            'torques': [0, 0, params['force'] * 0.1],
            'action_type': 'touch',
            'action_params': {
                'pressure': params['force'],
                'duration': params['duration'],
                'approach_speed': params['velocity']
            },
            'success': True,
            'reward': 1.0,
            'tags': [f'material_test', f'demo_{material}'],
            'notes': f'Basic perception test of {material}'
        }
    
    def _summarize_perception(self, result: Dict[str, Any], material: str) -> Dict[str, Any]:
        """Summarize perception results for display.
        
        Args:
            result: Raw perception result
            material: Material name
            
        Returns:
            Summary dictionary
        """
        summary = {
            'material': material,
            'experience_id': result.get('experience_id'),
            'processing_time': result.get('processing_time', 0),
            'has_tactile': result.get('tactile') is not None,
            'has_audio': result.get('audio') is not None,
            'has_prediction': result.get('prediction') is not None,
            'fused_embedding_size': len(result.get('fused_embedding', [])),
        }
        
        # Extract tactile properties
        if result.get('tactile'):
            tactile = result['tactile']
            summary['tactile'] = {
                'hardness': tactile.get('hardness', 0),
                'texture': tactile.get('texture_descriptor', 'unknown'),
                'temperature_feel': tactile.get('temperature_feel', 0),
                'grip_quality': tactile.get('grip_quality', 0),
                'total_force': tactile.get('total_force', 0),
                'contact_area': tactile.get('contact_area', 0),
                'pressure': tactile.get('pressure', 0)
            }
        
        # Extract audio properties  
        if result.get('audio'):
            audio = result['audio']
            summary['audio'] = {
                'amplitude': audio.get('amplitude', 0),
                'dominant_frequency': audio.get('dominant_frequency', 0),
                'harmonics_count': len(audio.get('harmonics', [])),
                'decay_rate': audio.get('decay_rate', 0)
            }
        
        # Extract prediction confidence
        if result.get('prediction'):
            pred = result['prediction']
            summary['prediction'] = {
                'confidence': pred.get('confidence', 0),
                'uncertainty_mean': np.mean(pred.get('uncertainty', [])) if pred.get('uncertainty') else 0
            }
        
        return summary
    
    def _display_material_results(self, material: str, summary: Dict[str, Any]):
        """Display formatted results for a material.
        
        Args:
            material: Material name
            summary: Perception summary
        """
        print(f"  Experience ID: {summary['experience_id']}")
        print(f"  Processing Time: {summary['processing_time']:.3f}s")
        print(f"  Embedding Size: {summary['fused_embedding_size']} dims")
        
        if summary.get('tactile'):
            tactile = summary['tactile']
            print(f"  \nTactile Properties:")
            print(f"    Hardness: {tactile['hardness']:.3f}")
            print(f"    Texture: {tactile['texture']}")
            print(f"    Temperature Feel: {tactile['temperature_feel']:.3f}")
            print(f"    Grip Quality: {tactile['grip_quality']:.3f}")
            print(f"    Contact Force: {tactile['total_force']:.2f} N")
            print(f"    Contact Area: {tactile['contact_area']:.4f} m²")
            print(f"    Pressure: {tactile['pressure']:.1f} Pa")
        
        if summary.get('audio'):
            audio = summary['audio']
            print(f"  \nAudio Properties:")
            print(f"    Amplitude: {audio['amplitude']:.3f}")
            print(f"    Dominant Freq: {audio['dominant_frequency']:.1f} Hz")
            print(f"    Harmonics: {audio['harmonics_count']}")
            print(f"    Decay Rate: {audio['decay_rate']:.3f}")
        
        if summary.get('prediction'):
            pred = summary['prediction']
            print(f"  \nPrediction:")
            print(f"    Confidence: {pred['confidence']:.3f}")
            print(f"    Uncertainty: {pred['uncertainty_mean']:.3f}")
    
    def test_prediction_accuracy(self, num_tests: int = 10) -> Dict[str, Any]:
        """Test prediction accuracy with sequential observations.
        
        Args:
            num_tests: Number of prediction tests to run
            
        Returns:
            Prediction accuracy metrics
        """
        logger.info(f"\n=== Testing Prediction Accuracy ({num_tests} tests) ===")
        
        predictions = []
        materials = ['steel', 'rubber', 'plastic']
        
        for i in range(num_tests):
            material = materials[i % len(materials)]
            
            try:
                # Create observation with action
                observation = self._create_material_observation(material)
                observation['action'] = {
                    'position': [0.1, 0, 0],
                    'orientation': [0, 0, 0.1],
                    'gripper': 0.5
                }
                
                # Process perception
                result = self.morpheus.perceive(observation)
                
                if result.get('prediction'):
                    pred = result['prediction']
                    predictions.append({
                        'material': material,
                        'confidence': pred['confidence'],
                        'uncertainty': np.mean(pred['uncertainty']) if pred.get('uncertainty') else 0
                    })
                    
                    logger.info(f"  Test {i+1}: {material} - Confidence: {pred['confidence']:.3f}")
                
            except Exception as e:
                logger.error(f"Prediction test {i+1} failed: {e}")
        
        # Calculate metrics
        if predictions:
            avg_confidence = np.mean([p['confidence'] for p in predictions])
            avg_uncertainty = np.mean([p['uncertainty'] for p in predictions])
            
            metrics = {
                'total_tests': len(predictions),
                'average_confidence': avg_confidence,
                'average_uncertainty': avg_uncertainty,
                'predictions': predictions
            }
            
            logger.info(f"  Average Confidence: {avg_confidence:.3f}")
            logger.info(f"  Average Uncertainty: {avg_uncertainty:.3f}")
            
            return metrics
        else:
            return {'total_tests': 0, 'error': 'No successful predictions'}
    
    def display_session_summary(self):
        """Display comprehensive session summary."""
        logger.info("\n=== Session Summary ===")
        
        try:
            summary = self.morpheus.get_session_summary()
            
            print(f"Session ID: {summary['session_id']}")
            print(f"System Uptime: {summary['system_uptime']:.1f} seconds")
            print(f"Total Perceptions: {summary['perception_count']}")
            print(f"Dream Sessions: {summary['dream_count']}")
            print(f"Error Count: {summary['error_count']}")
            print(f"Success Rate: {summary['success_rate']:.1%}")
            print(f"Average Reward: {summary['average_reward']:.2f}")
            print(f"Materials Explored: {summary['materials_explored']}")
            print(f"Action Types: {summary['action_types_used']}")
            print(f"Total Experiences: {summary['total_experiences']}")
            print(f"Learned Strategies: {summary['learned_strategies']}")
            
            print(f"\nActive Components:")
            for component, active in summary['components_active'].items():
                status = "✓" if active else "✗"
                print(f"  {status} {component.title()}")
                
        except Exception as e:
            logger.error(f"Failed to get session summary: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.morpheus:
            logger.info("Cleaning up MORPHEUS system...")
            self.morpheus.cleanup()
    
    def run_demo(self) -> bool:
        """Run complete basic perception demo.
        
        Returns:
            True if demo completed successfully
        """
        try:
            print("="*60)
            print("MORPHEUS Basic Perception Demo")  
            print("="*60)
            
            # Initialize system
            if not self.initialize_system():
                return False
            
            # Test material perception
            perception_results = self.test_material_perception()
            
            # Test prediction accuracy
            prediction_results = self.test_prediction_accuracy(5)
            
            # Display session summary
            self.display_session_summary()
            
            # Show final results summary
            print(f"\n=== Demo Results ===")
            print(f"Materials tested: {len(perception_results)}")
            print(f"Successful perceptions: {sum(1 for r in perception_results.values() if 'error' not in r)}")
            print(f"Prediction tests: {prediction_results.get('total_tests', 0)}")
            
            if prediction_results.get('average_confidence'):
                print(f"Average prediction confidence: {prediction_results['average_confidence']:.3f}")
            
            print(f"\nDemo completed successfully!")
            return True
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            return False
        finally:
            self.cleanup()


def main():
    """Main entry point for basic perception demo."""
    demo = BasicPerceptionDemo()
    
    success = demo.run_demo()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()