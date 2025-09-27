#!/usr/bin/env python3
"""Material exploration and learning demonstration.

This example shows how MORPHEUS explores material properties and learns
from interactions. It demonstrates:

- Material property prediction
- Interactive material exploration
- Learning from material interactions  
- Prediction accuracy improvement over time
- Material similarity analysis

Usage:
    python -m morpheus.examples.material_exploration
    
Requirements:
    - PostgreSQL database running
    - GASM-Robotics materials configuration
    - MORPHEUS configuration file
"""

import os
import sys
import logging
import time
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

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


@dataclass
class MaterialTest:
    """Single material interaction test."""
    material: str
    force: float
    velocity: float
    duration: float
    expected_hardness: float
    expected_friction: float


@dataclass
class ExplorationResults:
    """Results of material exploration session."""
    materials_tested: List[str]
    total_interactions: int
    prediction_accuracy: float
    learning_improvement: float
    material_similarities: Dict[str, List[str]]


class MaterialExplorationDemo:
    """Demonstrates MORPHEUS material exploration and learning capabilities."""
    
    def __init__(self, config_path: str = None, gasm_path: str = None):
        """Initialize material exploration demo.
        
        Args:
            config_path: Optional path to MORPHEUS config file
            gasm_path: Optional path to GASM-Robotics directory
        """
        self.config_path = config_path or self._find_default_config()
        self.gasm_path = gasm_path or self._find_gasm_path()
        self.morpheus = None
        self.interaction_history = []
        self.material_knowledge = defaultdict(list)
        
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
        """Initialize MORPHEUS system for material exploration.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("=== Initializing MORPHEUS for Material Exploration ===")
            
            # Database configuration
            db_config = {
                'host': os.getenv('DATABASE_HOST', 'localhost'),
                'port': int(os.getenv('DATABASE_PORT', 5432)),
                'database': os.getenv('DATABASE_NAME', 'morpheus'),
                'user': os.getenv('DATABASE_USER', 'morpheus_user'),
                'password': os.getenv('DATABASE_PASSWORD', 'morpheus_pass')
            }
            
            # Create MORPHEUS system
            self.morpheus = create_morpheus_system(
                config_path=self.config_path,
                gasm_path=self.gasm_path,
                database_config=db_config
            )
            
            logger.info("MORPHEUS system initialized for material exploration!")
            
            # Display available materials
            available_materials = list(self.morpheus.material_bridge.materials.keys())
            logger.info(f"Available materials: {available_materials}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MORPHEUS: {e}")
            return False
    
    def predict_material_interactions(self, material_pairs: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Predict outcomes of material-to-material interactions.
        
        Args:
            material_pairs: List of (material1, material2) pairs to test
            
        Returns:
            Dictionary with prediction results
        """
        if material_pairs is None:
            # Use default interesting material combinations
            material_pairs = [
                ('steel', 'steel'),      # Metal-on-metal
                ('rubber', 'steel'),     # Rubber grip on metal
                ('plastic', 'rubber'),   # Soft-soft interaction
                ('glass', 'steel'),      # Hard-hard interaction  
                ('wood', 'plastic'),     # Natural-synthetic
            ]
        
        logger.info("=== Material Interaction Predictions ===")
        results = {}
        
        for mat1, mat2 in material_pairs:
            logger.info(f"\n--- Predicting {mat1.upper()} + {mat2.upper()} ---")
            
            try:
                # Test different scenarios
                scenarios = [
                    {'force': 1.0, 'velocity': 0.01, 'impact_velocity': 0.1},    # Light touch
                    {'force': 5.0, 'velocity': 0.1, 'impact_velocity': 0.5},     # Normal contact
                    {'force': 10.0, 'velocity': 0.2, 'impact_velocity': 1.0},    # Strong contact
                ]
                
                scenario_results = []
                
                for i, scenario in enumerate(scenarios):
                    prediction = self.morpheus.predict_material_interaction(mat1, mat2, scenario)
                    scenario_results.append(prediction)
                    
                    self._display_interaction_prediction(f"Scenario {i+1}", prediction, scenario)
                
                results[f"{mat1}_{mat2}"] = scenario_results
                
                # Store for learning
                self.material_knowledge[mat1].extend(scenario_results)
                self.material_knowledge[mat2].extend(scenario_results)
                
            except Exception as e:
                logger.error(f"Failed to predict {mat1}+{mat2}: {e}")
                results[f"{mat1}_{mat2}"] = {'error': str(e)}
        
        return results
    
    def _display_interaction_prediction(self, scenario_name: str, prediction: Dict[str, Any], scenario: Dict[str, Any]):
        """Display formatted interaction prediction results."""
        print(f"\n  {scenario_name} (Force: {scenario['force']}N, Velocity: {scenario['velocity']}m/s):")
        
        if 'interaction' in prediction:
            interaction = prediction['interaction']
            print(f"    Combined Friction: {interaction['combined_friction']:.3f}")
            print(f"    Combined Restitution: {interaction['combined_restitution']:.3f}")
            print(f"    Expected Sound: {interaction['expected_sound_frequency']:.0f} Hz")
            print(f"    Good Grip: {'Yes' if interaction['grip_prediction'] else 'No'}")
            print(f"    Will Bounce: {'Yes' if interaction['bounce_prediction'] else 'No'}")
        
        if 'predicted_tactile' in prediction:
            tactile = prediction['predicted_tactile']
            print(f"    Predicted Feel:")
            print(f"      Hardness: {tactile.get('hardness', 0):.3f}")
            print(f"      Texture: {tactile.get('texture_roughness', 0):.3f}")
            print(f"      Deformation: {tactile.get('deformation_mm', 0):.3f} mm")
        
        if 'predicted_audio' in prediction:
            audio = prediction['predicted_audio']
            print(f"    Predicted Sound:")
            print(f"      Frequency: {audio.get('fundamental_freq', 0):.0f} Hz")
            print(f"      Amplitude: {audio.get('amplitude', 0):.3f}")
            print(f"      Brightness: {audio.get('brightness', 0):.3f}")
    
    def conduct_material_exploration(self, 
                                    materials: List[str] = None, 
                                    interactions_per_material: int = 10) -> ExplorationResults:
        """Conduct systematic material exploration with learning.
        
        Args:
            materials: Materials to explore (uses defaults if None)
            interactions_per_material: Number of interactions per material
            
        Returns:
            Exploration results summary
        """
        if materials is None:
            materials = ['steel', 'rubber', 'plastic', 'glass', 'wood', 'aluminum']
        
        logger.info(f"=== Conducting Material Exploration ===")
        logger.info(f"Materials: {materials}")
        logger.info(f"Interactions per material: {interactions_per_material}")
        
        exploration_start = time.time()
        total_interactions = 0
        initial_predictions = []
        final_predictions = []
        
        for material in materials:
            logger.info(f"\n--- Exploring {material.upper()} ---")
            
            # Initial prediction (before learning)
            try:
                initial_pred = self.morpheus.predict_material_interaction(
                    material, material, {'force': 5.0, 'velocity': 0.1}
                )
                initial_predictions.append(initial_pred)
            except:
                pass
            
            # Conduct multiple interactions
            for i in range(interactions_per_material):
                try:
                    # Vary interaction parameters
                    force = random.uniform(1.0, 15.0)
                    velocity = random.uniform(0.01, 0.3)
                    duration = random.uniform(0.5, 2.0)
                    
                    # Create observation
                    observation = {
                        'material': material,
                        'body_id': 1,
                        'robot_position': np.random.randn(3) * 0.1,
                        'robot_velocity': [velocity, 0, 0],
                        'contact_force': force,
                        'forces': [force, random.uniform(-1, 1), random.uniform(-1, 1)],
                        'torques': [0, 0, force * 0.05],
                        'action_type': random.choice(['touch', 'grasp', 'push', 'slide']),
                        'action_params': {
                            'pressure': force,
                            'duration': duration,
                            'approach_speed': velocity
                        },
                        'success': random.random() > 0.1,  # 90% success rate
                        'reward': random.uniform(-0.5, 1.5),
                        'tags': [f'exploration_{material}', f'iteration_{i}'],
                        'notes': f'Material exploration - {material} interaction {i+1}'
                    }
                    
                    # Process perception
                    result = self.morpheus.perceive(observation)
                    
                    # Store interaction
                    self.interaction_history.append({
                        'material': material,
                        'parameters': observation,
                        'result': result,
                        'timestamp': time.time()
                    })
                    
                    total_interactions += 1
                    
                    if (i + 1) % 5 == 0:
                        logger.info(f"  Completed {i + 1}/{interactions_per_material} interactions with {material}")
                    
                except Exception as e:
                    logger.warning(f"Interaction {i+1} with {material} failed: {e}")
            
            # Final prediction (after learning)  
            try:
                final_pred = self.morpheus.predict_material_interaction(
                    material, material, {'force': 5.0, 'velocity': 0.1}
                )
                final_predictions.append(final_pred)
            except:
                pass
        
        exploration_time = time.time() - exploration_start
        logger.info(f"\nExploration completed in {exploration_time:.1f} seconds")
        logger.info(f"Total interactions: {total_interactions}")
        
        # Analyze learning improvement
        learning_improvement = self._analyze_learning_improvement(initial_predictions, final_predictions)
        
        # Analyze material similarities
        material_similarities = self._analyze_material_similarities(materials)
        
        return ExplorationResults(
            materials_tested=materials,
            total_interactions=total_interactions,
            prediction_accuracy=0.8,  # Placeholder - would need ground truth for real accuracy
            learning_improvement=learning_improvement,
            material_similarities=material_similarities
        )
    
    def _analyze_learning_improvement(self, initial_preds: List[Dict], final_preds: List[Dict]) -> float:
        """Analyze improvement in predictions over time.
        
        Args:
            initial_preds: Predictions before learning
            final_preds: Predictions after learning
            
        Returns:
            Learning improvement ratio (0-1)
        """
        if not initial_preds or not final_preds:
            return 0.0
        
        try:
            # Compare prediction confidences as proxy for improvement
            initial_confidence = np.mean([p.get('confidence', 0) for p in initial_preds])
            final_confidence = np.mean([p.get('confidence', 0) for p in final_preds])
            
            improvement = (final_confidence - initial_confidence) / max(initial_confidence, 0.1)
            return max(0.0, min(1.0, improvement))
            
        except:
            return 0.0
    
    def _analyze_material_similarities(self, materials: List[str]) -> Dict[str, List[str]]:
        """Analyze similarities between materials based on interactions.
        
        Args:
            materials: List of materials to analyze
            
        Returns:
            Dictionary mapping each material to similar materials
        """
        similarities = {}
        
        try:
            # Get material properties for comparison
            material_props = {}
            for material in materials:
                props = self.morpheus.material_bridge.get_material(material)
                if props:
                    material_props[material] = props
            
            # Calculate similarity based on material properties
            for mat1 in materials:
                similar = []
                props1 = material_props.get(mat1)
                
                if props1:
                    for mat2 in materials:
                        if mat1 != mat2:
                            props2 = material_props.get(mat2)
                            if props2:
                                # Simple similarity based on physical properties
                                friction_sim = 1.0 - abs(props1.friction - props2.friction)
                                restitution_sim = 1.0 - abs(props1.restitution - props2.restitution)
                                density_sim = 1.0 - abs(props1.density - props2.density) / 10000.0
                                
                                avg_similarity = (friction_sim + restitution_sim + density_sim) / 3
                                
                                if avg_similarity > 0.7:  # Threshold for similarity
                                    similar.append(mat2)
                
                similarities[mat1] = similar
        
        except Exception as e:
            logger.warning(f"Failed to analyze material similarities: {e}")
        
        return similarities
    
    def test_prediction_accuracy_evolution(self, material: str = 'steel', num_tests: int = 20) -> Dict[str, Any]:
        """Test how prediction accuracy evolves with experience.
        
        Args:
            material: Material to test with
            num_tests: Number of prediction tests
            
        Returns:
            Prediction accuracy evolution data
        """
        logger.info(f"\n=== Testing Prediction Accuracy Evolution ({material}) ===")
        
        accuracy_evolution = []
        test_forces = np.linspace(1.0, 10.0, num_tests)
        
        for i, force in enumerate(test_forces):
            try:
                # Create test scenario
                scenario = {
                    'force': force,
                    'velocity': 0.1,
                    'impact_velocity': force * 0.1
                }
                
                # Get prediction
                prediction = self.morpheus.predict_material_interaction(material, material, scenario)
                
                # Conduct actual interaction for comparison
                observation = {
                    'material': material,
                    'body_id': 1,
                    'robot_position': [0, 0, 0.5],
                    'contact_force': force,
                    'forces': [force, 0, 0],
                    'action_type': 'touch',
                    'success': True,
                    'reward': 1.0
                }
                
                actual_result = self.morpheus.perceive(observation)
                
                # Compare prediction vs actual (simplified)
                predicted_confidence = prediction.get('confidence', 0)
                actual_success = actual_result.get('experience_id') is not None
                
                accuracy_evolution.append({
                    'test_number': i + 1,
                    'force': force,
                    'predicted_confidence': predicted_confidence,
                    'actual_success': actual_success,
                    'timestamp': time.time()
                })
                
                logger.info(f"  Test {i+1}: Force {force:.1f}N, Confidence: {predicted_confidence:.3f}")
                
            except Exception as e:
                logger.warning(f"Prediction accuracy test {i+1} failed: {e}")
        
        # Calculate overall accuracy trend
        if accuracy_evolution:
            confidences = [test['predicted_confidence'] for test in accuracy_evolution]
            accuracy_trend = np.polyfit(range(len(confidences)), confidences, 1)[0]  # Linear trend
            
            return {
                'material': material,
                'num_tests': len(accuracy_evolution),
                'accuracy_evolution': accuracy_evolution,
                'trend_slope': accuracy_trend,
                'final_confidence': confidences[-1] if confidences else 0,
                'improvement': confidences[-1] - confidences[0] if len(confidences) > 1 else 0
            }
        
        return {'material': material, 'num_tests': 0, 'error': 'No successful tests'}
    
    def generate_material_knowledge_report(self) -> Dict[str, Any]:
        """Generate comprehensive report of learned material knowledge.
        
        Returns:
            Knowledge report dictionary
        """
        logger.info("\n=== Generating Material Knowledge Report ===")
        
        try:
            # Analyze interaction history
            materials_analyzed = set()
            interaction_stats = defaultdict(lambda: {'count': 0, 'success_rate': 0, 'avg_reward': 0})
            
            for interaction in self.interaction_history:
                material = interaction['material']
                materials_analyzed.add(material)
                
                stats = interaction_stats[material]
                stats['count'] += 1
                
                params = interaction['parameters']
                stats['success_rate'] = (
                    stats['success_rate'] * (stats['count'] - 1) + 
                    (1 if params.get('success') else 0)
                ) / stats['count']
                
                stats['avg_reward'] = (
                    stats['avg_reward'] * (stats['count'] - 1) + 
                    params.get('reward', 0)
                ) / stats['count']
            
            # Get learned strategies
            strategies = self.morpheus.get_learned_strategies()
            
            # Generate knowledge summary
            knowledge_report = {
                'session_summary': self.morpheus.get_session_summary(),
                'materials_analyzed': list(materials_analyzed),
                'total_interactions': len(self.interaction_history),
                'interaction_statistics': dict(interaction_stats),
                'learned_strategies_count': len(strategies),
                'top_strategies': strategies[:5] if strategies else [],
                'material_knowledge_base': dict(self.material_knowledge),
                'generation_timestamp': time.time()
            }
            
            self._display_knowledge_report(knowledge_report)
            
            return knowledge_report
            
        except Exception as e:
            logger.error(f"Failed to generate knowledge report: {e}")
            return {'error': str(e)}
    
    def _display_knowledge_report(self, report: Dict[str, Any]):
        """Display formatted knowledge report."""
        print(f"\n{'='*60}")
        print(f"MATERIAL KNOWLEDGE REPORT")
        print(f"{'='*60}")
        
        print(f"\nMaterials Analyzed: {len(report['materials_analyzed'])}")
        for material in report['materials_analyzed']:
            print(f"  - {material}")
        
        print(f"\nInteraction Statistics:")
        print(f"  Total Interactions: {report['total_interactions']}")
        
        for material, stats in report['interaction_statistics'].items():
            print(f"  {material}:")
            print(f"    Interactions: {stats['count']}")
            print(f"    Success Rate: {stats['success_rate']:.1%}")
            print(f"    Average Reward: {stats['avg_reward']:.2f}")
        
        print(f"\nLearned Strategies: {report['learned_strategies_count']}")
        
        for i, strategy in enumerate(report['top_strategies'], 1):
            print(f"  {i}. {strategy.get('name', 'Unknown')} (Category: {strategy.get('category', 'N/A')})")
            print(f"     Improvement: {strategy.get('improvement_ratio', 0):.2%}")
            print(f"     Confidence: {strategy.get('confidence', 0):.2f}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.morpheus:
            logger.info("Cleaning up MORPHEUS system...")
            self.morpheus.cleanup()
    
    def run_demo(self) -> bool:
        """Run complete material exploration demo.
        
        Returns:
            True if demo completed successfully
        """
        try:
            print("="*80)
            print("MORPHEUS Material Exploration Demo")  
            print("="*80)
            
            # Initialize system
            if not self.initialize_system():
                return False
            
            # Phase 1: Material interaction predictions
            print(f"\n{'='*60}")
            print("PHASE 1: Material Interaction Predictions")
            print(f"{'='*60}")
            
            prediction_results = self.predict_material_interactions()
            
            # Phase 2: Material exploration with learning
            print(f"\n{'='*60}")
            print("PHASE 2: Interactive Material Exploration")
            print(f"{'='*60}")
            
            exploration_results = self.conduct_material_exploration(
                materials=['steel', 'rubber', 'plastic', 'glass'],
                interactions_per_material=8
            )
            
            # Phase 3: Prediction accuracy evolution
            print(f"\n{'='*60}")
            print("PHASE 3: Prediction Accuracy Evolution")
            print(f"{'='*60}")
            
            accuracy_results = self.test_prediction_accuracy_evolution('steel', 10)
            
            # Phase 4: Generate knowledge report
            print(f"\n{'='*60}")
            print("PHASE 4: Material Knowledge Analysis")
            print(f"{'='*60}")
            
            knowledge_report = self.generate_material_knowledge_report()
            
            # Final summary
            print(f"\n{'='*60}")
            print("DEMO COMPLETION SUMMARY")
            print(f"{'='*60}")
            
            print(f"Materials Explored: {len(exploration_results.materials_tested)}")
            print(f"Total Interactions: {exploration_results.total_interactions}")
            print(f"Learning Improvement: {exploration_results.learning_improvement:.2%}")
            print(f"Prediction Tests: {accuracy_results.get('num_tests', 0)}")
            
            if accuracy_results.get('improvement'):
                print(f"Prediction Improvement: {accuracy_results['improvement']:.3f}")
            
            print(f"Knowledge Base Entries: {len(knowledge_report.get('material_knowledge_base', {}))}")
            
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
    """Main entry point for material exploration demo."""
    demo = MaterialExplorationDemo()
    
    success = demo.run_demo()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()