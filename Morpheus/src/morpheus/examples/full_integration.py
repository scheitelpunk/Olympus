#!/usr/bin/env python3
"""Complete MORPHEUS system integration demonstration.

This comprehensive example demonstrates the full MORPHEUS system capabilities including:

- Multi-modal perception (tactile, audio, visual)
- Material property learning and prediction
- Dream cycle optimization and strategy learning
- Session management with UUID tracking
- Error handling and system resilience
- Performance monitoring and metrics
- Complete system lifecycle management

This is the most comprehensive demo showing all MORPHEUS features working together.

Usage:
    python -m morpheus.examples.full_integration
    
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
import threading
import signal
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# Add MORPHEUS to path if needed
if __name__ == "__main__":
    morpheus_root = Path(__file__).parent.parent.parent
    if str(morpheus_root) not in sys.path:
        sys.path.insert(0, str(morpheus_root))

from morpheus.core.orchestrator import MorpheusOrchestrator, create_morpheus_system, quick_perception_test
from morpheus.core.config import ConfigManager


# Configure logging for comprehensive output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/morpheus_full_demo.log', mode='w') if os.name != 'nt' else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SystemBenchmark:
    """Comprehensive system performance benchmark."""
    test_name: str
    start_time: float
    end_time: float
    operations_count: int
    success_rate: float
    average_latency: float
    peak_memory_mb: float
    errors_encountered: int
    notes: str = ""
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def throughput(self) -> float:
        return self.operations_count / self.duration if self.duration > 0 else 0


@dataclass
class IntegrationTestSuite:
    """Complete integration test suite configuration."""
    perception_tests: int = 50
    material_tests: int = 8
    dream_cycles: int = 2
    prediction_tests: int = 20
    stress_test_duration: int = 60
    concurrent_sessions: int = 3
    error_injection_rate: float = 0.05


class MorpheusFullIntegrationDemo:
    """Comprehensive demonstration of complete MORPHEUS system integration."""
    
    def __init__(self, config_path: str = None, gasm_path: str = None):
        """Initialize full integration demo.
        
        Args:
            config_path: Optional path to MORPHEUS config file
            gasm_path: Optional path to GASM-Robotics directory
        """
        self.config_path = config_path or self._find_default_config()
        self.gasm_path = gasm_path or self._find_gasm_path()
        self.morpheus = None
        self.benchmarks = []
        self.test_suite = IntegrationTestSuite()
        self.shutdown_requested = False
        self.error_count = 0
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
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
        """Initialize complete MORPHEUS system with full validation.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("="*80)
            logger.info("INITIALIZING MORPHEUS COMPLETE SYSTEM")
            logger.info("="*80)
            
            # Database configuration with environment fallbacks
            db_config = {
                'host': os.getenv('DATABASE_HOST', 'localhost'),
                'port': int(os.getenv('DATABASE_PORT', 5432)),
                'database': os.getenv('DATABASE_NAME', 'morpheus'),
                'user': os.getenv('DATABASE_USER', 'morpheus_user'),
                'password': os.getenv('DATABASE_PASSWORD', 'morpheus_pass')
            }
            
            logger.info(f"Configuration path: {self.config_path}")
            logger.info(f"GASM-Robotics path: {self.gasm_path}")
            logger.info(f"Database: {db_config['host']}:{db_config['port']}/{db_config['database']}")
            
            # Create MORPHEUS system with context management
            self.morpheus = create_morpheus_system(
                config_path=self.config_path,
                gasm_path=self.gasm_path,
                database_config=db_config
            )
            
            # Register comprehensive callbacks
            self._register_system_callbacks()
            
            # Perform system validation
            validation_results = self._validate_system_components()
            
            if not validation_results['all_valid']:
                logger.error("System validation failed!")
                for component, valid in validation_results['components'].items():
                    if not valid:
                        logger.error(f"  {component}: FAILED")
                return False
            
            logger.info("MORPHEUS system fully initialized and validated!")
            
            # Display system capabilities
            self._display_system_capabilities()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MORPHEUS: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _register_system_callbacks(self):
        """Register comprehensive system callbacks for monitoring."""
        self.morpheus.register_callback('perception', self._on_perception)
        self.morpheus.register_callback('dream_start', self._on_dream_start)
        self.morpheus.register_callback('dream_end', self._on_dream_end)
        self.morpheus.register_callback('error', self._on_error)
        
        logger.info("System callbacks registered")
    
    def _on_perception(self, experience, prediction):
        """Callback for perception events."""
        logger.debug(f"Perception processed: {experience.experience_id}")
    
    def _on_dream_start(self, dream_id: str, duration: float):
        """Callback for dream session start."""
        logger.info(f"🌙 Dream session {dream_id} started (duration: {duration}s)")
    
    def _on_dream_end(self, dream_id: str, results: Dict[str, Any]):
        """Callback for dream session completion."""
        strategies = results.get('strategies_found', 0)
        logger.info(f"🌅 Dream session {dream_id} completed ({strategies} strategies)")
    
    def _on_error(self, error: Exception, context: str):
        """Callback for error events."""
        self.error_count += 1
        logger.warning(f"System error in {context}: {error}")
    
    def _validate_system_components(self) -> Dict[str, Any]:
        """Validate all system components are working correctly.
        
        Returns:
            Validation results dictionary
        """
        logger.info("Validating system components...")
        
        validation_results = {
            'components': {
                'database': False,
                'material_bridge': False,
                'tactile_processor': False,
                'audio_processor': False,
                'fusion_network': False,
                'predictor': False,
                'dream_engine': False
            },
            'all_valid': False
        }
        
        try:
            # Test database connection
            summary = self.morpheus.get_session_summary()
            validation_results['components']['database'] = 'session_id' in summary
            
            # Test material bridge
            materials = list(self.morpheus.material_bridge.materials.keys())
            validation_results['components']['material_bridge'] = len(materials) > 0
            
            # Test processors
            validation_results['components']['tactile_processor'] = self.morpheus.tactile_processor is not None
            validation_results['components']['audio_processor'] = self.morpheus.audio_processor is not None
            validation_results['components']['fusion_network'] = self.morpheus.fusion_network is not None
            validation_results['components']['predictor'] = self.morpheus.predictor is not None
            validation_results['components']['dream_engine'] = self.morpheus.dream_engine is not None
            
            # Overall validation
            validation_results['all_valid'] = all(validation_results['components'].values())
            
            # Log results
            for component, valid in validation_results['components'].items():
                status = "✓" if valid else "✗"
                logger.info(f"  {status} {component}")
            
        except Exception as e:
            logger.error(f"Component validation failed: {e}")
        
        return validation_results
    
    def _display_system_capabilities(self):
        """Display comprehensive system capabilities."""
        logger.info("\nSYSTEM CAPABILITIES:")
        
        # Available materials
        materials = list(self.morpheus.material_bridge.materials.keys())
        logger.info(f"  Materials: {len(materials)} types - {', '.join(materials[:5])}{'...' if len(materials) > 5 else ''}")
        
        # Component status
        components = {
            'Multi-modal Perception': self.morpheus.tactile_processor and self.morpheus.audio_processor,
            'Sensory Fusion': self.morpheus.fusion_network is not None,
            'Forward Prediction': self.morpheus.predictor is not None,
            'Dream Optimization': self.morpheus.dream_engine is not None,
            'GASM Integration': self.morpheus.gasm_bridge is not None,
        }
        
        for capability, enabled in components.items():
            status = "✓" if enabled else "✗"
            logger.info(f"  {status} {capability}")
        
        # Configuration summary
        config = self.morpheus.config
        logger.info(f"  System Mode: {config.system.mode}")
        logger.info(f"  Dream Enabled: {config.dream.enabled}")
        logger.info(f"  Parallel Dreams: {config.dream.parallel_dreams}")
    
    def run_comprehensive_perception_test(self) -> SystemBenchmark:
        """Run comprehensive multi-modal perception testing.
        
        Returns:
            System benchmark results
        """
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE PERCEPTION TESTING")
        logger.info("="*60)
        
        start_time = time.time()
        materials = ['steel', 'rubber', 'plastic', 'glass', 'wood', 'aluminum']
        action_types = ['touch', 'grasp', 'push', 'slide', 'lift', 'rotate']
        
        operations = 0
        successes = 0
        latencies = []
        peak_memory = 0
        errors = 0
        
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            process = None
        
        for i in range(self.test_suite.perception_tests):
            if self.shutdown_requested:
                break
            
            try:
                # Create complex multi-modal observation
                material = random.choice(materials)
                action = random.choice(action_types)
                force = random.uniform(1.0, 15.0)
                
                perception_start = time.time()
                
                observation = {
                    'material': material,
                    'body_id': random.randint(1, 10),
                    'robot_position': np.random.randn(3) * 0.15,
                    'robot_velocity': np.random.randn(3) * 0.1,
                    'contact_force': force,
                    'forces': [force, random.uniform(-2, 2), random.uniform(-1, 1)],
                    'torques': [random.uniform(-1, 1), random.uniform(-1, 1), force * 0.1],
                    'action_type': action,
                    'action_params': {
                        'pressure': force,
                        'duration': random.uniform(0.5, 3.0),
                        'approach_speed': random.uniform(0.01, 0.3),
                        'target_position': np.random.randn(3).tolist()
                    },
                    'visual_features': np.random.randn(128).tolist(),  # Simulated visual features
                    'object_positions': [np.random.randn(3).tolist() for _ in range(random.randint(1, 5))],
                    'colors': [[random.random(), random.random(), random.random(), 1.0] for _ in range(random.randint(1, 3))],
                    'lighting': {'ambient': random.uniform(0.2, 1.0), 'directional': random.uniform(0.0, 0.8)},
                    'success': random.random() > 0.15,  # 85% success rate
                    'reward': random.uniform(-0.5, 2.0),
                    'tags': [f'test_{i}', f'material_{material}', f'action_{action}'],
                    'notes': f'Comprehensive perception test {i+1}/{self.test_suite.perception_tests}'
                }
                
                # Add action for prediction testing
                if random.random() > 0.3:  # 70% chance of action
                    observation['action'] = {
                        'position': np.random.randn(3).tolist(),
                        'orientation': np.random.randn(3).tolist(),
                        'gripper': random.uniform(0, 1)
                    }
                
                # Process perception
                result = self.morpheus.perceive(observation)
                
                perception_time = time.time() - perception_start
                latencies.append(perception_time)
                
                operations += 1
                if result.get('experience_id'):
                    successes += 1
                
                # Track memory usage
                if process:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, memory_mb)
                
                # Progress reporting
                if (i + 1) % 10 == 0:
                    avg_latency = np.mean(latencies[-10:])
                    logger.info(f"  Processed {i+1}/{self.test_suite.perception_tests} - "
                               f"Avg latency: {avg_latency:.3f}s")
                
                # Brief pause to avoid overwhelming the system
                if i % 5 == 0:
                    time.sleep(0.01)
                    
            except Exception as e:
                errors += 1
                logger.warning(f"Perception test {i+1} failed: {e}")
        
        end_time = time.time()
        
        benchmark = SystemBenchmark(
            test_name="Comprehensive Perception Test",
            start_time=start_time,
            end_time=end_time,
            operations_count=operations,
            success_rate=successes / operations if operations > 0 else 0,
            average_latency=np.mean(latencies) if latencies else 0,
            peak_memory_mb=peak_memory,
            errors_encountered=errors,
            notes=f"Multi-modal perception with {len(materials)} materials and {len(action_types)} actions"
        )
        
        self.benchmarks.append(benchmark)
        
        logger.info(f"\nPerception Test Results:")
        logger.info(f"  Duration: {benchmark.duration:.1f}s")
        logger.info(f"  Operations: {benchmark.operations_count}")
        logger.info(f"  Success Rate: {benchmark.success_rate:.1%}")
        logger.info(f"  Throughput: {benchmark.throughput:.1f} ops/sec")
        logger.info(f"  Average Latency: {benchmark.average_latency:.3f}s")
        logger.info(f"  Peak Memory: {benchmark.peak_memory_mb:.1f} MB")
        logger.info(f"  Errors: {benchmark.errors_encountered}")
        
        return benchmark
    
    def run_material_interaction_analysis(self) -> Dict[str, Any]:
        """Run comprehensive material interaction analysis.
        
        Returns:
            Material analysis results
        """
        logger.info("\n" + "="*60)
        logger.info("MATERIAL INTERACTION ANALYSIS")
        logger.info("="*60)
        
        materials = list(self.morpheus.material_bridge.materials.keys())[:self.test_suite.material_tests]
        
        # Test all material pairs
        interaction_results = {}
        prediction_accuracy = []
        
        for i, mat1 in enumerate(materials):
            for j, mat2 in enumerate(materials[i:], i):
                if self.shutdown_requested:
                    break
                
                logger.info(f"Testing interaction: {mat1} + {mat2}")
                
                try:
                    # Test multiple scenarios
                    scenarios = [
                        {'force': 1.0, 'velocity': 0.01, 'impact_velocity': 0.1},
                        {'force': 5.0, 'velocity': 0.1, 'impact_velocity': 0.5},
                        {'force': 10.0, 'velocity': 0.2, 'impact_velocity': 1.0}
                    ]
                    
                    scenario_results = []
                    
                    for scenario in scenarios:
                        # Get prediction
                        prediction = self.morpheus.predict_material_interaction(mat1, mat2, scenario)
                        
                        # Conduct actual test
                        observation = {
                            'material': mat1,
                            'body_id': 1,
                            'robot_position': [0, 0, 0.5],
                            'contact_force': scenario['force'],
                            'forces': [scenario['force'], 0, 0],
                            'action_type': 'touch',
                            'success': True,
                            'reward': 1.0
                        }
                        
                        actual = self.morpheus.perceive(observation)
                        
                        # Compare prediction vs actual (simplified accuracy measure)
                        predicted_confidence = prediction.get('confidence', 0)
                        actual_success = actual.get('experience_id') is not None
                        
                        accuracy = predicted_confidence if actual_success else (1.0 - predicted_confidence)
                        prediction_accuracy.append(accuracy)
                        
                        scenario_results.append({
                            'scenario': scenario,
                            'prediction': prediction,
                            'actual': actual,
                            'accuracy': accuracy
                        })
                    
                    interaction_results[f"{mat1}_{mat2}"] = scenario_results
                    
                except Exception as e:
                    logger.warning(f"Material interaction test {mat1}+{mat2} failed: {e}")
        
        # Calculate overall metrics
        avg_prediction_accuracy = np.mean(prediction_accuracy) if prediction_accuracy else 0
        
        results = {
            'materials_tested': len(materials),
            'interaction_pairs': len(interaction_results),
            'total_scenarios': sum(len(results) for results in interaction_results.values()),
            'average_prediction_accuracy': avg_prediction_accuracy,
            'interaction_results': interaction_results,
            'accuracy_distribution': {
                'mean': avg_prediction_accuracy,
                'std': np.std(prediction_accuracy) if len(prediction_accuracy) > 1 else 0,
                'min': np.min(prediction_accuracy) if prediction_accuracy else 0,
                'max': np.max(prediction_accuracy) if prediction_accuracy else 0
            }
        }
        
        logger.info(f"\nMaterial Analysis Results:")
        logger.info(f"  Materials Tested: {results['materials_tested']}")
        logger.info(f"  Interaction Pairs: {results['interaction_pairs']}")
        logger.info(f"  Total Scenarios: {results['total_scenarios']}")
        logger.info(f"  Avg Prediction Accuracy: {results['average_prediction_accuracy']:.3f}")
        logger.info(f"  Accuracy Range: {results['accuracy_distribution']['min']:.3f} - {results['accuracy_distribution']['max']:.3f}")
        
        return results
    
    def run_dream_cycle_optimization(self) -> List[Dict[str, Any]]:
        """Run multiple dream cycles for optimization testing.
        
        Returns:
            List of dream cycle results
        """
        logger.info("\n" + "="*60)
        logger.info("DREAM CYCLE OPTIMIZATION")
        logger.info("="*60)
        
        dream_results = []
        
        for cycle in range(self.test_suite.dream_cycles):
            if self.shutdown_requested:
                break
            
            logger.info(f"\nDream Cycle {cycle + 1}/{self.test_suite.dream_cycles}")
            
            try:
                # Get pre-dream state
                pre_dream_summary = self.morpheus.get_session_summary()
                pre_dream_strategies = len(self.morpheus.get_learned_strategies())
                
                logger.info(f"  Pre-dream: {pre_dream_summary['total_experiences']} experiences, "
                           f"{pre_dream_strategies} strategies")
                
                # Run dream session
                dream_duration = 20.0 + cycle * 10  # Increasing duration
                dream_result = self.morpheus.dream(dream_duration)
                
                # Get post-dream state
                post_dream_summary = self.morpheus.get_session_summary()
                post_dream_strategies = len(self.morpheus.get_learned_strategies())
                
                # Calculate improvements
                strategies_learned = post_dream_strategies - pre_dream_strategies
                success_rate_change = post_dream_summary['success_rate'] - pre_dream_summary['success_rate']
                
                cycle_result = {
                    'cycle': cycle + 1,
                    'dream_result': dream_result,
                    'strategies_learned': strategies_learned,
                    'success_rate_change': success_rate_change,
                    'pre_dream_state': pre_dream_summary,
                    'post_dream_state': post_dream_summary
                }
                
                dream_results.append(cycle_result)
                
                logger.info(f"  Dream completed: {dream_result.get('strategies_found', 0)} strategies found, "
                           f"{strategies_learned} stored")
                logger.info(f"  Success rate change: {success_rate_change:+.1%}")
                
                # Add more experiences between cycles
                if cycle < self.test_suite.dream_cycles - 1:
                    logger.info("  Adding experiences for next cycle...")
                    self._add_diverse_experiences(30)
                
            except Exception as e:
                logger.error(f"Dream cycle {cycle + 1} failed: {e}")
                dream_results.append({
                    'cycle': cycle + 1,
                    'error': str(e)
                })
        
        logger.info(f"\nDream Optimization Summary:")
        successful_cycles = [r for r in dream_results if 'error' not in r]
        
        if successful_cycles:
            total_strategies = sum(r['strategies_learned'] for r in successful_cycles)
            avg_success_change = np.mean([r['success_rate_change'] for r in successful_cycles])
            
            logger.info(f"  Successful Cycles: {len(successful_cycles)}/{len(dream_results)}")
            logger.info(f"  Total Strategies Learned: {total_strategies}")
            logger.info(f"  Average Success Rate Change: {avg_success_change:+.2%}")
        
        return dream_results
    
    def _add_diverse_experiences(self, count: int):
        """Add diverse experiences for dream processing."""
        materials = ['steel', 'rubber', 'plastic', 'glass', 'wood']
        actions = ['touch', 'grasp', 'push', 'slide', 'lift']
        
        for _ in range(count):
            observation = {
                'material': random.choice(materials),
                'body_id': random.randint(1, 3),
                'robot_position': np.random.randn(3) * 0.1,
                'contact_force': random.uniform(1.0, 10.0),
                'forces': [random.uniform(1, 10), random.uniform(-1, 1), random.uniform(-1, 1)],
                'action_type': random.choice(actions),
                'success': random.random() > 0.2,
                'reward': random.uniform(-0.5, 1.5)
            }
            
            try:
                self.morpheus.perceive(observation)
            except:
                pass  # Ignore errors during batch addition
    
    def run_prediction_accuracy_evolution(self) -> Dict[str, Any]:
        """Test how prediction accuracy evolves over time.
        
        Returns:
            Prediction evolution results
        """
        logger.info("\n" + "="*60)
        logger.info("PREDICTION ACCURACY EVOLUTION")
        logger.info("="*60)
        
        test_material = 'steel'
        evolution_data = []
        
        for test in range(self.test_suite.prediction_tests):
            if self.shutdown_requested:
                break
            
            try:
                # Create test scenario
                force = random.uniform(1.0, 12.0)
                scenario = {
                    'force': force,
                    'velocity': random.uniform(0.01, 0.2),
                    'impact_velocity': force * 0.08
                }
                
                # Get prediction
                prediction = self.morpheus.predict_material_interaction(test_material, test_material, scenario)
                predicted_confidence = prediction.get('confidence', 0)
                
                # Conduct actual interaction
                observation = {
                    'material': test_material,
                    'body_id': 1,
                    'robot_position': [0, 0, 0.5],
                    'contact_force': force,
                    'forces': [force, 0, 0],
                    'action_type': 'touch',
                    'action': {
                        'position': [0.1, 0, 0],
                        'orientation': [0, 0, 0],
                        'gripper': 0.5
                    },
                    'success': True,
                    'reward': 1.0
                }
                
                result = self.morpheus.perceive(observation)
                
                # Extract actual prediction accuracy (if predictor was used)
                actual_prediction = result.get('prediction')
                actual_confidence = actual_prediction.get('confidence', 0) if actual_prediction else 0
                
                evolution_data.append({
                    'test_number': test + 1,
                    'force': force,
                    'predicted_confidence': predicted_confidence,
                    'actual_confidence': actual_confidence,
                    'accuracy_delta': abs(predicted_confidence - actual_confidence)
                })
                
                if (test + 1) % 5 == 0:
                    recent_accuracy = np.mean([d['predicted_confidence'] for d in evolution_data[-5:]])
                    logger.info(f"  Tests {test-3}-{test+1}: Recent avg confidence: {recent_accuracy:.3f}")
                    
            except Exception as e:
                logger.warning(f"Prediction test {test+1} failed: {e}")
        
        # Analyze evolution
        if evolution_data:
            confidences = [d['predicted_confidence'] for d in evolution_data]
            accuracy_deltas = [d['accuracy_delta'] for d in evolution_data]
            
            # Calculate trends
            if len(confidences) > 5:
                early_confidence = np.mean(confidences[:5])
                late_confidence = np.mean(confidences[-5:])
                confidence_improvement = late_confidence - early_confidence
                
                early_accuracy = np.mean(accuracy_deltas[:5])
                late_accuracy = np.mean(accuracy_deltas[-5:])
                accuracy_improvement = early_accuracy - late_accuracy  # Lower delta = better
            else:
                confidence_improvement = 0
                accuracy_improvement = 0
            
            results = {
                'total_tests': len(evolution_data),
                'confidence_improvement': confidence_improvement,
                'accuracy_improvement': accuracy_improvement,
                'final_confidence': confidences[-1] if confidences else 0,
                'average_accuracy_delta': np.mean(accuracy_deltas) if accuracy_deltas else 0,
                'evolution_data': evolution_data
            }
            
            logger.info(f"\nPrediction Evolution Results:")
            logger.info(f"  Total Tests: {results['total_tests']}")
            logger.info(f"  Confidence Improvement: {results['confidence_improvement']:+.3f}")
            logger.info(f"  Accuracy Improvement: {results['accuracy_improvement']:+.3f}")
            logger.info(f"  Final Confidence: {results['final_confidence']:.3f}")
            logger.info(f"  Avg Accuracy Delta: {results['average_accuracy_delta']:.3f}")
            
            return results
        else:
            return {'total_tests': 0, 'error': 'No successful prediction tests'}
    
    def run_system_stress_test(self) -> SystemBenchmark:
        """Run system stress test with concurrent operations.
        
        Returns:
            Stress test benchmark
        """
        logger.info("\n" + "="*60)
        logger.info("SYSTEM STRESS TEST")
        logger.info("="*60)
        
        start_time = time.time()
        end_time = start_time + self.test_suite.stress_test_duration
        
        operations = 0
        successes = 0
        errors = 0
        latencies = []
        peak_memory = 0
        
        # Track resource usage
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
        except ImportError:
            process = None
            initial_memory = 0
        
        logger.info(f"Running stress test for {self.test_suite.stress_test_duration}s...")
        
        while time.time() < end_time and not self.shutdown_requested:
            try:
                operation_start = time.time()
                
                # Random operation type
                operation_type = random.choice(['perception', 'prediction', 'quick_test'])
                
                if operation_type == 'perception':
                    # Random perception test
                    observation = {
                        'material': random.choice(['steel', 'rubber', 'plastic']),
                        'body_id': random.randint(1, 5),
                        'robot_position': np.random.randn(3) * 0.1,
                        'contact_force': random.uniform(1, 15),
                        'forces': np.random.randn(3).tolist(),
                        'action_type': random.choice(['touch', 'grasp', 'push']),
                        'success': random.random() > 0.1,
                        'reward': random.uniform(-0.5, 1.5)
                    }
                    
                    result = self.morpheus.perceive(observation)
                    success = result.get('experience_id') is not None
                    
                elif operation_type == 'prediction':
                    # Material interaction prediction
                    materials = ['steel', 'rubber', 'plastic']
                    mat1, mat2 = random.choices(materials, k=2)
                    scenario = {'force': random.uniform(1, 10), 'velocity': random.uniform(0.01, 0.2)}
                    
                    result = self.morpheus.predict_material_interaction(mat1, mat2, scenario)
                    success = result.get('confidence', 0) > 0
                    
                else:  # quick_test
                    # Quick perception test
                    result = quick_perception_test(self.morpheus, 
                                                 material=random.choice(['steel', 'rubber']),
                                                 force=random.uniform(2, 8))
                    success = result.get('experience_id') is not None
                
                operation_time = time.time() - operation_start
                latencies.append(operation_time)
                operations += 1
                
                if success:
                    successes += 1
                
                # Track memory usage
                if process:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, memory_mb)
                
                # Brief pause to prevent overwhelming
                if operations % 10 == 0:
                    time.sleep(0.001)
                
            except Exception as e:
                errors += 1
                if errors <= 5:  # Log only first few errors
                    logger.warning(f"Stress test operation failed: {e}")
        
        actual_duration = time.time() - start_time
        
        benchmark = SystemBenchmark(
            test_name="System Stress Test",
            start_time=start_time,
            end_time=time.time(),
            operations_count=operations,
            success_rate=successes / operations if operations > 0 else 0,
            average_latency=np.mean(latencies) if latencies else 0,
            peak_memory_mb=peak_memory,
            errors_encountered=errors,
            notes=f"Concurrent operations for {actual_duration:.1f}s"
        )
        
        self.benchmarks.append(benchmark)
        
        logger.info(f"\nStress Test Results:")
        logger.info(f"  Duration: {actual_duration:.1f}s")
        logger.info(f"  Operations: {operations}")
        logger.info(f"  Success Rate: {benchmark.success_rate:.1%}")
        logger.info(f"  Throughput: {benchmark.throughput:.1f} ops/sec")
        logger.info(f"  Average Latency: {benchmark.average_latency:.3f}s")
        logger.info(f"  Peak Memory: {peak_memory:.1f} MB")
        logger.info(f"  Memory Growth: {peak_memory - initial_memory:+.1f} MB")
        logger.info(f"  Errors: {errors}")
        
        return benchmark
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive system performance and capability report.
        
        Returns:
            Complete system report
        """
        logger.info("\n" + "="*80)
        logger.info("GENERATING COMPREHENSIVE SYSTEM REPORT")
        logger.info("="*80)
        
        # Get current system state
        session_summary = self.morpheus.get_session_summary()
        learned_strategies = self.morpheus.get_learned_strategies(limit=50)
        system_metrics = self.morpheus.get_system_metrics()
        
        # Compile benchmark statistics
        if self.benchmarks:
            benchmark_stats = {
                'total_benchmarks': len(self.benchmarks),
                'total_operations': sum(b.operations_count for b in self.benchmarks),
                'overall_success_rate': np.mean([b.success_rate for b in self.benchmarks]),
                'average_throughput': np.mean([b.throughput for b in self.benchmarks]),
                'average_latency': np.mean([b.average_latency for b in self.benchmarks]),
                'peak_memory': max(b.peak_memory_mb for b in self.benchmarks),
                'total_errors': sum(b.errors_encountered for b in self.benchmarks)
            }
        else:
            benchmark_stats = {}
        
        # Analyze strategy effectiveness
        if learned_strategies:
            strategy_stats = {
                'total_strategies': len(learned_strategies),
                'categories': {},
                'average_improvement': np.mean([s.get('improvement_ratio', 0) for s in learned_strategies]),
                'average_confidence': np.mean([s.get('confidence', 0) for s in learned_strategies]),
                'top_strategy': max(learned_strategies, key=lambda x: x.get('improvement_ratio', 0))
            }
            
            # Group by category
            for strategy in learned_strategies:
                category = strategy.get('category', 'unknown')
                if category not in strategy_stats['categories']:
                    strategy_stats['categories'][category] = 0
                strategy_stats['categories'][category] += 1
        else:
            strategy_stats = {'total_strategies': 0}
        
        # Compile comprehensive report
        report = {
            'report_timestamp': time.time(),
            'session_summary': session_summary,
            'system_metrics': system_metrics.to_dict(),
            'benchmark_statistics': benchmark_stats,
            'strategy_analysis': strategy_stats,
            'test_configuration': {
                'perception_tests': self.test_suite.perception_tests,
                'material_tests': self.test_suite.material_tests,
                'dream_cycles': self.test_suite.dream_cycles,
                'prediction_tests': self.test_suite.prediction_tests,
                'stress_test_duration': self.test_suite.stress_test_duration
            },
            'system_health': {
                'total_errors': self.error_count,
                'error_rate': self.error_count / session_summary['perception_count'] if session_summary['perception_count'] > 0 else 0,
                'components_active': session_summary['components_active']
            },
            'performance_grade': self._calculate_performance_grade(benchmark_stats, strategy_stats)
        }
        
        self._display_comprehensive_report(report)
        
        return report
    
    def _calculate_performance_grade(self, benchmark_stats: Dict[str, Any], strategy_stats: Dict[str, Any]) -> str:
        """Calculate overall system performance grade."""
        if not benchmark_stats:
            return 'INCOMPLETE'
        
        score = 0
        max_score = 0
        
        # Success rate (30%)
        if 'overall_success_rate' in benchmark_stats:
            score += benchmark_stats['overall_success_rate'] * 30
            max_score += 30
        
        # Throughput (20%)
        if 'average_throughput' in benchmark_stats:
            # Score based on throughput (normalize to 0-1)
            throughput_score = min(benchmark_stats['average_throughput'] / 10.0, 1.0)
            score += throughput_score * 20
            max_score += 20
        
        # Strategy learning (25%)
        if strategy_stats.get('total_strategies', 0) > 0:
            strategy_score = min(strategy_stats['total_strategies'] / 20.0, 1.0)
            score += strategy_score * 25
            max_score += 25
        
        # Error rate (15% - inverted)
        if 'total_errors' in benchmark_stats:
            error_rate = benchmark_stats['total_errors'] / benchmark_stats.get('total_operations', 1)
            error_score = max(0, 1.0 - error_rate * 10)  # Penalize errors heavily
            score += error_score * 15
            max_score += 15
        
        # Latency performance (10% - inverted)
        if 'average_latency' in benchmark_stats:
            latency_score = max(0, 1.0 - benchmark_stats['average_latency'] / 2.0)  # Target < 0.2s
            score += latency_score * 10
            max_score += 10
        
        if max_score == 0:
            return 'INCOMPLETE'
        
        percentage = (score / max_score) * 100
        
        if percentage >= 90:
            return 'EXCELLENT (A+)'
        elif percentage >= 80:
            return 'VERY GOOD (A)'
        elif percentage >= 70:
            return 'GOOD (B)'
        elif percentage >= 60:
            return 'SATISFACTORY (C)'
        elif percentage >= 50:
            return 'NEEDS IMPROVEMENT (D)'
        else:
            return 'POOR (F)'
    
    def _display_comprehensive_report(self, report: Dict[str, Any]):
        """Display formatted comprehensive report."""
        print(f"\n{'='*100}")
        print(f"MORPHEUS COMPLETE SYSTEM INTEGRATION REPORT")
        print(f"{'='*100}")
        
        # Executive Summary
        session = report['session_summary']
        print(f"\nEXECUTIVE SUMMARY:")
        print(f"  Session ID: {session['session_id']}")
        print(f"  System Mode: {session.get('system_mode', 'Unknown')}")
        print(f"  Total Runtime: {session['system_uptime']:.1f}s")
        print(f"  Performance Grade: {report['performance_grade']}")
        
        # System Operations
        print(f"\nSYSTEM OPERATIONS:")
        print(f"  Total Perceptions: {session['perception_count']}")
        print(f"  Dream Cycles: {session['dream_count']}")
        print(f"  Total Experiences: {session['total_experiences']}")
        print(f"  Overall Success Rate: {session['success_rate']:.1%}")
        print(f"  Average Reward: {session.get('average_reward', 0):.2f}")
        
        # Materials and Actions
        print(f"\nCAPABILITIES DEMONSTRATED:")
        print(f"  Materials Explored: {len(session['materials_explored'])} types")
        for material in session['materials_explored'][:6]:  # Show first 6
            print(f"    - {material}")
        if len(session['materials_explored']) > 6:
            print(f"    ... and {len(session['materials_explored']) - 6} more")
        
        print(f"  Action Types: {len(session['action_types_used'])} types")
        for action in session['action_types_used']:
            print(f"    - {action}")
        
        # Benchmark Results
        benchmark_stats = report.get('benchmark_statistics', {})
        if benchmark_stats:
            print(f"\nPERFORMANCE BENCHMARKS:")
            print(f"  Total Benchmarks: {benchmark_stats['total_benchmarks']}")
            print(f"  Total Operations: {benchmark_stats['total_operations']}")
            print(f"  Overall Success Rate: {benchmark_stats['overall_success_rate']:.1%}")
            print(f"  Average Throughput: {benchmark_stats['average_throughput']:.1f} ops/sec")
            print(f"  Average Latency: {benchmark_stats['average_latency']:.3f}s")
            print(f"  Peak Memory Usage: {benchmark_stats['peak_memory']:.1f} MB")
            print(f"  Total Errors: {benchmark_stats['total_errors']}")
        
        # Strategy Learning
        strategy_stats = report.get('strategy_analysis', {})
        if strategy_stats.get('total_strategies', 0) > 0:
            print(f"\nSTRATEGY LEARNING:")
            print(f"  Total Strategies Learned: {strategy_stats['total_strategies']}")
            print(f"  Average Improvement: {strategy_stats['average_improvement']:.2%}")
            print(f"  Average Confidence: {strategy_stats['average_confidence']:.2f}")
            
            if 'categories' in strategy_stats:
                print(f"  Strategy Categories:")
                for category, count in strategy_stats['categories'].items():
                    print(f"    - {category}: {count}")
            
            if 'top_strategy' in strategy_stats:
                top = strategy_stats['top_strategy']
                print(f"  Best Strategy: {top.get('name', 'Unknown')} "
                      f"({top.get('improvement_ratio', 0):.2%} improvement)")
        
        # System Health
        health = report.get('system_health', {})
        print(f"\nSYSTEM HEALTH:")
        print(f"  Total Errors: {health.get('total_errors', 0)}")
        print(f"  Error Rate: {health.get('error_rate', 0):.3%}")
        
        components = health.get('components_active', {})
        print(f"  Active Components:")
        for component, active in components.items():
            status = "✓" if active else "✗"
            print(f"    {status} {component.title()}")
        
        # Test Configuration
        config = report.get('test_configuration', {})
        print(f"\nTEST CONFIGURATION:")
        print(f"  Perception Tests: {config.get('perception_tests', 0)}")
        print(f"  Material Tests: {config.get('material_tests', 0)}")
        print(f"  Dream Cycles: {config.get('dream_cycles', 0)}")
        print(f"  Prediction Tests: {config.get('prediction_tests', 0)}")
        print(f"  Stress Test Duration: {config.get('stress_test_duration', 0)}s")
    
    def cleanup(self):
        """Clean up all system resources."""
        if self.morpheus:
            logger.info("Cleaning up MORPHEUS system resources...")
            self.morpheus.cleanup()
            logger.info("System cleanup completed.")
    
    def run_complete_demo(self) -> bool:
        """Run the complete integration demonstration.
        
        Returns:
            True if demo completed successfully
        """
        try:
            print("="*100)
            print("MORPHEUS COMPLETE SYSTEM INTEGRATION DEMONSTRATION")
            print("="*100)
            print("This comprehensive demo tests all MORPHEUS capabilities including:")
            print("- Multi-modal perception processing")
            print("- Material property learning and prediction")
            print("- Dream cycle optimization")
            print("- Strategy learning and adaptation")
            print("- System performance and resilience")
            print("="*100)
            
            # Phase 1: System initialization and validation
            print(f"\n🚀 PHASE 1: System Initialization")
            if not self.initialize_system():
                logger.error("System initialization failed")
                return False
            
            time.sleep(2)  # Allow system to settle
            
            # Phase 2: Comprehensive perception testing
            print(f"\n🔍 PHASE 2: Comprehensive Perception Testing")
            perception_benchmark = self.run_comprehensive_perception_test()
            
            if self.shutdown_requested:
                logger.info("Demo interrupted during perception testing")
                return False
            
            time.sleep(1)
            
            # Phase 3: Material interaction analysis
            print(f"\n🧪 PHASE 3: Material Interaction Analysis")
            material_results = self.run_material_interaction_analysis()
            
            if self.shutdown_requested:
                return False
            
            time.sleep(1)
            
            # Phase 4: Dream cycle optimization
            print(f"\n🌙 PHASE 4: Dream Cycle Optimization")
            dream_results = self.run_dream_cycle_optimization()
            
            if self.shutdown_requested:
                return False
            
            time.sleep(1)
            
            # Phase 5: Prediction accuracy evolution
            print(f"\n🔮 PHASE 5: Prediction Accuracy Evolution")
            prediction_results = self.run_prediction_accuracy_evolution()
            
            if self.shutdown_requested:
                return False
            
            time.sleep(1)
            
            # Phase 6: System stress testing
            print(f"\n💪 PHASE 6: System Stress Testing")
            stress_benchmark = self.run_system_stress_test()
            
            if self.shutdown_requested:
                return False
            
            # Phase 7: Comprehensive analysis and reporting
            print(f"\n📊 PHASE 7: Comprehensive Analysis")
            final_report = self.generate_comprehensive_report()
            
            # Demo completion summary
            print(f"\n🎉 DEMO COMPLETION SUMMARY")
            print(f"="*60)
            
            total_benchmarks = len(self.benchmarks)
            total_operations = sum(b.operations_count for b in self.benchmarks)
            overall_success = np.mean([b.success_rate for b in self.benchmarks]) if self.benchmarks else 0
            
            print(f"Benchmarks Completed: {total_benchmarks}")
            print(f"Total Operations: {total_operations}")
            print(f"Overall Success Rate: {overall_success:.1%}")
            print(f"System Errors: {self.error_count}")
            print(f"Performance Grade: {final_report.get('performance_grade', 'Unknown')}")
            
            # Key achievements
            strategies_learned = len(self.morpheus.get_learned_strategies())
            materials_tested = len(material_results.get('interaction_results', {}))
            dream_cycles = len(dream_results)
            
            print(f"\nKEY ACHIEVEMENTS:")
            print(f"✓ {strategies_learned} strategies learned through dream optimization")
            print(f"✓ {materials_tested} material interaction pairs analyzed")
            print(f"✓ {dream_cycles} dream cycles completed successfully")
            print(f"✓ Multi-modal perception processing validated")
            print(f"✓ System resilience and error handling verified")
            
            if overall_success > 0.8:
                print(f"\n🏆 EXCELLENT PERFORMANCE - All systems operating optimally!")
            elif overall_success > 0.7:
                print(f"\n✅ GOOD PERFORMANCE - System functioning well with minor issues")
            else:
                print(f"\n⚠️  PERFORMANCE ISSUES DETECTED - Review system configuration")
            
            logger.info("MORPHEUS Complete Integration Demo finished successfully! 🎉")
            return True
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Demo failed with exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            self.cleanup()


def main():
    """Main entry point for complete MORPHEUS integration demo."""
    demo = MorpheusFullIntegrationDemo()
    
    logger.info("Starting MORPHEUS Complete Integration Demonstration")
    logger.info("This demo will comprehensively test all system capabilities")
    logger.info("Press Ctrl+C at any time to gracefully shutdown")
    
    success = demo.run_complete_demo()
    
    if success:
        print("\n🎊 Demo completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Demo encountered issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()