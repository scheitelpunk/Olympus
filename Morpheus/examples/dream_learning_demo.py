#!/usr/bin/env python3
"""
Dream Learning Demo for MORPHEUS System
Demonstrates the dream-based learning and optimization capabilities.

This example shows:
1. Experience collection from multiple scenarios
2. Dream session execution with strategy discovery
3. Strategy evaluation and ranking
4. Performance improvement over time
5. Knowledge consolidation
"""

import sys
import time
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add MORPHEUS to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morpheus.core.orchestrator import MorpheusOrchestrator
from morpheus.utils.demo_helpers import create_synthetic_observation, print_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DreamLearningDemo:
    """
    Demonstrates MORPHEUS dream-based learning capabilities.
    """
    
    def __init__(self):
        """Initialize demo."""
        self.morpheus = None
        self.experience_history = []
        self.performance_metrics = defaultdict(list)
        
        # Learning scenarios
        self.learning_scenarios = [
            {
                'name': 'Grip Optimization',
                'materials': ['steel', 'rubber', 'plastic'],
                'action_type': 'grip',
                'force_range': [3, 15],
                'success_factors': {'min_force': 5, 'material_bonus': {'rubber': 0.3}}
            },
            {
                'name': 'Gentle Touch',
                'materials': ['glass', 'plastic'],
                'action_type': 'touch',
                'force_range': [0.5, 3],
                'success_factors': {'max_force': 2, 'material_penalty': {'glass': 0.2}}
            },
            {
                'name': 'Impact Response',
                'materials': ['steel', 'aluminum', 'plastic'],
                'action_type': 'tap',
                'force_range': [1, 8],
                'success_factors': {'optimal_force': 4, 'tolerance': 2}
            }
        ]
        
    def setup_system(self):
        """Initialize MORPHEUS system."""
        print("=== Initializing MORPHEUS for Dream Learning ===")
        
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
            
            print(f"✓ MORPHEUS initialized for learning")
            print(f"✓ Session ID: {self.morpheus.session_id}")
            
        except Exception as e:
            print(f"✗ Failed to initialize MORPHEUS: {e}")
            sys.exit(1)
            
    def collect_initial_experiences(self, num_experiences: int = 150):
        """Collect initial set of experiences before learning."""
        print(f"\n=== Collecting {num_experiences} Initial Experiences ===")
        
        experiences_per_scenario = num_experiences // len(self.learning_scenarios)
        
        for scenario in self.learning_scenarios:
            print(f"\nCollecting experiences for: {scenario['name']}")
            
            scenario_experiences = []
            for i in range(experiences_per_scenario):
                # Generate random experience within scenario
                experience = self._generate_scenario_experience(scenario)
                
                # Process through MORPHEUS
                result = self.morpheus.perceive(experience)
                
                # Track experience
                experience_record = {
                    'scenario': scenario['name'],
                    'experience': experience,
                    'result': result,
                    'timestamp': time.time()
                }
                scenario_experiences.append(experience_record)
                self.experience_history.append(experience_record)
                
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i + 1}/{experiences_per_scenario}")
                    
            # Calculate baseline performance for scenario
            baseline_performance = self._calculate_scenario_performance(scenario_experiences)
            self.performance_metrics[scenario['name']].append({
                'phase': 'baseline',
                'performance': baseline_performance,
                'timestamp': time.time()
            })
            
            print(f"  Baseline performance: {baseline_performance:.1%}")
            
        print(f"\n✓ Collected {len(self.experience_history)} total experiences")
        
    def execute_dream_session(self, duration: float = 45):
        """Execute a dream session and analyze results."""
        print(f"\n=== Executing Dream Session ({duration}s) ===")
        
        # Get pre-dream strategy count
        strategies_before = self.morpheus.get_learned_strategies()
        print(f"Strategies before dream: {len(strategies_before)}")
        
        # Execute dream
        print("Entering dream state for strategy discovery...")
        start_time = time.time()
        
        dream_result = self.morpheus.dream(duration=duration)
        
        elapsed = time.time() - start_time
        print(f"\nDream session completed in {elapsed:.1f}s")
        
        # Analyze dream results
        self._analyze_dream_results(dream_result)
        
        # Get post-dream strategies
        strategies_after = self.morpheus.get_learned_strategies()
        print(f"Strategies after dream: {len(strategies_after)}")
        
        # Show new strategies discovered
        if len(strategies_after) > len(strategies_before):
            print(f"\n🧠 {len(strategies_after) - len(strategies_before)} new strategies discovered:")
            
            for i, strategy in enumerate(strategies_after[-5:], 1):  # Show last 5
                print(f"  {i}. {strategy['name']} (Category: {strategy['category']})")
                print(f"     Improvement: {strategy['improvement_ratio']:.1%}")
                print(f"     Confidence: {strategy['confidence']:.2f}")
                print(f"     Materials: {', '.join(strategy['applicable_materials'])}")
                
        return dream_result
        
    def test_learned_strategies(self, num_tests: int = 50):
        """Test performance after dream learning."""
        print(f"\n=== Testing Learned Strategies ({num_tests} tests) ===")
        
        # Get learned strategies
        strategies = self.morpheus.get_learned_strategies()
        
        if not strategies:
            print("No strategies learned yet")
            return
            
        # Test each scenario with learned strategies
        for scenario in self.learning_scenarios:
            print(f"\nTesting scenario: {scenario['name']}")
            
            # Find applicable strategies
            applicable_strategies = [
                s for s in strategies 
                if any(mat in s['applicable_materials'] for mat in scenario['materials'])
                or scenario['action_type'] in str(s.get('applicable_scenarios', []))
            ]
            
            print(f"  Applicable strategies: {len(applicable_strategies)}")
            
            # Generate test experiences
            test_experiences = []
            for i in range(num_tests // len(self.learning_scenarios)):
                experience = self._generate_scenario_experience(scenario, apply_strategies=True)
                result = self.morpheus.perceive(experience)
                
                test_record = {
                    'scenario': scenario['name'],
                    'experience': experience,
                    'result': result,
                    'timestamp': time.time()
                }
                test_experiences.append(test_record)
                
            # Calculate post-learning performance
            post_learning_performance = self._calculate_scenario_performance(test_experiences)
            
            # Record performance
            self.performance_metrics[scenario['name']].append({
                'phase': 'post_learning',
                'performance': post_learning_performance,
                'timestamp': time.time()
            })
            
            # Compare with baseline
            baseline_perf = self.performance_metrics[scenario['name']][0]['performance']
            improvement = post_learning_performance - baseline_perf
            
            print(f"  Baseline: {baseline_perf:.1%}")
            print(f"  Post-learning: {post_learning_performance:.1%}")
            print(f"  Improvement: {improvement:+.1%}")
            
    def demonstrate_iterative_learning(self, iterations: int = 3):
        """Demonstrate iterative learning through multiple dream cycles."""
        print(f"\n=== Iterative Learning Demonstration ({iterations} cycles) ===")
        
        for cycle in range(1, iterations + 1):
            print(f"\n--- Learning Cycle {cycle} ---")
            
            # Collect more experiences
            print("Collecting additional experiences...")
            self.collect_initial_experiences(num_experiences=75)
            
            # Dream session
            print(f"Dream cycle {cycle}...")
            dream_result = self.execute_dream_session(duration=30)
            
            # Test improvements
            print(f"Testing cycle {cycle} improvements...")
            self.test_learned_strategies(num_tests=30)
            
            # Brief analysis
            self._analyze_learning_progress(cycle)
            
            time.sleep(1)  # Brief pause between cycles
            
    def analyze_material_learning(self):
        """Analyze learning patterns across different materials."""
        print("\n=== Material Learning Analysis ===")
        
        strategies = self.morpheus.get_learned_strategies()
        
        # Group strategies by material
        material_strategies = defaultdict(list)
        for strategy in strategies:
            for material in strategy.get('applicable_materials', []):
                material_strategies[material].append(strategy)
                
        print("Material-specific learning:")
        print(f"{'Material':<10} {'Strategies':<10} {'Avg Improvement':<15} {'Best Strategy'}")
        print("-" * 65)
        
        for material, mat_strategies in material_strategies.items():
            if not mat_strategies:
                continue
                
            avg_improvement = np.mean([s['improvement_ratio'] for s in mat_strategies])
            best_strategy = max(mat_strategies, key=lambda x: x['improvement_ratio'])
            
            print(f"{material:<10} {len(mat_strategies):<10} "
                  f"{avg_improvement:<15.1%} {best_strategy['name']}")
                  
    def demonstrate_strategy_evolution(self):
        """Show how strategies evolve over time."""
        print("\n=== Strategy Evolution Analysis ===")
        
        strategies = self.morpheus.get_learned_strategies()
        
        # Group strategies by category
        categories = defaultdict(list)
        for strategy in strategies:
            categories[strategy.get('category', 'unknown')].append(strategy)
            
        print("Strategy categories discovered:")
        for category, cat_strategies in categories.items():
            print(f"\n{category.upper()} Strategies ({len(cat_strategies)}):")
            
            # Sort by confidence
            cat_strategies.sort(key=lambda x: x['confidence'], reverse=True)
            
            for i, strategy in enumerate(cat_strategies[:3], 1):  # Top 3
                print(f"  {i}. {strategy['name']}")
                print(f"     Confidence: {strategy['confidence']:.2f}")
                print(f"     Used {strategy.get('times_used', 0)} times")
                print(f"     Success rate: {strategy.get('success_rate', 0):.1%}")
                
    def generate_learning_report(self):
        """Generate comprehensive learning report."""
        print("\n=== Learning Performance Report ===")
        
        # Overall statistics
        total_experiences = len(self.experience_history)
        strategies = self.morpheus.get_learned_strategies()
        
        print(f"Total Experiences Processed: {total_experiences}")
        print(f"Total Strategies Learned: {len(strategies)}")
        
        if strategies:
            avg_confidence = np.mean([s['confidence'] for s in strategies])
            avg_improvement = np.mean([s['improvement_ratio'] for s in strategies])
            print(f"Average Strategy Confidence: {avg_confidence:.2f}")
            print(f"Average Performance Improvement: {avg_improvement:.1%}")
            
        # Performance by scenario
        print("\nPerformance by Scenario:")
        print(f"{'Scenario':<20} {'Baseline':<10} {'Current':<10} {'Improvement'}")
        print("-" * 55)
        
        for scenario_name, metrics in self.performance_metrics.items():
            if len(metrics) >= 2:
                baseline = metrics[0]['performance']
                current = metrics[-1]['performance'] 
                improvement = current - baseline
                
                print(f"{scenario_name:<20} {baseline:<10.1%} {current:<10.1%} {improvement:+.1%}")
                
        # Top strategies
        if strategies:
            print(f"\nTop 5 Strategies:")
            top_strategies = sorted(strategies, key=lambda x: x['improvement_ratio'], reverse=True)[:5]
            
            for i, strategy in enumerate(top_strategies, 1):
                print(f"  {i}. {strategy['name']} ({strategy['improvement_ratio']:+.1%})")
                
    def cleanup(self):
        """Clean up resources."""
        if self.morpheus:
            self.morpheus.cleanup()
            print("\n✓ System cleanup completed")
            
    def _generate_scenario_experience(self, scenario: Dict, apply_strategies: bool = False) -> Dict[str, Any]:
        """Generate experience based on scenario parameters."""
        material = random.choice(scenario['materials'])
        force = random.uniform(*scenario['force_range'])
        action_type = scenario['action_type']
        
        # Calculate success based on scenario rules
        success = self._calculate_scenario_success(scenario, material, force)
        
        # Apply learned strategies if requested
        if apply_strategies:
            force = self._apply_learned_force_optimization(material, action_type, force)
            
        return create_synthetic_observation(
            material=material,
            action_type=action_type,
            contact_force=force,
            robot_position=np.random.uniform(-0.1, 0.1, 3) + [0, 0, 0.5],
            success=success
        )
        
    def _calculate_scenario_success(self, scenario: Dict, material: str, force: float) -> bool:
        """Calculate success probability for scenario."""
        success_factors = scenario['success_factors']
        
        # Base success probability
        success_prob = 0.5
        
        # Apply scenario-specific rules
        if 'min_force' in success_factors:
            if force >= success_factors['min_force']:
                success_prob += 0.3
            else:
                success_prob -= 0.4
                
        if 'max_force' in success_factors:
            if force <= success_factors['max_force']:
                success_prob += 0.3
            else:
                success_prob -= 0.4
                
        if 'optimal_force' in success_factors:
            optimal = success_factors['optimal_force']
            tolerance = success_factors.get('tolerance', 1)
            distance = abs(force - optimal)
            if distance <= tolerance:
                success_prob += 0.4 * (1 - distance / tolerance)
            else:
                success_prob -= 0.3
                
        # Material-specific bonuses/penalties
        if 'material_bonus' in success_factors:
            success_prob += success_factors['material_bonus'].get(material, 0)
            
        if 'material_penalty' in success_factors:
            success_prob -= success_factors['material_penalty'].get(material, 0)
            
        # Add some randomness
        success_prob += random.uniform(-0.1, 0.1)
        
        return random.random() < max(0, min(1, success_prob))
        
    def _calculate_scenario_performance(self, experiences: List[Dict]) -> float:
        """Calculate performance metrics for a set of experiences."""
        if not experiences:
            return 0.0
            
        success_count = sum(1 for exp in experiences if exp['experience'].get('success', False))
        return success_count / len(experiences)
        
    def _apply_learned_force_optimization(self, material: str, action_type: str, original_force: float) -> float:
        """Apply learned strategies to optimize force."""
        strategies = self.morpheus.get_learned_strategies()
        
        # Find applicable strategies
        applicable = [
            s for s in strategies
            if material in s.get('applicable_materials', [])
            and action_type in str(s.get('applicable_scenarios', []))
        ]
        
        if not applicable:
            return original_force
            
        # Apply best strategy
        best_strategy = max(applicable, key=lambda x: x['confidence'])
        
        # Simple force modification based on strategy data
        strategy_data = best_strategy.get('strategy_data', {})
        if 'changes' in strategy_data and 'forces' in strategy_data['changes']:
            force_factor = strategy_data['changes']['forces'].get('factor', 1.0)
            return original_force * force_factor
            
        return original_force
        
    def _analyze_dream_results(self, dream_result: Dict):
        """Analyze and display dream session results."""
        print(f"\nDream Analysis:")
        print(f"  Experiences processed: {dream_result.get('experiences_processed', 0)}")
        print(f"  Strategies discovered: {dream_result.get('strategies_found', 0)}")
        print(f"  Strategies stored: {dream_result.get('strategies_stored', 0)}")
        print(f"  Processing time: {dream_result.get('time_elapsed', 0):.1f}s")
        
    def _analyze_learning_progress(self, cycle: int):
        """Analyze learning progress for current cycle."""
        print(f"\nCycle {cycle} Summary:")
        
        # Calculate overall improvement
        total_improvement = 0
        scenarios_with_data = 0
        
        for scenario_name, metrics in self.performance_metrics.items():
            if len(metrics) >= 2:
                baseline = metrics[0]['performance']
                current = metrics[-1]['performance']
                improvement = current - baseline
                total_improvement += improvement
                scenarios_with_data += 1
                
        if scenarios_with_data > 0:
            avg_improvement = total_improvement / scenarios_with_data
            print(f"  Average improvement: {avg_improvement:+.1%}")
        else:
            print("  No improvement data available yet")
            

def main():
    """Main demo function."""
    print("MORPHEUS Dream Learning Demonstration")
    print("=" * 50)
    
    demo = DreamLearningDemo()
    
    try:
        # Setup
        demo.setup_system()
        
        # Phase 1: Initial experience collection
        demo.collect_initial_experiences(num_experiences=100)
        
        # Phase 2: First dream session
        demo.execute_dream_session(duration=60)
        
        # Phase 3: Test learned strategies
        demo.test_learned_strategies(num_tests=60)
        
        # Phase 4: Iterative learning
        demo.demonstrate_iterative_learning(iterations=2)
        
        # Phase 5: Analysis
        demo.analyze_material_learning()
        demo.demonstrate_strategy_evolution()
        demo.generate_learning_report()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        logger.exception("Dream learning demo failed")
    finally:
        # Always cleanup
        demo.cleanup()
        
    print("\nDream learning demo completed! 🧠✨")


if __name__ == "__main__":
    main()