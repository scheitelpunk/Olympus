#!/usr/bin/env python3
"""Dream cycle demonstration for MORPHEUS system.

This example demonstrates the complete dream cycle workflow including:

- Experience collection from varied scenarios
- Dream state initiation and optimization
- Strategy discovery and learning
- Memory consolidation and knowledge extraction
- Performance improvement tracking

Usage:
    python -m morpheus.examples.dream_cycle_demo
    
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
from typing import Dict, List, Any, Optional
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
class DreamSessionMetrics:
    """Metrics for a dream session."""
    session_id: str
    duration: float
    experiences_processed: int
    strategies_discovered: int
    strategies_stored: int
    average_improvement: float
    best_improvement: float
    success_rate: float


@dataclass
class ExperienceScenario:
    """Defines a scenario for experience generation."""
    name: str
    materials: List[str]
    actions: List[str]
    force_range: tuple
    success_probability: float
    reward_range: tuple
    complexity: str  # 'simple', 'moderate', 'complex'


class DreamCycleDemo:
    """Demonstrates MORPHEUS dream cycle for experience replay and optimization."""
    
    def __init__(self, config_path: str = None, gasm_path: str = None):
        """Initialize dream cycle demo.
        
        Args:
            config_path: Optional path to MORPHEUS config file
            gasm_path: Optional path to GASM-Robotics directory
        """
        self.config_path = config_path or self._find_default_config()
        self.gasm_path = gasm_path or self._find_gasm_path()
        self.morpheus = None
        self.experience_scenarios = self._define_scenarios()
        self.dream_history = []
        
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
    
    def _define_scenarios(self) -> List[ExperienceScenario]:
        """Define diverse scenarios for experience generation."""
        return [
            ExperienceScenario(
                name="Basic Manipulation",
                materials=['steel', 'plastic', 'rubber'],
                actions=['grasp', 'lift', 'place'],
                force_range=(1.0, 8.0),
                success_probability=0.8,
                reward_range=(-0.2, 1.0),
                complexity='simple'
            ),
            ExperienceScenario(
                name="Material Interaction",
                materials=['steel', 'rubber', 'glass', 'wood'],
                actions=['touch', 'push', 'slide', 'rotate'],
                force_range=(0.5, 12.0),
                success_probability=0.7,
                reward_range=(-0.5, 1.5),
                complexity='moderate'
            ),
            ExperienceScenario(
                name="Complex Assembly",
                materials=['aluminum', 'plastic', 'rubber', 'steel'],
                actions=['grasp', 'rotate', 'slide', 'push'],
                force_range=(2.0, 15.0),
                success_probability=0.6,
                reward_range=(-1.0, 2.0),
                complexity='complex'
            ),
            ExperienceScenario(
                name="Delicate Handling",
                materials=['glass', 'plastic'],
                actions=['touch', 'grasp', 'lift'],
                force_range=(0.1, 3.0),
                success_probability=0.5,
                reward_range=(-2.0, 1.0),
                complexity='complex'
            ),
            ExperienceScenario(
                name="Heavy Duty Operations",
                materials=['steel', 'aluminum', 'wood'],
                actions=['push', 'lift', 'slide'],
                force_range=(5.0, 20.0),
                success_probability=0.75,
                reward_range=(-0.3, 1.2),
                complexity='moderate'
            )
        ]
    
    def initialize_system(self) -> bool:
        """Initialize MORPHEUS system for dream cycle testing.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("=== Initializing MORPHEUS for Dream Cycle Testing ===")
            
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
            
            logger.info("MORPHEUS system initialized for dream cycle testing!")
            
            # Register dream callbacks
            self.morpheus.register_callback('dream_start', self._on_dream_start)
            self.morpheus.register_callback('dream_end', self._on_dream_end)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MORPHEUS: {e}")
            return False
    
    def _on_dream_start(self, dream_id: str, duration: float):
        """Callback for dream session start."""
        logger.info(f"🌙 Dream session started: {dream_id} (duration: {duration}s)")
    
    def _on_dream_end(self, dream_id: str, results: Dict[str, Any]):
        """Callback for dream session end."""
        strategies = results.get('strategies_found', 0)
        logger.info(f"🌅 Dream session ended: {dream_id} ({strategies} strategies discovered)")
    
    def collect_diverse_experiences(self, 
                                  num_experiences: int = 150,
                                  scenario_distribution: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Collect diverse experiences across multiple scenarios.
        
        Args:
            num_experiences: Total number of experiences to collect
            scenario_distribution: Optional distribution weights for scenarios
            
        Returns:
            Experience collection summary
        """
        if scenario_distribution is None:
            # Default even distribution
            scenario_distribution = {scenario.name: 1.0 for scenario in self.experience_scenarios}
        
        logger.info(f"=== Collecting {num_experiences} Diverse Experiences ===")
        
        # Normalize distribution weights
        total_weight = sum(scenario_distribution.values())
        normalized_dist = {k: v/total_weight for k, v in scenario_distribution.items()}
        
        # Calculate experiences per scenario
        scenario_counts = {}
        remaining = num_experiences
        
        for i, (scenario_name, weight) in enumerate(normalized_dist.items()):
            if i == len(normalized_dist) - 1:  # Last scenario gets remainder
                scenario_counts[scenario_name] = remaining
            else:
                count = int(num_experiences * weight)
                scenario_counts[scenario_name] = count
                remaining -= count
        
        logger.info(f"Experience distribution: {scenario_counts}")
        
        collection_start = time.time()
        collected_experiences = []
        scenario_stats = defaultdict(lambda: {'success': 0, 'total': 0, 'avg_reward': 0})
        
        for scenario in self.experience_scenarios:
            if scenario.name not in scenario_counts:
                continue
                
            count = scenario_counts[scenario.name]
            logger.info(f"\n--- Collecting {count} experiences for '{scenario.name}' ---")
            
            for i in range(count):
                try:
                    experience = self._generate_experience(scenario)
                    result = self.morpheus.perceive(experience)
                    
                    # Update statistics
                    stats = scenario_stats[scenario.name]
                    stats['total'] += 1
                    if experience.get('success'):
                        stats['success'] += 1
                    
                    reward = experience.get('reward', 0)
                    stats['avg_reward'] = (stats['avg_reward'] * (stats['total'] - 1) + reward) / stats['total']
                    
                    collected_experiences.append({
                        'scenario': scenario.name,
                        'experience': experience,
                        'result': result
                    })
                    
                    if (i + 1) % 20 == 0:
                        logger.info(f"  Collected {i + 1}/{count} experiences")
                    
                except Exception as e:
                    logger.warning(f"Failed to collect experience {i+1} for {scenario.name}: {e}")
        
        collection_time = time.time() - collection_start
        
        # Calculate overall statistics
        total_collected = len(collected_experiences)
        overall_success = sum(stats['success'] for stats in scenario_stats.values())
        overall_success_rate = overall_success / total_collected if total_collected > 0 else 0
        
        logger.info(f"\nExperience collection completed in {collection_time:.1f}s")
        logger.info(f"Total experiences collected: {total_collected}")
        logger.info(f"Overall success rate: {overall_success_rate:.1%}")
        
        # Display per-scenario statistics
        for scenario_name, stats in scenario_stats.items():
            success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            logger.info(f"  {scenario_name}: {stats['total']} exp, {success_rate:.1%} success, {stats['avg_reward']:.2f} avg reward")
        
        return {
            'total_experiences': total_collected,
            'collection_time': collection_time,
            'scenario_statistics': dict(scenario_stats),
            'overall_success_rate': overall_success_rate,
            'experiences': collected_experiences
        }
    
    def _generate_experience(self, scenario: ExperienceScenario) -> Dict[str, Any]:
        """Generate a single experience based on scenario parameters.
        
        Args:
            scenario: Scenario definition
            
        Returns:
            Generated experience dictionary
        """
        # Select random parameters within scenario constraints
        material = random.choice(scenario.materials)
        action = random.choice(scenario.actions)
        force = random.uniform(*scenario.force_range)
        success = random.random() < scenario.success_probability
        reward = random.uniform(*scenario.reward_range) if success else random.uniform(-1.0, 0.0)
        
        # Add scenario-specific complexity
        if scenario.complexity == 'complex':
            # Add more parameters and noise for complex scenarios
            velocity = random.uniform(0.01, 0.3)
            duration = random.uniform(0.5, 3.0)
            position_noise = 0.2
        elif scenario.complexity == 'moderate':
            velocity = random.uniform(0.05, 0.2)
            duration = random.uniform(0.8, 2.0)
            position_noise = 0.1
        else:  # simple
            velocity = random.uniform(0.08, 0.15)
            duration = random.uniform(1.0, 1.5)
            position_noise = 0.05
        
        return {
            'material': material,
            'body_id': random.randint(1, 5),  # Multiple objects
            'robot_position': np.random.randn(3) * position_noise,
            'robot_velocity': [velocity * random.choice([-1, 1]), 0, 0],
            'contact_force': force,
            'forces': [force, random.uniform(-force*0.2, force*0.2), random.uniform(-force*0.1, force*0.1)],
            'torques': [random.uniform(-1, 1), random.uniform(-1, 1), force * random.uniform(-0.1, 0.1)],
            'action_type': action,
            'action_params': {
                'pressure': force,
                'duration': duration,
                'approach_speed': velocity,
                'target_material': material,
                'scenario_complexity': scenario.complexity
            },
            'success': success,
            'reward': reward,
            'tags': [f'scenario_{scenario.name.lower().replace(" ", "_")}', f'complexity_{scenario.complexity}', f'material_{material}'],
            'notes': f'Generated experience for {scenario.name} scenario'
        }
    
    def run_dream_session(self, 
                         duration: float = 45.0,
                         custom_config: Optional[Dict[str, Any]] = None) -> DreamSessionMetrics:
        """Run a single dream session with comprehensive monitoring.
        
        Args:
            duration: Dream duration in seconds
            custom_config: Optional custom dream configuration
            
        Returns:
            Dream session metrics
        """
        logger.info(f"=== Running Dream Session (Duration: {duration}s) ===")
        
        # Get baseline state
        pre_dream_summary = self.morpheus.get_session_summary()
        pre_dream_strategies = self.morpheus.get_learned_strategies()
        
        logger.info(f"Pre-dream state:")
        logger.info(f"  Perceptions: {pre_dream_summary['perception_count']}")
        logger.info(f"  Strategies: {len(pre_dream_strategies)}")
        logger.info(f"  Success Rate: {pre_dream_summary['success_rate']:.1%}")
        
        # Run dream session
        dream_start = time.time()
        
        try:
            dream_results = self.morpheus.dream(duration=duration)
            actual_duration = time.time() - dream_start
            
            # Get post-dream state
            post_dream_summary = self.morpheus.get_session_summary()
            post_dream_strategies = self.morpheus.get_learned_strategies()
            
            # Calculate metrics
            metrics = DreamSessionMetrics(
                session_id=dream_results['dream_id'],
                duration=actual_duration,
                experiences_processed=dream_results.get('experiences_processed', 0),
                strategies_discovered=dream_results.get('strategies_found', 0),
                strategies_stored=dream_results.get('strategies_stored', 0),
                average_improvement=0.0,  # Will be calculated from strategies
                best_improvement=0.0,
                success_rate=post_dream_summary['success_rate']
            )
            
            # Calculate improvement metrics from new strategies
            new_strategies = post_dream_strategies[len(pre_dream_strategies):]
            if new_strategies:
                improvements = [s.get('improvement_ratio', 0) for s in new_strategies]
                metrics.average_improvement = np.mean(improvements)
                metrics.best_improvement = np.max(improvements)
            
            self.dream_history.append(metrics)
            
            logger.info(f"Dream session completed!")
            logger.info(f"  Actual Duration: {actual_duration:.1f}s")
            logger.info(f"  Experiences Processed: {metrics.experiences_processed}")
            logger.info(f"  Strategies Discovered: {metrics.strategies_discovered}")
            logger.info(f"  Strategies Stored: {metrics.strategies_stored}")
            logger.info(f"  Average Improvement: {metrics.average_improvement:.2%}")
            logger.info(f"  Best Improvement: {metrics.best_improvement:.2%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Dream session failed: {e}")
            # Return failed metrics
            return DreamSessionMetrics(
                session_id=f"failed_{time.time():.0f}",
                duration=time.time() - dream_start,
                experiences_processed=0,
                strategies_discovered=0,
                strategies_stored=0,
                average_improvement=0.0,
                best_improvement=0.0,
                success_rate=pre_dream_summary['success_rate']
            )
    
    def run_multiple_dream_cycles(self, 
                                 num_cycles: int = 3,
                                 dream_duration: float = 30.0,
                                 experience_batches: int = 50) -> List[DreamSessionMetrics]:
        """Run multiple dream cycles with experience collection between them.
        
        Args:
            num_cycles: Number of dream cycles to run
            dream_duration: Duration of each dream session
            experience_batches: Experiences to collect between cycles
            
        Returns:
            List of dream session metrics
        """
        logger.info(f"=== Running {num_cycles} Dream Cycles ===")
        logger.info(f"Dream duration: {dream_duration}s each")
        logger.info(f"Experience batches: {experience_batches} between cycles")
        
        cycle_results = []
        
        for cycle in range(num_cycles):
            logger.info(f"\n{'='*50}")
            logger.info(f"DREAM CYCLE {cycle + 1}/{num_cycles}")
            logger.info(f"{'='*50}")
            
            # Collect more experiences between cycles (except first)
            if cycle > 0:
                logger.info(f"\n--- Collecting Additional Experiences ---")
                self.collect_diverse_experiences(experience_batches)
                
                # Wait a bit for consolidation
                time.sleep(2)
            
            # Run dream session
            logger.info(f"\n--- Dream Session {cycle + 1} ---")
            metrics = self.run_dream_session(dream_duration)
            cycle_results.append(metrics)
            
            # Analyze progress
            if cycle > 0:
                self._analyze_cycle_progress(cycle_results)
        
        return cycle_results
    
    def _analyze_cycle_progress(self, cycle_results: List[DreamSessionMetrics]):
        """Analyze progress across dream cycles."""
        logger.info("\n--- Dream Cycle Progress Analysis ---")
        
        if len(cycle_results) < 2:
            return
        
        # Calculate trends
        strategies_trend = [m.strategies_discovered for m in cycle_results]
        improvement_trend = [m.average_improvement for m in cycle_results]
        success_rate_trend = [m.success_rate for m in cycle_results]
        
        logger.info(f"Strategies discovered per cycle: {strategies_trend}")
        logger.info(f"Average improvement per cycle: {[f'{x:.2%}' for x in improvement_trend]}")
        logger.info(f"Success rate per cycle: {[f'{x:.1%}' for x in success_rate_trend]}")
        
        # Calculate overall trends
        if len(strategies_trend) > 1:
            strategy_growth = strategies_trend[-1] - strategies_trend[0]
            improvement_growth = improvement_trend[-1] - improvement_trend[0]
            success_growth = success_rate_trend[-1] - success_rate_trend[0]
            
            logger.info(f"\nOverall Progress:")
            logger.info(f"  Strategy Discovery: {strategy_growth:+d} strategies")
            logger.info(f"  Improvement Growth: {improvement_growth:+.2%}")
            logger.info(f"  Success Rate Change: {success_growth:+.1%}")
    
    def analyze_learned_strategies(self) -> Dict[str, Any]:
        """Analyze all learned strategies and their effectiveness.
        
        Returns:
            Strategy analysis results
        """
        logger.info("\n=== Analyzing Learned Strategies ===")
        
        try:
            # Get all strategies
            all_strategies = self.morpheus.get_learned_strategies(limit=100)
            
            if not all_strategies:
                logger.warning("No learned strategies found")
                return {'strategies': 0, 'analysis': 'No strategies to analyze'}
            
            # Categorize strategies
            category_stats = defaultdict(lambda: {'count': 0, 'avg_improvement': 0, 'avg_confidence': 0})
            material_strategies = defaultdict(list)
            
            for strategy in all_strategies:
                category = strategy.get('category', 'unknown')
                improvement = strategy.get('improvement_ratio', 0)
                confidence = strategy.get('confidence', 0)
                
                stats = category_stats[category]
                stats['count'] += 1
                stats['avg_improvement'] = (stats['avg_improvement'] * (stats['count'] - 1) + improvement) / stats['count']
                stats['avg_confidence'] = (stats['avg_confidence'] * (stats['count'] - 1) + confidence) / stats['count']
                
                # Group by applicable materials
                for material in strategy.get('applicable_materials', []):
                    material_strategies[material].append(strategy)
            
            # Find top strategies
            top_strategies = sorted(all_strategies, key=lambda x: x.get('improvement_ratio', 0), reverse=True)[:10]
            
            # Generate analysis report
            analysis = {
                'total_strategies': len(all_strategies),
                'categories': dict(category_stats),
                'material_coverage': {material: len(strategies) for material, strategies in material_strategies.items()},
                'top_strategies': top_strategies,
                'average_improvement': np.mean([s.get('improvement_ratio', 0) for s in all_strategies]),
                'average_confidence': np.mean([s.get('confidence', 0) for s in all_strategies])
            }
            
            self._display_strategy_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Strategy analysis failed: {e}")
            return {'error': str(e)}
    
    def _display_strategy_analysis(self, analysis: Dict[str, Any]):
        """Display formatted strategy analysis."""
        print(f"\n{'='*60}")
        print(f"STRATEGY ANALYSIS REPORT")
        print(f"{'='*60}")
        
        print(f"\nOverall Statistics:")
        print(f"  Total Strategies: {analysis['total_strategies']}")
        print(f"  Average Improvement: {analysis['average_improvement']:.2%}")
        print(f"  Average Confidence: {analysis['average_confidence']:.2f}")
        
        print(f"\nStrategies by Category:")
        for category, stats in analysis['categories'].items():
            print(f"  {category.title()}:")
            print(f"    Count: {stats['count']}")
            print(f"    Avg Improvement: {stats['avg_improvement']:.2%}")
            print(f"    Avg Confidence: {stats['avg_confidence']:.2f}")
        
        print(f"\nMaterial Coverage:")
        for material, count in analysis['material_coverage'].items():
            print(f"  {material}: {count} strategies")
        
        print(f"\nTop 5 Strategies:")
        for i, strategy in enumerate(analysis['top_strategies'][:5], 1):
            print(f"  {i}. {strategy.get('name', 'Unknown')}")
            print(f"     Category: {strategy.get('category', 'N/A')}")
            print(f"     Improvement: {strategy.get('improvement_ratio', 0):.2%}")
            print(f"     Confidence: {strategy.get('confidence', 0):.2f}")
            print(f"     Materials: {', '.join(strategy.get('applicable_materials', []))}")
    
    def generate_dream_cycle_report(self) -> Dict[str, Any]:
        """Generate comprehensive dream cycle report.
        
        Returns:
            Complete dream cycle analysis report
        """
        logger.info("\n=== Generating Dream Cycle Report ===")
        
        try:
            # Get current system state
            session_summary = self.morpheus.get_session_summary()
            strategy_analysis = self.analyze_learned_strategies()
            
            # Analyze dream history
            if self.dream_history:
                total_dream_time = sum(d.duration for d in self.dream_history)
                total_experiences_processed = sum(d.experiences_processed for d in self.dream_history)
                total_strategies_discovered = sum(d.strategies_discovered for d in self.dream_history)
                avg_strategies_per_dream = total_strategies_discovered / len(self.dream_history)
                
                dream_efficiency = total_strategies_discovered / total_dream_time if total_dream_time > 0 else 0
            else:
                total_dream_time = 0
                total_experiences_processed = 0
                total_strategies_discovered = 0
                avg_strategies_per_dream = 0
                dream_efficiency = 0
            
            report = {
                'session_summary': session_summary,
                'dream_cycles': len(self.dream_history),
                'total_dream_time': total_dream_time,
                'total_experiences_processed': total_experiences_processed,
                'total_strategies_discovered': total_strategies_discovered,
                'average_strategies_per_dream': avg_strategies_per_dream,
                'dream_efficiency': dream_efficiency,  # strategies per second
                'strategy_analysis': strategy_analysis,
                'dream_history': [
                    {
                        'session_id': d.session_id,
                        'duration': d.duration,
                        'strategies_discovered': d.strategies_discovered,
                        'average_improvement': d.average_improvement
                    }
                    for d in self.dream_history
                ],
                'generation_timestamp': time.time()
            }
            
            self._display_dream_cycle_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate dream cycle report: {e}")
            return {'error': str(e)}
    
    def _display_dream_cycle_report(self, report: Dict[str, Any]):
        """Display formatted dream cycle report."""
        print(f"\n{'='*80}")
        print(f"DREAM CYCLE COMPREHENSIVE REPORT")
        print(f"{'='*80}")
        
        # System overview
        session = report['session_summary']
        print(f"\nSystem Overview:")
        print(f"  Session ID: {session['session_id']}")
        print(f"  System Uptime: {session['system_uptime']:.1f}s")
        print(f"  Total Perceptions: {session['perception_count']}")
        print(f"  Total Experiences: {session['total_experiences']}")
        print(f"  Overall Success Rate: {session['success_rate']:.1%}")
        
        # Dream cycle summary
        print(f"\nDream Cycle Summary:")
        print(f"  Dream Cycles Completed: {report['dream_cycles']}")
        print(f"  Total Dream Time: {report['total_dream_time']:.1f}s")
        print(f"  Experiences Processed: {report['total_experiences_processed']}")
        print(f"  Strategies Discovered: {report['total_strategies_discovered']}")
        print(f"  Avg Strategies/Dream: {report['average_strategies_per_dream']:.1f}")
        print(f"  Dream Efficiency: {report['dream_efficiency']:.2f} strategies/second")
        
        # Dream history
        if report['dream_history']:
            print(f"\nDream Session History:")
            for i, dream in enumerate(report['dream_history'], 1):
                print(f"  Dream {i}: {dream['duration']:.1f}s, "
                      f"{dream['strategies_discovered']} strategies, "
                      f"{dream['average_improvement']:.2%} avg improvement")
        
        # Materials and strategies
        strategy_analysis = report.get('strategy_analysis', {})
        if strategy_analysis and 'material_coverage' in strategy_analysis:
            print(f"\nMaterial Strategy Coverage:")
            for material, count in strategy_analysis['material_coverage'].items():
                print(f"  {material}: {count} strategies")
    
    def cleanup(self):
        """Clean up resources."""
        if self.morpheus:
            logger.info("Cleaning up MORPHEUS system...")
            self.morpheus.cleanup()
    
    def run_demo(self) -> bool:
        """Run complete dream cycle demonstration.
        
        Returns:
            True if demo completed successfully
        """
        try:
            print("="*80)
            print("MORPHEUS Dream Cycle Demonstration")  
            print("="*80)
            
            # Initialize system
            if not self.initialize_system():
                return False
            
            # Phase 1: Initial experience collection
            print(f"\n{'='*60}")
            print("PHASE 1: Initial Experience Collection")
            print(f"{'='*60}")
            
            initial_collection = self.collect_diverse_experiences(100)
            
            # Get baseline metrics
            baseline_summary = self.morpheus.get_session_summary()
            baseline_strategies = len(self.morpheus.get_learned_strategies())
            
            print(f"\nBaseline State:")
            print(f"  Experiences: {baseline_summary['total_experiences']}")
            print(f"  Success Rate: {baseline_summary['success_rate']:.1%}")
            print(f"  Strategies: {baseline_strategies}")
            
            # Phase 2: Multiple dream cycles
            print(f"\n{'='*60}")
            print("PHASE 2: Dream Cycle Execution")
            print(f"{'='*60}")
            
            dream_results = self.run_multiple_dream_cycles(
                num_cycles=3,
                dream_duration=25.0,
                experience_batches=40
            )
            
            # Phase 3: Strategy analysis
            print(f"\n{'='*60}")
            print("PHASE 3: Strategy Analysis")
            print(f"{'='*60}")
            
            strategy_analysis = self.analyze_learned_strategies()
            
            # Phase 4: Final comprehensive report
            print(f"\n{'='*60}")
            print("PHASE 4: Comprehensive Analysis")
            print(f"{'='*60}")
            
            final_report = self.generate_dream_cycle_report()
            
            # Demo completion summary
            final_summary = self.morpheus.get_session_summary()
            final_strategies = len(self.morpheus.get_learned_strategies())
            
            print(f"\n{'='*80}")
            print("DEMO COMPLETION SUMMARY")
            print(f"{'='*80}")
            
            print(f"Initial State:")
            print(f"  Experiences: {baseline_summary['total_experiences']}")
            print(f"  Success Rate: {baseline_summary['success_rate']:.1%}")
            print(f"  Strategies: {baseline_strategies}")
            
            print(f"\nFinal State:")
            print(f"  Experiences: {final_summary['total_experiences']}")
            print(f"  Success Rate: {final_summary['success_rate']:.1%}")
            print(f"  Strategies: {final_strategies}")
            
            print(f"\nImprovement:")
            print(f"  Experience Growth: +{final_summary['total_experiences'] - baseline_summary['total_experiences']}")
            print(f"  Success Rate Change: {final_summary['success_rate'] - baseline_summary['success_rate']:+.1%}")
            print(f"  Strategies Learned: +{final_strategies - baseline_strategies}")
            
            print(f"\nDream Cycles Completed: {len(dream_results)}")
            print(f"Total Dream Time: {sum(d.duration for d in dream_results):.1f}s")
            print(f"Strategies per Dream: {sum(d.strategies_discovered for d in dream_results) / len(dream_results):.1f}")
            
            print(f"\nDemo completed successfully! 🎉")
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
    """Main entry point for dream cycle demo."""
    demo = DreamCycleDemo()
    
    success = demo.run_demo()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()