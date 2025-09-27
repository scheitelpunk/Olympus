#!/usr/bin/env python3
"""
Spatial Agent Demo Script

This script demonstrates the capabilities of the spatial agent system
with various example tasks.
"""

import sys
import time
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.spatial_agent import (
    SpatialAgent, 
    SimulationConfig, 
    SimulationMode
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_basic_placement():
    """Demo: Basic object placement"""
    print("\n" + "🎯 DEMO 1: Basic Object Placement")
    print("-" * 50)
    
    config = SimulationConfig(max_steps=200)
    agent = SpatialAgent(config, SimulationMode.HEADLESS)
    
    try:
        result = agent.execute_task("Place a red box on the conveyor belt")
        
        print(f"✓ Task: {result['task']}")
        print(f"✓ Success: {result['success']}")
        print(f"✓ Objects created: {result['entities']}")
        print(f"✓ Simulation steps: {result['steps']}")
        print(f"✓ Execution time: {result['execution_time']:.2f}s")
        
        metrics = result['final_metrics']
        print("\n📊 Final Metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.3f}")
            
    finally:
        agent.cleanup()
    
    return result


def demo_multi_object_arrangement():
    """Demo: Multiple object spatial arrangement"""
    print("\n" + "🎯 DEMO 2: Multi-Object Arrangement")
    print("-" * 50)
    
    config = SimulationConfig(max_steps=400)
    agent = SpatialAgent(config, SimulationMode.HEADLESS)
    
    try:
        result = agent.execute_task(
            "Arrange a red box, blue cube, and green sphere in a triangular pattern"
        )
        
        print(f"✓ Task: {result['task']}")
        print(f"✓ Success: {result['success']}")
        print(f"✓ Objects created: {result['entities']}")
        
        # Analyze spatial efficiency over time
        if result['step_results']:
            initial_metrics = result['step_results'][0]['metrics']
            final_metrics = result['step_results'][-1]['metrics']
            
            print("\n📈 Improvement Analysis:")
            print(f"   Initial constraint satisfaction: {initial_metrics.get('constraint_satisfaction', 0):.3f}")
            print(f"   Final constraint satisfaction: {final_metrics.get('constraint_satisfaction', 0):.3f}")
            print(f"   Spatial efficiency: {final_metrics.get('spatial_efficiency', 0):.3f}")
            
    finally:
        agent.cleanup()
    
    return result


def demo_physics_interaction():
    """Demo: Physics-based interactions"""
    print("\n" + "🎯 DEMO 3: Physics Interactions")
    print("-" * 50)
    
    config = SimulationConfig(
        max_steps=800,
        time_step=1./120.,  # Higher precision physics
        gravity=(0., 0., -9.81)
    )
    agent = SpatialAgent(config, SimulationMode.HEADLESS)
    
    try:
        result = agent.execute_task(
            "Stack three boxes vertically and observe them settle under gravity"
        )
        
        print(f"✓ Task: {result['task']}")
        print(f"✓ Success: {result['success']}")
        print(f"✓ Physics simulation steps: {result['steps']}")
        
        # Analyze collisions over time
        collision_steps = [step for step in result['step_results'] if len(step['collisions']) > 0]
        print(f"✓ Steps with collisions: {len(collision_steps)}")
        
        if collision_steps:
            total_collisions = sum(len(step['collisions']) for step in collision_steps)
            print(f"✓ Total collision events: {total_collisions}")
            
    finally:
        agent.cleanup()
    
    return result


def demo_constraint_satisfaction():
    """Demo: Constraint-based spatial reasoning"""
    print("\n" + "🎯 DEMO 4: Constraint Satisfaction")
    print("-" * 50)
    
    config = SimulationConfig(max_steps=300)
    agent = SpatialAgent(config, SimulationMode.HEADLESS)
    
    try:
        # Test different constraint types
        tasks = [
            "Place a box exactly 20cm away from a sphere",
            "Arrange three cubes in a straight line with equal spacing",
            "Position objects so they form a perfect square pattern"
        ]
        
        results = []
        for i, task in enumerate(tasks, 1):
            print(f"\n   Subtask {i}: {task}")
            result = agent.execute_task(task)
            results.append(result)
            
            constraint_satisfaction = result['final_metrics'].get('constraint_satisfaction', 0)
            print(f"   ✓ Constraint satisfaction: {constraint_satisfaction:.3f}")
            print(f"   ✓ Success: {result['success']}")
        
        # Overall performance
        avg_satisfaction = sum(r['final_metrics'].get('constraint_satisfaction', 0) for r in results) / len(results)
        success_rate = sum(1 for r in results if r['success']) / len(results)
        
        print(f"\n📊 Overall Performance:")
        print(f"   Average constraint satisfaction: {avg_satisfaction:.3f}")
        print(f"   Success rate: {success_rate:.1%}")
        
    finally:
        agent.cleanup()
    
    return results


def demo_vision_integration():
    """Demo: Vision pipeline integration"""
    print("\n" + "🎯 DEMO 5: Vision Integration")
    print("-" * 50)
    
    config = SimulationConfig(
        max_steps=200,
        width=320,  # Smaller for faster processing
        height=240
    )
    agent = SpatialAgent(config, SimulationMode.HEADLESS)
    
    try:
        result = agent.execute_task("Place objects and analyze the scene with computer vision")
        
        print(f"✓ Task: {result['task']}")
        print(f"✓ Success: {result['success']}")
        
        # Analyze vision processing results
        vision_steps = [step for step in result['step_results'] if step['vision']['objects']]
        print(f"✓ Steps with vision data: {len(vision_steps)}")
        
        if vision_steps:
            objects_detected = [len(step['vision']['objects']) for step in vision_steps]
            avg_objects = sum(objects_detected) / len(objects_detected)
            print(f"✓ Average objects detected per frame: {avg_objects:.1f}")
            
            # Show details from last vision step
            last_vision = vision_steps[-1]['vision']
            print(f"✓ Final vision analysis:")
            for i, obj in enumerate(last_vision['objects']):
                print(f"     Object {i}: area={obj['area']:.0f}px, centroid={obj['centroid']}")
        
    finally:
        agent.cleanup()
    
    return result


def demo_performance_analysis():
    """Demo: Performance analysis across multiple tasks"""
    print("\n" + "🎯 DEMO 6: Performance Analysis")
    print("-" * 50)
    
    config = SimulationConfig(max_steps=150)
    agent = SpatialAgent(config, SimulationMode.HEADLESS)
    
    try:
        # Execute a series of increasingly complex tasks
        task_series = [
            "Place a single box",
            "Arrange two objects side by side",
            "Create a triangle with three spheres", 
            "Stack four cubes in a pyramid",
            "Arrange five objects in a pentagon pattern"
        ]
        
        print("Executing task series...")
        for i, task in enumerate(task_series, 1):
            print(f"   Task {i}/{len(task_series)}: {task}")
            result = agent.execute_task(task)
            success_icon = "✓" if result['success'] else "✗"
            print(f"   {success_icon} Result: {result['final_metrics'].get('constraint_satisfaction', 0):.3f}")
        
        # Get comprehensive performance summary
        summary = agent.get_performance_summary()
        
        print(f"\n📊 Performance Summary:")
        print(f"   Total tasks executed: {summary.get('total_tasks', 0)}")
        print(f"   Overall success rate: {summary.get('success_rate', 0):.1%}")
        
        if 'constraint_satisfaction' in summary:
            cs = summary['constraint_satisfaction']
            print(f"   Constraint satisfaction: {cs['mean']:.3f} ± {cs['std']:.3f}")
            print(f"   Range: [{cs['min']:.3f}, {cs['max']:.3f}]")
        
        if 'spatial_efficiency' in summary:
            se = summary['spatial_efficiency']
            print(f"   Spatial efficiency: {se['mean']:.3f} ± {se['std']:.3f}")
        
    finally:
        agent.cleanup()
    
    return summary


def demo_error_handling():
    """Demo: Error handling and robustness"""
    print("\n" + "🎯 DEMO 7: Error Handling & Robustness")
    print("-" * 50)
    
    config = SimulationConfig(max_steps=100)
    agent = SpatialAgent(config, SimulationMode.HEADLESS)
    
    try:
        # Test edge cases and error conditions
        edge_cases = [
            "",  # Empty input
            "Nonsensical gibberish with no spatial meaning",  # No spatial content
            "Place 1000 objects in impossible configurations",  # Impossible task
            "   ",  # Whitespace only
            "Move the invisible quantum box to the 5th dimension",  # Nonsensical
        ]
        
        print("Testing edge cases and error handling...")
        
        for i, test_case in enumerate(edge_cases, 1):
            display_case = test_case if test_case.strip() else "<empty/whitespace>"
            print(f"   Test {i}: '{display_case[:40]}{'...' if len(display_case) > 40 else ''}'")
            
            try:
                result = agent.execute_task(test_case)
                status = "✓ Handled gracefully" if not result.get('error') else f"⚠ Error: {result['error'][:30]}..."
                print(f"   {status}")
            except Exception as e:
                print(f"   ⚠ Exception caught: {str(e)[:30]}...")
        
        print("\n✓ All edge cases handled without crashes")
        
    finally:
        agent.cleanup()
    
    return True


def run_all_demos():
    """Run all demonstration scenarios"""
    print("=" * 60)
    print("🚀 SPATIAL AGENT COMPREHENSIVE DEMO")
    print("=" * 60)
    print("This demo showcases the capabilities of the Spatial Agent")
    print("system with PyBullet physics simulation and GASM integration.")
    print("=" * 60)
    
    demos = [
        demo_basic_placement,
        demo_multi_object_arrangement, 
        demo_physics_interaction,
        demo_constraint_satisfaction,
        demo_vision_integration,
        demo_performance_analysis,
        demo_error_handling
    ]
    
    start_time = time.time()
    results = []
    
    for demo in demos:
        try:
            result = demo()
            results.append((demo.__name__, True, result))
            time.sleep(0.5)  # Brief pause between demos
        except Exception as e:
            logger.error(f"Demo {demo.__name__} failed: {e}")
            results.append((demo.__name__, False, str(e)))
    
    end_time = time.time()
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 DEMO SUMMARY")
    print("=" * 60)
    
    successful_demos = sum(1 for _, success, _ in results if success)
    total_demos = len(results)
    
    for name, success, result in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        demo_name = name.replace('demo_', '').replace('_', ' ').title()
        print(f"   {status}: {demo_name}")
    
    print(f"\nResults: {successful_demos}/{total_demos} demos passed ({successful_demos/total_demos*100:.1f}%)")
    print(f"Total execution time: {end_time - start_time:.1f} seconds")
    
    if successful_demos == total_demos:
        print("\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("The Spatial Agent system is ready for use.")
    else:
        print(f"\n⚠️  {total_demos - successful_demos} demo(s) encountered issues.")
        print("Check the logs above for details.")
    
    print("\n" + "=" * 60)
    return successful_demos == total_demos


if __name__ == "__main__":
    success = run_all_demos()
    sys.exit(0 if success else 1)