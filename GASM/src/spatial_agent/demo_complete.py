#!/usr/bin/env python3
"""
Complete demonstration of the 2D Spatial Agent capabilities
Showcases all features: constraint parsing, optimization, visualization, and video export
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from spatial_agent.agent_loop_2d import SpatialAgent2D

def demo_basic_constraints():
    """Demonstrate basic spatial constraints"""
    print("🎯 Demo 1: Basic Spatial Constraints")
    print("-" * 40)
    
    test_cases = [
        ("box above robot", "Vertical positioning"),
        ("robot left of box", "Horizontal positioning"),
        ("box near sensor", "Distance minimization"),
        ("robot far from box", "Distance maximization")
    ]
    
    for text, description in test_cases:
        print(f"\n📝 Task: {text}")
        print(f"📋 Description: {description}")
        
        agent = SpatialAgent2D(
            scene_width=10.0,
            scene_height=8.0,
            max_iterations=25,
            convergence_threshold=1e-3
        )
        
        results = agent.run(
            text_description=text,
            enable_visualization=False,
            save_video=False,
            seed=42
        )
        
        print(f"✅ Result: {'Success' if results['success'] else 'Partial'}")
        print(f"🔄 Iterations: {results['iterations']}")
        print(f"📊 Score: {results['final_score']:.4f}")
        print(f"⚡ Time: {results['execution_time']:.2f}s")
        
        print("📍 Final positions:")
        for entity, pos in results['entities'].items():
            print(f"  {entity}: ({pos[0]:.2f}, {pos[1]:.2f})")
        
        agent.close()
        time.sleep(1)  # Brief pause between demos


def demo_complex_constraints():
    """Demonstrate complex multi-entity constraints"""
    print("\n🎯 Demo 2: Complex Multi-Entity Constraints")
    print("-" * 40)
    
    complex_tasks = [
        ("box above robot and robot left of sensor", "Multi-constraint coordination"),
        ("box near robot but far from sensor", "Conflicting objectives"),
        ("robot between box and sensor", "Positional mediation")
    ]
    
    for text, description in complex_tasks:
        print(f"\n📝 Task: {text}")
        print(f"📋 Description: {description}")
        
        agent = SpatialAgent2D(
            scene_width=12.0,
            scene_height=10.0,
            max_iterations=40,
            convergence_threshold=1e-3
        )
        
        results = agent.run(
            text_description=text,
            enable_visualization=False,
            save_video=False,
            seed=123
        )
        
        print(f"✅ Result: {'Success' if results['success'] else 'Partial'}")
        print(f"🔄 Iterations: {results['iterations']}")
        print(f"📊 Score: {results['final_score']:.4f}")
        print(f"⚠️ Constraint Violation: {results['final_constraint_violation']:.4f}")
        
        print("📍 Final positions:")
        for entity, pos in results['entities'].items():
            print(f"  {entity}: ({pos[0]:.2f}, {pos[1]:.2f})")
        
        # Calculate actual relationships
        entities = list(results['entities'].items())
        if len(entities) >= 2:
            pos1, pos2 = entities[0][1], entities[1][1]
            distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
            print(f"📏 Distance between {entities[0][0]} and {entities[1][0]}: {distance:.2f}")
        
        agent.close()
        time.sleep(1)


def demo_with_visualization():
    """Demonstrate with real-time visualization"""
    print("\n🎯 Demo 3: Real-time Visualization")
    print("-" * 40)
    
    print("📝 Task: robot near sensor with live visualization")
    print("💡 This will show the optimization process in real-time")
    print("🎬 Close the visualization window when ready to continue")
    
    try:
        agent = SpatialAgent2D(
            scene_width=10.0,
            scene_height=8.0,
            max_iterations=30,
            convergence_threshold=1e-3
        )
        
        results = agent.run(
            text_description="robot near sensor",
            enable_visualization=True,  # Show live visualization
            save_video=False,
            seed=456
        )
        
        print(f"\n✅ Visualization demo completed!")
        print(f"🔄 Iterations: {results['iterations']}")
        print(f"📊 Final Score: {results['final_score']:.4f}")
        
        agent.close()
        
    except Exception as e:
        print(f"⚠️  Visualization demo failed: {e}")
        print("💡 This might happen in headless environments")


def demo_video_export():
    """Demonstrate video/GIF export functionality"""
    print("\n🎯 Demo 4: Video Export")
    print("-" * 40)
    
    print("📝 Task: box above robot with video export")
    print("🎥 This will create an animated GIF of the optimization process")
    
    agent = SpatialAgent2D(
        scene_width=10.0,
        scene_height=8.0,
        max_iterations=25,
        convergence_threshold=1e-3
    )
    
    output_file = "spatial_agent_demo.gif"
    
    results = agent.run(
        text_description="box above robot",
        enable_visualization=False,  # No live display for video export
        save_video=True,  # Enable video export
        seed=789
    )
    
    print(f"✅ Animation export completed!")
    print(f"🔄 Iterations: {results['iterations']}")
    print(f"📊 Final Score: {results['final_score']:.4f}")
    
    # Check if file was created
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"🎥 Video saved: {output_file} ({file_size} bytes)")
    else:
        print(f"⚠️  Video file not found: {output_file}")
    
    agent.close()


def demo_parameter_effects():
    """Demonstrate the effect of different parameters"""
    print("\n🎯 Demo 5: Parameter Effects")
    print("-" * 40)
    
    task = "robot near sensor"
    
    parameter_configs = [
        {"max_iterations": 10, "name": "Quick (10 iter)"},
        {"max_iterations": 50, "name": "Thorough (50 iter)"},
        {"convergence_threshold": 1e-2, "name": "Relaxed threshold"},
        {"convergence_threshold": 1e-4, "name": "Strict threshold"},
    ]
    
    base_config = {
        "scene_width": 8.0,
        "scene_height": 6.0,
        "max_iterations": 25,
        "convergence_threshold": 1e-3
    }
    
    print(f"📝 Task: {task}")
    
    for config in parameter_configs:
        print(f"\n🔧 Configuration: {config['name']}")
        
        # Merge configurations
        full_config = {**base_config, **{k: v for k, v in config.items() if k != 'name'}}
        
        agent = SpatialAgent2D(**full_config)
        
        start_time = time.time()
        results = agent.run(
            text_description=task,
            enable_visualization=False,
            save_video=False,
            seed=999
        )
        actual_time = time.time() - start_time
        
        print(f"  ✅ {'Success' if results['success'] else 'Partial'}")
        print(f"  🔄 Iterations: {results['iterations']}")
        print(f"  📊 Score: {results['final_score']:.4f}")
        print(f"  ⚡ Time: {actual_time:.2f}s")
        
        agent.close()


def demo_edge_cases():
    """Demonstrate handling of edge cases"""
    print("\n🎯 Demo 6: Edge Cases and Error Handling")
    print("-" * 40)
    
    edge_cases = [
        ("impossible constraint", "Testing nonsensical input"),
        ("", "Testing empty input"),
        ("box", "Testing single entity"),
        ("very long sentence with many words that may not make much sense spatially", 
         "Testing complex parsing"),
    ]
    
    for text, description in edge_cases:
        print(f"\n📝 Task: '{text}'")
        print(f"📋 Description: {description}")
        
        try:
            agent = SpatialAgent2D(
                scene_width=8.0,
                scene_height=6.0,
                max_iterations=15,
                convergence_threshold=1e-2
            )
            
            results = agent.run(
                text_description=text,
                enable_visualization=False,
                save_video=False,
                seed=111
            )
            
            print(f"✅ Handled gracefully")
            print(f"🔄 Iterations: {results['iterations']}")
            print(f"📊 Score: {results['final_score']:.4f}")
            
            agent.close()
            
        except Exception as e:
            print(f"⚠️  Error (expected): {str(e)[:50]}...")


def demo_performance_benchmark():
    """Benchmark performance across multiple runs"""
    print("\n🎯 Demo 7: Performance Benchmark")
    print("-" * 40)
    
    task = "box above robot"
    num_runs = 5
    
    print(f"📝 Task: {task}")
    print(f"🔢 Runs: {num_runs}")
    
    results_list = []
    total_time = 0
    
    for run in range(num_runs):
        print(f"\n🔄 Run {run + 1}/{num_runs}")
        
        agent = SpatialAgent2D(
            scene_width=8.0,
            scene_height=6.0,
            max_iterations=20,
            convergence_threshold=1e-3
        )
        
        start_time = time.time()
        results = agent.run(
            text_description=task,
            enable_visualization=False,
            save_video=False,
            seed=run * 42  # Different seed each run
        )
        run_time = time.time() - start_time
        total_time += run_time
        
        results_list.append({
            'success': results['success'],
            'iterations': results['iterations'],
            'score': results['final_score'],
            'time': run_time
        })
        
        print(f"  ✅ {'Success' if results['success'] else 'Partial'}")
        print(f"  🔄 {results['iterations']} iter, 📊 {results['final_score']:.4f}, ⚡ {run_time:.2f}s")
        
        agent.close()
    
    # Calculate statistics
    success_rate = sum(1 for r in results_list if r['success']) / num_runs
    avg_iterations = sum(r['iterations'] for r in results_list) / num_runs
    avg_score = sum(r['score'] for r in results_list) / num_runs
    avg_time = total_time / num_runs
    
    print(f"\n📊 Benchmark Results:")
    print(f"  🎯 Success Rate: {success_rate:.1%}")
    print(f"  🔄 Avg Iterations: {avg_iterations:.1f}")
    print(f"  📊 Avg Score: {avg_score:.4f}")
    print(f"  ⚡ Avg Time: {avg_time:.2f}s")
    print(f"  ⚡ Total Time: {total_time:.2f}s")


def main():
    """Run complete demonstration suite"""
    parser = argparse.ArgumentParser(
        description="Complete demonstration of 2D Spatial Agent capabilities"
    )
    parser.add_argument(
        '--demo', 
        type=int, 
        choices=range(1, 8), 
        help='Run specific demo (1-7), or all demos if not specified'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick versions of demos (fewer iterations)'
    )
    
    args = parser.parse_args()
    
    print("🚀 2D Spatial Agent - Complete Demonstration")
    print("=" * 60)
    print("This demo showcases all capabilities of the spatial reasoning system:")
    print("• Natural language constraint parsing")
    print("• Real-time GASM-based optimization")
    print("• Live visualization and animation")
    print("• Robust error handling and edge cases")
    print("• Performance benchmarking")
    print("=" * 60)
    
    if args.demo:
        # Run specific demo
        demos = [
            demo_basic_constraints,
            demo_complex_constraints,
            demo_with_visualization,
            demo_video_export,
            demo_parameter_effects,
            demo_edge_cases,
            demo_performance_benchmark
        ]
        
        print(f"Running Demo {args.demo}")
        demos[args.demo - 1]()
    
    else:
        # Run all demos
        try:
            demo_basic_constraints()
            demo_complex_constraints()
            demo_parameter_effects()
            demo_edge_cases()
            demo_performance_benchmark()
            
            # Interactive demos (might fail in headless environments)
            try:
                demo_with_visualization()
                demo_video_export()
            except Exception as e:
                print(f"⚠️  Interactive demos skipped: {e}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Demo interrupted by user")
            return 1
        
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("\n" + "=" * 60)
    print("🎉 Demonstration completed successfully!")
    print("\n📖 To run the agent manually:")
    print("  python agent_loop_2d.py --text 'your constraint here'")
    print("\n📚 For more options:")
    print("  python agent_loop_2d.py --help")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())