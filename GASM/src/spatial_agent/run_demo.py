#!/usr/bin/env python3
"""
Quick demo runner for 2D Spatial Agent
Provides easy access to common use cases
"""

import sys
import os
import argparse

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

def run_quick_demo():
    """Run a quick demonstration"""
    # Direct import from current directory
    try:
        from agent_loop_2d import SpatialAgent2D
    except ImportError:
        # Try alternative import paths
        sys.path.insert(0, current_dir)
        from agent_loop_2d import SpatialAgent2D
    
    print("🚀 Quick Demo - 2D Spatial Agent")
    print("=" * 50)
    
    # Simple example
    print("📝 Task: 'box above robot'")
    
    agent = SpatialAgent2D(
        scene_width=10.0,
        scene_height=8.0,
        max_iterations=20,
        convergence_threshold=1e-3
    )
    
    results = agent.run(
        text_description="box above robot",
        enable_visualization=True,
        save_video=False,
        seed=42
    )
    
    print("\n" + "=" * 50)
    print("✅ Results:")
    print(f"  Success: {results['success']}")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Final Score: {results['final_score']:.4f}")
    print(f"  Execution Time: {results['execution_time']:.2f}s")
    
    print("\n📍 Final Positions:")
    for entity, pos in results['entities'].items():
        print(f"  {entity}: ({pos[0]:.2f}, {pos[1]:.2f})")
    
    agent.close()
    
    print("\n🎉 Demo completed! Close the visualization window when ready.")
    input("Press Enter to continue...")


def run_examples():
    """Run predefined examples"""
    from spatial_agent.agent_loop_2d import SpatialAgent2D
    
    examples = [
        ("box above robot", "Basic vertical constraint"),
        ("robot near sensor", "Distance minimization"),
        ("box left of robot", "Horizontal positioning"),
        ("robot far from box", "Distance maximization")
    ]
    
    print("🎯 Running Example Tasks")
    print("=" * 50)
    
    for i, (task, description) in enumerate(examples, 1):
        print(f"\n{i}. {description}")
        print(f"   Task: '{task}'")
        
        agent = SpatialAgent2D(max_iterations=15)
        
        results = agent.run(
            text_description=task,
            enable_visualization=False,
            save_video=False,
            seed=42 + i
        )
        
        print(f"   Result: {'✅ Success' if results['success'] else '⚠️  Partial'}")
        print(f"   Score: {results['final_score']:.3f}, Time: {results['execution_time']:.1f}s")
        
        agent.close()
    
    print("\n🎉 All examples completed!")


def run_interactive():
    """Interactive mode - ask user for constraints"""
    from spatial_agent.agent_loop_2d import SpatialAgent2D
    
    print("🎮 Interactive Mode")
    print("=" * 50)
    print("Enter spatial constraints in natural language.")
    print("Examples: 'box above robot', 'robot near sensor', 'box left of robot'")
    print("Type 'quit' to exit.\n")
    
    agent = SpatialAgent2D(max_iterations=25)
    
    while True:
        try:
            user_input = input("🤖 Enter constraint: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                print("Please enter a constraint or 'quit' to exit.")
                continue
            
            print(f"🔄 Processing: '{user_input}'")
            
            results = agent.run(
                text_description=user_input,
                enable_visualization=True,
                save_video=False
            )
            
            print(f"✅ {'Success' if results['success'] else 'Partial'} - "
                  f"Score: {results['final_score']:.3f}, "
                  f"Iterations: {results['iterations']}")
            
            for entity, pos in results['entities'].items():
                print(f"  {entity}: ({pos[0]:.2f}, {pos[1]:.2f})")
            
            print()
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    agent.close()
    print("👋 Goodbye!")


def run_performance_test():
    """Run performance benchmark"""
    from spatial_agent.agent_loop_2d import SpatialAgent2D
    import time
    
    print("📊 Performance Benchmark")
    print("=" * 50)
    
    task = "box above robot"
    num_runs = 10
    
    print(f"Task: '{task}'")
    print(f"Runs: {num_runs}")
    print()
    
    total_time = 0
    success_count = 0
    iteration_sum = 0
    score_sum = 0
    
    for i in range(num_runs):
        agent = SpatialAgent2D(max_iterations=20)
        
        start_time = time.time()
        results = agent.run(
            text_description=task,
            enable_visualization=False,
            save_video=False,
            seed=i * 42
        )
        run_time = time.time() - start_time
        
        total_time += run_time
        if results['success']:
            success_count += 1
        iteration_sum += results['iterations']
        score_sum += results['final_score']
        
        print(f"Run {i+1:2d}: {'✅' if results['success'] else '⚠️ '} "
              f"{results['iterations']:2d} iter, "
              f"{results['final_score']:.3f} score, "
              f"{run_time:.2f}s")
        
        agent.close()
    
    print("\n📈 Summary:")
    print(f"  Success Rate: {success_count/num_runs:.1%}")
    print(f"  Avg Iterations: {iteration_sum/num_runs:.1f}")
    print(f"  Avg Score: {score_sum/num_runs:.3f}")
    print(f"  Avg Time: {total_time/num_runs:.2f}s")
    print(f"  Total Time: {total_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Quick demo runner for 2D Spatial Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modes:
  demo        - Quick interactive demonstration
  examples    - Run predefined example tasks  
  interactive - Interactive mode (enter your own constraints)
  benchmark   - Performance benchmark
  
Examples:
  python run_demo.py demo
  python run_demo.py examples
  python run_demo.py interactive
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['demo', 'examples', 'interactive', 'benchmark'],
        help='Demo mode to run'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'demo':
            run_quick_demo()
        elif args.mode == 'examples':
            run_examples()
        elif args.mode == 'interactive':
            run_interactive()
        elif args.mode == 'benchmark':
            run_performance_test()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())