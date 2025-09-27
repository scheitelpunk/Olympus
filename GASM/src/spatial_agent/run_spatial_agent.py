#!/usr/bin/env python3
"""
3D Spatial Agent Launcher Script

This script provides easy access to all spatial agent functionality:
- Quick task execution
- Demo scenarios
- Performance benchmarks  
- System diagnostics
- Interactive mode

Usage Examples:
    python run_spatial_agent.py
    python run_spatial_agent.py --quick "place box near sphere"
    python run_spatial_agent.py --demo
    python run_spatial_agent.py --benchmark
    python run_spatial_agent.py --interactive
    python run_spatial_agent.py --install-deps
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from agent_loop_pybullet import (
        main as agent_main, run_demo_scenarios, 
        check_system_requirements, install_dependencies,
        SimulationMode, SimulationConfig, SpatialAgent
    )
    from test_pybullet_agent import main as test_main
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Agent import failed: {e}")
    AGENT_AVAILABLE = False


def quick_task(task_description: str, render_mode: str = "headless", steps: int = 500):
    """Execute a quick spatial task"""
    print(f"🚀 Quick Task Execution")
    print(f"📝 Task: {task_description}")
    print(f"🎮 Render Mode: {render_mode}")
    print("-" * 40)
    
    if not AGENT_AVAILABLE:
        print("❌ Agent not available. Please install dependencies first.")
        return False
    
    try:
        config = SimulationConfig()
        config.max_steps = steps
        
        mode = SimulationMode(render_mode)
        agent = SpatialAgent(config, mode)
        
        result = agent.execute_task(task_description, max_steps=steps)
        
        # Print results
        success_icon = "✅" if result['success'] else "❌"
        print(f"\n{success_icon} Result: {result['success']}")
        print(f"🎲 Objects: {result['objects_created']}")
        print(f"⏱️  Steps: {result['steps']}")
        print(f"🏆 Constraint Satisfaction: {result['final_metrics'].get('constraint_satisfaction', 0):.3f}")
        
        agent.cleanup()
        return result['success']
        
    except Exception as e:
        print(f"❌ Task failed: {e}")
        return False


def interactive_mode():
    """Run interactive spatial agent mode"""
    print("🎮 Interactive Spatial Agent Mode")
    print("=" * 40)
    print("Enter spatial tasks or commands:")
    print("  'help' - Show help")
    print("  'demo' - Run demo scenarios")
    print("  'benchmark' - Run benchmarks")
    print("  'test' - Run test suite")
    print("  'gui' - Switch to GUI mode")
    print("  'quit' or 'exit' - Exit")
    print("-" * 40)
    
    if not AGENT_AVAILABLE:
        print("❌ Agent not available. Install dependencies with: python run_spatial_agent.py --install-deps")
        return
    
    # Initialize agent
    config = SimulationConfig()
    config.max_steps = 1000
    current_mode = SimulationMode.HEADLESS
    agent = None
    
    try:
        while True:
            try:
                user_input = input("\n🤖 Spatial Agent > ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                elif user_input.lower() == 'help':
                    print("""
Available commands:
  help          - Show this help
  demo          - Run demo scenarios  
  benchmark     - Run performance benchmarks
  test          - Run test suite
  gui           - Switch to GUI mode
  headless      - Switch to headless mode
  record        - Switch to recording mode
  config        - Show current configuration
  clear         - Clear screen
  quit/exit     - Exit interactive mode

Spatial tasks (examples):
  "place red box near blue sphere"
  "sort three objects by size"
  "robot arm picks up cylinder"
  "arrange boxes in a line with equal spacing"
""")
                
                elif user_input.lower() == 'demo':
                    print("Running demo scenarios...")
                    if agent:
                        agent.cleanup()
                    run_demo_scenarios()
                    agent = None
                
                elif user_input.lower() == 'benchmark':
                    print("Running benchmarks...")
                    if agent:
                        agent.cleanup()
                    # Would implement benchmark here
                    agent = None
                
                elif user_input.lower() == 'test':
                    print("Running test suite...")
                    if agent:
                        agent.cleanup()
                    test_main()
                    agent = None
                
                elif user_input.lower() == 'gui':
                    current_mode = SimulationMode.GUI
                    print("🖼️  Switched to GUI mode")
                    if agent:
                        agent.cleanup()
                        agent = None
                
                elif user_input.lower() == 'headless':
                    current_mode = SimulationMode.HEADLESS
                    print("💻 Switched to headless mode")
                    if agent:
                        agent.cleanup()
                        agent = None
                
                elif user_input.lower() == 'record':
                    current_mode = SimulationMode.RECORD
                    print("🎬 Switched to recording mode")
                    if agent:
                        agent.cleanup()
                        agent = None
                
                elif user_input.lower() == 'config':
                    print(f"Current Configuration:")
                    print(f"  Mode: {current_mode.value}")
                    print(f"  Max Steps: {config.max_steps}")
                    print(f"  Time Step: {config.time_step:.6f}")
                    print(f"  Resolution: {config.width}x{config.height}")
                
                elif user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                
                else:
                    # Treat as spatial task
                    print(f"🎯 Executing task: {user_input}")
                    
                    # Initialize agent if needed
                    if agent is None:
                        agent = SpatialAgent(config, current_mode)
                    
                    result = agent.execute_task(user_input)
                    
                    # Show results
                    success_icon = "✅" if result['success'] else "❌"
                    print(f"{success_icon} Success: {result['success']}")
                    print(f"🎲 Objects: {result.get('objects_created', 0)}")
                    print(f"⏱️  Time: {result['execution_time']:.2f}s")
                    
                    if result.get('final_metrics'):
                        metrics = result['final_metrics']
                        print(f"📊 Constraint Satisfaction: {metrics.get('constraint_satisfaction', 0):.3f}")
                        print(f"🛡️  Collision Free: {metrics.get('collision_free', 0):.3f}")
                    
                    # Wait for GUI input if in GUI mode
                    if current_mode == SimulationMode.GUI:
                        input("Press Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n⏹️  Interrupted. Type 'quit' to exit.")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    finally:
        if agent:
            print("🧹 Cleaning up...")
            agent.cleanup()
        print("👋 Goodbye!")


def run_system_diagnostics():
    """Run comprehensive system diagnostics"""
    print("🔧 System Diagnostics")
    print("=" * 30)
    
    # Check requirements
    print("\n📋 Checking Requirements...")
    requirements_ok = check_system_requirements()
    
    # Test basic functionality
    print(f"\n🧪 Basic Functionality Test...")
    if AGENT_AVAILABLE:
        try:
            config = SimulationConfig()
            config.max_steps = 5
            agent = SpatialAgent(config, SimulationMode.HEADLESS)
            
            result = agent.execute_task("test task", max_steps=5)
            agent.cleanup()
            
            if result.get('success') is not None:
                print("✅ Agent initialization successful")
            else:
                print("⚠️  Agent initialization completed with warnings")
        except Exception as e:
            print(f"❌ Agent test failed: {e}")
    else:
        print("❌ Agent not available")
    
    # Performance info
    print(f"\n⚡ Performance Information...")
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        print(f"  CPU Cores: {cpu_count}")
        print(f"  RAM Total: {memory.total // (1024**3)} GB")
        print(f"  RAM Available: {memory.available // (1024**3)} GB")
        print(f"  RAM Usage: {memory.percent}%")
    except ImportError:
        print("  Performance monitoring not available (psutil not installed)")
    
    return requirements_ok


def main():
    """Main launcher interface"""
    parser = argparse.ArgumentParser(
        description="3D Spatial Agent Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_spatial_agent.py                          # Interactive mode
  python run_spatial_agent.py --quick "place box"     # Quick task
  python run_spatial_agent.py --demo                   # Run demos
  python run_spatial_agent.py --benchmark              # Benchmarks
  python run_spatial_agent.py --test                   # Run tests
  python run_spatial_agent.py --install-deps           # Install dependencies
  python run_spatial_agent.py --diagnostics            # System diagnostics
        """)
    
    parser.add_argument("--quick", type=str, metavar="TASK",
                       help="Execute quick spatial task")
    parser.add_argument("--render", choices=["headless", "gui", "record"], 
                       default="headless", help="Rendering mode for quick tasks")
    parser.add_argument("--steps", type=int, default=500,
                       help="Maximum steps for quick tasks")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo scenarios")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Run performance benchmarks")
    parser.add_argument("--test", action="store_true",
                       help="Run test suite")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive mode")
    parser.add_argument("--install-deps", action="store_true",
                       help="Install missing dependencies")
    parser.add_argument("--diagnostics", action="store_true",
                       help="Run system diagnostics")
    parser.add_argument("--version", action="store_true",
                       help="Show version information")
    
    args = parser.parse_args()
    
    # Show header
    print("=" * 60)
    print("🤖 3D SPATIAL AGENT WITH PYBULLET PHYSICS")
    print("=" * 60)
    
    # Handle version
    if args.version:
        print("3D Spatial Agent v1.0.0")
        print("PyBullet Physics Integration")
        print("GASM Spatial Reasoning")
        print("Advanced Trajectory Planning")
        return
    
    # Handle dependency installation
    if args.install_deps:
        print("📦 Installing Dependencies...")
        install_dependencies()
        return
    
    # Handle system diagnostics
    if args.diagnostics:
        requirements_ok = run_system_diagnostics()
        if not requirements_ok:
            print(f"\n💡 Install missing dependencies with:")
            print(f"   python run_spatial_agent.py --install-deps")
        return
    
    # Handle test suite
    if args.test:
        print("🧪 Running Test Suite...")
        if AGENT_AVAILABLE:
            exit_code = test_main()
            sys.exit(exit_code)
        else:
            print("❌ Tests not available. Install dependencies first.")
            return
    
    # Handle demo scenarios
    if args.demo:
        print("🎭 Running Demo Scenarios...")
        if AGENT_AVAILABLE:
            run_demo_scenarios()
        else:
            print("❌ Demo not available. Install dependencies first.")
        return
    
    # Handle benchmarks
    if args.benchmark:
        print("🏁 Running Performance Benchmarks...")
        if AGENT_AVAILABLE:
            # Import and run benchmarks
            from test_pybullet_agent import run_benchmark_tests
            run_benchmark_tests()
        else:
            print("❌ Benchmarks not available. Install dependencies first.")
        return
    
    # Handle quick task
    if args.quick:
        success = quick_task(args.quick, args.render, args.steps)
        sys.exit(0 if success else 1)
    
    # Default: interactive mode or show help
    if args.interactive or len(sys.argv) == 1:
        interactive_mode()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()