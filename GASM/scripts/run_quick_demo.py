#!/usr/bin/env python3
"""
Quick demo runner for GASM Spatial Agent - Fixed Import Version
Run from project root directory
"""

import sys
import os

# Add paths for module imports
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, 'src')
spatial_agent_dir = os.path.join(src_dir, 'spatial_agent')

sys.path.insert(0, src_dir)
sys.path.insert(0, spatial_agent_dir)

def run_demo():
    """Run the 2D spatial agent demo"""
    print("🚀 GASM Spatial Agent - Quick Demo")
    print("=" * 50)
    
    try:
        # Import the demo function directly
        import importlib.util
        
        # Load agent_loop_2d module
        spec = importlib.util.spec_from_file_location(
            "agent_loop_2d", 
            os.path.join(spatial_agent_dir, "agent_loop_2d.py")
        )
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        
        print("✅ Modules loaded successfully")
        print("\n📝 Running: 'Place box above robot'")
        
        # Create and run the agent
        if hasattr(agent_module, 'main'):
            # Call main function directly
            import argparse
            
            # Mock command line arguments
            sys.argv = [
                'agent_loop_2d.py',
                '--text', 'Place box above robot',
                '--steps', '20',
                '--no_visualization',  # Disable GUI for quick demo
                '--seed', '42'
            ]
            
            agent_module.main()
        else:
            print("❌ No main function found in agent_loop_2d.py")
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're running from the project root directory")
        print("2. Install dependencies: pip install -r requirements.txt") 
        print("3. Try: python -m src.spatial_agent.agent_loop_2d --help")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        print("🔧 Try running the agent directly:")
        print("python src/spatial_agent/agent_loop_2d.py --text 'box above robot' --no_visualization")

def run_test_import():
    """Test if all modules can be imported"""
    print("🧪 Testing Module Imports")
    print("=" * 30)
    
    modules_to_test = [
        'gasm_bridge',
        'utils_se3', 
        'metrics',
        'planner',
        'vision'
    ]
    
    for module_name in modules_to_test:
        try:
            module_path = os.path.join(spatial_agent_dir, f"{module_name}.py")
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"✅ {module_name}")
        except Exception as e:
            print(f"❌ {module_name}: {str(e)[:50]}...")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GASM Spatial Agent Demo Runner")
    parser.add_argument('command', choices=['demo', 'test', 'help'], 
                       help='Command to run')
    
    if len(sys.argv) == 1:
        sys.argv.append('demo')  # Default to demo
        
    args = parser.parse_args()
    
    if args.command == 'demo':
        run_demo()
    elif args.command == 'test':
        run_test_import()
    elif args.command == 'help':
        print("Available commands:")
        print("  demo - Run 2D spatial agent demonstration")  
        print("  test - Test module imports")
        print("\nDirect usage:")
        print("  python src/spatial_agent/agent_loop_2d.py --text 'box above robot'")

if __name__ == "__main__":
    main()