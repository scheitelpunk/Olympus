#!/usr/bin/env python3
"""
Windows-friendly installation script for GASM-Roboting
Handles problematic dependencies gracefully
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors gracefully"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def main():
    print("🚀 GASM-Roboting Windows Installation")
    print("=" * 50)
    
    # Step 1: Install minimal requirements
    print("\n📦 Installing core dependencies...")
    success = run_command(
        "pip install numpy matplotlib scipy fastapi uvicorn psutil pyyaml pytest black",
        "Core dependencies"
    )
    
    if not success:
        print("❌ Core installation failed. Please check your Python environment.")
        return
    
    # Step 2: Optional dependencies with fallbacks
    optional_deps = [
        ("torch", "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"),
        ("opencv-python", "pip install opencv-python"),
        ("pybullet", "pip install pybullet"),
        ("pillow", "pip install pillow"),
        ("transformers", "pip install transformers")
    ]
    
    print("\n🎯 Installing optional dependencies...")
    for name, cmd in optional_deps:
        success = run_command(cmd, f"Optional: {name}")
        if not success:
            print(f"⚠️  {name} installation failed - will use fallbacks")
    
    # Step 3: Test imports
    print("\n🧪 Testing imports...")
    test_imports = [
        "numpy",
        "matplotlib", 
        "scipy",
        "fastapi"
    ]
    
    all_good = True
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            all_good = False
    
    # Step 4: Check GASM modules
    print("\n🔍 Testing GASM modules...")
    sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
    
    gasm_modules = [
        "spatial_agent.gasm_bridge",
        "spatial_agent.utils_se3",
        "spatial_agent.metrics",
        "spatial_agent.planner"
    ]
    
    for module in gasm_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"⚠️  {module} - {str(e)[:50]}...")
    
    # Final status
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 Installation completed successfully!")
        print("\n🚀 Ready to run:")
        print("python run_quick_demo.py demo")
        print("python src/spatial_agent/agent_loop_2d.py --text 'box above robot' --no_visualization")
    else:
        print("⚠️  Installation completed with some optional dependencies missing.")
        print("   Core functionality should still work with fallbacks.")

if __name__ == "__main__":
    main()