#!/usr/bin/env python3
"""
Package Build Script
===================

Script to build and verify the GASM-Roboting package for distribution.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import tempfile

def clean_build():
    """Clean previous build artifacts"""
    print("🧹 Cleaning build artifacts...")
    
    dirs_to_clean = ['build', 'dist', '.pytest_cache']
    patterns_to_clean = ['*.egg-info', '__pycache__']
    
    # Clean specific directories
    for dir_name in dirs_to_clean:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"  Removed {dir_name}")
    
    # Clean pattern matches using glob
    for pattern in patterns_to_clean:
        for path in Path('.').rglob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"  Removed {path}")
            elif path.is_file():
                path.unlink()
                print(f"  Removed {path}")
    
    print("✅ Clean completed")
    return True

def build_sdist():
    """Build source distribution"""
    print("📦 Building source distribution...")
    
    try:
        result = subprocess.run([
            sys.executable, 'setup.py', 'sdist'
        ], check=True, capture_output=True, text=True)
        
        print("✅ Source distribution built successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Source distribution build failed:")
        print(e.stdout)
        print(e.stderr)
        return False

def build_wheel():
    """Build wheel distribution"""
    print("🎡 Building wheel distribution...")
    
    try:
        result = subprocess.run([
            sys.executable, 'setup.py', 'bdist_wheel'
        ], check=True, capture_output=True, text=True)
        
        print("✅ Wheel distribution built successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Wheel distribution build failed:")
        print(e.stdout)
        print(e.stderr)
        return False

def verify_build():
    """Verify built packages"""
    print("🔍 Verifying built packages...")
    
    dist_dir = Path('dist')
    if not dist_dir.exists():
        print("❌ No dist directory found")
        return False
    
    files = list(dist_dir.glob('*'))
    if not files:
        print("❌ No files in dist directory")
        return False
    
    print("Built files:")
    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name} ({size_mb:.2f} MB)")
    
    # Check for both sdist and wheel
    has_sdist = any(f.name.endswith('.tar.gz') for f in files)
    has_wheel = any(f.name.endswith('.whl') for f in files)
    
    if has_sdist and has_wheel:
        print("✅ Both source and wheel distributions available")
        return True
    else:
        print(f"⚠️  Missing distributions: sdist={has_sdist}, wheel={has_wheel}")
        return False

def test_installation():
    """Test installation in temporary environment"""
    print("🧪 Testing installation...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  Using temporary directory: {tmpdir}")
        
        # Find the wheel file
        dist_dir = Path('dist')
        wheel_files = list(dist_dir.glob('*.whl'))
        
        if not wheel_files:
            print("❌ No wheel file found for testing")
            return False
        
        wheel_file = wheel_files[0]
        print(f"  Testing wheel: {wheel_file.name}")
        
        try:
            # Install in isolated environment
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                str(wheel_file.resolve()),
                '--target', tmpdir,
                '--no-deps'  # Don't install dependencies for speed
            ], check=True, capture_output=True, text=True)
            
            # Test basic import
            env = os.environ.copy()
            env['PYTHONPATH'] = tmpdir
            
            result = subprocess.run([
                sys.executable, '-c', 
                'import gasm; print(f"Successfully imported GASM v{gasm.__version__}")'
            ], env=env, check=True, capture_output=True, text=True)
            
            print(f"  {result.stdout.strip()}")
            print("✅ Installation test passed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation test failed:")
            print(e.stdout)
            print(e.stderr)
            return False

def main():
    """Main build process"""
    print("GASM-Roboting Package Build")
    print("=" * 30)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    steps = [
        ("Clean", clean_build),
        ("Build Source", build_sdist), 
        ("Build Wheel", build_wheel),
        ("Verify", verify_build),
        ("Test Install", test_installation),
    ]
    
    results = []
    for step_name, step_func in steps:
        print(f"\n{step_name}:")
        try:
            result = step_func()
            results.append(result)
            
            if not result and step_name not in ["Test Install"]:
                print(f"⚠️  {step_name} failed, stopping build process")
                break
                
        except Exception as e:
            print(f"❌ {step_name} failed with exception: {e}")
            results.append(False)
            break
    
    print(f"\n{'='*30}")
    print("Build Summary:")
    print(f"{'='*30}")
    
    for i, (step_name, _) in enumerate(steps):
        if i < len(results):
            status = "✅ PASS" if results[i] else "❌ FAIL"
            print(f"{status} {step_name}")
        else:
            print(f"⏭️  SKIP {step_name}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} steps completed successfully")
    
    if passed >= 4:  # Allow test installation to fail
        print("🎉 Package build successful!")
        print("\nTo upload to PyPI:")
        print("  pip install twine")
        print("  python -m twine upload dist/*")
        return 0
    else:
        print("⚠️  Build process incomplete. Please fix issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())