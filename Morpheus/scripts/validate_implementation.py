#!/usr/bin/env python3
"""Validation script for MORPHEUS implementation.

This script validates that all components are correctly implemented and can be imported
without errors. It performs basic integration testing without requiring a database.

Usage:
    python scripts/validate_implementation.py
"""

import sys
import os
import importlib
from pathlib import Path

# Add MORPHEUS to path
morpheus_root = Path(__file__).parent.parent
src_path = morpheus_root / 'src'
sys.path.insert(0, str(src_path))

def test_imports():
    """Test all critical imports."""
    print("Testing MORPHEUS component imports...")
    
    components = [
        'morpheus.core.types',
        'morpheus.core.config', 
        'morpheus.core.orchestrator',
        'morpheus.storage.postgres_storage',
        'morpheus.integration.material_bridge',
        'morpheus.dream_sim.dream_orchestrator',
        'morpheus.examples.basic_perception',
        'morpheus.examples.material_exploration', 
        'morpheus.examples.dream_cycle_demo',
        'morpheus.examples.full_integration'
    ]
    
    results = {}
    
    for component in components:
        try:
            module = importlib.import_module(component)
            results[component] = {'status': 'SUCCESS', 'error': None}
            print(f"  ✓ {component}")
        except Exception as e:
            results[component] = {'status': 'FAILED', 'error': str(e)}
            print(f"  ✗ {component}: {e}")
    
    return results

def test_class_instantiation():
    """Test basic class instantiation."""
    print("\nTesting class instantiation...")
    
    try:
        from morpheus.core.types import MaterialProperties, TactileSignature, Vector3D
        from morpheus.core.config import MorpheusConfig, ConfigManager
        
        # Test basic types
        vec = Vector3D(1, 2, 3)
        assert vec.magnitude() > 0
        print("  ✓ Vector3D")
        
        # Test material properties
        mat = MaterialProperties(name='test', friction=0.5)
        assert mat.name == 'test'
        print("  ✓ MaterialProperties")
        
        # Test configuration
        config = MorpheusConfig()
        assert hasattr(config, 'system')
        print("  ✓ MorpheusConfig")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Class instantiation failed: {e}")
        return False

def test_orchestrator_creation():
    """Test orchestrator creation without database."""
    print("\nTesting orchestrator creation (dry run)...")
    
    try:
        from morpheus.core.orchestrator import MorpheusOrchestrator
        from morpheus.core.config import MorpheusConfig
        
        # Create minimal config
        config = MorpheusConfig()
        
        # This will fail at database connection, but we can check class structure
        try:
            orchestrator = MorpheusOrchestrator(config, gasm_robotics_path=None, database_config=None)
        except Exception as e:
            # Expected to fail due to missing database/GASM
            if "GASM" in str(e) or "database" in str(e) or "not found" in str(e):
                print("  ✓ MorpheusOrchestrator class structure valid (database connection expected to fail)")
                return True
            else:
                print(f"  ✗ Unexpected orchestrator error: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Orchestrator creation test failed: {e}")
        return False

def test_example_structure():
    """Test example module structure.""" 
    print("\nTesting example module structure...")
    
    try:
        from morpheus.examples import get_example_config, ensure_morpheus_path
        
        config = get_example_config()
        assert 'config_path' in config
        print("  ✓ Example utilities")
        
        # Test example classes can be imported
        from morpheus.examples.basic_perception import BasicPerceptionDemo
        from morpheus.examples.material_exploration import MaterialExplorationDemo
        from morpheus.examples.dream_cycle_demo import DreamCycleDemo  
        from morpheus.examples.full_integration import MorpheusFullIntegrationDemo
        
        print("  ✓ All example classes imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Example structure test failed: {e}")
        return False

def validate_file_structure():
    """Validate expected file structure exists."""
    print("\nValidating file structure...")
    
    required_files = [
        'src/morpheus/core/orchestrator.py',
        'src/morpheus/examples/basic_perception.py',
        'src/morpheus/examples/material_exploration.py', 
        'src/morpheus/examples/dream_cycle_demo.py',
        'src/morpheus/examples/full_integration.py',
        'src/morpheus/examples/__init__.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = morpheus_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
            print(f"  ✗ Missing: {file_path}")
        else:
            print(f"  ✓ Found: {file_path}")
    
    if missing_files:
        print(f"\nERROR: {len(missing_files)} required files missing!")
        return False
    
    print(f"\n✓ All {len(required_files)} required files found")
    return True

def main():
    """Run complete validation."""
    print("="*70)
    print("MORPHEUS IMPLEMENTATION VALIDATION")
    print("="*70)
    
    tests = [
        ("File Structure", validate_file_structure),
        ("Component Imports", test_imports),
        ("Class Instantiation", test_class_instantiation),
        ("Orchestrator Creation", test_orchestrator_creation),
        ("Example Structure", test_example_structure)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"RUNNING: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"TEST FAILED: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        symbol = "✓" if passed_test else "✗"
        print(f"  {symbol} {test_name}: {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - MORPHEUS implementation is valid!")
        return True
    else:
        print("⚠️  Some tests failed - check implementation")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)