#!/usr/bin/env python3
"""Basic validation script for MORPHEUS implementation.

This script performs basic structural validation without requiring
external dependencies like torch, pydantic, etc.
"""

import sys
import ast
import traceback
from pathlib import Path

# Add MORPHEUS to path
morpheus_root = Path(__file__).parent.parent
src_path = morpheus_root / 'src'
sys.path.insert(0, str(src_path))

def validate_python_syntax(file_path):
    """Validate that a Python file has correct syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Parse error: {e}"

def validate_file_structure():
    """Validate expected file structure exists."""
    print("Validating file structure...")
    
    required_files = [
        'src/morpheus/core/orchestrator.py',
        'src/morpheus/examples/basic_perception.py',
        'src/morpheus/examples/material_exploration.py', 
        'src/morpheus/examples/dream_cycle_demo.py',
        'src/morpheus/examples/full_integration.py',
        'src/morpheus/examples/__init__.py'
    ]
    
    missing_files = []
    syntax_errors = []
    
    for file_path in required_files:
        full_path = morpheus_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
            print(f"  ✗ Missing: {file_path}")
        else:
            # Check syntax
            valid, error = validate_python_syntax(full_path)
            if valid:
                print(f"  ✓ Valid: {file_path}")
            else:
                print(f"  ✗ Syntax Error: {file_path} - {error}")
                syntax_errors.append((file_path, error))
    
    if missing_files:
        print(f"\nERROR: {len(missing_files)} required files missing!")
        return False
    
    if syntax_errors:
        print(f"\nERROR: {len(syntax_errors)} files have syntax errors!")
        for file_path, error in syntax_errors:
            print(f"  {file_path}: {error}")
        return False
    
    print(f"\n✓ All {len(required_files)} required files found and syntactically valid")
    return True

def validate_key_classes():
    """Validate key classes can be defined (without external deps)."""
    print("\nValidating key class definitions...")
    
    # Mock external dependencies
    class MockTorch:
        class nn:
            class Module:
                def __init__(self): pass
            class Sequential:
                def __init__(self, *args): pass
            class Linear:
                def __init__(self, *args, **kwargs): pass
            class ReLU:
                def __init__(self, *args, **kwargs): pass
            class Softmax:
                def __init__(self, *args, **kwargs): pass
            class ModuleList:
                def __init__(self, *args): pass
    
    class MockPydantic:
        class BaseModel:
            def __init__(self, **kwargs): pass
            def dict(self): return {}
        class Field:
            def __init__(self, *args, **kwargs): pass
        class validator:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, func): return func
        class BaseSettings(BaseModel): pass
        class env_settings:
            class BaseSettings(BaseModel): pass
    
    # Monkey patch imports
    sys.modules['torch'] = MockTorch
    sys.modules['torch.nn'] = MockTorch.nn
    sys.modules['pydantic'] = MockPydantic
    sys.modules['pydantic.env_settings'] = MockPydantic.env_settings
    sys.modules['psycopg2'] = type('MockPsycopg2', (), {})()
    
    try:
        # Test core types
        from morpheus.core.types import MaterialProperties, TactileSignature, Vector3D
        vec = Vector3D(1, 2, 3)
        assert vec.magnitude() > 0
        print("  ✓ Core types")
        
        # Test basic material bridge (doesn't need external deps)
        from morpheus.integration.material_bridge import MaterialBridge
        print("  ✓ Material bridge")
        
        # Test orchestrator class structure
        from morpheus.core.orchestrator import MorpheusOrchestrator
        print("  ✓ Main orchestrator")
        
        # Test example classes
        from morpheus.examples.basic_perception import BasicPerceptionDemo
        from morpheus.examples.material_exploration import MaterialExplorationDemo
        from morpheus.examples.dream_cycle_demo import DreamCycleDemo
        from morpheus.examples.full_integration import MorpheusFullIntegrationDemo
        print("  ✓ Example demos")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Class validation failed: {e}")
        traceback.print_exc()
        return False

def validate_imports_structure():
    """Validate import structure is correct."""
    print("\nValidating import structure...")
    
    try:
        # Test that we can at least parse the modules
        from morpheus.examples import get_example_config
        config = get_example_config()
        assert isinstance(config, dict)
        print("  ✓ Examples package")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Import structure validation failed: {e}")
        return False

def check_implementation_completeness():
    """Check that implementation appears complete."""
    print("\nChecking implementation completeness...")
    
    # Check orchestrator has key methods
    orchestrator_file = morpheus_root / 'src/morpheus/core/orchestrator.py'
    with open(orchestrator_file, 'r') as f:
        content = f.read()
    
    required_methods = [
        'def perceive(',
        'def dream(',
        'def predict_material_interaction(',
        'def get_session_summary(',
        'def cleanup('
    ]
    
    missing_methods = []
    for method in required_methods:
        if method not in content:
            missing_methods.append(method)
    
    if missing_methods:
        print(f"  ✗ Orchestrator missing methods: {missing_methods}")
        return False
    
    print("  ✓ Orchestrator has all required methods")
    
    # Check examples have main functions
    example_files = [
        'basic_perception.py',
        'material_exploration.py',
        'dream_cycle_demo.py',
        'full_integration.py'
    ]
    
    for example_file in example_files:
        example_path = morpheus_root / 'src/morpheus/examples' / example_file
        with open(example_path, 'r') as f:
            content = f.read()
        
        if 'def main():' not in content:
            print(f"  ✗ {example_file} missing main() function")
            return False
        
        if '__name__ == "__main__":' not in content:
            print(f"  ✗ {example_file} missing main guard")
            return False
    
    print("  ✓ All examples have proper main functions")
    
    return True

def main():
    """Run basic validation."""
    print("="*70)
    print("MORPHEUS BASIC IMPLEMENTATION VALIDATION")
    print("="*70)
    print("This validation checks structure without requiring external dependencies.")
    
    tests = [
        ("File Structure & Syntax", validate_file_structure),
        ("Key Class Definitions", validate_key_classes),
        ("Import Structure", validate_imports_structure),
        ("Implementation Completeness", check_implementation_completeness)
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
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*70}")
    print("BASIC VALIDATION SUMMARY")
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
        print("\n🎉 ALL BASIC TESTS PASSED!")
        print("MORPHEUS implementation structure is valid.")
        print("\nNOTE: Full functionality requires:")
        print("  - PostgreSQL database (docker-compose up)")
        print("  - Python dependencies (pip install -r requirements.txt)")
        print("  - GASM-Robotics materials configuration")
        return True
    else:
        print("\n⚠️  Some tests failed - check implementation")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)