#!/usr/bin/env python3
"""
Validation script for spatial agent metrics module
Tests core functionality without requiring external dependencies
"""

import sys
import os
import math
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def create_mock_tensor(values):
    """Create a mock tensor-like object for testing"""
    class MockTensor:
        def __init__(self, data):
            if isinstance(data, (list, tuple)):
                self.data = list(data)
            else:
                self.data = [data]
        
        def __getitem__(self, idx):
            return self.data[idx]
        
        def __len__(self):
            return len(self.data)
        
        def norm(self):
            return math.sqrt(sum(x*x for x in self.data))
        
        def __sub__(self, other):
            return MockTensor([a - b for a, b in zip(self.data, other.data)])
        
        def __truediv__(self, scalar):
            return MockTensor([x / scalar for x in self.data])
        
        def dot(self, other):
            return sum(a * b for a, b in zip(self.data, other.data))
        
        def cpu(self):
            return self
        
        def item(self):
            return self.data[0] if len(self.data) == 1 else self.data
        
        def tolist(self):
            return self.data
        
        @property
        def device(self):
            return 'cpu'
    
    return MockTensor(values)

def validate_imports():
    """Validate that the metrics module can be imported"""
    try:
        from spatial_agent.metrics import (
            SpatialMetricsCalculator,
            PoseError,
            ConstraintScore,
            ConstraintType,
            ToleranceConfig
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def validate_basic_functionality():
    """Validate basic functionality of the metrics module"""
    try:
        # Mock torch module
        import torch
    except ImportError:
        # Create a mock torch module for testing
        class MockTorch:
            @staticmethod
            def tensor(data):
                return create_mock_tensor(data)
            
            @staticmethod
            def zeros(size):
                return create_mock_tensor([0.0] * size)
            
            @staticmethod
            def as_tensor(data, **kwargs):
                return create_mock_tensor(data)
            
            @staticmethod
            def norm(tensor):
                return tensor.norm()
        
        sys.modules['torch'] = MockTorch()
    
    from spatial_agent.metrics import (
        SpatialMetricsCalculator,
        PoseError,
        ConstraintScore,
        ConstraintType,
        ToleranceConfig
    )
    
    # Test ToleranceConfig
    config = ToleranceConfig()
    print(f"✓ ToleranceConfig created: position_tol={config.position_tolerance}")
    
    # Test SpatialMetricsCalculator
    calculator = SpatialMetricsCalculator(config)
    print("✓ SpatialMetricsCalculator created")
    
    # Test constraint types
    constraint_types = list(ConstraintType)
    print(f"✓ Found {len(constraint_types)} constraint types: {[ct.value for ct in constraint_types]}")
    
    return True

def validate_module_completeness():
    """Validate that all required methods are implemented"""
    from spatial_agent.metrics import SpatialMetricsCalculator
    
    calculator = SpatialMetricsCalculator()
    required_methods = [
        'pose_error',
        'constraint_score', 
        'is_done',
        'accumulate_statistics',
        'reset_statistics',
        'get_detailed_report'
    ]
    
    missing_methods = []
    for method_name in required_methods:
        if not hasattr(calculator, method_name):
            missing_methods.append(method_name)
        elif callable(getattr(calculator, method_name)):
            print(f"✓ Method '{method_name}' is implemented")
        else:
            missing_methods.append(method_name + " (not callable)")
    
    if missing_methods:
        print(f"✗ Missing methods: {missing_methods}")
        return False
    
    print("✓ All required methods are implemented")
    return True

def validate_constraint_evaluation_methods():
    """Validate that all constraint evaluation methods are implemented"""
    from spatial_agent.metrics import SpatialMetricsCalculator
    
    calculator = SpatialMetricsCalculator()
    constraint_methods = [
        '_evaluate_distance_constraint',
        '_evaluate_angle_constraint',
        '_evaluate_above_constraint',
        '_evaluate_collision_constraint',
        '_evaluate_orientation_constraint',
        '_evaluate_velocity_constraint',
        '_evaluate_workspace_constraint'
    ]
    
    missing_methods = []
    for method_name in constraint_methods:
        if not hasattr(calculator, method_name):
            missing_methods.append(method_name)
        elif callable(getattr(calculator, method_name)):
            print(f"✓ Constraint method '{method_name}' is implemented")
        else:
            missing_methods.append(method_name + " (not callable)")
    
    if missing_methods:
        print(f"✗ Missing constraint evaluation methods: {missing_methods}")
        return False
    
    print("✓ All constraint evaluation methods are implemented")
    return True

def validate_utility_methods():
    """Validate that utility methods for geometric computations are implemented"""
    from spatial_agent.metrics import SpatialMetricsCalculator
    
    calculator = SpatialMetricsCalculator()
    utility_methods = [
        '_parse_pose_input',
        '_calculate_rotation_error',
        '_euler_to_quaternion',
        '_quaternion_to_rotation_matrix',
        '_quaternion_multiply',
        '_quaternion_conjugate',
        '_calculate_trend'
    ]
    
    missing_methods = []
    for method_name in utility_methods:
        if not hasattr(calculator, method_name):
            missing_methods.append(method_name)
        elif callable(getattr(calculator, method_name)):
            print(f"✓ Utility method '{method_name}' is implemented")
        else:
            missing_methods.append(method_name + " (not callable)")
    
    if missing_methods:
        print(f"✗ Missing utility methods: {missing_methods}")
        return False
    
    print("✓ All utility methods are implemented")
    return True

def validate_convenience_functions():
    """Validate that convenience functions are available"""
    try:
        from spatial_agent.metrics import (
            calculate_pose_error,
            evaluate_constraints,
            check_convergence
        )
        print("✓ All convenience functions are available")
        return True
    except ImportError as e:
        print(f"✗ Missing convenience functions: {e}")
        return False

def main():
    """Main validation function"""
    print("=== Spatial Agent Metrics Module Validation ===\n")
    
    validations = [
        ("Import validation", validate_imports),
        ("Basic functionality", validate_basic_functionality),
        ("Module completeness", validate_module_completeness),
        ("Constraint evaluation methods", validate_constraint_evaluation_methods),
        ("Utility methods", validate_utility_methods),
        ("Convenience functions", validate_convenience_functions)
    ]
    
    passed = 0
    total = len(validations)
    
    for validation_name, validation_func in validations:
        print(f"\n--- {validation_name} ---")
        try:
            if validation_func():
                passed += 1
            else:
                print(f"✗ {validation_name} failed")
        except Exception as e:
            print(f"✗ {validation_name} failed with exception: {e}")
    
    print(f"\n=== Validation Results ===")
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All validations passed! The metrics module appears to be complete.")
        return True
    else:
        print("⚠️  Some validations failed. Check the implementation.")
        return False

if __name__ == '__main__':
    main()