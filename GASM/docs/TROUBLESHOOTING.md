# GASM Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide covers common issues, error messages, debugging techniques, and solutions for the GASM (Geometric Assembly State Machine) system. It's organized by component and severity to help you quickly identify and resolve problems.

## Table of Contents

1. [Quick Diagnosis](#quick-diagnosis)
2. [Installation and Setup Issues](#installation-and-setup-issues)
3. [Runtime Errors](#runtime-errors)
4. [Performance Issues](#performance-issues)
5. [Integration Problems](#integration-problems)
6. [Mathematical and Numerical Issues](#mathematical-and-numerical-issues)
7. [Hardware and Environment Issues](#hardware-and-environment-issues)
8. [Debugging Tools and Techniques](#debugging-tools-and-techniques)
9. [FAQ](#faq)

## Quick Diagnosis

### System Health Check

Run this diagnostic script to quickly identify common issues:

```python
#!/usr/bin/env python3
"""
GASM System Diagnostic Tool
Run this script to identify common configuration and runtime issues
"""

import sys
import os
import importlib
import subprocess
import platform
from pathlib import Path

class GASMDiagnostic:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = []
    
    def run_full_diagnostic(self):
        """Run complete system diagnostic"""
        print("🔧 GASM System Diagnostic")
        print("=" * 50)
        
        self.check_python_version()
        self.check_dependencies()
        self.check_cuda_availability()
        self.check_gasm_installation()
        self.check_file_permissions()
        self.check_memory_availability()
        self.test_basic_functionality()
        
        self.print_summary()
    
    def check_python_version(self):
        """Check Python version compatibility"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.issues.append(f"Python version {version.major}.{version.minor} is too old. Requires Python >= 3.8")
        else:
            self.info.append(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    
    def check_dependencies(self):
        """Check required dependencies"""
        required_packages = [
            ('torch', 'PyTorch'),
            ('numpy', 'NumPy'),
            ('scipy', 'SciPy'),
            ('matplotlib', 'Matplotlib')
        ]
        
        optional_packages = [
            ('torch_geometric', 'PyTorch Geometric'),
            ('geomstats', 'Geomstats'),
            ('pybullet', 'PyBullet'),
            ('cv2', 'OpenCV')
        ]
        
        for module_name, display_name in required_packages:
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, '__version__', 'unknown')
                self.info.append(f"✓ {display_name} {version}")
            except ImportError:
                self.issues.append(f"❌ Missing required package: {display_name}")
        
        for module_name, display_name in optional_packages:
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, '__version__', 'unknown')
                self.info.append(f"✓ {display_name} {version} (optional)")
            except ImportError:
                self.warnings.append(f"⚠️  Optional package not found: {display_name}")
    
    def check_cuda_availability(self):
        """Check CUDA and GPU availability"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.info.append(f"✓ CUDA available with {gpu_count} GPU(s)")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory // 1024**3
                    self.info.append(f"  GPU {i}: {gpu_name} ({gpu_memory}GB)")
            else:
                self.warnings.append("⚠️  CUDA not available, using CPU only")
        except Exception as e:
            self.warnings.append(f"⚠️  Could not check CUDA: {e}")
    
    def check_gasm_installation(self):
        """Check GASM package installation"""
        try:
            from gasm_bridge import create_bridge
            self.info.append("✓ GASM bridge module accessible")
        except ImportError as e:
            self.issues.append(f"❌ GASM bridge not found: {e}")
        
        try:
            from gasm_core import EnhancedGASM
            self.info.append("✓ GASM core module accessible")
        except ImportError as e:
            self.issues.append(f"❌ GASM core not found: {e}")
        
        try:
            from utils_se3 import SE3Utils
            self.info.append("✓ SE(3) utilities accessible")
        except ImportError as e:
            self.issues.append(f"❌ SE(3) utils not found: {e}")
    
    def check_file_permissions(self):
        """Check file system permissions"""
        test_paths = [
            Path.cwd(),
            Path.home() / '.gasm',
            Path('/tmp') if platform.system() != 'Windows' else Path.cwd() / 'temp'
        ]
        
        for path in test_paths:
            if path.exists():
                if os.access(path, os.R_OK | os.W_OK):
                    self.info.append(f"✓ Read/write access to {path}")
                else:
                    self.issues.append(f"❌ No read/write access to {path}")
            else:
                self.warnings.append(f"⚠️  Path does not exist: {path}")
    
    def check_memory_availability(self):
        """Check system memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 2.0:
                self.issues.append(f"❌ Low memory: {available_gb:.1f}GB available (minimum 2GB recommended)")
            elif available_gb < 4.0:
                self.warnings.append(f"⚠️  Limited memory: {available_gb:.1f}GB available (4GB+ recommended)")
            else:
                self.info.append(f"✓ Available memory: {available_gb:.1f}GB")
        except ImportError:
            self.warnings.append("⚠️  Could not check memory (psutil not available)")
    
    def test_basic_functionality(self):
        """Test basic GASM functionality"""
        try:
            from gasm_bridge import create_bridge
            bridge = create_bridge({'device': 'cpu', 'fallback_mode': True})
            result = bridge.process("test instruction")
            
            if isinstance(result, dict) and 'success' in result:
                self.info.append("✓ Basic GASM processing functional")
            else:
                self.warnings.append("⚠️  GASM processing returned unexpected format")
                
        except Exception as e:
            self.issues.append(f"❌ Basic GASM test failed: {e}")
    
    def print_summary(self):
        """Print diagnostic summary"""
        print("\n" + "=" * 50)
        print("📋 DIAGNOSTIC SUMMARY")
        print("=" * 50)
        
        if self.issues:
            print("\n🚨 CRITICAL ISSUES (must fix):")
            for issue in self.issues:
                print(f"  {issue}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS (recommended to fix):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.info:
            print("\n✅ SYSTEM INFO:")
            for info in self.info:
                print(f"  {info}")
        
        # Overall status
        print("\n" + "=" * 50)
        if self.issues:
            print("❌ SYSTEM STATUS: Issues found - see above for details")
            print("📖 Refer to troubleshooting guide for solutions")
        elif self.warnings:
            print("⚠️  SYSTEM STATUS: Warnings present but functional")
        else:
            print("✅ SYSTEM STATUS: All checks passed")
        
        return len(self.issues) == 0

if __name__ == "__main__":
    diagnostic = GASMDiagnostic()
    success = diagnostic.run_full_diagnostic()
    sys.exit(0 if success else 1)
```

### Quick Error Code Reference

| Error Code | Component | Description | Quick Fix |
|------------|-----------|-------------|-----------|
| GASM-001 | Bridge | Module import failed | Check PYTHONPATH and installation |
| GASM-002 | Core | CUDA initialization failed | Use CPU mode or check CUDA drivers |
| GASM-003 | SE3Utils | Invalid matrix dimensions | Validate input matrix shapes |
| GASM-004 | Planner | Constraint conflict | Review constraint compatibility |
| GASM-005 | Metrics | Numerical instability | Increase tolerance parameters |

## Installation and Setup Issues

### Issue: Import Errors

**Symptoms:**
```python
ModuleNotFoundError: No module named 'gasm_bridge'
ImportError: cannot import name 'create_bridge' from 'gasm_bridge'
```

**Diagnosis:**
```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Check if GASM is in Python path
python -c "import sys; print('\n'.join(sys.path))"

# Check installation
pip list | grep gasm
```

**Solutions:**

1. **Development Installation:**
```bash
# Install in development mode
cd /path/to/GASM-Roboting
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/GASM-Roboting/src"
```

2. **Virtual Environment Issues:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Reinstall in virtual environment
pip install -e .
```

3. **Python Path Issues:**
```python
# Add to Python path at runtime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
```

### Issue: Dependency Conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently have a solution
ERROR: Could not find a version that satisfies the requirement
```

**Solutions:**

1. **Clean Environment:**
```bash
# Create fresh environment
python -m venv gasm_env
source gasm_env/bin/activate
pip install --upgrade pip setuptools wheel

# Install with specific versions
pip install torch==1.13.0
pip install -r requirements.txt
```

2. **Conflict Resolution:**
```bash
# Check conflicting packages
pip check

# Install with no dependencies first
pip install --no-deps package_name

# Resolve manually
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: CUDA Version Mismatch

**Symptoms:**
```
RuntimeError: CUDA runtime error (35) : CUDA driver version is insufficient for CUDA runtime version
```

**Diagnosis:**
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

**Solutions:**

1. **Install Compatible PyTorch:**
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

2. **Force CPU Mode:**
```python
# In your code, force CPU usage
bridge = create_bridge({'device': 'cpu'})
```

## Runtime Errors

### Issue: GASM Processing Failures

**Symptoms:**
```python
result = bridge.process("place box above table")
# result['success'] is False
# result['error_message'] contains error details
```

**Debugging Steps:**

1. **Enable Debug Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable GASM debugging
bridge = create_bridge({
    'debug': True,
    'fallback_mode': True,
    'timeout_seconds': 30
})
```

2. **Check Input Validation:**
```python
def debug_processing(text):
    print(f"Input text: '{text}'")
    print(f"Text length: {len(text)}")
    print(f"Text type: {type(text)}")
    
    # Check for common issues
    if not text or not isinstance(text, str):
        print("❌ Invalid input text")
        return
    
    if len(text) > 10000:
        print("⚠️  Very long text, may cause timeout")
    
    result = bridge.process(text)
    
    print(f"Success: {result['success']}")
    if not result['success']:
        print(f"Error: {result.get('error_message')}")
        print(f"Debug info: {result.get('debug_info')}")
    
    return result
```

**Common Fixes:**

1. **Text Processing Issues:**
```python
# Clean and validate input text
def clean_input_text(text):
    if not isinstance(text, str):
        text = str(text)
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    
    # Limit length
    if len(text) > 1000:
        text = text[:1000]
        logging.warning("Text truncated to 1000 characters")
    
    return text
```

2. **Timeout Issues:**
```python
# Increase timeout for complex instructions
bridge = create_bridge({
    'timeout_seconds': 60,  # Increase from default 30
    'max_retries': 3
})
```

### Issue: SE(3) Validation Errors

**Symptoms:**
```
SE3ValidationError: Matrix not orthogonal. Error: 0.001234
SE3ValidationError: Determinant not 1. Det: 0.999
```

**Solutions:**

1. **Increase Tolerance:**
```python
from utils_se3 import SE3Utils

# Use higher tolerance for validation
SE3Utils.validate_rotation_matrix(R, tolerance=1e-4)  # Default: 1e-6
```

2. **Matrix Cleanup:**
```python
def fix_rotation_matrix(R):
    """Clean up rotation matrix to ensure SO(3) properties"""
    # Use SVD to project to nearest rotation matrix
    U, S, Vh = np.linalg.svd(R)
    R_clean = U @ Vh
    
    # Ensure proper rotation (det = +1)
    if np.linalg.det(R_clean) < 0:
        R_clean[:, -1] *= -1
    
    return R_clean
```

3. **Gradual Orthogonalization:**
```python
def orthogonalize_gradually(R, steps=10):
    """Gradually orthogonalize matrix to avoid numerical shocks"""
    R_current = R.copy()
    
    for i in range(steps):
        # Move towards orthogonal matrix
        R_next = 0.5 * (R_current + np.linalg.inv(R_current).T)
        R_current = R_next
    
    return fix_rotation_matrix(R_current)
```

### Issue: Constraint Conflicts

**Symptoms:**
```
ConstraintConflictError: Constraints cannot be simultaneously satisfied
Planning failed: No valid path found
```

**Debugging:**

1. **Constraint Analysis:**
```python
def analyze_constraints(constraints):
    """Analyze constraint compatibility"""
    print("Constraint Analysis:")
    print("-" * 30)
    
    for i, constraint in enumerate(constraints):
        print(f"Constraint {i}: {constraint.type}")
        print(f"  Subject: {constraint.subject}")
        print(f"  Target: {constraint.target}")
        print(f"  Parameters: {constraint.parameters}")
        print(f"  Priority: {constraint.priority}")
        
        # Check for conflicts
        for j, other_constraint in enumerate(constraints[i+1:], i+1):
            if check_constraint_conflict(constraint, other_constraint):
                print(f"  ⚠️  Potential conflict with constraint {j}")
    
    return True

def check_constraint_conflict(c1, c2):
    """Check if two constraints might conflict"""
    # Example: can't be both above and below same object
    if (c1.type == "above" and c2.type == "below" and
        c1.subject == c2.subject and c1.target == c2.target):
        return True
    
    # Add more conflict detection logic
    return False
```

2. **Constraint Relaxation:**
```python
def relax_constraints(constraints, relaxation_factor=1.5):
    """Relax constraint tolerances to resolve conflicts"""
    relaxed = []
    
    for constraint in constraints:
        new_constraint = constraint.copy()
        
        # Increase tolerances
        if 'tolerance' in new_constraint.parameters:
            new_constraint.parameters['tolerance'] *= relaxation_factor
        
        # Reduce priorities for less critical constraints
        if new_constraint.priority < 0.8:
            new_constraint.priority *= 0.8
        
        relaxed.append(new_constraint)
    
    return relaxed
```

## Performance Issues

### Issue: Slow Processing

**Symptoms:**
- Processing takes > 10 seconds for simple instructions
- High CPU/memory usage
- System becomes unresponsive

**Diagnosis:**

1. **Performance Profiling:**
```python
import time
import cProfile
import pstats
from memory_profiler import profile

@profile
def profile_gasm_processing(text):
    """Profile memory usage during processing"""
    bridge = create_bridge({'device': 'cpu'})
    
    start_time = time.time()
    result = bridge.process(text)
    end_time = time.time()
    
    print(f"Processing time: {end_time - start_time:.2f}s")
    return result

# CPU profiling
def profile_cpu_usage(text):
    profiler = cProfile.Profile()
    profiler.enable()
    
    bridge = create_bridge({'device': 'cpu'})
    result = bridge.process(text)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return result
```

2. **Memory Monitoring:**
```python
import psutil
import gc

def monitor_memory():
    """Monitor memory usage during processing"""
    process = psutil.Process()
    
    def get_memory():
        return process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Initial memory: {get_memory():.1f} MB")
    
    # Your GASM operations here
    bridge = create_bridge()
    
    print(f"After initialization: {get_memory():.1f} MB")
    
    for i in range(10):
        result = bridge.process(f"test instruction {i}")
        if i % 3 == 0:
            gc.collect()  # Force garbage collection
        print(f"After iteration {i}: {get_memory():.1f} MB")
```

**Solutions:**

1. **Enable GPU Acceleration:**
```python
# Use GPU if available
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
bridge = create_bridge({
    'device': device,
    'precision': 'float16' if device == 'cuda' else 'float32'
})
```

2. **Batch Processing:**
```python
def process_batch(instructions, batch_size=10):
    """Process multiple instructions efficiently"""
    bridge = create_bridge({'cache_enabled': True})
    
    results = []
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i + batch_size]
        
        # Process batch
        batch_results = []
        for instruction in batch:
            result = bridge.process(instruction)
            batch_results.append(result)
        
        results.extend(batch_results)
        
        # Clear caches periodically
        if i % (batch_size * 5) == 0:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results
```

3. **Optimize Configuration:**
```python
# Performance-optimized configuration
config = {
    'device': 'cuda',
    'precision': 'float16',
    'max_iterations': 100,  # Reduce from default
    'convergence_threshold': 1e-2,  # Relax from 1e-3
    'cache_enabled': True,
    'batch_size': 32,
    'num_workers': 4
}

bridge = create_bridge(config)
```

### Issue: Memory Leaks

**Symptoms:**
- Memory usage continuously increases
- Out of memory errors after multiple operations
- System slowdown over time

**Detection:**
```python
import tracemalloc

def detect_memory_leaks():
    """Detect memory leaks in GASM operations"""
    tracemalloc.start()
    
    # Initial snapshot
    snapshot1 = tracemalloc.take_snapshot()
    
    # Perform operations
    bridge = create_bridge()
    for i in range(100):
        result = bridge.process(f"test instruction {i}")
        del result  # Explicit cleanup
    
    # Final snapshot
    snapshot2 = tracemalloc.take_snapshot()
    
    # Compare snapshots
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("Top 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)
```

**Solutions:**

1. **Explicit Cleanup:**
```python
class GASMManager:
    """Context manager for GASM operations"""
    
    def __init__(self, config=None):
        self.config = config
        self.bridge = None
    
    def __enter__(self):
        self.bridge = create_bridge(self.config)
        return self.bridge
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.bridge, 'cleanup'):
            self.bridge.cleanup()
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Usage
with GASMManager({'device': 'cuda'}) as bridge:
    result = bridge.process("place box above table")
```

2. **Memory Pool Management:**
```python
class MemoryPool:
    """Simple memory pool for tensor operations"""
    
    def __init__(self, max_size=1000):
        self.pool = {}
        self.max_size = max_size
        self.current_size = 0
    
    def get_tensor(self, shape, dtype=torch.float32):
        key = (shape, dtype)
        
        if key in self.pool and self.pool[key]:
            return self.pool[key].pop()
        
        return torch.zeros(shape, dtype=dtype)
    
    def return_tensor(self, tensor):
        if self.current_size >= self.max_size:
            return  # Pool is full
        
        key = (tensor.shape, tensor.dtype)
        if key not in self.pool:
            self.pool[key] = []
        
        tensor.zero_()  # Clear data
        self.pool[key].append(tensor)
        self.current_size += 1
```

## Integration Problems

### Issue: ROS Integration Failures

**Symptoms:**
```
rospy.ROSException: Unable to communicate with master
ImportError: No module named 'rospy'
```

**Solutions:**

1. **ROS Environment Setup:**
```bash
# Source ROS setup
source /opt/ros/noetic/setup.bash  # or your ROS version

# Check ROS environment
echo $ROS_MASTER_URI
echo $ROS_PACKAGE_PATH

# Install Python dependencies
pip install rospkg rospy-message-converter
```

2. **Robust ROS Integration:**
```python
import sys
import time

class RobustROSIntegration:
    def __init__(self, retry_count=5, retry_delay=1.0):
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.node_initialized = False
    
    def initialize_node(self, node_name):
        """Initialize ROS node with retries"""
        for attempt in range(self.retry_count):
            try:
                import rospy
                rospy.init_node(node_name, anonymous=True)
                self.node_initialized = True
                rospy.loginfo(f"ROS node '{node_name}' initialized")
                return True
                
            except Exception as e:
                rospy.logwarn(f"ROS initialization attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                else:
                    rospy.logerr("Failed to initialize ROS node")
                    return False
        
        return False
    
    def safe_publish(self, publisher, message, max_retries=3):
        """Safely publish with error handling"""
        if not self.node_initialized:
            return False
        
        for attempt in range(max_retries):
            try:
                publisher.publish(message)
                return True
            except Exception as e:
                rospy.logwarn(f"Publish attempt {attempt + 1} failed: {e}")
                time.sleep(0.1)
        
        return False
```

### Issue: Hardware Communication Errors

**Symptoms:**
```
ConnectionRefusedError: [Errno 111] Connection refused
TimeoutError: Robot communication timeout
```

**Solutions:**

1. **Connection Retry Logic:**
```python
import socket
import time
from contextlib import contextmanager

@contextmanager
def robust_robot_connection(host, port, timeout=10, max_retries=5):
    """Robust robot connection with retry logic"""
    connection = None
    
    for attempt in range(max_retries):
        try:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.settimeout(timeout)
            connection.connect((host, port))
            
            print(f"Connected to robot at {host}:{port}")
            yield connection
            break
            
        except (ConnectionRefusedError, socket.timeout) as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            if connection:
                connection.close()
                connection = None
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise ConnectionError(f"Failed to connect after {max_retries} attempts")
        
        except Exception as e:
            if connection:
                connection.close()
            raise e
    
    finally:
        if connection:
            connection.close()

# Usage
try:
    with robust_robot_connection('192.168.1.100', 30002) as conn:
        # Send robot commands
        conn.send(b"get_actual_tcp_pose()\n")
        response = conn.recv(1024)
except ConnectionError as e:
    print(f"Robot connection failed: {e}")
```

2. **Hardware Health Monitoring:**
```python
class HardwareMonitor:
    """Monitor hardware connections and status"""
    
    def __init__(self):
        self.connections = {}
        self.monitoring = False
    
    def add_device(self, name, check_function, critical=True):
        """Add device to monitor"""
        self.connections[name] = {
            'check_function': check_function,
            'critical': critical,
            'status': 'unknown',
            'last_check': None,
            'error_count': 0
        }
    
    def check_all_devices(self):
        """Check status of all monitored devices"""
        results = {}
        
        for name, device in self.connections.items():
            try:
                status = device['check_function']()
                device['status'] = 'online' if status else 'offline'
                device['error_count'] = 0
            except Exception as e:
                device['status'] = 'error'
                device['error_count'] += 1
                print(f"Device {name} error: {e}")
            
            device['last_check'] = time.time()
            results[name] = device['status']
        
        return results
    
    def get_system_health(self):
        """Get overall system health status"""
        statuses = [dev['status'] for dev in self.connections.values()]
        critical_devices = [name for name, dev in self.connections.items() 
                          if dev['critical'] and dev['status'] != 'online']
        
        if critical_devices:
            return 'critical', critical_devices
        elif 'error' in statuses:
            return 'degraded', []
        else:
            return 'healthy', []

# Example usage
def check_robot_connection():
    """Example device check function"""
    try:
        # Attempt simple connection test
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect(('192.168.1.100', 30002))
        return True
    except:
        return False

monitor = HardwareMonitor()
monitor.add_device('robot_arm', check_robot_connection, critical=True)
health_status, issues = monitor.get_system_health()
```

## Mathematical and Numerical Issues

### Issue: Numerical Instability

**Symptoms:**
```
RuntimeWarning: invalid value encountered in divide
RuntimeWarning: overflow encountered in exp
nan values in computation results
```

**Solutions:**

1. **Safe Mathematical Operations:**
```python
import numpy as np
from numpy import inf, nan

def safe_divide(numerator, denominator, fallback=0.0, epsilon=1e-12):
    """Safe division with fallback for zero denominator"""
    denominator = np.asarray(denominator)
    mask = np.abs(denominator) < epsilon
    
    result = np.divide(numerator, denominator, 
                      out=np.full_like(numerator, fallback), 
                      where=~mask)
    return result

def safe_sqrt(x, epsilon=1e-12):
    """Safe square root that handles negative values"""
    x = np.asarray(x)
    return np.sqrt(np.maximum(x, epsilon))

def safe_arccos(x, epsilon=1e-8):
    """Safe arccos that handles values outside [-1, 1]"""
    x = np.asarray(x)
    x = np.clip(x, -1 + epsilon, 1 - epsilon)
    return np.arccos(x)

def safe_log(x, epsilon=1e-12):
    """Safe logarithm that handles zero and negative values"""
    x = np.asarray(x)
    return np.log(np.maximum(x, epsilon))
```

2. **Numerical Stability in SE(3) Operations:**
```python
def stable_rodrigues_formula(omega, threshold=1e-8):
    """Numerically stable Rodrigues formula"""
    theta = np.linalg.norm(omega)
    
    if theta < threshold:
        # Use Taylor series expansion for small angles
        # R = I + [omega]_× + (1/2)[omega]_×² + ...
        omega_skew = skew_symmetric(omega)
        R = (np.eye(3) + omega_skew + 
             0.5 * omega_skew @ omega_skew)
    else:
        # Standard Rodrigues formula
        axis = omega / theta
        axis_skew = skew_symmetric(axis)
        
        R = (np.eye(3) + 
             np.sin(theta) * axis_skew +
             (1 - np.cos(theta)) * axis_skew @ axis_skew)
    
    # Ensure orthogonality
    return orthogonalize_matrix(R)

def orthogonalize_matrix(R, method='qr'):
    """Orthogonalize matrix using stable methods"""
    if method == 'qr':
        Q, _ = np.linalg.qr(R)
        return Q
    elif method == 'svd':
        U, _, Vt = np.linalg.svd(R)
        return U @ Vt
    else:
        raise ValueError(f"Unknown method: {method}")
```

### Issue: Constraint Solver Convergence

**Symptoms:**
```
ConvergenceWarning: Maximum iterations reached without convergence
Optimization failed: Constraint solver did not converge
```

**Solutions:**

1. **Adaptive Convergence Criteria:**
```python
class AdaptiveConstraintSolver:
    def __init__(self):
        self.max_iterations = 1000
        self.tolerance = 1e-6
        self.adaptive_tolerance = True
    
    def solve_constraints(self, initial_poses, constraints):
        """Solve constraints with adaptive convergence"""
        poses = initial_poses.copy()
        tolerance = self.tolerance
        
        for iteration in range(self.max_iterations):
            # Compute constraint violations
            violations = self.compute_violations(poses, constraints)
            max_violation = np.max(np.abs(violations))
            
            # Adaptive tolerance adjustment
            if self.adaptive_tolerance and iteration > 100:
                if max_violation < tolerance * 10:
                    tolerance *= 1.5  # Relax tolerance
                elif max_violation > tolerance * 100:
                    tolerance *= 0.5  # Tighten tolerance
            
            # Check convergence
            if max_violation < tolerance:
                return poses, True, iteration
            
            # Update poses
            gradients = self.compute_gradients(poses, constraints)
            step_size = self.line_search(poses, gradients, constraints)
            poses = self.update_poses(poses, gradients, step_size)
            
            # Progress monitoring
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: max violation = {max_violation:.6f}")
        
        return poses, False, self.max_iterations
```

2. **Multi-Scale Optimization:**
```python
def multiscale_optimization(poses, constraints):
    """Optimize at multiple scales for better convergence"""
    scales = [10.0, 5.0, 2.0, 1.0, 0.5]  # Coarse to fine
    
    current_poses = poses.copy()
    
    for scale in scales:
        print(f"Optimizing at scale {scale}")
        
        # Scale constraints
        scaled_constraints = scale_constraints(constraints, scale)
        
        # Optimize at current scale
        solver = ConstraintSolver(max_iterations=200)
        current_poses, converged, iterations = solver.solve(
            current_poses, scaled_constraints
        )
        
        if not converged:
            print(f"Warning: Scale {scale} did not converge")
        
        print(f"Scale {scale}: converged in {iterations} iterations")
    
    return current_poses

def scale_constraints(constraints, scale_factor):
    """Scale constraint tolerances and weights"""
    scaled = []
    
    for constraint in constraints:
        new_constraint = constraint.copy()
        
        # Scale tolerances
        if 'tolerance' in new_constraint.parameters:
            new_constraint.parameters['tolerance'] *= scale_factor
        
        # Adjust weights (inverse scaling)
        new_constraint.priority /= scale_factor
        
        scaled.append(new_constraint)
    
    return scaled
```

## Hardware and Environment Issues

### Issue: GPU Memory Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
RuntimeError: CUDA error: device-side assert triggered
```

**Solutions:**

1. **GPU Memory Management:**
```python
import torch
import gc

class GPUMemoryManager:
    def __init__(self):
        self.memory_fraction = 0.8  # Use 80% of GPU memory
        self.cleanup_threshold = 0.9  # Clean when 90% used
    
    def setup_gpu_memory(self):
        """Setup GPU memory management"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Set memory fraction if supported
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
    
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return 0, 0
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return allocated, cached, total
    
    def cleanup_if_needed(self):
        """Clean up GPU memory if usage is high"""
        if not torch.cuda.is_available():
            return
        
        allocated, cached, total = self.get_memory_usage()
        usage_ratio = cached / total
        
        if usage_ratio > self.cleanup_threshold:
            print(f"GPU memory usage high ({usage_ratio:.1%}), cleaning up...")
            gc.collect()
            torch.cuda.empty_cache()
            
            # Check after cleanup
            allocated, cached, total = self.get_memory_usage()
            print(f"After cleanup: {cached/total:.1%} GPU memory used")
    
    @contextmanager
    def gpu_memory_context(self):
        """Context manager for GPU memory operations"""
        try:
            self.setup_gpu_memory()
            yield
        finally:
            self.cleanup_if_needed()

# Usage
memory_manager = GPUMemoryManager()

with memory_manager.gpu_memory_context():
    # Your GASM operations here
    bridge = create_bridge({'device': 'cuda'})
    result = bridge.process("complex instruction")
```

2. **Batch Size Adaptation:**
```python
def adaptive_batch_processing(data, initial_batch_size=32):
    """Adaptively reduce batch size if GPU memory runs out"""
    batch_size = initial_batch_size
    
    while batch_size >= 1:
        try:
            # Process in batches
            results = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_result = process_batch_gpu(batch)
                results.extend(batch_result)
            
            return results
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                batch_size //= 2
                print(f"GPU OOM, reducing batch size to {batch_size}")
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise e
    
    raise RuntimeError("Could not process even with batch_size=1")
```

### Issue: Multi-Threading Problems

**Symptoms:**
```
RuntimeError: CUDA context error
Threading conflicts in PyTorch operations
Deadlocks in multi-process execution
```

**Solutions:**

1. **Thread-Safe GASM Operations:**
```python
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class ThreadSafeGASM:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.thread_local = threading.local()
        self.lock = threading.Lock()
    
    def get_bridge(self):
        """Get thread-local GASM bridge"""
        if not hasattr(self.thread_local, 'bridge'):
            # Create bridge for this thread
            config = {
                'device': 'cpu',  # Use CPU for thread safety
                'fallback_mode': True
            }
            self.thread_local.bridge = create_bridge(config)
        
        return self.thread_local.bridge
    
    def process_single(self, instruction):
        """Process single instruction in thread-safe manner"""
        bridge = self.get_bridge()
        
        with self.lock:  # Ensure thread safety
            result = bridge.process(instruction)
        
        return result
    
    def process_multiple(self, instructions):
        """Process multiple instructions concurrently"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_instruction = {
                executor.submit(self.process_single, instruction): instruction
                for instruction in instructions
            }
            
            # Collect results
            for future in as_completed(future_to_instruction):
                instruction = future_to_instruction[future]
                try:
                    result = future.result(timeout=30)
                    results.append((instruction, result))
                except Exception as e:
                    print(f"Error processing '{instruction}': {e}")
                    results.append((instruction, {'success': False, 'error': str(e)}))
        
        return results
```

2. **Process-Based Parallelism:**
```python
import multiprocessing as mp
from multiprocessing import Pool, Queue, Process

def process_instruction_worker(instruction_queue, result_queue):
    """Worker process for instruction processing"""
    # Create bridge in worker process
    bridge = create_bridge({'device': 'cpu'})
    
    while True:
        try:
            instruction = instruction_queue.get(timeout=1)
            if instruction is None:  # Poison pill
                break
            
            result = bridge.process(instruction)
            result_queue.put((instruction, result))
            
        except Exception as e:
            result_queue.put((instruction, {'success': False, 'error': str(e)}))

def parallel_processing_multiprocess(instructions, num_processes=4):
    """Process instructions using multiple processes"""
    instruction_queue = Queue()
    result_queue = Queue()
    
    # Add instructions to queue
    for instruction in instructions:
        instruction_queue.put(instruction)
    
    # Add poison pills
    for _ in range(num_processes):
        instruction_queue.put(None)
    
    # Start worker processes
    processes = []
    for _ in range(num_processes):
        p = Process(target=process_instruction_worker, 
                   args=(instruction_queue, result_queue))
        p.start()
        processes.append(p)
    
    # Collect results
    results = []
    for _ in instructions:
        result = result_queue.get()
        results.append(result)
    
    # Wait for processes to finish
    for p in processes:
        p.join()
    
    return results
```

## Debugging Tools and Techniques

### Interactive Debugging

```python
def interactive_debug_session():
    """Start interactive debugging session for GASM"""
    import pdb
    import IPython
    
    # Setup debugging environment
    bridge = create_bridge({'debug': True, 'device': 'cpu'})
    
    print("🐛 GASM Interactive Debug Session")
    print("Available objects:")
    print("  bridge - GASM bridge instance")
    print("  test_instruction() - test with sample instruction")
    print("  analyze_result(result) - analyze processing result")
    
    def test_instruction(text="place box above table"):
        """Test instruction processing with debugging"""
        print(f"Testing: '{text}'")
        result = bridge.process(text)
        print(f"Success: {result['success']}")
        if not result['success']:
            print(f"Error: {result.get('error_message')}")
        return result
    
    def analyze_result(result):
        """Analyze processing result"""
        print("Result Analysis:")
        print(f"  Success: {result['success']}")
        print(f"  Constraints: {len(result.get('constraints', []))}")
        print(f"  Poses: {len(result.get('target_poses', {}))}")
        print(f"  Confidence: {result.get('confidence', 0):.3f}")
        print(f"  Time: {result.get('execution_time', 0):.3f}s")
        
        if result.get('debug_info'):
            print("Debug Info:")
            for key, value in result['debug_info'].items():
                print(f"  {key}: {value}")
    
    # Start IPython session with local variables
    IPython.embed(locals())

# Usage: Call this function to start debugging
# interactive_debug_session()
```

### Logging Configuration

```python
import logging
import sys
from datetime import datetime

def setup_comprehensive_logging():
    """Setup comprehensive logging for GASM debugging"""
    
    # Create logger
    logger = logging.getLogger('gasm')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler
    log_filename = f"gasm_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    
    # Formatters
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    print(f"Debug logging enabled. Log file: {log_filename}")
    return logger

# Enable debug logging
debug_logger = setup_comprehensive_logging()
```

## FAQ

### Q: GASM processing returns success=False with no error message

**A:** This usually indicates a silent failure in the processing pipeline. Enable debug logging and check:

```python
bridge = create_bridge({
    'debug': True,
    'fallback_mode': True,
    'verbose': True
})

result = bridge.process(your_text)
print("Debug info:", result.get('debug_info'))
```

### Q: SE(3) operations produce NaN or infinite values

**A:** Check input data ranges and use safe mathematical operations:

```python
# Validate input poses
def validate_poses(poses):
    for i, pose in enumerate(poses):
        if np.any(np.isnan(pose)) or np.any(np.isinf(pose)):
            raise ValueError(f"Pose {i} contains NaN or infinite values")
        
        if pose.shape != (4, 4):
            raise ValueError(f"Pose {i} has wrong shape: {pose.shape}")

# Use safe operations
from utils_se3 import SE3Utils
SE3Utils.validate_homogeneous_matrix(pose, tolerance=1e-4)
```

### Q: High memory usage that doesn't decrease

**A:** Implement explicit memory management:

```python
import gc
import torch

def cleanup_memory():
    """Clean up memory after GASM operations"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Use after batch operations
for i in range(num_batches):
    # ... process batch ...
    if i % 10 == 0:  # Clean every 10 batches
        cleanup_memory()
```

### Q: Constraints seem to be ignored

**A:** Check constraint priorities and compatibility:

```python
def debug_constraints(constraints):
    """Debug constraint application"""
    print(f"Number of constraints: {len(constraints)}")
    
    for i, constraint in enumerate(constraints):
        print(f"Constraint {i}:")
        print(f"  Type: {constraint.type}")
        print(f"  Priority: {constraint.priority}")
        print(f"  Active: {constraint.active}")
        print(f"  Parameters: {constraint.parameters}")
        
        # Check if constraint is reasonable
        if constraint.priority < 0.1:
            print(f"  ⚠️  Very low priority, may be ignored")
        
        if not constraint.active:
            print(f"  ⚠️  Constraint is inactive")
```

### Q: Robot integration fails intermittently

**A:** Implement robust communication:

```python
def robust_robot_command(command, max_retries=3):
    """Send robot command with retry logic"""
    for attempt in range(max_retries):
        try:
            response = send_robot_command(command)
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
```

This troubleshooting guide should help you identify and resolve most common issues with the GASM system. For complex problems not covered here, consider enabling debug logging and examining the detailed error information provided.