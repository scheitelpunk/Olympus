# OLYMPUS Comprehensive Test Suite

## 🎯 Overview

This comprehensive test suite validates all aspects of Project OLYMPUS with special focus on **safety-critical components** that must achieve 100% test coverage to ensure human safety is never compromised.

## 🚨 Critical Safety Requirements

### 100% Coverage Required Components
- **Asimov Kernel** (`olympus/ethical_core/asimov_kernel.py`) - Ethical decision engine
- **Action Filter** (`olympus/safety_layer/action_filter.py`) - Multi-layer safety validation
- **Human Protection** (`olympus/safety_layer/human_protection.py`) - Human safety systems
- **Fail-Safe Systems** (`olympus/safety_layer/fail_safe.py`) - Emergency procedures

### Coverage Requirements by Category
- **Safety Tests**: 100% coverage (MANDATORY)
- **Ethical Tests**: 100% coverage (MANDATORY)
- **Unit Tests**: >90% coverage (REQUIRED)
- **Integration Tests**: >85% coverage (REQUIRED)
- **Performance Tests**: >70% coverage (RECOMMENDED)

## 📁 Test Suite Structure

```
tests/
├── conftest.py                 # Global test configuration and fixtures
├── pytest.ini                 # PyTest configuration
├── requirements-test.txt       # Test dependencies
├── unit/                       # Unit tests (90% coverage required)
│   ├── ethical_core/          # Asimov Kernel tests (100% coverage)
│   ├── safety_layer/          # Safety system tests (100% coverage)
│   ├── core/                  # Core orchestrator tests
│   └── modules/               # Module-specific tests
├── integration/               # Integration tests (85% coverage)
│   └── test_system_integration.py
├── performance/               # Performance and load tests
│   └── test_load_testing.py
├── stress/                    # Failure injection and chaos tests
│   └── test_failure_injection.py
├── reports/                   # Test reports and coverage data
└── fixtures/                  # Test data and mock objects
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Install project dependencies
pip install -r requirements.txt
```

### 2. Run All Tests
```bash
# Run comprehensive test suite
python scripts/run_tests.py

# Run specific categories
python scripts/run_tests.py --categories safety ethical unit
```

### 3. Generate Coverage Report
```bash
# Generate HTML coverage report
pytest --cov=src/olympus --cov-report=html tests/

# View coverage report
open tests/reports/htmlcov/index.html
```

## 📊 Test Categories

### 🛡️ Safety-Critical Tests (`@pytest.mark.safety`)
**Coverage Requirement: 100%**

Tests for components that directly impact human safety:
- Asimov Kernel integrity validation
- Action filtering and blocking
- Emergency stop procedures
- Human proximity detection
- Fail-safe mechanism activation

```bash
# Run only safety tests
pytest -m safety --cov-fail-under=100
```

### ⚖️ Ethical Compliance Tests (`@pytest.mark.ethical`)
**Coverage Requirement: 100%**

Tests for ethical decision-making systems:
- Three Laws of Robotics enforcement
- Human override validation
- Ethical conflict resolution
- Moral reasoning verification

```bash
# Run only ethical tests
pytest -m ethical --cov-fail-under=100
```

### 🧪 Unit Tests (`@pytest.mark.unit`)
**Coverage Requirement: >90%**

Isolated component testing:
- Individual function testing
- Class method validation
- Edge case handling
- Error condition testing

```bash
# Run unit tests
pytest tests/unit/ --cov-fail-under=90
```

### 🔗 Integration Tests (`@pytest.mark.integration`)
**Coverage Requirement: >85%**

Cross-component interaction testing:
- Component integration flows
- End-to-end scenarios
- Communication protocols
- System coordination

```bash
# Run integration tests
pytest tests/integration/ --cov-fail-under=85
```

### 🚄 Performance Tests (`@pytest.mark.performance`)
**Coverage Requirement: >70%**

System performance validation:
- Response time benchmarks
- Throughput measurements
- Resource usage monitoring
- Scalability testing

```bash
# Run performance tests with benchmarks
pytest tests/performance/ --benchmark-only
```

### 💥 Stress Tests (`@pytest.mark.stress`)
**Failure injection and chaos testing:**
- Component failure simulation
- Network partition testing
- Resource exhaustion scenarios
- Race condition detection

```bash
# Run stress tests
pytest tests/stress/ --maxfail=5
```

## 🎛️ Test Execution Options

### Basic Execution
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/olympus

# Run specific test file
pytest tests/unit/ethical_core/test_asimov_kernel.py
```

### Advanced Options
```bash
# Parallel execution
pytest -n auto

# Fail fast (stop on first failure)
pytest -x

# Verbose output
pytest -v

# Show local variables in tracebacks
pytest -l

# Run only failed tests from last run
pytest --lf

# Generate multiple report formats
pytest --cov=src/olympus \
       --cov-report=html \
       --cov-report=xml \
       --cov-report=term-missing \
       --html=tests/reports/report.html
```

### Test Selection
```bash
# Run by markers
pytest -m "safety or ethical"
pytest -m "not performance"

# Run by keywords
pytest -k "asimov"
pytest -k "filter and not integration"

# Run by test node ID
pytest tests/unit/ethical_core/test_asimov_kernel.py::TestAsimovKernelInitialization::test_kernel_initialization_default
```

## 📋 Test Configuration

### PyTest Configuration (`pytest.ini`)
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    safety: Safety-critical tests (must achieve 100% coverage)
    ethical: Ethical compliance tests (must achieve 100% coverage)
    slow: Slow running tests (>1 second)
```

### Coverage Configuration
```ini
[coverage:run]
source = src/olympus
omit = 
    */tests/*
    */test_*
    */__pycache__/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    if __name__ == .__main__.:
```

## 🔧 Custom Test Utilities

### Fixtures (`conftest.py`)
- `asimov_kernel`: Pre-configured AsimovKernel instance
- `action_filter`: Pre-configured ActionFilter instance
- `orchestrator`: Initialized OLYMPUS orchestrator
- `safe_action_context`: Valid action context for testing
- `unsafe_action_context`: Invalid action context for testing
- `performance_metrics`: Performance measurement utilities

### Test Data Generators
- `TestDataGenerator.generate_action_contexts(count)`
- `TestDataGenerator.generate_action_requests(count)`
- Mock objects for various components

### Async Test Helpers
- `async_wait_for_condition(condition, timeout)`
- `create_mock_component(name, methods)`

## 📊 Coverage Analysis

### View Coverage Report
```bash
# Generate and view HTML report
pytest --cov=src/olympus --cov-report=html tests/
open tests/reports/htmlcov/index.html
```

### Check Specific Component Coverage
```bash
# Check Asimov Kernel coverage (must be 100%)
pytest --cov=src/olympus/ethical_core/asimov_kernel --cov-report=term-missing tests/unit/ethical_core/

# Check Action Filter coverage (must be 100%)
pytest --cov=src/olympus/safety_layer/action_filter --cov-report=term-missing tests/unit/safety_layer/
```

### Coverage Requirements Check
```bash
# Verify safety-critical coverage
python scripts/run_tests.py --categories safety ethical
```

## 🚨 Failure Analysis

### Common Failure Scenarios
1. **Safety Test Failures**: Critical - deployment blocked
2. **Coverage Below Requirements**: Warning - investigate missing tests
3. **Performance Regressions**: Monitor - may require optimization
4. **Intermittent Failures**: Debug - check race conditions

### Debug Test Failures
```bash
# Run with debugging
pytest --pdb tests/unit/ethical_core/test_asimov_kernel.py

# Show full traceback
pytest --tb=long

# Show local variables
pytest -l

# Capture and show print statements
pytest -s
```

## 🔄 Continuous Integration

### GitHub Actions Workflow
The test suite integrates with GitHub Actions for automated testing:

1. **Safety-Critical Tests**: Run first, block on failure
2. **Unit/Integration Tests**: Comprehensive coverage analysis
3. **Performance Tests**: Benchmark tracking
4. **Security Tests**: Vulnerability scanning
5. **Stress Tests**: Chaos engineering validation

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## 📈 Performance Benchmarks

### Benchmark Execution
```bash
# Run performance benchmarks
pytest tests/performance/ --benchmark-only --benchmark-sort=mean

# Save benchmark results
pytest tests/performance/ --benchmark-json=benchmarks.json
```

### Performance Requirements
- **Asimov Evaluation**: <1ms per evaluation
- **Action Filtering**: <10ms per action
- **System Response**: <100ms for simple operations
- **Throughput**: >100 actions/second under normal load

## 🛠️ Test Development Guidelines

### Writing Safety-Critical Tests
1. **100% Coverage Required**: Every line must be tested
2. **Edge Cases**: Test all boundary conditions
3. **Failure Modes**: Test system behavior under failures
4. **Human Safety**: Validate all human protection mechanisms

### Test Naming Convention
```python
def test_[component]_[scenario]_[expected_outcome]():
    """
    Test [component] [scenario] results in [expected_outcome].
    
    This test validates that [detailed description].
    """
```

### Test Structure (Arrange-Act-Assert)
```python
def test_asimov_kernel_first_law_blocks_harmful_action():
    """Test First Law blocks physically harmful actions."""
    # Arrange
    kernel = AsimovKernel()
    harmful_context = ActionContext(
        action_type=ActionType.PHYSICAL,
        description="Harmful action that could injure humans"
    )
    
    # Act
    result = kernel.evaluate_action(harmful_context)
    
    # Assert
    assert result.result == EthicalResult.DENIED
    assert 1 in result.violated_laws  # First Law violation
    
    # Cleanup
    kernel.stop_integrity_monitoring()
```

## 📞 Support and Troubleshooting

### Common Issues

#### Test Discovery Problems
```bash
# Fix Python path issues
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

# Verify test discovery
pytest --collect-only
```

#### Coverage Issues
```bash
# Clean coverage data
coverage erase

# Debug coverage collection
pytest --cov=src/olympus --cov-report=term-missing -v
```

#### Performance Test Issues
```bash
# Install performance dependencies
pip install pytest-benchmark psutil

# Run single performance test
pytest tests/performance/test_load_testing.py::TestAsimovKernelPerformance::test_single_evaluation_performance -v
```

### Getting Help
1. Check test logs in `tests/reports/`
2. Run with verbose output (`pytest -v`)
3. Use debugger for complex failures (`pytest --pdb`)
4. Review GitHub Actions logs for CI failures

## 🎯 Test Quality Metrics

### Success Criteria
- ✅ **All safety tests pass** (100% success rate)
- ✅ **Safety components have 100% coverage**
- ✅ **Overall coverage >90%**
- ✅ **No critical security vulnerabilities**
- ✅ **Performance benchmarks within acceptable ranges**

### Quality Gates
1. **Safety Gate**: 100% safety/ethical test coverage
2. **Quality Gate**: >90% overall test coverage
3. **Performance Gate**: All benchmarks pass requirements
4. **Security Gate**: No high-severity vulnerabilities

---

## 🚀 Ready to Test?

Run the complete test suite:
```bash
python scripts/run_tests.py
```

**Remember: OLYMPUS safety depends on comprehensive testing. Every test matters for human safety!**