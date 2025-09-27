# CLI Validation Report
==================================================

## Executive Summary
- **Total Tests:** 30
- **Passed:** 16 ✅
- **Failed:** 2 ❌
- **Skipped:** 12 ⏭️
- **Success Rate:** 53.3%

## Environment
- **Platform:** Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39
- **Python Version:** 3.12.3
- **Python Executable:** /usr/bin/python3
- **Project Root:** /mnt/c/dev/coding/GASM-Roboting

## Scripts Discovered
- ✅ **agent_loop_2d**: `/mnt/c/dev/coding/GASM-Roboting/src/spatial_agent/agent_loop_2d.py`
- ✅ **agent_loop_pybullet**: `/mnt/c/dev/coding/GASM-Roboting/src/spatial_agent/agent_loop_pybullet.py`
- ✅ **demo**: `/mnt/c/dev/coding/GASM-Roboting/src/spatial_agent/demo.py`
- ✅ **demo_complete**: `/mnt/c/dev/coding/GASM-Roboting/src/spatial_agent/demo_complete.py`
- ✅ **run_demo**: `/mnt/c/dev/coding/GASM-Roboting/src/spatial_agent/run_demo.py`
- ✅ **run_spatial_agent**: `/mnt/c/dev/coding/GASM-Roboting/src/spatial_agent/run_spatial_agent.py`
- ✅ **test_agent_2d**: `/mnt/c/dev/coding/GASM-Roboting/src/spatial_agent/test_agent_2d.py`
- ✅ **test_pybullet_agent**: `/mnt/c/dev/coding/GASM-Roboting/src/spatial_agent/test_pybullet_agent.py`
- ✅ **validate_metrics**: `/mnt/c/dev/coding/GASM-Roboting/src/spatial_agent/validate_metrics.py`

## Results by Category
### File Existence
**Status:** 9/9 passed, 0 failed, 0 skipped

### Help Messages
**Status:** 4/9 passed, 1 failed, 4 skipped
**Critical Issues:**
- `help_test_pybullet_agent`: test_pose_creation (__main__.TestSE3Pose.test_pose_creation)
Test SE3Pose creation and normalization ... ok
test_pose_matrix_conversion (__main__.Test
**Dependency Issues:**
- `help_agent_loop_2d`: Missing dependencies (acceptable in some environments)
- `help_demo_complete`: Missing dependencies (acceptable in some environments)
- `help_test_agent_2d`: Missing dependencies (acceptable in some environments)

### Argument Validation
**Status:** 0/5 passed, 0 failed, 5 skipped
**Dependency Issues:**
- `arg_validation_no_arguments`: Missing Python dependencies
- `arg_validation_missing_text_argument`: Missing Python dependencies
- `arg_validation_invalid_steps_negative`: Missing Python dependencies

### Pybullet Interface
**Status:** 1/2 passed, 1 failed, 0 skipped
**Critical Issues:**
- `pybullet_headless_mode`: usage: agent_loop_pybullet.py [-h] [--text TEXT] [--steps STEPS]
                              [--use_vision] [--render {headless,gui,record}]
       

### Cross Platform
**Status:** 2/2 passed, 0 failed, 0 skipped

### Entry Points
**Status:** 0/3 passed, 0 failed, 3 skipped

## Recommendations
### 🔧 Needs Attention
Several issues need to be addressed before production use.

### Immediate Actions Needed:
- 🐛 Fix failed test cases
- 📝 Improve error messages and user feedback
- ✅ Add more comprehensive input validation
### Environment Setup:
- 📦 Install missing dependencies: `pip install torch matplotlib pybullet opencv-python`
- 🐳 Consider using containerized environments for consistent testing
- 🔧 Set up proper development environment

### Development Best Practices:
- 🧪 Implement continuous integration testing
- 📚 Create comprehensive user documentation
- 🔄 Add automated regression testing
- 🎯 Test on multiple Python versions and platforms
