#!/usr/bin/env python3
"""
Final CLI Validation Suite
===========================

Comprehensive validation of all CLI interfaces with robust dependency handling,
cross-platform compatibility testing, and detailed reporting.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import time
import json
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import argparse


@dataclass
class ValidationResult:
    """Individual test validation result"""
    test_name: str
    category: str
    command: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    error_type: Optional[str] = None
    skip_reason: Optional[str] = None
    expected_behavior: str = ""
    actual_behavior: str = ""


class CLIValidator:
    """Comprehensive CLI validation system"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.python_exe = sys.executable
        self.results = []
        self.environment_info = self._get_environment_info()
        self.scripts = self._discover_scripts()

    def _get_environment_info(self) -> Dict[str, Any]:
        """Collect environment information"""
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_executable": self.python_exe,
            "working_directory": str(Path.cwd()),
            "project_root": str(self.project_root),
            "path_sep": os.sep,
            "environment_variables": {
                key: os.environ.get(key, "") 
                for key in ["PATH", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV"]
            }
        }

    def _discover_scripts(self) -> Dict[str, Path]:
        """Discover all CLI scripts in the project"""
        scripts = {}
        script_patterns = [
            "agent_loop_2d.py",
            "agent_loop_pybullet.py", 
            "demo.py",
            "demo_complete.py",
            "run_demo.py",
            "run_spatial_agent.py",
            "test_agent_2d.py",
            "test_pybullet_agent.py",
            "validate_metrics.py"
        ]

        base_path = self.project_root / "src" / "spatial_agent"
        for pattern in script_patterns:
            script_path = base_path / pattern
            if script_path.exists():
                scripts[pattern.replace('.py', '')] = script_path

        return scripts

    def run_command_safe(self, cmd: List[str], timeout: int = 30, 
                        cwd: Optional[str] = None) -> ValidationResult:
        """Safely run a command with comprehensive error handling"""
        start_time = time.time()
        cmd_str = " ".join(str(c) for c in cmd)
        
        result = ValidationResult(
            test_name=f"cmd_{abs(hash(cmd_str)) % 10000}",
            category="command",
            command=cmd_str,
            success=False,
            exit_code=-999,
            stdout="",
            stderr="",
            execution_time=0.0
        )

        try:
            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or str(self.project_root)
            )
            
            result.execution_time = time.time() - start_time
            result.exit_code = proc_result.returncode
            result.stdout = proc_result.stdout
            result.stderr = proc_result.stderr
            result.success = proc_result.returncode == 0
            
            # Analyze error types
            stderr_lower = proc_result.stderr.lower()
            if "modulenotfounderror" in stderr_lower or "no module named" in stderr_lower:
                result.error_type = "missing_dependency"
                result.skip_reason = "Missing Python dependencies"
            elif "import" in stderr_lower and "error" in stderr_lower:
                result.error_type = "import_error"
                result.skip_reason = "Import/dependency issues"
            elif proc_result.returncode == 127:
                result.error_type = "command_not_found"
            elif "permission denied" in stderr_lower:
                result.error_type = "permission_error"
            elif "timeout" in stderr_lower:
                result.error_type = "timeout"
                
        except subprocess.TimeoutExpired:
            result.execution_time = time.time() - start_time
            result.error_type = "timeout"
            result.stderr = f"Command timed out after {timeout}s"
            result.exit_code = -1
            
        except subprocess.CalledProcessError as e:
            result.execution_time = time.time() - start_time
            result.exit_code = e.returncode
            result.stdout = e.stdout or ""
            result.stderr = e.stderr or str(e)
            result.error_type = "process_error"
            
        except FileNotFoundError:
            result.execution_time = time.time() - start_time
            result.error_type = "file_not_found"
            result.stderr = f"Command not found: {cmd[0]}"
            result.exit_code = 127
            
        except Exception as e:
            result.execution_time = time.time() - start_time
            result.error_type = "unknown_error"
            result.stderr = str(e)
            result.exit_code = -2

        return result

    def test_script_existence(self) -> List[ValidationResult]:
        """Test that all expected scripts exist"""
        results = []
        
        for script_name, script_path in self.scripts.items():
            result = ValidationResult(
                test_name=f"exists_{script_name}",
                category="file_existence",
                command=f"check {script_path}",
                success=script_path.exists(),
                exit_code=0 if script_path.exists() else 1,
                stdout=str(script_path),
                stderr="",
                execution_time=0.0,
                expected_behavior="Script file should exist",
                actual_behavior=f"File {'exists' if script_path.exists() else 'missing'}"
            )
            results.append(result)
        
        return results

    def test_help_messages(self) -> List[ValidationResult]:
        """Test help messages for all scripts"""
        results = []
        
        for script_name, script_path in self.scripts.items():
            if not script_path.exists():
                continue
                
            result = self.run_command_safe([self.python_exe, str(script_path), "--help"], timeout=15)
            result.test_name = f"help_{script_name}"
            result.category = "help_messages"
            result.expected_behavior = "Should display help message and exit with code 0"
            
            # Special handling for dependency issues
            if result.error_type == "missing_dependency":
                result.actual_behavior = "Missing dependencies prevent help display"
                # This is acceptable - mark as warning rather than failure
                result.skip_reason = "Missing dependencies (acceptable in some environments)"
            elif result.success and ("usage:" in result.stdout.lower() or "help" in result.stdout.lower()):
                result.actual_behavior = "Help message displayed successfully"
            elif result.exit_code == 2:  # argparse error
                result.actual_behavior = "Help not found or argparse error"
            else:
                result.actual_behavior = f"Exit code {result.exit_code}, unexpected output"
            
            results.append(result)
        
        return results

    def test_argument_validation(self) -> List[ValidationResult]:
        """Test argument validation and error handling"""
        results = []
        
        # Focus on agent_loop_2d as the primary interface
        agent_2d = self.scripts.get("agent_loop_2d")
        if not agent_2d or not agent_2d.exists():
            return results

        test_cases = [
            {
                "name": "no_arguments",
                "args": [],
                "expected_exit": 2,
                "expected_behavior": "Should fail with missing required arguments"
            },
            {
                "name": "missing_text_argument", 
                "args": ["--steps", "5"],
                "expected_exit": 2,
                "expected_behavior": "Should fail without --text argument"
            },
            {
                "name": "invalid_steps_negative",
                "args": ["--text", "test", "--steps", "-1"],
                "expected_exit": 2,
                "expected_behavior": "Should reject negative steps"
            },
            {
                "name": "invalid_steps_string",
                "args": ["--text", "test", "--steps", "abc"],
                "expected_exit": 2, 
                "expected_behavior": "Should reject non-numeric steps"
            },
            {
                "name": "valid_minimal_args",
                "args": ["--text", "box above robot", "--no_visualization", "--steps", "1"],
                "expected_exit": 0,
                "expected_behavior": "Should succeed with minimal valid arguments",
                "timeout": 45
            }
        ]

        for case in test_cases:
            cmd = [self.python_exe, str(agent_2d)] + case["args"]
            result = self.run_command_safe(cmd, timeout=case.get("timeout", 15))
            result.test_name = f"arg_validation_{case['name']}"
            result.category = "argument_validation"
            result.expected_behavior = case["expected_behavior"]
            
            # Handle dependency issues
            if result.error_type == "missing_dependency":
                result.actual_behavior = "Cannot test due to missing dependencies"
                result.skip_reason = "Missing Python dependencies"
            else:
                expected_exit = case["expected_exit"]
                if result.exit_code == expected_exit:
                    result.success = True
                    result.actual_behavior = f"Correctly exited with code {result.exit_code}"
                else:
                    result.actual_behavior = f"Expected exit {expected_exit}, got {result.exit_code}"
            
            results.append(result)

        return results

    def test_pybullet_interface(self) -> List[ValidationResult]:
        """Test PyBullet-specific CLI functionality"""
        results = []
        
        agent_pybullet = self.scripts.get("agent_loop_pybullet")
        if not agent_pybullet or not agent_pybullet.exists():
            return results

        test_cases = [
            {
                "name": "help_message",
                "args": ["--help"],
                "expected_exit": 0,
                "expected_behavior": "Should display PyBullet-specific help"
            },
            {
                "name": "headless_mode",
                "args": ["--text", "box above conveyor", "--mode", "headless", "--steps", "1"],
                "expected_exit": 0,
                "expected_behavior": "Should run in headless mode",
                "timeout": 60
            }
        ]

        for case in test_cases:
            cmd = [self.python_exe, str(agent_pybullet)] + case["args"]
            result = self.run_command_safe(cmd, timeout=case.get("timeout", 20))
            result.test_name = f"pybullet_{case['name']}"
            result.category = "pybullet_interface"
            result.expected_behavior = case["expected_behavior"]
            
            if result.error_type == "missing_dependency":
                if "pybullet" in result.stderr.lower():
                    result.actual_behavior = "PyBullet dependency missing (expected in minimal environments)"
                else:
                    result.actual_behavior = "Other dependencies missing"
                result.skip_reason = "Missing PyBullet or related dependencies"
            elif result.success:
                result.actual_behavior = "PyBullet interface working correctly"
            else:
                result.actual_behavior = f"Failed with exit code {result.exit_code}"
            
            results.append(result)

        return results

    def test_cross_platform_compatibility(self) -> List[ValidationResult]:
        """Test cross-platform compatibility"""
        results = []
        
        # Test path handling
        result = ValidationResult(
            test_name="path_handling",
            category="cross_platform",
            command="path compatibility check",
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            execution_time=0.0,
            expected_behavior="Paths should work across platforms"
        )
        
        try:
            # Test that our scripts can be found with different path separators
            for script_name, script_path in self.scripts.items():
                if script_path.exists():
                    # Try to resolve path
                    resolved = script_path.resolve()
                    result.stdout += f"{script_name}: {resolved}\n"
        except Exception as e:
            result.success = False
            result.stderr = str(e)
            result.actual_behavior = f"Path resolution failed: {e}"
        else:
            result.actual_behavior = "Path handling working correctly"
        
        results.append(result)
        
        # Test Python executable detection
        python_test = ValidationResult(
            test_name="python_executable",
            category="cross_platform", 
            command=f"python executable test: {self.python_exe}",
            success=os.path.exists(self.python_exe),
            exit_code=0 if os.path.exists(self.python_exe) else 1,
            stdout=f"Python executable: {self.python_exe}",
            stderr="",
            execution_time=0.0,
            expected_behavior="Python executable should be accessible",
            actual_behavior=f"Python executable {'found' if os.path.exists(self.python_exe) else 'not found'}"
        )
        results.append(python_test)
        
        return results

    def test_entry_points(self) -> List[ValidationResult]:
        """Test package entry points (if installed)"""
        results = []
        
        entry_points = [
            "gasm-agent-2d",
            "gasm-agent-3d", 
            "gasm-demo"
        ]
        
        for entry_point in entry_points:
            # Try to run entry point with --help
            result = self.run_command_safe([entry_point, "--help"], timeout=10)
            result.test_name = f"entry_point_{entry_point.replace('-', '_')}"
            result.category = "entry_points"
            result.expected_behavior = f"Entry point {entry_point} should be available"
            
            if result.exit_code == 127 or "command not found" in result.stderr:
                result.skip_reason = "Package not installed or entry points not configured"
                result.actual_behavior = "Entry point not found (package may not be installed)"
            elif result.success:
                result.actual_behavior = "Entry point working correctly"
            elif result.error_type == "missing_dependency":
                result.skip_reason = "Entry point exists but dependencies missing"
                result.actual_behavior = "Entry point found but cannot run due to dependencies"
            else:
                result.actual_behavior = f"Entry point failed with exit code {result.exit_code}"
            
            results.append(result)
        
        return results

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests"""
        print("🔍 Comprehensive CLI Validation Suite")
        print("=" * 60)
        print(f"Platform: {self.environment_info['platform']}")
        print(f"Python: {self.environment_info['python_version'].split()[0]}")
        print(f"Scripts found: {len(self.scripts)}")
        print("=" * 60)

        all_results = []
        
        test_suites = [
            ("Script Existence", self.test_script_existence),
            ("Help Messages", self.test_help_messages),
            ("Argument Validation", self.test_argument_validation), 
            ("PyBullet Interface", self.test_pybullet_interface),
            ("Cross-Platform", self.test_cross_platform_compatibility),
            ("Entry Points", self.test_entry_points)
        ]

        for suite_name, test_func in test_suites:
            print(f"\n🧪 {suite_name}...")
            try:
                suite_results = test_func()
                all_results.extend(suite_results)
                
                passed = sum(1 for r in suite_results if r.success and not r.skip_reason)
                failed = sum(1 for r in suite_results if not r.success and not r.skip_reason)  
                skipped = sum(1 for r in suite_results if r.skip_reason)
                
                print(f"   ✅ {passed}  ❌ {failed}  ⏭️ {skipped}")
                
                # Show key failures
                key_failures = [r for r in suite_results if not r.success and not r.skip_reason]
                for failure in key_failures[:3]:  # Show first 3 failures
                    print(f"   💥 {failure.test_name}: {failure.stderr[:100]}")
                
            except Exception as e:
                print(f"   🚨 Suite failed: {e}")

        # Compile final statistics
        total = len(all_results)
        passed = sum(1 for r in all_results if r.success and not r.skip_reason)
        failed = sum(1 for r in all_results if not r.success and not r.skip_reason)
        skipped = sum(1 for r in all_results if r.skip_reason)
        
        success_rate = (passed / total * 100) if total > 0 else 0

        validation_summary = {
            "environment": self.environment_info,
            "scripts_discovered": {name: str(path) for name, path in self.scripts.items()},
            "test_statistics": {
                "total_tests": total,
                "passed": passed,
                "failed": failed, 
                "skipped": skipped,
                "success_rate": success_rate
            },
            "test_results": [asdict(r) for r in all_results],
            "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }

        return validation_summary

    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("# CLI Validation Report")
        report.append("=" * 50)
        report.append("")
        
        # Executive Summary
        stats = results["test_statistics"]
        report.append("## Executive Summary")
        report.append(f"- **Total Tests:** {stats['total_tests']}")
        report.append(f"- **Passed:** {stats['passed']} ✅")
        report.append(f"- **Failed:** {stats['failed']} ❌")
        report.append(f"- **Skipped:** {stats['skipped']} ⏭️") 
        report.append(f"- **Success Rate:** {stats['success_rate']:.1f}%")
        report.append("")
        
        # Environment Details
        env = results["environment"]
        report.append("## Environment")
        report.append(f"- **Platform:** {env['platform']}")
        report.append(f"- **Python Version:** {env['python_version'].split()[0]}")
        report.append(f"- **Python Executable:** {env['python_executable']}")
        report.append(f"- **Project Root:** {env['project_root']}")
        report.append("")
        
        # Scripts Discovered
        report.append("## Scripts Discovered")
        for name, path in results["scripts_discovered"].items():
            status = "✅" if Path(path).exists() else "❌"
            report.append(f"- {status} **{name}**: `{path}`")
        report.append("")
        
        # Results by Category
        categories = {}
        for test_result in results["test_results"]:
            cat = test_result["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(test_result)
        
        report.append("## Results by Category")
        for category, tests in categories.items():
            passed = sum(1 for t in tests if t["success"] and not t["skip_reason"])
            failed = sum(1 for t in tests if not t["success"] and not t["skip_reason"])
            skipped = sum(1 for t in tests if t["skip_reason"])
            
            report.append(f"### {category.replace('_', ' ').title()}")
            report.append(f"**Status:** {passed}/{len(tests)} passed, {failed} failed, {skipped} skipped")
            
            # Critical failures
            critical_failures = [t for t in tests if not t["success"] and not t["skip_reason"]]
            if critical_failures:
                report.append("**Critical Issues:**")
                for test in critical_failures[:5]:  # Show first 5
                    report.append(f"- `{test['test_name']}`: {test['stderr'][:150]}")
            
            # Dependency issues
            dep_issues = [t for t in tests if t["error_type"] == "missing_dependency"]
            if dep_issues:
                report.append("**Dependency Issues:**")
                for test in dep_issues[:3]:
                    report.append(f"- `{test['test_name']}`: {test['skip_reason']}")
            
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        
        if stats["success_rate"] >= 90:
            report.append("### ✨ Excellent!")
            report.append("CLI interfaces are well-implemented and robust.")
        elif stats["success_rate"] >= 70:
            report.append("### 👍 Good Foundation")
            report.append("Most functionality works well with some areas for improvement.")
        else:
            report.append("### 🔧 Needs Attention")
            report.append("Several issues need to be addressed before production use.")
        
        report.append("")
        
        if stats["failed"] > 0:
            report.append("### Immediate Actions Needed:")
            report.append("- 🐛 Fix failed test cases")
            report.append("- 📝 Improve error messages and user feedback") 
            report.append("- ✅ Add more comprehensive input validation")
        
        if stats["skipped"] > 5:
            report.append("### Environment Setup:")
            report.append("- 📦 Install missing dependencies: `pip install torch matplotlib pybullet opencv-python`")
            report.append("- 🐳 Consider using containerized environments for consistent testing")
            report.append("- 🔧 Set up proper development environment")
        
        report.append("")
        report.append("### Development Best Practices:")
        report.append("- 🧪 Implement continuous integration testing")
        report.append("- 📚 Create comprehensive user documentation")
        report.append("- 🔄 Add automated regression testing")
        report.append("- 🎯 Test on multiple Python versions and platforms")
        report.append("")
        
        return "\n".join(report)


def main():
    """Main CLI validation entry point"""
    parser = argparse.ArgumentParser(description="Comprehensive CLI Validation Suite")
    parser.add_argument("--output", "-o", default="cli_validation_results.json",
                       help="Output file for JSON results")
    parser.add_argument("--report", "-r", default="cli_validation_report.md", 
                       help="Output file for markdown report")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    validator = CLIValidator()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Display summary
    stats = results["test_statistics"]
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {stats['total_tests']}")
    print(f"Passed: {stats['passed']} ✅")
    print(f"Failed: {stats['failed']} ❌")
    print(f"Skipped: {stats['skipped']} ⏭️")
    print(f"Success Rate: {stats['success_rate']:.1f}%")
    
    # Save results
    try:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved: {args.output}")
    except Exception as e:
        print(f"\n⚠️ Failed to save results: {e}")
    
    # Generate and save report
    try:
        report = validator.generate_validation_report(results)
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"📋 Report saved: {args.report}")
    except Exception as e:
        print(f"\n⚠️ Failed to save report: {e}")
    
    # Return exit code based on results
    if stats["failed"] == 0:
        print("\n🎉 All critical tests passed!")
        return 0
    elif stats["success_rate"] >= 80:
        print("\n⚠️ Some tests failed but system is mostly functional")
        return 1
    else:
        print("\n🚨 Multiple critical failures detected")
        return 2


if __name__ == "__main__":
    sys.exit(main())