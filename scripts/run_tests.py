#!/usr/bin/env python3
"""
OLYMPUS Test Runner and Coverage Analysis

This script provides comprehensive test execution and coverage analysis
for the OLYMPUS project, with special focus on safety-critical components
that must achieve 100% test coverage.

Features:
- Automated test discovery and execution
- Coverage analysis with detailed reporting
- Safety-critical component validation
- Performance benchmark execution
- Test result aggregation and reporting
- CI/CD integration support
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import coverage


class OlympusTestRunner:
    """Main test runner for OLYMPUS testing suite."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / "tests"
        self.src_dir = project_root / "src"
        self.reports_dir = self.test_dir / "reports"
        
        # Ensure reports directory exists
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Safety-critical components requiring 100% coverage
        self.safety_critical_modules = [
            "src/olympus/ethical_core/asimov_kernel.py",
            "src/olympus/safety_layer/action_filter.py",
            "src/olympus/safety_layer/human_protection.py",
            "src/olympus/safety_layer/fail_safe.py"
        ]
        
        # Test categories and their requirements
        self.test_categories = {
            "safety": {"min_coverage": 100, "required": True},
            "ethical": {"min_coverage": 100, "required": True}, 
            "unit": {"min_coverage": 90, "required": True},
            "integration": {"min_coverage": 85, "required": True},
            "performance": {"min_coverage": 70, "required": False}
        }
    
    def run_all_tests(self, 
                      categories: Optional[List[str]] = None,
                      parallel: bool = True,
                      generate_reports: bool = True) -> Dict:
        """
        Run all test categories with comprehensive reporting.
        
        Args:
            categories: Specific test categories to run (None = all)
            parallel: Enable parallel test execution
            generate_reports: Generate detailed coverage and test reports
            
        Returns:
            Dict containing test results and coverage information
        """
        print("🚀 Starting OLYMPUS Comprehensive Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        results = {
            "timestamp": datetime.now().isoformat(),
            "categories": {},
            "overall_coverage": 0.0,
            "safety_critical_coverage": {},
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": []
        }
        
        # Determine which categories to run
        if categories is None:
            categories = list(self.test_categories.keys())
        
        # Run each test category
        for category in categories:
            print(f"\n📋 Running {category.upper()} tests...")
            category_result = self._run_category_tests(category, parallel)
            results["categories"][category] = category_result
            
            # Update overall statistics
            results["passed"] += category_result.get("passed", 0)
            results["failed"] += category_result.get("failed", 0)
            results["skipped"] += category_result.get("skipped", 0)
            
            if category_result.get("errors"):
                results["errors"].extend(category_result["errors"])
        
        # Generate comprehensive coverage report
        if generate_reports:
            coverage_data = self._generate_coverage_report()
            results.update(coverage_data)
            
            # Check safety-critical coverage
            safety_coverage = self._check_safety_critical_coverage()
            results["safety_critical_coverage"] = safety_coverage
            
            # Generate HTML and JSON reports
            self._generate_html_report()
            self._generate_json_report(results)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        results["execution_time"] = execution_time
        
        # Print summary
        self._print_test_summary(results)
        
        return results
    
    def _run_category_tests(self, category: str, parallel: bool = True) -> Dict:
        """Run tests for a specific category."""
        category_config = self.test_categories.get(category, {})
        min_coverage = category_config.get("min_coverage", 80)
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            f"--tb=short",
            f"--cov=src/olympus",
            f"--cov-report=",  # Disable intermediate reports
            f"-m", category,
            "-v"
        ]
        
        # Add parallel execution if enabled
        if parallel:
            cmd.extend(["-n", "auto"])
        
        # Add test directory
        if category == "unit":
            cmd.append(str(self.test_dir / "unit"))
        elif category == "integration":
            cmd.append(str(self.test_dir / "integration"))
        elif category == "performance":
            cmd.append(str(self.test_dir / "performance"))
        elif category in ["safety", "ethical"]:
            # Safety and ethical tests are throughout the suite
            cmd.append(str(self.test_dir))
        else:
            cmd.append(str(self.test_dir))
        
        # Execute tests
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Parse results
            return self._parse_pytest_output(result, category)
            
        except Exception as e:
            return {
                "category": category,
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "coverage": 0.0,
                "errors": [f"Failed to run {category} tests: {str(e)}"]
            }
    
    def _parse_pytest_output(self, result: subprocess.CompletedProcess, category: str) -> Dict:
        """Parse pytest output to extract test statistics."""
        output = result.stdout + result.stderr
        lines = output.split('\n')
        
        passed = 0
        failed = 0
        skipped = 0
        errors = []
        
        # Parse test results from output
        for line in lines:
            if "passed" in line and "failed" in line:
                # Parse summary line like "5 failed, 20 passed, 3 skipped"
                parts = line.split(',')
                for part in parts:
                    part = part.strip()
                    if part.endswith('passed'):
                        passed = int(part.split()[0])
                    elif part.endswith('failed'):
                        failed = int(part.split()[0])
                    elif part.endswith('skipped'):
                        skipped = int(part.split()[0])
            elif "ERROR" in line or "FAILED" in line:
                errors.append(line.strip())
        
        return {
            "category": category,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "return_code": result.returncode,
            "errors": errors[:10]  # Limit error messages
        }
    
    def _generate_coverage_report(self) -> Dict:
        """Generate comprehensive coverage report."""
        print("\n📊 Generating coverage report...")
        
        # Run coverage analysis
        cmd = [
            sys.executable, "-m", "pytest",
            f"--cov=src/olympus",
            f"--cov-report=html:{self.reports_dir}/htmlcov",
            f"--cov-report=xml:{self.reports_dir}/coverage.xml",
            f"--cov-report=term-missing",
            "--cov-branch",
            str(self.test_dir),
            "-q"  # Quiet mode for coverage run
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Parse coverage percentage
            overall_coverage = self._extract_coverage_percentage(result.stdout)
            
            return {
                "overall_coverage": overall_coverage,
                "coverage_report_generated": True
            }
            
        except Exception as e:
            return {
                "overall_coverage": 0.0,
                "coverage_report_generated": False,
                "coverage_error": str(e)
            }
    
    def _check_safety_critical_coverage(self) -> Dict:
        """Check coverage for safety-critical components."""
        safety_coverage = {}
        
        # Load coverage data
        try:
            cov = coverage.Coverage()
            cov.load()
            
            for module_path in self.safety_critical_modules:
                full_path = self.project_root / module_path
                if full_path.exists():
                    try:
                        # Get coverage data for this module
                        analysis = cov.analysis2(str(full_path))
                        total_lines = len(analysis[1])  # executable lines
                        missing_lines = len(analysis[3])  # missing lines
                        
                        if total_lines > 0:
                            coverage_percent = ((total_lines - missing_lines) / total_lines) * 100
                        else:
                            coverage_percent = 100.0  # No executable lines
                        
                        safety_coverage[module_path] = {
                            "coverage": coverage_percent,
                            "total_lines": total_lines,
                            "missing_lines": missing_lines,
                            "meets_requirement": coverage_percent >= 100.0
                        }
                        
                    except Exception as e:
                        safety_coverage[module_path] = {
                            "coverage": 0.0,
                            "error": str(e),
                            "meets_requirement": False
                        }
        
        except Exception as e:
            print(f"⚠️  Warning: Could not analyze safety-critical coverage: {e}")
        
        return safety_coverage
    
    def _extract_coverage_percentage(self, output: str) -> float:
        """Extract overall coverage percentage from pytest-cov output."""
        lines = output.split('\n')
        for line in lines:
            if 'TOTAL' in line and '%' in line:
                # Extract percentage from line like "TOTAL     1234    56    95%"
                parts = line.split()
                for part in parts:
                    if part.endswith('%'):
                        return float(part[:-1])
        return 0.0
    
    def _generate_html_report(self) -> None:
        """Generate HTML test report."""
        print("📝 Generating HTML test report...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            f"--html={self.reports_dir}/test_report.html",
            "--self-contained-html",
            str(self.test_dir),
            "-q"
        ]
        
        try:
            subprocess.run(cmd, cwd=self.project_root, check=False)
            print(f"✅ HTML report generated: {self.reports_dir}/test_report.html")
        except Exception as e:
            print(f"⚠️  Warning: Could not generate HTML report: {e}")
    
    def _generate_json_report(self, results: Dict) -> None:
        """Generate JSON test report for CI/CD integration."""
        report_path = self.reports_dir / "test_results.json"
        
        try:
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ JSON report generated: {report_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not generate JSON report: {e}")
    
    def _print_test_summary(self, results: Dict) -> None:
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("🎯 OLYMPUS TEST SUITE SUMMARY")
        print("=" * 60)
        
        # Overall statistics
        total_tests = results["passed"] + results["failed"] + results["skipped"]
        success_rate = (results["passed"] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 Test Results:")
        print(f"   ✅ Passed: {results['passed']}")
        print(f"   ❌ Failed: {results['failed']}")
        print(f"   ⏭️  Skipped: {results['skipped']}")
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        print(f"   ⏱️  Execution Time: {results['execution_time']:.2f}s")
        
        # Coverage information
        print(f"\n📋 Coverage Analysis:")
        print(f"   📊 Overall Coverage: {results['overall_coverage']:.1f}%")
        
        # Safety-critical coverage
        safety_coverage = results.get("safety_critical_coverage", {})
        if safety_coverage:
            print(f"\n🛡️  Safety-Critical Components:")
            all_meet_requirements = True
            
            for module, data in safety_coverage.items():
                coverage_pct = data.get("coverage", 0.0)
                meets_req = data.get("meets_requirement", False)
                status = "✅" if meets_req else "❌"
                
                module_name = Path(module).name
                print(f"   {status} {module_name}: {coverage_pct:.1f}%")
                
                if not meets_req:
                    all_meet_requirements = False
            
            if all_meet_requirements:
                print("   🎉 All safety-critical components meet 100% coverage requirement!")
            else:
                print("   ⚠️  Some safety-critical components need additional test coverage!")
        
        # Category results
        print(f"\n📋 Test Categories:")
        for category, data in results["categories"].items():
            passed = data.get("passed", 0)
            failed = data.get("failed", 0)
            total = passed + failed + data.get("skipped", 0)
            
            if total > 0:
                success = (passed / total) * 100
                status = "✅" if failed == 0 else "❌"
                print(f"   {status} {category.upper()}: {success:.1f}% ({passed}/{total})")
        
        # Errors
        if results["errors"]:
            print(f"\n⚠️  Errors ({len(results['errors'])}):")
            for error in results["errors"][:5]:  # Show first 5 errors
                print(f"   • {error}")
            
            if len(results["errors"]) > 5:
                print(f"   • ... and {len(results['errors']) - 5} more errors")
        
        # Final assessment
        print(f"\n🏆 FINAL ASSESSMENT:")
        
        critical_failure = False
        
        # Check safety-critical requirements
        for module, data in safety_coverage.items():
            if not data.get("meets_requirement", False):
                print(f"   ❌ CRITICAL: {Path(module).name} does not meet 100% coverage requirement")
                critical_failure = True
        
        # Check overall success
        if results["failed"] > 0:
            print(f"   ❌ {results['failed']} test(s) failed")
            critical_failure = True
        
        if results["overall_coverage"] < 90.0:
            print(f"   ⚠️  Overall coverage ({results['overall_coverage']:.1f}%) below 90% target")
        
        if not critical_failure:
            print("   🎉 ALL TESTS PASSED - OLYMPUS IS READY FOR DEPLOYMENT!")
        else:
            print("   🚨 CRITICAL ISSUES DETECTED - DEPLOYMENT NOT RECOMMENDED!")
        
        print("=" * 60)
    
    def run_performance_benchmarks(self) -> Dict:
        """Run performance benchmarks with detailed analysis."""
        print("\n🚀 Running Performance Benchmarks...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "-m", "performance",
            "--benchmark-only",
            "--benchmark-json=" + str(self.reports_dir / "benchmarks.json"),
            str(self.test_dir / "performance"),
            "-v"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Load benchmark results
            benchmark_file = self.reports_dir / "benchmarks.json"
            if benchmark_file.exists():
                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)
                
                return {
                    "benchmarks_run": True,
                    "benchmark_data": benchmark_data,
                    "return_code": result.return_code
                }
            else:
                return {
                    "benchmarks_run": False,
                    "error": "Benchmark results file not generated"
                }
                
        except Exception as e:
            return {
                "benchmarks_run": False,
                "error": str(e)
            }


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="OLYMPUS Test Suite Runner")
    
    parser.add_argument(
        "--categories",
        nargs="*",
        choices=["safety", "ethical", "unit", "integration", "performance"],
        help="Test categories to run (default: all)"
    )
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel test execution"
    )
    
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip report generation"
    )
    
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Run performance benchmarks"
    )
    
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = OlympusTestRunner(args.project_root)
    
    try:
        # Run main test suite
        results = runner.run_all_tests(
            categories=args.categories,
            parallel=not args.no_parallel,
            generate_reports=not args.no_reports
        )
        
        # Run benchmarks if requested
        if args.benchmarks:
            benchmark_results = runner.run_performance_benchmarks()
            results["benchmarks"] = benchmark_results
        
        # Exit with appropriate code
        if results["failed"] > 0:
            sys.exit(1)
        
        # Check safety-critical coverage
        safety_coverage = results.get("safety_critical_coverage", {})
        for module_data in safety_coverage.values():
            if not module_data.get("meets_requirement", False):
                print("🚨 CRITICAL: Safety components do not meet coverage requirements!")
                sys.exit(2)
        
        print("✅ All tests passed successfully!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()