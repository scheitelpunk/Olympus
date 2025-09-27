#!/usr/bin/env python3
"""
GASM-Roboting Environment Setup Script
=====================================

Automated environment setup and dependency management for GASM-Roboting.
This script provides easy installation and configuration of the package.

Usage:
    python scripts/setup_environment.py [OPTIONS]
    gasm-setup [OPTIONS]  # After package installation

Options:
    --mode {development,production,minimal}
        Installation mode (default: production)
    --extras EXTRAS
        Comma-separated list of extra dependencies
    --gpu
        Enable GPU support optimizations
    --vision
        Include computer vision dependencies
    --robotics
        Include robotics dependencies (PyBullet, etc.)
    --check-only
        Only check current environment, don't install
    --verbose
        Enable verbose output
    --no-confirm
        Skip confirmation prompts
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Set
import platform
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    """Main environment setup class"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.platform = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.project_root = Path(__file__).parent.parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.setup_file = self.project_root / "setup.py"
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        logger.info(f"Checking Python version: {sys.version}")
        
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        
        logger.info(f"✅ Python {self.python_version} is compatible")
        return True
    
    def check_system_dependencies(self) -> Dict[str, bool]:
        """Check system-level dependencies"""
        logger.info("Checking system dependencies...")
        
        dependencies = {
            "git": self._command_exists("git"),
            "curl": self._command_exists("curl") or self._command_exists("wget"),
        }
        
        # Platform-specific checks
        if self.platform == "linux":
            dependencies.update({
                "build-essential": self._command_exists("gcc"),
                "python-dev": True,  # Assume available if Python is working
            })
        elif self.platform == "darwin":  # macOS
            dependencies.update({
                "xcode-tools": self._command_exists("clang"),
            })
        
        # Report results
        for dep, available in dependencies.items():
            status = "✅" if available else "❌"
            logger.info(f"{status} {dep}: {'Available' if available else 'Missing'}")
        
        return dependencies
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH"""
        try:
            subprocess.run([command, "--version"], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def install_base_dependencies(self, mode: str = "production") -> bool:
        """Install base Python dependencies"""
        logger.info(f"Installing base dependencies for {mode} mode...")
        
        try:
            # Upgrade pip first
            self._run_pip_command(["install", "--upgrade", "pip", "setuptools", "wheel"])
            
            # Install package in editable mode for development
            if mode == "development":
                logger.info("Installing package in editable mode...")
                self._run_pip_command(["install", "-e", "."])
            else:
                # Install from requirements.txt
                if self.requirements_file.exists():
                    logger.info("Installing from requirements.txt...")
                    self._run_pip_command(["install", "-r", str(self.requirements_file)])
                else:
                    logger.warning("requirements.txt not found, installing minimal dependencies")
                    self._install_minimal_dependencies()
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install base dependencies: {e}")
            return False
    
    def install_optional_extras(self, extras: List[str], gpu: bool = False) -> bool:
        """Install optional extra dependencies"""
        if not extras:
            return True
        
        logger.info(f"Installing optional extras: {', '.join(extras)}")
        
        try:
            # Build extras specification
            extras_spec = ",".join(extras)
            if gpu and "geometric" in extras:
                logger.info("Installing GPU-optimized geometric libraries...")
                # Install PyTorch with CUDA support first
                self._install_pytorch_cuda()
            
            # Install the package with extras
            self._run_pip_command(["install", f".[{extras_spec}]"])
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install extras {extras}: {e}")
            return False
    
    def _install_pytorch_cuda(self) -> None:
        """Install PyTorch with CUDA support"""
        logger.info("Installing PyTorch with CUDA support...")
        
        # Detect CUDA version (simplified)
        cuda_available = self._command_exists("nvcc")
        if cuda_available:
            logger.info("CUDA detected, installing PyTorch with CUDA support")
            self._run_pip_command([
                "install", 
                "torch", 
                "torchvision", 
                "torchaudio", 
                "--index-url", 
                "https://download.pytorch.org/whl/cu118"  # CUDA 11.8
            ])
        else:
            logger.info("CUDA not detected, installing CPU-only PyTorch")
            self._run_pip_command(["install", "torch", "torchvision", "torchaudio"])
    
    def _install_minimal_dependencies(self) -> None:
        """Install minimal required dependencies"""
        minimal_deps = [
            "torch>=2.0.0",
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "matplotlib>=3.5.0",
            "pillow>=8.0.0",
            "psutil>=5.9.0",
        ]
        
        logger.info("Installing minimal dependencies...")
        for dep in minimal_deps:
            self._run_pip_command(["install", dep])
    
    def _run_pip_command(self, args: List[str]) -> None:
        """Run a pip command with proper logging"""
        cmd = [sys.executable, "-m", "pip"] + args
        
        if self.verbose:
            logger.debug(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=not self.verbose)
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)
    
    def verify_installation(self) -> Dict[str, bool]:
        """Verify that key components are installed and working"""
        logger.info("Verifying installation...")
        
        verification_results = {}
        
        # Test core imports
        core_modules = [
            ("torch", "PyTorch"),
            ("numpy", "NumPy"), 
            ("scipy", "SciPy"),
            ("matplotlib", "Matplotlib"),
        ]
        
        for module, name in core_modules:
            try:
                __import__(module)
                verification_results[name] = True
                logger.info(f"✅ {name} imported successfully")
            except ImportError as e:
                verification_results[name] = False
                logger.error(f"❌ Failed to import {name}: {e}")
        
        # Test GASM core
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            from gasm.core import GASM
            verification_results["GASM Core"] = True
            logger.info("✅ GASM core imported successfully")
        except ImportError as e:
            verification_results["GASM Core"] = False
            logger.error(f"❌ Failed to import GASM core: {e}")
        
        # Test spatial agent
        try:
            from spatial_agent import agent_loop_2d
            verification_results["Spatial Agent"] = True
            logger.info("✅ Spatial Agent imported successfully")
        except ImportError as e:
            verification_results["Spatial Agent"] = False
            logger.error(f"❌ Failed to import Spatial Agent: {e}")
        
        return verification_results
    
    def create_config_files(self) -> None:
        """Create default configuration files"""
        logger.info("Creating default configuration files...")
        
        # Create .env file
        env_file = self.project_root / ".env"
        if not env_file.exists():
            env_content = f"""# GASM-Roboting Environment Configuration
# Generated by setup_environment.py

# Python path
PYTHONPATH={self.project_root}/src

# GASM Configuration
GASM_DEBUG=false
GASM_DEVICE=auto
GASM_PRECISION=float32

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Visualization
MATPLOTLIB_BACKEND=Agg
PLOT_DPI=100
"""
            env_file.write_text(env_content)
            logger.info(f"Created environment file: {env_file}")
    
    def generate_install_report(self, results: Dict[str, bool]) -> None:
        """Generate installation report"""
        logger.info("\n" + "="*60)
        logger.info("GASM-Roboting Installation Report")
        logger.info("="*60)
        
        success_count = sum(results.values())
        total_count = len(results)
        
        logger.info(f"Python Version: {self.python_version}")
        logger.info(f"Platform: {self.platform}")
        logger.info(f"Project Root: {self.project_root}")
        logger.info(f"Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        logger.info("")
        
        for component, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{status} {component}")
        
        if success_count == total_count:
            logger.info("\n🎉 Installation completed successfully!")
            logger.info("You can now use the gasm-* command line tools:")
            logger.info("  gasm-agent-2d --help")
            logger.info("  gasm-agent-3d --help")
            logger.info("  gasm-demo --help")
        else:
            logger.warning(f"\n⚠️  Installation completed with {total_count - success_count} issues.")
            logger.warning("Please check the error messages above and install missing dependencies.")


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description="GASM-Roboting Environment Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--mode",
        choices=["development", "production", "minimal"],
        default="production",
        help="Installation mode (default: production)"
    )
    
    parser.add_argument(
        "--extras",
        type=str,
        help="Comma-separated list of extra dependencies (e.g., geometric,vision,robotics)"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU support optimizations"
    )
    
    parser.add_argument(
        "--vision",
        action="store_true", 
        help="Include computer vision dependencies"
    )
    
    parser.add_argument(
        "--robotics",
        action="store_true",
        help="Include robotics dependencies"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check current environment, don't install"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompts"
    )
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = EnvironmentSetup(verbose=args.verbose)
    
    # Check Python version
    if not setup.check_python_version():
        sys.exit(1)
    
    # Check system dependencies
    system_deps = setup.check_system_dependencies()
    missing_deps = [dep for dep, available in system_deps.items() if not available]
    
    if missing_deps and not args.no_confirm:
        logger.warning(f"Missing system dependencies: {', '.join(missing_deps)}")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            sys.exit(1)
    
    # Build extras list
    extras = []
    if args.extras:
        extras.extend(args.extras.split(","))
    if args.vision:
        extras.append("vision")
    if args.robotics:
        extras.append("robotics")
    if args.gpu:
        extras.append("geometric")
    
    extras = list(set(extras))  # Remove duplicates
    
    if args.check_only:
        # Only verify current installation
        results = setup.verify_installation()
        setup.generate_install_report(results)
        return
    
    # Confirm installation
    if not args.no_confirm:
        logger.info(f"Installation plan:")
        logger.info(f"  Mode: {args.mode}")
        logger.info(f"  Extras: {', '.join(extras) if extras else 'None'}")
        logger.info(f"  GPU support: {args.gpu}")
        
        response = input("\nProceed with installation? [Y/n]: ")
        if response.lower() in ['n', 'no']:
            logger.info("Installation cancelled by user")
            sys.exit(0)
    
    # Install base dependencies
    if not setup.install_base_dependencies(args.mode):
        logger.error("Failed to install base dependencies")
        sys.exit(1)
    
    # Install optional extras
    if extras:
        if not setup.install_optional_extras(extras, args.gpu):
            logger.error("Failed to install optional extras")
            sys.exit(1)
    
    # Create configuration files
    setup.create_config_files()
    
    # Verify installation
    results = setup.verify_installation()
    setup.generate_install_report(results)
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()