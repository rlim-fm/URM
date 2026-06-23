#!/usr/bin/env python3
"""
URM Visualization Setup and Verification Script

This script will:
1. Check all dependencies
2. Create necessary directories
3. Test imports
4. Provide setup instructions
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text):
    print(f"\n▶ {text}")


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name.replace('-', '_')
    try:
        __import__(import_name)
        return True, None
    except ImportError as e:
        return False, str(e)


def check_command(command):
    """Check if a system command is available."""
    try:
        result = subprocess.run([command, '--version'], capture_output=True, timeout=2)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def main():
    print_header("URM VISUALIZATION SETUP & VERIFICATION")

    repo_dir = Path(__file__).parent
    os.chdir(repo_dir)

    # Check core files
    print_section("Checking Core Files")
    required_files = [
        'visualization.py',
        'train.py',
        'viz_utils.py',
        'requirements.txt',
        'example_urm_visualization.py',
    ]

    all_files_present = True
    for fname in required_files:
        fpath = repo_dir / fname
        status = "✓" if fpath.exists() else "✗"
        print(f"  {status} {fname}")
        if not fpath.exists():
            all_files_present = False

    if not all_files_present:
        print("\n✗ Some files are missing!")
        return 1

    # Check directories
    print_section("Creating Directories")
    dirs_to_create = ['visualizations', 'train_out', 'checkpoints']
    for dirname in dirs_to_create:
        dirpath = repo_dir / dirname
        dirpath.mkdir(exist_ok=True)
        print(f"  ✓ {dirname}/")

    # Check Python packages
    print_section("Checking Python Packages")

    critical_packages = [
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('h5py', 'h5py'),
    ]

    optional_packages = [
        ('torch', 'torch'),
        ('pydantic', 'pydantic'),
        ('omegaconf', 'omegaconf'),
        ('hydra-core', 'hydra'),
    ]

    missing_critical = []
    for pkg, imp in critical_packages:
        installed, _ = check_package(pkg, imp)
        status = "✓" if installed else "✗"
        print(f"  {status} {pkg}")
        if not installed:
            missing_critical.append(pkg)

    print()
    for pkg, imp in optional_packages:
        installed, _ = check_package(pkg, imp)
        status = "✓" if installed else "○"
        note = "(optional)" if not installed else ""
        print(f"  {status} {pkg} {note}")

    # Check system commands
    print_section("Checking System Dependencies")
    ffmpeg_ok = check_command('ffmpeg')
    status = "✓" if ffmpeg_ok else "✗"
    print(f"  {status} ffmpeg")

    if not ffmpeg_ok:
        print("\n    Note: ffmpeg not found. MP4 generation won't work.")
        print("    Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")

    # Test imports
    print_section("Testing Python Imports")
    try:
        from visualization import Visualizer, Visualization
        print("  ✓ visualization module")
    except Exception as e:
        print(f"  ✗ visualization module: {e}")
        return 1

    try:
        from viz_utils import MetricsCollector
        print("  ✓ viz_utils module")
    except Exception as e:
        print(f"  ✗ viz_utils module: {e}")
        return 1

    # Final status
    print_header("SETUP VERIFICATION")

    if missing_critical:
        print(f"✗ MISSING CRITICAL PACKAGES: {', '.join(missing_critical)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing_critical)}")
        return 1

    print("✓ ALL CHECKS PASSED!")
    print("\nYou can now run:")
    print("  1. python3 example_urm_visualization.py")
    print("  2. bash scripts/run_visualization_example.sh")
    print("  3. bash scripts/URM_sudoku.sh")

    print("\nDocumentation:")
    print("  - QUICK_REFERENCE.md (start here!)")
    print("  - VISUALIZATION_GUIDE.md (complete guide)")
    print("  - INTEGRATION_SUMMARY.md (integration details)")

    print("\n" + "=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
