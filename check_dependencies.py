#!/usr/bin/env python3
"""
Validate that all required dependencies are installed for URM with visualizations.
"""

import sys
from pathlib import Path

def check_dependency(package_name: str, import_name: str = None) -> bool:
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name.replace('-', '_')

    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def main():
    print("=" * 70)
    print("URM DEPENDENCY CHECKER")
    print("=" * 70)

    # Core dependencies
    core_deps = [
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('tqdm', 'tqdm'),
        ('pydantic', 'pydantic'),
        ('omegaconf', 'omegaconf'),
        ('hydra-core', 'hydra'),
        ('yaml', 'yaml'),
    ]

    # Visualization dependencies
    viz_deps = [
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('h5py', 'h5py'),
    ]

    # Training dependencies
    training_deps = [
        ('adam-atan2-pytorch', 'adam_atan2_pytorch'),
        ('einops', 'einops'),
        ('coolname', 'coolname'),
        ('wandb', 'wandb'),
        ('huggingface_hub', 'huggingface_hub'),
    ]

    all_missing = []

    # Check core
    print("\n[Core Dependencies]")
    for pkg, imp in core_deps:
        status = "✓" if check_dependency(pkg, imp) else "✗"
        print(f"  {status} {pkg}")
        if status == "✗":
            all_missing.append(pkg)

    # Check visualization
    print("\n[Visualization Dependencies]")
    for pkg, imp in viz_deps:
        status = "✓" if check_dependency(pkg, imp) else "✗"
        print(f"  {status} {pkg}")
        if status == "✗":
            all_missing.append(pkg)

    # Check training
    print("\n[Training Dependencies]")
    for pkg, imp in training_deps:
        status = "✓" if check_dependency(pkg, imp) else "✗"
        print(f"  {status} {pkg}")
        if status == "✗":
            all_missing.append(pkg)

    # Check system dependencies
    print("\n[System Dependencies]")
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=2)
        print("  ✓ ffmpeg (required for MP4 generation)")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  ✗ ffmpeg (required for MP4 generation)")
        print("    Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")

    # Summary
    print("\n" + "=" * 70)

    if all_missing:
        print(f"MISSING PACKAGES: {len(all_missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(all_missing)}")
        print("\nOr install all dependencies:")
        print("  pip install -r requirements.txt")
        return 1
    else:
        print("✓ ALL DEPENDENCIES INSTALLED")
        print("\nYou can now run:")
        print("  python example_urm_visualization.py")
        print("Or:")
        print("  bash scripts/run_visualization_example.sh")
        return 0

if __name__ == '__main__':
    sys.exit(main())
