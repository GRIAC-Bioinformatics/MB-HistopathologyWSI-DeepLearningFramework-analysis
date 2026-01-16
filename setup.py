#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for ImageRecognition project.

This script helps configure the environment and validate the setup.
Run this after cloning the repository to ensure everything is configured correctly.

Usage:
    python setup.py              # Run full setup with validation
    python setup.py --check      # Only check environment without modifications
    python setup.py --help       # Show help message
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).resolve().parent


def check_python_version():
    """Check Python version is compatible."""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"❌ Python {required_version[0]}.{required_version[1]}+ required, "
              f"but {current_version[0]}.{current_version[1]} found")
        return False
    
    print(f"✅ Python version: {sys.version}")
    return True


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {device_name}")
            return True
        else:
            print("⚠️  CUDA not available. Training will be slow on CPU.")
            return True  # Not a fatal error
    except ImportError:
        print("⚠️  PyTorch not installed. Install with: pip install -r requirements.txt")
        return True  # Not a fatal error during setup


def check_dependencies():
    """Check if required dependencies are installed."""
    required = ['torch', 'numpy', 'pandas', 'cv2', 'sklearn', 'PIL']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            # Handle package name differences
            if package == 'cv2':
                missing.append('opencv-python')
            elif package == 'PIL':
                missing.append('pillow')
            else:
                missing.append(package)
    
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    print("✅ Core dependencies installed")
    return True


def check_directory_structure(project_root):
    """Check that required directories exist."""
    required_dirs = [
        '1_data',
        '2_preprocessing',
        '3_model',
        '3_model/config',
        '3_model/base_model_architectures',
        '4_evaluation',
        '5_results',
        'config',
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"⚠️  Missing directories: {', '.join(missing_dirs)}")
        return False
    
    print("✅ Directory structure valid")
    return True


def check_required_files(project_root):
    """Check that required files exist."""
    required_files = {
        '1_data/mapping.csv': 'Patient metadata file',
        '3_model/config/config_default.json': 'Default training configuration',
        'requirements.txt': 'Python dependencies',
    }
    
    optional_files = {
        '1_data/patient_partitions_424242.xlsx': 'Patient train/val/test splits',
        '3_model/base_model_architectures/Inception-ResNet-V2_model.pth': 'Pre-trained model weights',
    }
    
    all_found = True
    
    for file_path, description in required_files.items():
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path} ({description})")
            all_found = False
    
    print("\nOptional files:")
    for file_path, description in optional_files.items():
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ Found: {file_path}")
        else:
            print(f"⚠️  Missing: {file_path} ({description})")
    
    return all_found


def create_directories(project_root):
    """Create necessary directories if they don't exist."""
    directories = [
        '5_results/completed_grid_runs',
        '5_results/learning_evaluation',
        '3_model/base_model_architectures',
        'config',
    ]
    
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")


def setup_environment_variable():
    """Print instructions for setting environment variable."""
    project_root = get_project_root()
    
    print("\n" + "=" * 60)
    print("ENVIRONMENT VARIABLE SETUP")
    print("=" * 60)
    print("\nTo configure the base directory, set the environment variable:")
    print(f"\n  export IMAGE_RECOGNITION_BASE_DIR=\"{project_root}\"")
    print("\nAdd this to your shell configuration file (~/.bashrc, ~/.zshrc) for persistence.")
    print("\nAlternatively, the config module will auto-detect the base directory")
    print("when running scripts from within the repository.")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Install dependencies:
   pip install -r requirements.txt

2. Download pre-trained model weights:
   - Download Inception-ResNet-V2 weights
   - Save to: 3_model/base_model_architectures/Inception-ResNet-V2_model.pth
   - See README.md for download instructions

3. Prepare data:
   - Obtain WSI data from the repository specified in the manuscript
   - Place mapping.csv in 1_data/
   - Run preprocessing pipeline (see PIPELINE_GUIDE.md)

4. Configure training:
   - Edit 3_model/config/config_custom.json
   - Or use config_grid.json for grid search

5. Start training:
   python 3_model/train_nn_inner_outer_gpu_pytorch.py

For detailed instructions, see:
- README.md: Overview and setup
- PIPELINE_GUIDE.md: Step-by-step pipeline execution
- REPRODUCTION_GUIDE.md: Reproducing manuscript results
""")


def main():
    parser = argparse.ArgumentParser(
        description='Setup and validate ImageRecognition project environment'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Only check environment without making modifications'
    )
    args = parser.parse_args()
    
    project_root = get_project_root()
    
    print("=" * 60)
    print("ImageRecognition Project Setup")
    print("=" * 60)
    print(f"\nProject root: {project_root}\n")
    
    # Run checks
    checks_passed = True
    checks_passed &= check_python_version()
    checks_passed &= check_directory_structure(project_root)
    checks_passed &= check_required_files(project_root)
    
    print("\n" + "-" * 60)
    
    # Check dependencies (non-fatal)
    check_dependencies()
    check_cuda()
    
    if not args.check:
        print("\n" + "-" * 60)
        create_directories(project_root)
    
    setup_environment_variable()
    
    if checks_passed:
        print("\n✅ Basic setup validation passed!")
    else:
        print("\n⚠️  Some checks failed. Please review the output above.")
    
    print_next_steps()
    
    return 0 if checks_passed else 1


if __name__ == '__main__':
    sys.exit(main())
