# -*- coding: utf-8 -*-
"""
Path configuration and validation for ImageRecognition project.

This module handles:
- Environment variable configuration (IMAGE_RECOGNITION_BASE_DIR)
- Automatic base directory detection
- Path validation
- Dynamic path generation for different data types and thresholds

Environment Variables:
    IMAGE_RECOGNITION_BASE_DIR: Base directory for the project (optional)
        If not set, automatically detects based on this file's location.
    
Example:
    export IMAGE_RECOGNITION_BASE_DIR=/path/to/ImageRecognition
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def _get_base_dir() -> Path:
    """
    Get the base directory for the project.
    
    Priority:
    1. IMAGE_RECOGNITION_BASE_DIR environment variable
    2. Auto-detect from this file's location
    
    Returns:
        Path: Absolute path to project root directory
    """
    # Check environment variable first
    env_base = os.environ.get('IMAGE_RECOGNITION_BASE_DIR')
    if env_base:
        base = Path(env_base).resolve()
        if base.exists():
            return base
        else:
            logger.warning(
                f"IMAGE_RECOGNITION_BASE_DIR={env_base} does not exist. "
                "Falling back to auto-detection."
            )
    
    # Auto-detect: this file is in config/, so parent is base
    return Path(__file__).resolve().parent.parent


# =============================================================================
# Base Paths
# =============================================================================

BASE_DIR = _get_base_dir()
DATA_DIR = BASE_DIR / '1_data'
PREPROCESSING_DIR = BASE_DIR / '2_preprocessing'
MODEL_DIR = BASE_DIR / '3_model'
EVALUATION_DIR = BASE_DIR / '4_evaluation'
RESULTS_DIR = BASE_DIR / '5_results'
SECRETS_DIR = BASE_DIR / 'secrets'

# Data subdirectories
PATCHES_DIR = DATA_DIR / 'patches_120x120'
MODEL_WEIGHTS_DIR = MODEL_DIR / 'base_model_architectures'
CONFIG_DIR = MODEL_DIR / 'config'

# Key files
MAPPING_FILE = DATA_DIR / 'mapping.csv'
PATIENT_PARTITIONS_FILE = DATA_DIR / 'patient_partitions_424242.xlsx'


# =============================================================================
# Dynamic Path Generators
# =============================================================================

def get_patches_path(
    window_size: tuple = (120, 120),
    patch_type: str = 'original'
) -> Path:
    """
    Get path to patches directory.
    
    Args:
        window_size: Tuple of (width, height) for patch size
        patch_type: Type of patches ('original', or imputation method)
    
    Returns:
        Path to patches directory
    """
    w, h = window_size
    patches_base = DATA_DIR / f'patches_{w}x{h}'
    
    if patch_type == 'original':
        return patches_base / 'patches_original'
    return patches_base / patch_type


def get_imputed_patches_path(
    processing_method: str,
    threshold: float,
    window_size: tuple = (120, 120)
) -> Path:
    """
    Get path to imputed patches directory.
    
    Args:
        processing_method: Imputation method ('black', 'random_image', 
                          'random_dataset', 'no_masking')
        threshold: Tissue threshold (0.2, 0.3, ..., 0.8)
        window_size: Tuple of (width, height) for patch size
    
    Returns:
        Path to imputed patches directory
    """
    w, h = window_size
    patches_base = DATA_DIR / f'patches_{w}x{h}'
    
    if processing_method == 'no_masking':
        return patches_base / f'patches_cutoff_{processing_method}_{threshold}'
    return patches_base / f'patches_cutoff_{processing_method}_imputed_{threshold}'


def get_results_path(
    data_imputation_type: str,
    threshold: float
) -> Path:
    """
    Get path to results directory for a specific configuration.
    
    Args:
        data_imputation_type: Imputation method
        threshold: Tissue threshold
    
    Returns:
        Path to results directory
    """
    return RESULTS_DIR / data_imputation_type / f'threshold_{threshold}'


def get_model_path(model_name: str = 'Inception-ResNet-V2') -> Path:
    """
    Get path to pre-trained model weights.
    
    Args:
        model_name: Name of the model architecture
    
    Returns:
        Path to model weights file
    """
    return MODEL_WEIGHTS_DIR / f'{model_name}_model.pth'


def get_config_path(config_name: str = 'config_default.json') -> Path:
    """
    Get path to configuration file.
    
    Args:
        config_name: Name of config file
    
    Returns:
        Path to config file
    """
    return CONFIG_DIR / config_name


# =============================================================================
# Path Validation
# =============================================================================

def validate_paths(
    required: List[str] = None,
    raise_error: bool = True
) -> dict:
    """
    Validate that required paths exist.
    
    Args:
        required: List of path types to validate. Options:
                 'base', 'data', 'model', 'results', 'weights', 'mapping'
                 If None, validates all.
        raise_error: If True, raises FileNotFoundError for missing paths
    
    Returns:
        Dict with path names and their existence status
    
    Raises:
        FileNotFoundError: If raise_error=True and required paths don't exist
    """
    path_map = {
        'base': BASE_DIR,
        'data': DATA_DIR,
        'preprocessing': PREPROCESSING_DIR,
        'model': MODEL_DIR,
        'evaluation': EVALUATION_DIR,
        'results': RESULTS_DIR,
        'weights': MODEL_WEIGHTS_DIR,
        'mapping': MAPPING_FILE,
        'config': CONFIG_DIR,
    }
    
    if required is None:
        required = list(path_map.keys())
    
    results = {}
    missing = []
    
    for name in required:
        if name not in path_map:
            logger.warning(f"Unknown path type: {name}")
            continue
        
        path = path_map[name]
        exists = path.exists()
        results[name] = {'path': path, 'exists': exists}
        
        if not exists:
            missing.append((name, path))
    
    if missing and raise_error:
        missing_str = '\n'.join([f"  - {name}: {path}" for name, path in missing])
        raise FileNotFoundError(
            f"Required paths not found:\n{missing_str}\n\n"
            f"Please ensure:\n"
            f"1. You're running from the correct directory\n"
            f"2. Or set IMAGE_RECOGNITION_BASE_DIR environment variable\n"
            f"3. Required data/model files have been downloaded\n"
            f"\nCurrent BASE_DIR: {BASE_DIR}"
        )
    
    return results


def validate_data_exists(
    processing_method: str,
    threshold: float,
    window_size: tuple = (120, 120),
    require_splits: bool = True
) -> bool:
    """
    Validate that processed data exists for a configuration.
    
    Args:
        processing_method: Imputation method
        threshold: Tissue threshold
        window_size: Patch size
        require_splits: If True, also checks for train/val/test directories
    
    Returns:
        True if data exists
    
    Raises:
        FileNotFoundError: If data doesn't exist
    """
    data_path = get_imputed_patches_path(processing_method, threshold, window_size)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data not found at: {data_path}\n"
            f"Please run preprocessing pipeline first:\n"
            f"  python 2_preprocessing/patch_splitting/3_data_imputation.py\n"
            f"  python 2_preprocessing/read_and_split_data.py"
        )
    
    if require_splits:
        for split in ['train_dir', 'val_dir', 'test_dir']:
            split_path = data_path / split
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Data split not found at: {split_path}\n"
                    f"Please run data splitting:\n"
                    f"  python 2_preprocessing/read_and_split_data.py"
                )
    
    return True


def validate_model_weights(model_name: str = 'Inception-ResNet-V2') -> bool:
    """
    Validate that model weights exist.
    
    Args:
        model_name: Name of the model architecture
    
    Returns:
        True if weights exist
    
    Raises:
        FileNotFoundError: If weights don't exist with download instructions
    """
    model_path = get_model_path(model_name)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at: {model_path}\n\n"
            f"Please download the pre-trained weights:\n"
            f"1. Download from: https://github.com/rwightman/pytorch-image-models\n"
            f"   or use timm library: timm.create_model('{model_name.lower()}', pretrained=True)\n"
            f"2. Save to: {model_path}\n\n"
            f"See README.md for detailed instructions."
        )
    
    return True


# =============================================================================
# Environment Setup
# =============================================================================

def setup_environment(verbose: bool = True) -> dict:
    """
    Setup and validate the environment for running scripts.
    
    This function:
    1. Validates base paths exist
    2. Creates necessary directories
    3. Adds project root to Python path
    4. Returns environment information
    
    Args:
        verbose: If True, prints setup information
    
    Returns:
        Dict with environment information
    """
    info = {
        'base_dir': str(BASE_DIR),
        'python_version': sys.version,
        'env_var_set': os.environ.get('IMAGE_RECOGNITION_BASE_DIR') is not None,
    }
    
    # Add project root to Python path if not already there
    base_str = str(BASE_DIR)
    if base_str not in sys.path:
        sys.path.insert(0, base_str)
        info['added_to_path'] = True
    else:
        info['added_to_path'] = False
    
    # Create results directories if they don't exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'learning_evaluation').mkdir(exist_ok=True)
    (RESULTS_DIR / 'completed_grid_runs').mkdir(exist_ok=True)
    
    # Validate core paths (don't require data/weights to exist)
    try:
        validate_paths(['base', 'preprocessing', 'model', 'evaluation'], raise_error=True)
        info['paths_valid'] = True
    except FileNotFoundError as e:
        info['paths_valid'] = False
        info['path_error'] = str(e)
    
    if verbose:
        print(f"ImageRecognition Environment Setup")
        print(f"=" * 50)
        print(f"Base Directory: {BASE_DIR}")
        print(f"Environment Variable Set: {info['env_var_set']}")
        print(f"Paths Valid: {info['paths_valid']}")
        if info['added_to_path']:
            print(f"Added to Python path: {base_str}")
        print(f"=" * 50)
    
    return info


# =============================================================================
# Convenience functions for scripts
# =============================================================================

def get_relative_path(absolute_path: Union[str, Path]) -> Path:
    """
    Convert absolute path to relative path from BASE_DIR.
    
    Args:
        absolute_path: Absolute path to convert
    
    Returns:
        Relative path from BASE_DIR
    """
    return Path(absolute_path).relative_to(BASE_DIR)


def resolve_path(relative_path: Union[str, Path]) -> Path:
    """
    Resolve a relative path to absolute path from BASE_DIR.
    
    Args:
        relative_path: Path relative to BASE_DIR
    
    Returns:
        Absolute path
    """
    return (BASE_DIR / relative_path).resolve()


# Print info when module is imported directly
if __name__ == '__main__':
    setup_environment(verbose=True)
    print(f"\nPath Information:")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  MODEL_DIR: {MODEL_DIR}")
    print(f"  RESULTS_DIR: {RESULTS_DIR}")
    print(f"  MAPPING_FILE: {MAPPING_FILE}")
