# -*- coding: utf-8 -*-
"""
Central configuration module for ImageRecognition project.

This module provides centralized path management, environment variable handling,
and path validation for all scripts in the project.

Usage:
    from config import paths, validate_paths
    
    # Get paths
    base_dir = paths.BASE_DIR
    data_dir = paths.DATA_DIR
    
    # Validate paths exist
    validate_paths(['data', 'model'])
"""

from .paths import (
    BASE_DIR,
    DATA_DIR,
    PREPROCESSING_DIR,
    MODEL_DIR,
    EVALUATION_DIR,
    RESULTS_DIR,
    SECRETS_DIR,
    PATCHES_DIR,
    MODEL_WEIGHTS_DIR,
    MAPPING_FILE,
    get_patches_path,
    get_imputed_patches_path,
    get_results_path,
    get_model_path,
    get_config_path,
    validate_paths,
    validate_data_exists,
    validate_model_weights,
    setup_environment,
)

__all__ = [
    'BASE_DIR',
    'DATA_DIR',
    'PREPROCESSING_DIR',
    'MODEL_DIR',
    'EVALUATION_DIR',
    'RESULTS_DIR',
    'SECRETS_DIR',
    'PATCHES_DIR',
    'MODEL_WEIGHTS_DIR',
    'MAPPING_FILE',
    'get_patches_path',
    'get_imputed_patches_path',
    'get_results_path',
    'get_model_path',
    'get_config_path',
    'validate_paths',
    'validate_data_exists',
    'validate_model_weights',
    'setup_environment',
]
