# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end deep learning pipeline for analyzing tissue compartments in lung histopathology whole slide images (WSI). Classifies Inside vs Outside airway regions, with GRAD-CAM and Integrated Gradients to understand how models learn from masked/imputed image regions. Accompanies van Breugel et al., 2026.

## Setup & Environment

```bash
./setup.sh                 # Full setup: venv, deps, directory creation
./setup.sh --check         # Validate environment only
source venv/bin/activate   # Activate virtualenv
```

Optional env var: `export IMAGE_RECOGNITION_BASE_DIR=/path/to/project` (auto-detected if unset).

GPU with CUDA required for practical training. Verify: `python -c "import torch; print(torch.cuda.is_available())"`.

## Pipeline Commands

Run sequentially from project root:

```bash
# 1. Preprocessing
python 2_preprocessing/patch_splitting/1_create_data_set.py      # WSI + masks -> compartments
python 2_preprocessing/patch_splitting/2_sliding_window.py        # Extract 120x120 patches
python 2_preprocessing/patch_splitting/3_data_imputation.py       # Apply masking/imputation
python 2_preprocessing/read_and_split_data.py                     # Patient-level train/val/test split

# 2. Training
python 3_model/train_nn_inner_outer_gpu_pytorch.py                # Train model (reads config/)

# 3. Evaluation
python 4_evaluation/evaluate_learning_ability.py                  # GRAD-CAM + Integrated Gradients
python 4_evaluation/plot_plus_statistical_test.py                 # Publication plots with stats
python 4_evaluation/visualize_compartment_differences.py          # Statistical compartment analysis
```

## Architecture

### Five-stage numbered directory structure

- `1_data/` - Patient metadata (`mapping.csv`), partition files, raw/processed patches
- `2_preprocessing/` - Patch extraction, imputation strategies, data splitting
- `3_model/` - PyTorch training with Inception-ResNet-V2 (via `timm`), config files, pre-trained weights
- `4_evaluation/` - GRAD-CAM, Integrated Gradients (`captum`), noise resistance, mixed-effects stats
- `5_results/` - Organized as `{imputation_type}/threshold_{value}/{timestamp}_{run_id}/`

### Central path module (`config/`)

All scripts import paths from `config/paths.py`. Key exports: `BASE_DIR`, `DATA_DIR`, `MODEL_DIR`, `RESULTS_DIR`, `EVALUATION_DIR`, `get_imputed_patches_path()`, `get_results_path()`, `validate_paths()`, `setup_environment()`.

Standard import pattern used across all scripts:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BASE_DIR, DATA_DIR, RESULTS_DIR, setup_environment
setup_environment(verbose=False)
```

### Training configuration

Configs in `3_model/config/`:
- `config_default.json` - Base hyperparameters (always loaded first)
- `config_custom.json` - Single run overrides
- `config_grid.json` - Grid search parameter combinations

Key defaults: Inception-ResNet-V2, 224x224 input (resized from 120x120 patches), lr=0.001, batch_size=64, 100 epochs, heavy augmentation, mixed precision, ReduceLROnPlateau with patience=10.

### Data imputation strategies (core concept)

Four approaches to handle artifact-masked regions, each producing a separate dataset:
- **black** - Replace masked pixels with black
- **random_image** - Replace with random pixels from same image
- **random_dataset** - Replace with random patches from dataset
- **no_masking** - Original unmodified patches

## Critical Design Decisions

- **Patient-level splits** (not patch/WSI-level) prevent data leakage. Defined in `patient_partitions_424242.xlsx`.
- **Random seed 424242** used everywhere for reproducibility. Alternative seeds (1, 3) for sensitivity analysis.
- **Tissue thresholds** (0.2-0.8) filter patches with insufficient tissue coverage.
- Model checkpoints saved as `best_auc_model.pth`, `best_loss_model.pth`, `final_model.pth`.
- WandB integration for experiment tracking (optional but recommended).

## Key Dependencies

PyTorch 2.0+, timm (Inception-ResNet-V2), pytorch-grad-cam, captum (Integrated Gradients), statsmodels (mixed-effects models), hydra-core, wandb, scikit-image, opencv-python, imbalanced-learn. Full list in `requirements.txt`.

## No Test Suite

This is a research codebase without automated tests or CI/CD. Validation is done via `setup.py --check` and example results in `5_results/example/`.
