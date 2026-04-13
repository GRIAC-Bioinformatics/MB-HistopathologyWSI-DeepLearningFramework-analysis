# Pipeline Guide

## Quick Start

```bash
# Setup
./setup.sh

# Preprocessing
python 2_preprocessing/patch_splitting/1_create_data_set.py
python 2_preprocessing/patch_splitting/2_sliding_window.py
python 2_preprocessing/patch_splitting/3_data_imputation.py
python 2_preprocessing/read_and_split_data.py

# Training
python 3_model/train_nn_inner_outer_gpu_pytorch.py

# Evaluation
python 4_evaluation/evaluate_learning_ability.py
python 4_evaluation/plot_plus_statistical_test.py
```

## Data Flow

```
WSI + Masks → Patches → Imputed Patches → Train/Val/Test → Model → Evaluation
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `1_create_data_set.py` | Process WSI and masks into compartments |
| `2_sliding_window.py` | Extract 120x120 patches |
| `3_data_imputation.py` | Apply masking and imputation strategies |
| `read_and_split_data.py` | Create patient-level train/val/test splits |
| `train_nn_inner_outer_gpu_pytorch.py` | Train Inception-ResNet-V2 model |
| `evaluate_learning_ability.py` | GRAD-CAM and Integrated Gradients analysis |
| `plot_plus_statistical_test.py` | Generate publication plots |
| `train_shuffled_labels.py` | Ablation: shuffled-label negative control (GPU) |
| `train_texture_baseline.py` | Ablation: GLCM+LBP+SVM texture baseline (CPU) |
| `train_simple_cnn.py` | Ablation: simple CNN from scratch (GPU) |
