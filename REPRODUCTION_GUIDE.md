# Reproduction Guide

## Prerequisites

- Access to WSI data (contact corresponding author)
- GPU with CUDA support
- Python environment with dependencies installed

## Data Preparation

1. **Obtain data**: Contact corresponding author for WSI data access
2. **Place files**: `1_data/mapping.csv`, `1_data/patient_partitions_424242.xlsx`

3. **Run preprocessing**:
```bash
python 2_preprocessing/patch_splitting/1_create_data_set.py
python 2_preprocessing/patch_splitting/2_sliding_window.py
python 2_preprocessing/patch_splitting/3_data_imputation.py
python 2_preprocessing/read_and_split_data.py
```

## Model Training

1. **Configure**: Edit `3_model/config/config_custom.json`
   - Set `data_imputation_type: "random_image"`
   - Set `threshold: 0.2`
   - Set `random_seed: 424242`

2. **Train model**:
```bash
python 3_model/train_nn_inner_outer_gpu_pytorch.py
```

## Evaluation

1. **Run evaluation**:
```bash
python 4_evaluation/evaluate_learning_ability.py
```

2. **Generate plots**:
```bash
python 4_evaluation/plot_plus_statistical_test.py
```

## Expected Results

- **Test AUC**: ~0.82-0.84
- **Training time**: 10-20 hours on RTX 3090

## Ablation Study

Two additional experiments validate the model's learning signal. Both use the same patient-level split (`patient_partitions_424242.xlsx`) and the same preprocessed patches (random_image imputation, threshold 0.2).

1. **Shuffled-label negative control** (requires GPU):
```bash
python 3_model/train_shuffled_labels.py
```
Trains the same model on randomly permuted labels. Expected test AUC ~0.50.

2. **Texture baseline — GLCM + LBP + SVM** (CPU only):
```bash
python 3_model/train_texture_baseline.py
```
Classical texture features + linear SVM. Expected test AUC ~0.71.

3. **Simple CNN baseline — no pretraining** (requires GPU):
```bash
python 3_model/train_simple_cnn.py
```
Lightweight 4-block CNN (~390K params) trained from scratch. Expected test AUC ~0.76.

See `ABLATION_RESULTS.md` for full details and results.

## Notes

- Use exact patient partitions from `1_data/patient_partitions_424242.xlsx`
- Random seed must be 424242 for reproducibility
- Contact corresponding author for data access
