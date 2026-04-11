# Ablation Study — Implementation Log and Results

This document records the ablation experiments implemented to address reviewer comments for van Breugel et al., 2026 (Scientific Reports). See `ABLATION_STUDY.md` for the original design rationale.

## Experiments implemented

### Experiment 1: Shuffled-Label Negative Control

**Script:** `3_model/train_shuffled_labels.py`

**What it does:** Trains the same Inception-ResNet-V2 model with the same hyperparameters and the same preprocessed patches (random_image imputation, threshold 0.2), but with randomly permuted class labels (Inside vs Outside). This is a standard negative control — if performance drops to chance level (AUC ~0.50), it confirms the original model learned from genuine tissue signal.

**Design:**
- Wrapper around the existing training pipeline — imports `build_model`, `train_model`, `evaluate_model`, etc. from `train_nn_inner_outer_gpu_pytorch.py`
- Loads the same config via `load_custom_config()` (reads `config_custom.json`)
- Creates ImageFolder datasets from the same `train_dir/val_dir/test_dir` directories
- Permutes labels in-place using `np.random.RandomState(424242)` — this is the only difference from the main training script
- Uses the same patient-level split from `patient_partitions_424242.xlsx` — critical for a valid comparison
- Uses the same 100 epochs and 20-epoch warmup as the reported CNN — identical training duration for an airtight comparison
- Results saved to `5_results/ablation/shuffled_labels/`
- WandB project: `ImageRecognition-Ablation`

**How to run (requires GPU):**
```bash
python 3_model/train_shuffled_labels.py
```

**Expected result:** Test AUC ~0.50 (chance level).

**Status:** Script implemented. Awaiting GPU execution.

---

### Experiment 2: Texture-Only Baseline (GLCM + LBP + SVM)

**Script:** `3_model/train_texture_baseline.py`

**What it does:** Classical, non-deep-learning baseline that classifies the same patches using 18 handcrafted texture features and a linear SVM. Contextualises how much performance comes from the deep learning architecture vs. simple texture statistics.

**Design:**
- Fully standalone script — no PyTorch, no GPU required
- Loads patches from the same `train_dir/val_dir/test_dir` directories as the CNN (same patient-level split)
- Feature extraction per patch (18 features total):
  - GLCM (8 features): contrast, energy, homogeneity, correlation at distances [1, 3], averaged over 4 angles. Computed via `skimage.feature.graycomatrix` / `graycoprops`.
  - LBP (10 features): local binary pattern histogram with P=8, R=1, uniform encoding, 10 bins. Computed via `skimage.feature.local_binary_pattern`.
- Classifier: `sklearn.svm.LinearSVC(max_iter=10000, random_state=424242)` wrapped in `CalibratedClassifierCV(cv=5)` for probability output. Regularisation parameter C selected from {0.01, 0.1, 1, 10, 100} on the validation set.
- Features standardized with `StandardScaler`
- Results saved to `5_results/ablation/texture_baseline/texture_baseline_results.json`

**How to run (CPU only):**
```bash
python 3_model/train_texture_baseline.py
```

Note: texture features are computed on the same 224x224 preprocessed patches as the CNN (resized from 120x120 by `read_and_split_data.py`). This is conservative — upscaling smooths local gradients, which if anything slightly favours the texture baseline.

**Result (random_image imputation, threshold 0.2):**

| Metric | Value |
|--------|-------|
| Validation AUC | 0.7043 |
| Test AUC | **0.7141** |
| Train patches | 26,152 (10,987 Inside + 15,165 Outside) |
| Val patches | 8,752 (3,906 Inside + 4,846 Outside) |
| Test patches | 9,919 (4,404 Inside + 5,515 Outside) |

Test set classification report:
```
              precision    recall  f1-score   support
      Inside       0.62      0.62      0.62      4404
     Outside       0.70      0.70      0.70      5515
    accuracy                           0.66      9919
```

**Interpretation:** The texture baseline achieves AUC 0.71, confirming that basic texture differences exist between compartments (expected — they are anatomically different), but the CNN (AUC 0.84) captures substantially more than simple texture statistics. This supports the value of the deep learning approach.

---

## Ablation summary table

| Experiment | What it tests | Test AUC |
|---|---|---|
| No preprocessing (nonmasked) | Raw classification ability | 0.86 (existing) |
| Limited preprocessing | Basic cleaning only | Existing (Figure 4) |
| Full pipeline (random imputation) | Best preprocessing config | 0.84 (existing) |
| **Shuffled labels** | Genuine tissue signal? | ~0.50 (pending GPU run) |
| **Texture baseline (GLCM+LBP+SVM)** | DL vs. classical features | **0.71** |

## Bug fix during implementation

While running the imputation pipeline locally, a bug was found in `2_preprocessing/patch_splitting/3_data_imputation.py` at line 314. The subfolder extraction used `os.path.basename(os.path.dirname(directory))` which returned the grandparent directory name (`patches_original`) instead of the compartment name (`Inside`/`Outside`). This caused all imputed patches to be saved into a single directory without class separation.

**Fix:** Changed to `os.path.basename(directory)` which correctly returns `Inside` or `Outside`.

## Files created/modified

| File | Action | Description |
|------|--------|-------------|
| `3_model/train_shuffled_labels.py` | Created | Shuffled-label negative control |
| `3_model/train_texture_baseline.py` | Created | GLCM + LBP + SVM baseline |
| `2_preprocessing/patch_splitting/3_data_imputation.py` | Bug fix | Subfolder name extraction |
| `5_results/ablation/texture_baseline/texture_baseline_results.json` | Generated | Texture baseline results |
| `ABLATION_RESULTS.md` | Created | This document |
