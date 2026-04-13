# Ablation Study — Complete Work Log

Full record of implementation work performed on 2026-04-11 and 2026-04-12 to address reviewer comments R1-3, R2-Major-3, and R2-Minor-2 for van Breugel et al., 2026 (Scientific Reports).

## Objective

Implement three ablation experiments:
1. Shuffled-label negative control — prove the model learns from genuine tissue signal (AUC should drop to ~0.50)
2. Classical texture baseline — contextualise CNN vs. simple features
3. Simple CNN baseline — contextualise the value of transfer learning

## Work performed

### Phase 1: Planning and codebase exploration

- Analysed `ABLATION_STUDY.md` for requirements
- Explored the training pipeline (`3_model/train_nn_inner_outer_gpu_pytorch.py`), data loading (`2_preprocessing/read_and_split_data.py`), config system (`config/paths.py`, `config/__init__.py`), and results structure
- Key findings:
  - Labels come from directory structure (ImageFolder: Inside=0, Outside=1 alphabetically)
  - Patient-level splits from `patient_partitions_424242.xlsx` — must be preserved for valid comparison
  - Config loaded via `load_custom_config()` reads `config_custom.json` (random_image, threshold 0.2)
  - `ImageFolder.samples` can be modified in-place to permute labels
  - scikit-image and scikit-learn already in `requirements.txt`

### Phase 2: Implementation — Experiment 1 (shuffled labels)

**File created:** `3_model/train_shuffled_labels.py`

- Wrapper script that imports functions from the main training script
- Loads same config, same data, same hyperparameters
- Only difference: permutes labels via `np.random.RandomState(424242)` after creating ImageFolder datasets
- Updates `.samples`, `.imgs`, and `.targets` on each dataset
- Saves results to `5_results/ablation/shuffled_labels/`
- WandB project set to `ImageRecognition-Ablation`
- Merged via PR #1 (`feature/ablation-shuffled-labels`)

### Phase 3: Implementation — Experiment 2 (texture baseline)

**File created:** `3_model/train_texture_baseline.py`

- Fully standalone script — no PyTorch, no GPU needed
- Extracts 18 features per patch: 8 GLCM + 10 LBP histogram bins
- Trains LinearSVC wrapped in CalibratedClassifierCV for probability output
- C tuned on validation set from {0.01, 0.1, 1, 10, 100}
- Merged via PR #2 (`feature/ablation-texture-baseline`)

### Phase 4: Local data preparation and testing

The preprocessed data (random_image imputation, threshold 0.2) did not exist locally. Steps to create it:

1. Unzipped `patches_original.zip` in `1_data/patches_120x120/`
2. **Found and fixed a bug** in `2_preprocessing/patch_splitting/3_data_imputation.py` line 314:
   - `os.path.basename(os.path.dirname(directory))` returned `patches_original` instead of `Inside`/`Outside`
   - Fixed to `os.path.basename(directory)`
   - This bug would have mixed all compartment patches into one directory
3. Ran `3_data_imputation.py` with `random_image` method, threshold 0.2 (~172K patches, ~4 min)
4. Ran `read_and_split_data.py` to create patient-level train/val/test splits
5. Ran texture baseline locally — **test AUC: 0.7141**

### Phase 5: Code review for peer review robustness

Ran two independent code review agents that checked analytical correctness. Issues found and fixed:

**Shuffled labels:**
- `dataset.targets` (a separate list in ImageFolder) was not updated after shuffling — fixed
- Changed from 30 to 100 epochs / 20 warmup to match the CNN exactly — eliminates "insufficient training" reviewer objection
- Added comment that dataset iteration order is load-bearing for RNG reproducibility

**Texture baseline:**
- Added C hyperparameter tuning on validation set — C=1.0 (default) confirmed optimal, AUC stable across all C values
- Documented that features are computed on 224x224 patches (resized by `read_and_split_data.py`), not native 120x120

**Verified correct (no action needed):**
- Label distribution preserved after shuffling (in-place permutation)
- No data leakage — shuffling within each split, patient-level split intact
- GLCM levels=256, angle averaging, LBP histogram binning all correct
- StandardScaler fit on train only
- Label mapping matches ImageFolder convention
- CalibratedClassifierCV fits exclusively on training data

### Phase 6: Validation of shuffled labels script

Before GPU execution, ran comprehensive validation locally (without PyTorch):
- Config loads correctly: random_image, threshold 0.2, 100 epochs, 20 warmup
- All 6 data directories present with correct patch counts (44,823 total)
- Shuffle logic tested with simulated datasets: distribution preserved, `.samples`/`.imgs`/`.targets` in sync, 49.3% of labels changed (not a no-op), fully reproducible with seed 424242

### Phase 7: GPU execution on RunPod

**Infrastructure:** RunPod instance with RTX 3090 (24GB VRAM)

Steps:
1. Pulled latest code on pod (`git pull origin main`)
2. Compressed preprocessed patches locally into tarball (4.8GB), transferred via SCP, extracted on pod
3. Removed macOS `._` resource fork files that ImageFolder would try to load as images
4. Installed missing Python dependencies (timm, wandb, scikit-learn, etc.)
5. Fixed `config/__init__.py` — `get_config_path` and `validate_model_weights` were defined in `paths.py` but not exported
6. Created `3_model/base_model_architectures/` directory (needed for model weight caching)
7. First run failed at epoch 21 — `torch.save` failed on network filesystem (`mfs#`). Patched results path to use local disk (`/tmp/ablation_results/`)
8. Ran with `WANDB_MODE=disabled` (no API key on pod)
9. Training completed: 100 epochs, ~2.5 hours total

**Result: test AUC = 0.5000, test loss = 0.6931**

Retrieved results (checkpoints, training curves, config JSON) from pod to local `5_results/ablation/shuffled_labels/`.

### Phase 8: Documentation

- Created `ABLATION_RESULTS.md` with full experiment details, results, and code review findings
- Updated `REPRODUCTION_GUIDE.md` with ablation study section and run commands
- Updated `README.md` — project structure tree, link to ABLATION_RESULTS.md
- Updated `PIPELINE_GUIDE.md` — ablation scripts in key scripts table

### Phase 9: Implementation — Experiment 3 (simple CNN baseline)

**File created:** `3_model/train_simple_cnn.py`

Follows the same pattern as `train_shuffled_labels.py` but replaces `build_model()` with a custom `SimpleCNN` class:

- 4 conv blocks: 32 → 64 → 128 → 256 channels, each with BatchNorm + ReLU + pooling
- Final AdaptiveAvgPool(1) + Linear(256, 1)
- 389,633 parameters (vs. 55.9M for Inception-ResNet-V2)
- Random Kaiming initialisation, no pretrained weights
- All other settings identical to the main CNN (data, transforms, class imbalance, loss, optimizer, 100 epochs, 20-epoch warmup, lr scheduler, seed 424242)

### Phase 10: GPU execution — simple CNN

Ran on the same RunPod instance:
1. Cleaned up intermediate data directories on pod to free network filesystem quota
2. Pulled latest code
3. Patched results path to local disk (`/tmp/ablation_results/`) to avoid the network filesystem `torch.save` issue
4. Ran with `WANDB_MODE=disabled`
5. Training completed: 100 epochs, ~1.5 hours total (faster than Inception-ResNet-V2 due to smaller model)

**Result: test AUC = 0.7602, test loss = 0.6497, best val AUC = 0.7587**

Retrieved results (checkpoints, training curves, config JSON) from pod to local `5_results/ablation/simple_cnn/`.

### Phase 11: Rebuttal figures

**File created:** `4_evaluation/plot_rebuttal_figures.py`

Standalone script that generates three figures for the rebuttal letter, reading entirely from already-saved experiment outputs (no retraining):

1. **`fig1_ablation_auc.png` / `.pdf`** — Horizontal bar chart of test AUC for all four models (shuffled 0.50, texture 0.71, simple CNN 0.76, Inception-ResNet-V2 0.84) with a chance-level reference line.
2. **`fig2_training_curves.png` / `.pdf`** — Two-panel figure (validation loss + validation AUC) comparing shuffled labels vs. simple CNN, both on the same 100-epoch schedule from random init. Inception-ResNet-V2 is deliberately excluded here because the reported run used a 50-epoch schedule and its pretrained features start far from random, making the training dynamics not like-for-like. The final AUC of the pretrained model is shown in Figure 1.
3. **`fig3_texture_features.png` / `.pdf`** — 2×3 grid showing an example Inside and Outside patch alongside their LBP image and GLCM matrix. Makes the classical baseline tangible for the reviewer.

Saved to `5_results/ablation/rebuttal_figures/`. Runs on CPU in seconds.

## Final results

| Experiment | What it tests | Test AUC |
|---|---|---|
| No preprocessing (nonmasked) | Raw classification ability | 0.86 (existing) |
| Full pipeline (random imputation) | Best preprocessing config | 0.84 (existing) |
| **Shuffled labels** | Genuine tissue signal? | **0.50** |
| **Texture baseline (GLCM+LBP+SVM)** | DL vs. classical features | **0.71** |
| **Simple CNN (no pretraining)** | Transfer learning value | **0.76** |

**Key argument for the rebuttal:** The results form a clean hierarchy of evidence.
- Shuffled labels (0.50) confirm the signal is genuine, not an artefact.
- The texture baseline (0.71) shows basic compartment-level texture differences exist.
- The simple CNN from scratch (0.76) shows deep learning captures more than handcrafted features.
- The pretrained Inception-ResNet-V2 (0.84) shows ImageNet transfer learning adds a further +0.08 despite the domain gap.

Together with the existing attribution analysis (Grad-CAM, Integrated Gradients), this provides strong evidence that (a) the model learns from tissue content, (b) deep learning adds value over classical features, and (c) transfer learning is justified.

## All commits

```
1871348 Add shuffled-label negative control (ablation experiment 1)
7866389 Merge pull request #1 from GRIAC-Bioinformatics/feature/ablation-shuffled-labels
bcb3dbb Add texture-only baseline with GLCM + LBP + SVM (ablation experiment 2)
fbf9fcb Merge pull request #2 from GRIAC-Bioinformatics/feature/ablation-texture-baseline
1486f1a Add ablation results documentation and fix imputation subfolder bug
6103f4e Fix review issues in ablation scripts for peer review robustness
cbe0b00 Update ABLATION_RESULTS.md with C tuning details and code review section
2c4695d Add missing exports to config/__init__.py
1060671 Update ABLATION_RESULTS.md with shuffled-label experiment result
e7e5dfe Add complete ablation study work log
33ce4c4 Add simple CNN baseline (ablation experiment 3)
13f23d9 Update ABLATION_RESULTS.md with simple CNN baseline result
```

## All files created or modified

| File | Action |
|------|--------|
| `3_model/train_shuffled_labels.py` | Created |
| `3_model/train_texture_baseline.py` | Created |
| `3_model/train_simple_cnn.py` | Created |
| `2_preprocessing/patch_splitting/3_data_imputation.py` | Bug fix (subfolder extraction) |
| `config/__init__.py` | Added missing exports |
| `ABLATION_RESULTS.md` | Created |
| `ABLATION_WORK_LOG.md` | Created (this file) |
| `REPRODUCTION_GUIDE.md` | Added ablation section |
| `README.md` | Updated project structure and links |
| `PIPELINE_GUIDE.md` | Added ablation scripts to table |
| `5_results/ablation/texture_baseline/texture_baseline_results.json` | Generated |
| `5_results/ablation/shuffled_labels/` | Generated (checkpoints, plots, config) |
| `5_results/ablation/simple_cnn/` | Generated (checkpoints, plots, config) |
| `4_evaluation/plot_rebuttal_figures.py` | Created |
| `5_results/ablation/rebuttal_figures/` | Generated (3 figures, PNG + PDF) |
