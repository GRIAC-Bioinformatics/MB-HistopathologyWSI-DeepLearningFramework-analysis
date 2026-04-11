# Ablation Study — Instructions for Implementation

## Why we need this

We are revising a paper submitted to Scientific Reports (Nature group) after peer review. Both reviewers independently asked for stronger evidence that our deep learning model genuinely learns from the extracellular matrix (ECM) tissue signal, rather than from artefacts introduced by our preprocessing pipeline (masking, imputation, threshold selection).

Specifically, Reviewer 1 (Comment R1-3) asks for "at least one stronger control/ablation," and Reviewer 2 (Comment R2-Major-3) asks for "simple ablation analyses — such as removing imputation, masking, or ECM-pixel thresholding steps individually." Reviewer 2 also separately asks whether simpler baseline models were considered (R2-Minor-2).

This is the hardest ask in the review, and arguably the most important one to get right for acceptance.

## What we already have

The manuscript already contains results that partially address this, but they are scattered and not framed as a formal ablation. We have:

1. **No preprocessing (nonmasked original patches):** The default Inception-ResNet-V2 model trained on completely unprocessed patches yields **test AUC 0.86**. This is mentioned in the Discussion (around line 687 of the current manuscript).

2. **Limited preprocessing pipeline:** No random imputation, basic RGB-based RBC removal only. Results are shown in Figure 4 and compared against the extensive pipeline.

3. **Full preprocessing pipeline (random pixel imputation):** The optimised model achieves **test AUC 0.84**.

4. **Threshold sweep:** Supplementary Figure 2 shows validation AUC across thresholds 0.2–0.8, demonstrating that lower thresholds (0.2–0.4, AUC ~0.85) outperform higher ones (0.8, AUC ~0.77).

5. **Attribution analysis:** Grad-CAM and Integrated Gradients show the preprocessed model attends more to tissue pixels than to masked/imputed pixels. This is the core of Figures 4–5 and Supplementary Figures.

The first task is to **consolidate these into a clear ablation table** in the manuscript. But we also need two new experiments:

## New experiment 1: Shuffled-label negative control

### What it is
Train the exact same model (same architecture, same hyperparameters, same preprocessed patches) but with **randomly shuffled compartment labels** (Inside vs Outside airway). This is a standard negative control in machine learning.

### Why it matters
If the model achieves AUC ≈ 0.50 on the shuffled data, it proves the original model's performance (AUC 0.84) comes from genuine label-associated signal in the tissue, not from data leakage, preprocessing artefacts, or spurious correlations in the pipeline. This is a very clean, hard-to-argue-with control.

### How to implement it
- Use the existing training pipeline (`3_model/train_nn_inner_outer_gpu_pytorch.py`)
- Use the same config as the final reported model: `random_image` imputation, threshold `0.2`, optimised hyperparameters
- The only change: **shuffle the labels** in the data loading step. After `read_and_split_data.py` produces the train/val/test splits, randomly permute the class labels (Inside ↔ Outside) within each partition. Use the same random seed (424242) for the shuffle so it's reproducible.
- Train for the same number of epochs (or until early stopping kicks in — the model likely won't converge meaningfully)
- Report the test AUC. We expect it to be ~0.50 (chance level for binary classification)

### What to save
- Test AUC, validation AUC, training curves (to show the model doesn't learn)
- A brief note on whether it converged or oscillated

### Implementation note
The simplest approach is probably to add a config flag like `"shuffle_labels": true` or just create a small wrapper script that loads the data, shuffles labels, and passes it to the existing training loop. Don't overthink this — it's a one-line change to the data loading.

## New experiment 2: Texture-only baseline (GLCM + LBP + SVM)

### What it is
A classical, non-deep-learning baseline that classifies the same patches using handcrafted texture features. This contextualises how much of the performance comes from the deep learning architecture vs. simple texture statistics that any model could pick up.

### Why it matters
If the texture baseline achieves, say, AUC 0.65–0.70, it shows that basic texture differences exist between compartments (which is expected — they're anatomically different), but that the deep learning model captures something beyond simple texture. If it performs close to the CNN, that would suggest the CNN isn't adding much. Either way, it's informative.

### How to implement it

**Features to extract (per patch):**
- **GLCM (Grey-Level Co-occurrence Matrix):** Compute on the greyscale version of each patch. Extract four standard properties: contrast, energy (ASM), homogeneity, and correlation. Use `skimage.feature.graycomatrix` and `skimage.feature.graycoprops`. Use distances [1, 3] and angles [0, π/4, π/2, 3π/4], then average across angles to get one value per distance per property. That gives 4 properties × 2 distances = 8 features.
- **LBP (Local Binary Pattern):** Compute using `skimage.feature.local_binary_pattern` with radius=1, n_points=8 (the simplest standard setting). Take the histogram of the LBP image (typically 10 bins for uniform LBP) as the feature vector. That gives ~10 features.

Total: ~18 features per patch. This is deliberately simple.

**Classifier:**
- `sklearn.svm.LinearSVC` or `sklearn.svm.SVC(kernel='linear')` with default parameters, or light cross-validation on C
- Use the **exact same patient-level train/val/test split** as the CNN. This is critical — load the same partition file (`patient_partitions_424242.xlsx`) and the same patches
- Train on the training set, pick C on the validation set if you want, report AUC on the test set
- Use `sklearn.metrics.roc_auc_score` with `decision_function` output for AUC

**Patches to use:**
- Use the same preprocessed patches as the CNN (random_image imputation, threshold 0.2)
- The patches are 120×120 pixels. No need to resize to 224×224 (that's only for Inception-ResNet)

### What to save
- Test AUC, classification report
- Optionally: confusion matrix, to see if one compartment is easier than the other

### Implementation note
This should be a standalone Python script — it doesn't need GPU, doesn't need PyTorch, and can run on a laptop in minutes. Keep it simple: load patches → convert to greyscale → extract features → train SVM → report AUC. A single file, maybe 100–150 lines.

## How these results fit into the paper

In the rebuttal and revised manuscript, the ablation story becomes:

| Experiment | What it tests | Expected AUC |
|---|---|---|
| No preprocessing (nonmasked) | Does preprocessing help or hurt raw classification? | 0.86 (existing) |
| Limited preprocessing | Does random imputation add value over basic cleaning? | Existing (Figure 4) |
| Full pipeline (random imputation) | Best preprocessing configuration | 0.84 (existing) |
| **Shuffled labels (new)** | Is the model learning from genuine tissue signal? | ~0.50 |
| **Texture baseline (new)** | Does deep learning add value over classical texture? | ~0.65–0.75 (hypothesis) |

The key argument: the full pipeline achieves AUC 0.84, the shuffled-label model performs at chance (confirming genuine signal), and the texture baseline performs worse (confirming the CNN captures something beyond basic texture). The slightly higher AUC of the nonmasked model (0.86) is expected — it has access to more information — but the attribution analysis shows the preprocessed model focuses on tissue rather than masked regions, which is the methodological point of the paper.

## Practical notes

- **Random seed:** Use 424242 everywhere, consistent with the rest of the pipeline.
- **Patient-level splits:** Must use the same splits. Load from `patient_partitions_424242.xlsx` via the existing data loading code.
- **Config:** The final reported model uses `random_image` imputation, threshold `0.2`. The config system is in `3_model/config/`.
- **Paths:** All scripts import paths from `config/paths.py`. Follow the existing pattern:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  from config import BASE_DIR, DATA_DIR, RESULTS_DIR, setup_environment
  setup_environment(verbose=False)
  ```
- **Output:** Save results to `5_results/` following the existing directory structure. Consider a subfolder like `5_results/ablation/` for these experiments.
- **Dependencies:** GLCM/LBP from `scikit-image`, SVM from `scikit-learn` — both already in `requirements.txt`.

## Priority

1. **Shuffled labels** — easiest, most impactful, do this first
2. **Texture baseline** — straightforward, do second
3. **Consolidate existing results into ablation table** — this is a manuscript/writing task, not a coding task
