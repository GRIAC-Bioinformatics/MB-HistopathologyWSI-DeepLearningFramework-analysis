# Rebuttal Figures

Three simple figures generated from the completed ablation experiments to support the rebuttal letter for van Breugel et al., 2026 (Scientific Reports).

## How to regenerate

```bash
python 4_evaluation/plot_rebuttal_figures.py
```

Runs on CPU in seconds. All input data is already on disk — no retraining is needed.

## The figures

### `fig1_ablation_auc` — Ablation AUC comparison

Horizontal bar chart of test AUC for all four models with a chance-level reference line.

**How it's built:** Test AUC values are read directly from the completed experiment outputs:
- **Shuffled labels (0.50)**: from `5_results/ablation/shuffled_labels/<run>/` (training log print)
- **Texture baseline (0.71)**: from `5_results/ablation/texture_baseline/texture_baseline_results.json`
- **Simple CNN (0.76)**: from `5_results/ablation/simple_cnn/<run>/` (training log print)
- **Inception-ResNet-V2 (0.84)**: the paper's reported test AUC for the manuscript-selected run

Matplotlib horizontal `barh`, 4 colours (grey → light/mid/dark blue), dashed red line at 0.50.

### `fig2_training_curves` — Training dynamics (shuffled vs simple CNN)

Two-panel figure showing validation loss (left) and validation AUC (right) across 100 epochs, for the shuffled-label control and the simple CNN.

**How it's built:** Training histories are loaded from the pickled outputs saved by `summarize_model()` in the main training script:
- `5_results/ablation/shuffled_labels/<run>/loss_data.pkl` + `auc_data.pkl`
- `5_results/ablation/simple_cnn/<run>/loss_data.pkl` + `auc_data.pkl`

The script uses `_find_run_dir()` to locate the single run directory under each ablation folder. Each pickle contains `{'train_loss': [...], 'val_loss': [...]}` or `{'train_auc': [...], 'val_auc': [...]}`.

Reference lines: `ln(2) ≈ 0.693` on the loss panel (theoretical random-binary loss) and `0.50` on the AUC panel (chance).

**Note on Inception-ResNet-V2 exclusion:** The manuscript-selected Inception-ResNet-V2 run used a 50-epoch schedule, not 100, so plotting it here would look like early stopping. It is also pretrained, so it starts at val AUC ~0.73 at epoch 1 — the training dynamics aren't like-for-like with from-scratch models. Its final test AUC is shown in `fig1` instead, which is the appropriate place for the comparison.

### `fig3_texture_features` — Texture feature visualisation

2×3 grid showing an example Inside and Outside patch alongside their LBP image and GLCM matrix. Makes the classical baseline tangible.

**How it's built:**
- **Patches**: `_pick_example_patches()` picks a deterministic patch (the middle file, alphabetically) from `1_data/patches_120x120/patches_cutoff_random_image_imputed_0.2/test_dir/{Inside,Outside}/`.
- **Original patch**: RGB, loaded with PIL.
- **LBP image**: `skimage.feature.local_binary_pattern(gray, P=8, R=1, method='uniform')` — exactly the same settings as the texture baseline. Displayed with viridis colormap.
- **GLCM matrix**: `skimage.feature.graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)` and then `[:, :, 0, 0]`. Log-scaled (`np.log1p`) with magma colormap because the raw counts span many orders of magnitude.

The full texture baseline uses 8 GLCM features (4 properties × 2 distances, averaged over 4 angles) and 10 LBP histogram bins — this figure shows just one representative (distance=1, angle=0) for visualisation.

## Output

Each figure is saved as both PNG (for documents) and PDF (for vector graphics).

| File | Format | Purpose |
|------|--------|---------|
| `fig1_ablation_auc.png` / `.pdf` | bar chart | Hero plot: final AUCs at a glance |
| `fig2_training_curves.png` / `.pdf` | line plot | Shows the negative control working |
| `fig3_texture_features.png` / `.pdf` | image grid | Makes the texture baseline concrete |

## Source

The plotting script is at `4_evaluation/plot_rebuttal_figures.py`. It is standalone — no PyTorch, no GPU, no retraining. All dependencies (`matplotlib`, `numpy`, `PIL`, `scikit-image`) are already in `requirements.txt`.
