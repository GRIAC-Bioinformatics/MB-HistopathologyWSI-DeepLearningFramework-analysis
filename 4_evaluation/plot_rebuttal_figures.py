"""
Rebuttal figures for the Scientific Reports submission.

Produces three simple figures based on the completed ablation experiments:
  1. Ablation AUC comparison (bar chart)
  2. Training curves — shuffled labels vs simple CNN vs main CNN
  3. Texture feature visualisation (patch + LBP + GLCM)

All data is read from existing run directories — no retraining needed.
Runs on CPU in seconds. Saves PNG + PDF to 5_results/ablation/rebuttal_figures/.
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from config import RESULTS_DIR, DATA_DIR, setup_environment

setup_environment(verbose=False)

OUT_DIR = RESULTS_DIR / 'ablation' / 'rebuttal_figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    """Save a figure as both PNG and PDF."""
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'{name}.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved {name}.png and {name}.pdf")


# ---------------------------------------------------------------------------
# Figure 1: Ablation AUC comparison
# ---------------------------------------------------------------------------

def fig1_ablation_auc():
    """Horizontal bar chart of test AUCs for all four models."""
    models = [
        ('Shuffled labels\n(negative control)', 0.50, '#9e9e9e'),
        ('Texture baseline\n(GLCM + LBP + SVM)', 0.71, '#90caf9'),
        ('Simple CNN\n(no pretraining)',         0.76, '#42a5f5'),
        ('Inception-ResNet-V2\n(ImageNet pretrained)', 0.84, '#1565c0'),
    ]
    labels = [m[0] for m in models]
    aucs   = [m[1] for m in models]
    colors = [m[2] for m in models]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(models))
    bars = ax.barh(y, aucs, color=colors, edgecolor='black', linewidth=0.6)

    ax.axvline(0.50, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Chance (0.50)')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Test AUC')
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    ax.set_title('Ablation study — Test AUC by model', pad=10)
    ax.legend(loc='lower right', fontsize=9)

    for bar, auc in zip(bars, aucs):
        ax.text(auc + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{auc:.2f}', va='center', fontsize=10, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    save_fig(fig, 'fig1_ablation_auc')


# ---------------------------------------------------------------------------
# Figure 2: Training curves
# ---------------------------------------------------------------------------

def _load_history(run_dir):
    """Load train/val loss and AUC from pickled history files."""
    with open(run_dir / 'loss_data.pkl', 'rb') as f:
        loss = pickle.load(f)
    with open(run_dir / 'auc_data.pkl', 'rb') as f:
        auc = pickle.load(f)
    return loss, auc


def _find_run_dir(base):
    """Return the single run directory under an ablation results base."""
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one run dir under {base}, found {len(candidates)}")
    return candidates[0]


def fig2_training_curves():
    """Two-panel figure: loss (left), validation AUC (right) for three models."""
    shuffled_dir = _find_run_dir(RESULTS_DIR / 'ablation' / 'shuffled_labels')
    simple_dir   = _find_run_dir(RESULTS_DIR / 'ablation' / 'simple_cnn')
    main_dir     = RESULTS_DIR / 'example'

    shuffled_loss, shuffled_auc = _load_history(shuffled_dir)
    simple_loss,   simple_auc   = _load_history(simple_dir)
    main_loss,     main_auc     = _load_history(main_dir)

    models = [
        ('Shuffled labels',      shuffled_loss, shuffled_auc, '#9e9e9e'),
        ('Simple CNN',            simple_loss,   simple_auc,   '#42a5f5'),
        ('Inception-ResNet-V2',  main_loss,     main_auc,     '#1565c0'),
    ]

    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, loss, auc, colour in models:
        epochs = np.arange(1, len(loss['val_loss']) + 1)
        ax_loss.plot(epochs, loss['val_loss'], label=name, color=colour, linewidth=1.8)
        ax_auc.plot(epochs, auc['val_auc'],   label=name, color=colour, linewidth=1.8)

    ax_loss.axhline(np.log(2), color='red', linestyle='--', linewidth=1, alpha=0.5,
                    label=f'ln(2) ≈ {np.log(2):.3f} (random)')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Validation loss')
    ax_loss.set_title('Validation loss over training')
    ax_loss.legend(loc='upper right', fontsize=9)
    ax_loss.spines['top'].set_visible(False)
    ax_loss.spines['right'].set_visible(False)

    ax_auc.axhline(0.50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Chance (0.50)')
    ax_auc.set_xlabel('Epoch')
    ax_auc.set_ylabel('Validation AUC')
    ax_auc.set_ylim(0.45, 0.95)
    ax_auc.set_title('Validation AUC over training')
    ax_auc.legend(loc='lower right', fontsize=9)
    ax_auc.spines['top'].set_visible(False)
    ax_auc.spines['right'].set_visible(False)

    fig.suptitle('Training dynamics — the shuffled-label model never learns', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig2_training_curves')


# ---------------------------------------------------------------------------
# Figure 3: Texture feature visualisation
# ---------------------------------------------------------------------------

def _extract_glcm_matrix(img_gray):
    """GLCM contrast matrix at distance=1, angle=0 — for visualisation."""
    glcm = graycomatrix(img_gray, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    return glcm[:, :, 0, 0]  # shape (256, 256)


def _extract_lbp_image(img_gray):
    """Uniform LBP image, same settings as the texture baseline."""
    return local_binary_pattern(img_gray, P=8, R=1, method='uniform')


def _pick_example_patches():
    """Return (inside_path, outside_path) as representative examples."""
    base = DATA_DIR / 'patches_120x120' / 'patches_cutoff_random_image_imputed_0.2' / 'test_dir'
    inside_files  = sorted((base / 'Inside').glob('*.png'))
    outside_files = sorted((base / 'Outside').glob('*.png'))
    # Pick a deterministic example near the middle of the list
    return inside_files[len(inside_files) // 2], outside_files[len(outside_files) // 2]


def fig3_texture_features():
    """2×3 grid: patch, LBP image, GLCM contrast — for Inside and Outside."""
    inside_path, outside_path = _pick_example_patches()

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))

    for row, (label, path) in enumerate([('Inside (submucosa)', inside_path),
                                         ('Outside (adventitia)', outside_path)]):
        rgb  = np.array(Image.open(path).convert('RGB'))
        gray = np.array(Image.open(path).convert('L'))

        lbp  = _extract_lbp_image(gray)
        glcm = _extract_glcm_matrix(gray)

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f'{label}\nOriginal patch' if row == 0 else 'Original patch')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(lbp, cmap='viridis')
        axes[row, 1].set_title('LBP image (P=8, R=1, uniform)' if row == 0 else 'LBP image')
        axes[row, 1].axis('off')

        im = axes[row, 2].imshow(np.log1p(glcm), cmap='magma')
        axes[row, 2].set_title('GLCM (log-scaled, d=1, θ=0)' if row == 0 else 'GLCM (log-scaled)')
        axes[row, 2].axis('off')

    # Add row labels on the left
    for row, label in enumerate(['Inside', 'Outside']):
        axes[row, 0].text(-0.1, 0.5, label, transform=axes[row, 0].transAxes,
                          fontsize=12, fontweight='bold', rotation=90,
                          va='center', ha='center')

    fig.suptitle('Example patches and their handcrafted texture features\n'
                 '(the texture baseline uses GLCM + LBP statistics from images like these)',
                 fontsize=11, y=1.00)
    fig.tight_layout()
    save_fig(fig, 'fig3_texture_features')


def main():
    print(f"Writing rebuttal figures to {OUT_DIR}")
    print("\nFigure 1: Ablation AUC comparison")
    fig1_ablation_auc()
    print("\nFigure 2: Training curves")
    fig2_training_curves()
    print("\nFigure 3: Texture feature visualisation")
    fig3_texture_features()
    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
