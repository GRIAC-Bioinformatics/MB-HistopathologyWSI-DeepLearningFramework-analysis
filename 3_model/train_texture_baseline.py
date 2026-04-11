"""
Texture-Only Baseline (GLCM + LBP + SVM) — Ablation Study Experiment 2

Classical, non-deep-learning baseline that classifies the same patches using
handcrafted texture features. Contextualises how much performance comes from
the deep learning architecture vs. simple texture statistics.

Features: 8 GLCM features (contrast, energy, homogeneity, correlation at
distances 1 and 3, averaged over 4 angles) + 10 LBP histogram bins = 18 total.
Classifier: LinearSVC with Platt scaling (CalibratedClassifierCV).

IMPORTANT — Same patient-level split:
    This script loads patches from the same train_dir/val_dir/test_dir directories
    as the main training script. These directories were created by read_and_split_data.py
    using patient-level partitions from patient_partitions_424242.xlsx (seed 424242).
    This guarantees that the same patients (and therefore the same patches) appear in
    train, validation, and test as in the reported CNN results. Using a different split
    would confound the comparison — any performance difference could be attributed to
    the split rather than the model architecture.

NOTE — Patch resolution:
    The train/val/test directories contain patches resized to 224x224 by
    read_and_split_data.py (the CNN requires this resolution for Inception-ResNet).
    Texture features are therefore computed on 224x224 images, not the original
    120x120 patches. This is conservative: upscaling smooths local gradients, which
    if anything slightly favours the texture baseline. The CNN comparison remains valid
    because both methods operate on the same preprocessed images.

No GPU required. Runs on CPU in under a minute.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import json
from datetime import datetime

from config import RESULTS_DIR, get_imputed_patches_path, setup_environment

setup_environment(verbose=False)

SEED = 424242
# Same config as the final reported CNN model
IMPUTATION_TYPE = 'random_image'
THRESHOLD = 0.2
WINDOW_SIZE = (120, 120)


def extract_features(image_path):
    """Extract GLCM (8) + LBP (10) = 18 texture features from a single patch."""
    img = np.array(Image.open(image_path).convert('L'))

    # GLCM: distances [1, 3], angles [0, pi/4, pi/2, 3pi/4]
    # Average across angles for each distance -> 4 properties x 2 distances = 8 features
    glcm = graycomatrix(img, distances=[1, 3],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=True)
    glcm_feats = np.concatenate([
        graycoprops(glcm, prop).mean(axis=1)  # mean over angles -> 2 values (one per distance)
        for prop in ('contrast', 'energy', 'homogeneity', 'correlation')
    ])  # 8 features

    # LBP: P=8 neighbours, R=1, uniform encoding -> 10 histogram bins
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)

    return np.concatenate([glcm_feats, lbp_hist])  # 18 features


def load_split(split_dir):
    """Load all patches from a split directory and extract features.

    Uses alphabetical label ordering (Inside=0, Outside=1) to match
    PyTorch ImageFolder convention used by the CNN.
    """
    X, y = [], []
    for label_name, label_int in [('Inside', 0), ('Outside', 1)]:
        class_dir = os.path.join(split_dir, label_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Expected directory not found: {class_dir}")
        filenames = sorted(f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        print(f"  {label_name}: {len(filenames)} patches")
        for fname in filenames:
            X.append(extract_features(os.path.join(class_dir, fname)))
            y.append(label_int)
    return np.array(X), np.array(y)


def main():
    data_path = get_imputed_patches_path(IMPUTATION_TYPE, THRESHOLD, WINDOW_SIZE)

    print("Extracting features from train split...")
    X_train, y_train = load_split(str(data_path / 'train_dir'))
    print("Extracting features from val split...")
    X_val, y_val = load_split(str(data_path / 'val_dir'))
    print("Extracting features from test split...")
    X_test, y_test = load_split(str(data_path / 'test_dir'))

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Select regularisation strength C on the validation set
    C_candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_C, best_C_auc = None, -1
    print("\nTuning C on validation set...")
    for C in C_candidates:
        svm = LinearSVC(C=C, max_iter=10000, random_state=SEED)
        clf = CalibratedClassifierCV(svm, cv=5)
        clf.fit(X_train, y_train)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        print(f"  C={C:<6}  val AUC={auc:.4f}")
        if auc > best_C_auc:
            best_C, best_C_auc = C, auc

    # Retrain with best C and evaluate on test set
    print(f"\nBest C={best_C} (val AUC={best_C_auc:.4f}). Training final model...")
    svm = LinearSVC(C=best_C, max_iter=10000, random_state=SEED)
    clf = CalibratedClassifierCV(svm, cv=5)
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    test_auc = roc_auc_score(y_test, test_probs)

    test_preds = clf.predict(X_test)

    print(f"\n{'='*50}")
    print(f"TEXTURE BASELINE RESULTS (GLCM + LBP + SVM)")
    print(f"  Validation AUC: {val_auc:.4f}")
    print(f"  Test AUC:       {test_auc:.4f}")
    print(f"{'='*50}\n")
    print("Test set classification report:")
    print(classification_report(y_test, test_preds, target_names=['Inside', 'Outside']))

    # Save results
    output_dir = RESULTS_DIR / 'ablation' / 'texture_baseline'
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'val_auc': float(val_auc),
        'test_auc': float(test_auc),
        'best_C': float(best_C),
        'C_candidates': C_candidates,
        'n_train': int(len(y_train)),
        'n_val': int(len(y_val)),
        'n_test': int(len(y_test)),
        'features': 'GLCM(8) + LBP(10) = 18 features',
        'classifier': 'LinearSVC + CalibratedClassifierCV(cv=5)',
        'imputation_type': IMPUTATION_TYPE,
        'threshold': THRESHOLD,
        'random_seed': SEED,
        'timestamp': datetime.now().isoformat(),
    }

    results_file = output_dir / 'texture_baseline_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
