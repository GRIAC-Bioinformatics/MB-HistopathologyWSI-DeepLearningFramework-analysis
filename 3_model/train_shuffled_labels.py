"""
Shuffled-Label Negative Control — Ablation Study Experiment 1

Trains the exact same Inception-ResNet-V2 model with the exact same hyperparameters
and the exact same preprocessed patches, but with randomly permuted class labels
(Inside vs Outside). This is a standard negative control: if the original model's
performance (AUC ~0.84) comes from genuine tissue signal, then training on shuffled
labels should yield AUC ~0.50 (chance level).

IMPORTANT — Same patient-level split:
    This script loads patches from the same train_dir/val_dir/test_dir directories
    as the main training script. These directories were created by read_and_split_data.py
    using patient-level partitions from patient_partitions_424242.xlsx (seed 424242).
    This guarantees that the same patients (and therefore the same patches) appear in
    train, validation, and test as in the reported CNN results. The ONLY difference is
    that the labels are randomly permuted within each split. Using a different split
    would confound the comparison — any performance difference could be attributed to
    the split rather than the label shuffle.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import wandb
import json
import uuid
from datetime import datetime

from config import BASE_DIR, RESULTS_DIR, get_imputed_patches_path, setup_environment

setup_environment(verbose=False)

# Import reusable functions from the main training script
from train_nn_inner_outer_gpu_pytorch import (
    build_model,
    train_model,
    evaluate_model,
    handle_class_imbalance,
    get_augmentation_transforms,
    summarize_model,
    load_custom_config,
    create_output_folder,
)

SEED = 424242
# Reduced epochs — the model will not converge on shuffled labels,
# so there is no need to train for the full 100 epochs.
EPOCHS = 30
WARMUP_EPOCHS = 5


def shuffle_dataset_labels(dataset, rng):
    """Randomly permute the labels of an ImageFolder dataset in-place.

    This breaks the association between image content and class label
    while preserving the same set of images and the same label distribution.
    """
    labels = [label for _, label in dataset.samples]
    rng.shuffle(labels)
    dataset.samples = [(path, label) for (path, _), label in zip(dataset.samples, labels)]
    dataset.imgs = dataset.samples  # ImageFolder uses both .samples and .imgs


def main():
    config = load_custom_config()

    # Override training duration
    config['epochs'] = EPOCHS
    config['warmup_epochs'] = WARMUP_EPOCHS

    # Save results separately from real training runs
    config['results_path'] = str(RESULTS_DIR / 'ablation' / 'shuffled_labels')

    run_id = str(uuid.uuid4())
    config['run_id'] = run_id
    if config.get('base_dir') is None:
        config['base_dir'] = str(BASE_DIR)

    # --- Data loading (identical to main training script) ---
    # Patches come from the same patient-level split directories used by the CNN.
    config['data_path'] = str(get_imputed_patches_path(
        config['data_imputation_type'],
        config['threshold'],
        tuple(config['org_size']),
    ))

    train_transforms = get_augmentation_transforms(config['augmentation'], config['size'])
    val_transforms = transforms.Compose([
        transforms.Resize(config['size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(os.path.join(config['data_path'], 'train_dir'), transform=train_transforms)
    val_dataset = ImageFolder(os.path.join(config['data_path'], 'val_dir'), transform=val_transforms)
    test_dataset = ImageFolder(os.path.join(config['data_path'], 'test_dir'), transform=val_transforms)

    # --- Shuffle labels (the only difference from the main training script) ---
    # A single RandomState seeded with 424242 shuffles train, then val, then test
    # in that order, making the permutation fully reproducible.
    rng = np.random.RandomState(SEED)
    for dataset in [train_dataset, val_dataset, test_dataset]:
        shuffle_dataset_labels(dataset, rng)

    # --- Training setup (identical to main training script) ---
    config['output_path'] = create_output_folder(config)

    train_loader = handle_class_imbalance(train_dataset, config)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model, trainable_params = build_model(config)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(trainable_params, lr=config['lr'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        wandb.init(
            project="ImageRecognition-Ablation",
            config=config,
            name=f"shuffled_labels_{run_id}",
            dir=config['output_path'],
            reinit=True,
        )

        history, best_epoch_auc, best_epoch_loss = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            config['epochs'], device, config,
        )

        # Load best checkpoint and evaluate on test set
        checkpoint_path = os.path.join(config['output_path'], 'best_auc_model.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_val_auc = checkpoint['val_auc']
        else:
            best_val_auc = None

        test_loss, test_auc = evaluate_model(model, test_loader, criterion, device,
                                             use_mixed_precision=config['mixed_precision'])

        print(f"\n{'='*50}")
        print(f"SHUFFLED-LABEL NEGATIVE CONTROL RESULTS")
        print(f"  Test AUC:  {test_auc:.4f}  (expected ~0.50)")
        print(f"  Test Loss: {test_loss:.4f}")
        if best_val_auc is not None:
            print(f"  Best Val AUC: {best_val_auc:.4f}")
        print(f"{'='*50}\n")

        wandb.log({"test_loss": test_loss, "test_auc": test_auc})
        wandb.run.summary.update({
            "experiment": "shuffled_labels",
            "test_auc": test_auc,
            "test_loss": test_loss,
            "best_val_auc": best_val_auc,
            "seed": SEED,
        })

        summarize_model(config['output_path'], model, history)

        config_file = os.path.join(config['output_path'], f"{run_id}_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
