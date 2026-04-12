"""
Simple CNN Baseline — Ablation Study Experiment 3

Trains a lightweight 4-block CNN from scratch (no pretrained weights) on the
exact same data, splits, and hyperparameters as the Inception-ResNet-V2 model.
This contextualises the value of transfer learning: if the simple CNN falls
between the texture baseline (AUC 0.71) and the pretrained model (AUC 0.84),
it shows that (a) deep learning adds value over handcrafted features and
(b) ImageNet pretraining provides a meaningful further boost.

Architecture (~1M parameters):
    Conv2d(3,32) → BN → ReLU → MaxPool
    Conv2d(32,64) → BN → ReLU → MaxPool
    Conv2d(64,128) → BN → ReLU → MaxPool
    Conv2d(128,256) → BN → ReLU → AdaptiveAvgPool(1)
    Flatten → Linear(256, 1)

IMPORTANT — Same patient-level split:
    This script loads patches from the same train_dir/val_dir/test_dir directories
    as the main training script. These directories were created by read_and_split_data.py
    using patient-level partitions from patient_partitions_424242.xlsx (seed 424242).
    This guarantees that the same patients (and therefore the same patches) appear in
    train, validation, and test as in the reported CNN results. The ONLY differences
    are the architecture and the absence of pretrained weights.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    train_model,
    evaluate_model,
    handle_class_imbalance,
    get_augmentation_transforms,
    summarize_model,
    load_custom_config,
    create_output_folder,
)


class SimpleCNN(nn.Module):
    """4-block CNN with ~1M parameters. No pretrained weights (Kaiming init)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224x224 → 112x112
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 112x112 → 56x56
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 56x56 → 28x28
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: 28x28 → 1x1
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def main():
    config = load_custom_config()

    # Override model name and results path
    config['tf_model'] = 'SimpleCNN'
    config['results_path'] = str(RESULTS_DIR / 'ablation' / 'simple_cnn')

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

    # --- Training setup (identical to main training script) ---
    config['output_path'] = create_output_folder(config)

    train_loader = handle_class_imbalance(train_dataset, config)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Build simple CNN (no pretrained weights, Kaiming init by default)
    model = SimpleCNN()
    trainable_params = model.parameters()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nSimpleCNN: {total_params:,} parameters (all trainable)")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(trainable_params, lr=config['lr'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        wandb.init(
            project="ImageRecognition-Ablation",
            config=config,
            name=f"simple_cnn_{run_id}",
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
        print(f"SIMPLE CNN BASELINE RESULTS")
        print(f"  Test AUC:       {test_auc:.4f}")
        print(f"  Test Loss:      {test_loss:.4f}")
        if best_val_auc is not None:
            print(f"  Best Val AUC:   {best_val_auc:.4f}")
        print(f"  Parameters:     {total_params:,}")
        print(f"{'='*50}\n")

        wandb.log({"test_loss": test_loss, "test_auc": test_auc})
        wandb.run.summary.update({
            "experiment": "simple_cnn",
            "test_auc": test_auc,
            "test_loss": test_loss,
            "best_val_auc": best_val_auc,
            "total_params": total_params,
        })

        summarize_model(config['output_path'], model, history)

        config_file = os.path.join(config['output_path'], f"{run_id}_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
