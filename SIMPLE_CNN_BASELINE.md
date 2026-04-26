# Simple CNN Baseline — Experiment Plan

## Why this experiment exists

Reviewer 2-Minor-2 asked: *"Clarify whether simpler baseline models were evaluated. Classical texture-based methods or lightweight CNNs could highlight advantages of the proposed framework."*

We already have a classical texture baseline (GLCM+LBP+SVM, AUC 0.71). This experiment adds the missing piece: a **lightweight CNN** trained from scratch (no ImageNet pretraining). It completes the hierarchy of evidence:

| Model | Pretraining | Test AUC |
|---|---|---|
| GLCM+LBP+SVM | None | 0.71 |
| **Simple CNN (this experiment)** | **None** | **?** |
| Inception-ResNet-V2 | ImageNet | 0.84 |

If the simple CNN falls between 0.71 and 0.84, it shows that (a) deep learning adds value over handcrafted features, and (b) the pretrained Inception-ResNet-V2 adds further value via transfer learning. This directly supports the paper's claim that transfer learning is beneficial given the dissimilarity between ImageNet and histopathological ECM images.

## Experimental design

### Architecture

Use a small, standard CNN with no pretrained weights. Something like 3-4 conv blocks:

```
Conv2d(3, 32, 3, padding=1) → BatchNorm → ReLU → MaxPool(2)
Conv2d(32, 64, 3, padding=1) → BatchNorm → ReLU → MaxPool(2)
Conv2d(64, 128, 3, padding=1) → BatchNorm → ReLU → MaxPool(2)
Conv2d(128, 256, 3, padding=1) → BatchNorm → ReLU → AdaptiveAvgPool(1)
Flatten → Linear(256, 1)
```

This is intentionally simple — roughly 1M parameters vs Inception-ResNet-V2's ~56M. The point is not to match performance but to show the transfer learning advantage.

**Do NOT use timm or any pretrained weights.** Define the architecture explicitly in the script. Initialise with PyTorch defaults (Kaiming).

### Training setup — match the main CNN exactly

These settings MUST be identical to the Inception-ResNet-V2 experiment to ensure a fair comparison:

- **Data**: Same preprocessed patches (random_image imputation, threshold 0.2)
- **Patient-level split**: Same `train_dir/val_dir/test_dir` from `read_and_split_data.py`
- **Input size**: 224×224 (same resize)
- **Epochs**: 100
- **Warmup**: 20 epochs
- **Batch size**: 64
- **Learning rate**: 0.001 with ReduceLROnPlateau (patience=10, factor=0.1)
- **Loss**: BCEWithLogitsLoss
- **Class imbalance**: WeightedRandomSampler (oversample)
- **Augmentation**: heavy (same transforms as main training)
- **Mixed precision**: yes
- **Seed**: 424242
- **Optimizer**: Adam (same as main script)

### What to change

Only TWO things differ from the main training:

1. **Architecture**: Simple CNN defined above instead of Inception-ResNet-V2
2. **No pretrained weights**: Random (Kaiming) initialisation

Everything else stays the same.

## Implementation approach

### Script: `3_model/train_simple_cnn.py`

Follow the pattern of `train_shuffled_labels.py`: import reusable functions from the main training script.

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BASE_DIR, RESULTS_DIR, setup_environment
setup_environment(verbose=False)

# Import training utilities from main script
from train_nn_inner_outer_gpu_pytorch import (
    train_model, evaluate_model, handle_class_imbalance,
    get_augmentation_transforms, load_custom_config, create_output_folder
)
```

Key steps:

1. Load config via `load_custom_config()`
2. Override `config['tf_model']` to a custom name (e.g., `"SimpleCNN"`)
3. Define the `SimpleCNN` class as a `nn.Module`
4. Load data from the same `train_dir/val_dir/test_dir` directories using `ImageFolder`
5. Apply the same transforms (`get_augmentation_transforms("heavy", (224, 224))`)
6. Handle class imbalance (`handle_class_imbalance(train_dataset, config)`)
7. Call `train_model()` — check if it accepts a model directly or if you need to refactor. If `train_model()` calls `build_model()` internally, you may need to either:
   - Pass the model as an argument, or
   - Duplicate the training loop (simpler and avoids touching the main script)
8. Call `evaluate_model()` for final test set evaluation
9. Save results to `5_results/ablation/simple_cnn/`

**Important**: Read `train_model()` carefully to understand its signature. If it builds the model internally, it may be cleaner to write a self-contained training loop (copy the relevant parts) rather than monkeypatching. The shuffled labels script is a good reference for how to handle this — it imports `build_model` and calls it explicitly with the config.

### WandB

- Project: `ImageRecognition-Ablation`
- Run name: `simple_cnn_baseline`

### Output directory

```
5_results/ablation/simple_cnn/
├── training_history.json
├── best_auc_model.pth
├── final_metrics.json     # test AUC, test loss, classification report
└── training_curves.png    # optional
```

## What to report

Save at minimum:
- Test AUC (the primary metric)
- Test loss
- Validation AUC (for completeness)
- Patch counts per split (train/val/test, Inside/Outside)
- Architecture summary (parameter count)

## How to run

```bash
# Requires GPU
cd /path/to/project
source venv/bin/activate
python 3_model/train_simple_cnn.py
```

## Expected result

The simple CNN should achieve something between the texture baseline (0.71) and the pretrained model (0.84), likely around 0.75-0.80. This would demonstrate:

1. A CNN from scratch already outperforms handcrafted texture features (deep learning adds value)
2. Transfer learning from ImageNet provides a meaningful boost even for histopathology (justifies Inception-ResNet-V2 choice)

If the simple CNN matches or beats Inception-ResNet-V2, that would actually also be interesting — it would suggest the task doesn't require such a large architecture. Either outcome is scientifically informative and strengthens the paper.

## Context for the manuscript

This result slots into the rebuttal at R1-3 (line 3, alongside the texture baseline) and R2-Minor-2. The ablation summary table becomes:

| Experiment | Test AUC | What it shows |
|---|---|---|
| GLCM+LBP+SVM | 0.71 | Classical features capture some signal |
| Simple CNN (no pretraining) | ? | DL adds value over handcrafted features |
| Inception-ResNet-V2 (pretrained) | 0.84 | Transfer learning adds further value |
| Shuffled labels | 0.50 | Signal is genuine, not artefact |

Manuscript additions would go in:
- **Methods, Evaluation**: One sentence describing the simple CNN architecture
- **Results, after Table 1**: One sentence with the AUC
- **Discussion**: Update the ablation paragraph to include the simple CNN result
