# Example Training Run

This directory contains an example training run to demonstrate the structure and contents of a typical model training output.

## Contents

This example run includes:

- **Configuration file**: `*_config.json` - Complete training configuration used for this run
- **Training curves**: 
  - `auc_development.png` - Validation AUC over training epochs
  - `loss_development.png` - Training and validation loss curves
  - `acc_class0_development.png` - Accuracy for class 0 (Outside compartment)
  - `acc_class1_development.png` - Accuracy for class 1 (Inside compartment)
- **Training data**: `*_data.pkl` - Pickled data used to generate the plots
- **Evaluation results**: `evaluation_*.csv` - Model evaluation metrics on validation/test sets

## Note

- **Model weights** (`.pth` files) are excluded from this example due to size (200+ MB each)
- **WandB logs** are excluded as they are environment-specific
- To use this example, you would need to train your own model or download the model weights separately

## Training Configuration

This example was trained with:
- **Model**: Inception-ResNet-V2
- **Data imputation**: random_image
- **Tissue threshold**: 0.3
- **Patch size**: 120x120 (resized to 224x224 for model input)
- **Full fine-tuning**: Yes (all layers trainable)

See the `*_config.json` file for complete hyperparameters and settings.

