# Image Recognition for Lung Histopathology Analysis

A comprehensive deep learning pipeline for analyzing tissue compartments in lung histopathology images. This project focuses on distinguishing between tissue and artifact regions, with particular emphasis on understanding how neural networks learn from masked and imputed image regions using techniques like GRAD-CAM and Integrated Gradients.

## TL;DR

**What**: Deep learning pipeline for lung tissue analysis using GRAD-CAM and Integrated Gradients

**Requirements**: Python 3.8+, GPU with CUDA, 16GB+ RAM, 50GB+ storage

**Quick Setup**:
```bash
git clone <repo> ImageRecognition
cd ImageRecognition
./setup.sh
python 3_model/train_nn_inner_outer_gpu_pytorch.py
```

**Data**: Contact corresponding author for 162 WSIs from 75 patients

**Citation**: van Breugel et al., 2026

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Installation & Setup](#installation--setup)
5. [Workflow Guide](#workflow-guide)
6. [Results & Outputs](#results--outputs)
7. [Configuration & Optimization](#configuration--optimization)
8. [Troubleshooting](#troubleshooting)
9. [Data Availability](#data-availability)
10. [Code Availability](#code-availability)
11. [Citation](#citation)
12. [Reproducing Manuscript Results](#reproducing-manuscript-results)
13. [Contact](#contact)

**Additional Documentation**:
- **[PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)**: Detailed step-by-step pipeline execution guide
- **[REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)**: Complete guide for reproducing manuscript results

## Overview

This repository provides a complete end-to-end pipeline for:

- **Preprocessing**: Patch extraction from whole slide images (WSI), data imputation strategies (black masking, random imputation), and dataset preparation
- **Model Training**: PyTorch-based neural network training with configurable architectures (Inception-ResNet-V2, etc.)
- **Evaluation**: Comprehensive analysis including:
  - GRAD-CAM visualization for model attention
  - Integrated Gradients for attribution analysis
  - Noise resistance testing
  - Statistical evaluation with mixed-effects models
  - Compartment difference analysis

### Key Features

- Patient-level data splitting to prevent data leakage
- Multiple data imputation strategies for handling masked regions
- Configurable model architectures and training hyperparameters
- Comprehensive evaluation framework with visualization tools
- Experiment tracking with WandB integration

## Project Structure

```
ImageRecognition/
├── 1_data/                              # Data files and mappings
│   ├── mapping.csv                       # Patient metadata mapping
│   └── patient_partitions_*.xlsx        # Patient partition files
│
├── 2_preprocessing/                      # Data preprocessing scripts
│   ├── patch_splitting/                 # Patch extraction and processing
│   │   ├── 1_create_data_set.py        # Create dataset from WSI
│   │   ├── 2_sliding_window.py         # Sliding window patch extraction
│   │   ├── 3_data_imputation.py        # Apply data imputation strategies
│   │   ├── 3b_generative_fill_image.py # Generative inpainting (experimental)
│   │   ├── 4_check_quantity_per_intersection.py  # Quality control
│   │   ├── 6_map_airway_accuracy_to_WSI.py      # Map predictions to WSI
│   │   ├── 8_aggregate_on_patient_level.py      # Patient-level aggregation
│   │   └── run_preprocess_pipeline.sh   # Pipeline execution script
│   ├── qupath_annotation/               # QuPath annotation scripts (.groovy)
│   └── read_and_split_data.py           # Split data into train/val/test sets
│
├── 3_model/                             # Model training
│   ├── base_model_architectures/        # Pre-trained model weights (.pth)
│   ├── config/                          # Configuration files
│   │   ├── config_default.json         # Default training configuration
│   │   ├── config_custom.json          # Custom single-run configuration
│   │   ├── config_grid.json            # Grid search configuration
│   │   └── config_hydra.yaml           # Hydra configuration
│   └── train_nn_inner_outer_gpu_pytorch.py  # Main training script (PyTorch)
│
├── 4_evaluation/                        # Evaluation and visualization
│   ├── config/                          # Evaluation configuration files
│   ├── evaluate_learning_ability.py    # Main evaluation framework
│   ├── visualize_compartment_differences.py  # Statistical analysis
│   ├── plot_plus_statistical_test.py   # Statistical plotting
│   ├── plot_information_scores.py      # Information score visualization
│   ├── plot_information_scores_black_relative.py  # Relative attribution plots
│   └── visualize_inner_outer.py        # Visualization tools
│
├── 5_results/                            # All output results (see Results section)
│   ├── {data_type}/                     # Results by imputation type
│   │   └── threshold_{threshold}/      # Results by tissue threshold
│   ├── learning_evaluation/             # Evaluation analysis results
│   └── completed_grid_runs/            # Grid search tracking
│
└── secrets/                             # Sensitive files (see secrets/README.md)
    └── README.md                        # Instructions for secret files
```

## Requirements

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended: 12GB+ VRAM)
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ for code, models, and data
- **OS**: Linux (tested on Ubuntu), macOS, or Windows with WSL

### Python Dependencies

See `requirements.txt` for complete list. Key dependencies:

- **Deep Learning**: PyTorch, TensorFlow (for legacy components)
- **Image Processing**: OpenCV, scikit-image, PIL
- **Data Science**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Experiment Tracking**: wandb
- **Statistical Analysis**: statsmodels

**Note**: For reproducibility, specific version numbers should be pinned. Current `requirements.txt` contains unpinned versions. For manuscript reproduction, see `PUBLICATION_READINESS.md` for version requirements.

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Installation & Setup

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url> ImageRecognition
   cd ImageRecognition
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set up secrets** (see `secrets/README.md` for details):
   - Place required secret files in `secrets/` directory
   - These files are excluded from version control

5. **Verify installation**:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### RunPod GPU Environment Setup

This project has been tested on RunPod cloud GPU instances. Follow these steps:

#### Step 1: Create RunPod Instance

1. Go to [RunPod Dashboard](https://www.runpod.io/) and log in
2. Click **"Pods"** → **"Deploy Pod"**
3. **Select GPU Template**:
   - Recommended: PyTorch 2.0+ or TensorFlow with CUDA 11.8+
   - Choose GPU: RTX 3090 (24GB), RTX 4090 (24GB), or A100 (40GB/80GB)
   - Minimum: RTX 3060 (12GB) for smaller models
4. **Configure Storage**:
   - Minimum 50GB for code and models
   - Additional storage for data (mount via Network Volumes)
5. **Set Network Volume** (optional): For persistent storage across pod restarts

#### Step 2: Mount Cloud Storage (GCP) via RunPod GUI

1. In RunPod dashboard, go to your pod's **"Network Volumes"** or **"Storage"** section
2. Click **"Add Network Volume"** or **"Mount Storage"**
3. For **Google Cloud Storage**:
   - Select "Google Cloud Storage" as storage type
   - Enter your GCS bucket name
   - Configure mount point (e.g., `/workspace/gcs-data` or `/data`)
   - Authenticate using service account JSON key (upload via GUI) or OAuth
4. The mount will be automatically available when the pod starts

#### Step 3: Connect via SSH

1. In RunPod dashboard, find your pod's **"Connect"** or **"SSH"** section
2. Copy the SSH command (e.g., `ssh root@ssh.runpod.io -p <port>`)
3. **Set up SSH key** (recommended):
   - Go to **"Settings"** → **"SSH Keys"** in RunPod dashboard
   - Add your public SSH key (from `secrets/ssh_key.ssh.pub`)
4. Connect:
   ```bash
   ssh root@ssh.runpod.io -p <port>
   # Or with SSH key:
   ssh -i secrets/ssh_key.ssh root@ssh.runpod.io -p <port>
   ```

#### Step 4: Clone Repository and Setup

```bash
cd /workspace
git clone <your-repo-url> ImageRecognition
cd ImageRecognition

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

#### Step 5: Access Mounted Data

```bash
# Check mount (path depends on your RunPod configuration)
ls -lh /workspace/gcs-data  # or /data
# Verify data access
ls -lh /workspace/ImageRecognition/1_data/
```

#### Step 6: Configure Paths

- Most scripts use `/workspace/ImageRecognition` as base path (works out of the box)
- If data is mounted elsewhere, update paths in:
  - `2_preprocessing/patch_splitting/*.py`
  - `3_model/train_nn_inner_outer_gpu_pytorch.py`
  - `4_evaluation/*.py`

#### Step 7: Set Up WandB (Optional)

```bash
wandb login
# Enter your API key when prompted
```

## Workflow Guide

This section provides an overview of the complete workflow. For detailed step-by-step instructions with explanations of what happens between each step, see **[PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)**.

### Step 1: Data Preparation

#### 1.1 Prepare Input Data

**Prerequisites** (manual steps in QuPath, not automated):
- Whole slide images (WSI) scanned and stored (BigTIFF format, 0.25 μm/pixel)
- Manual annotation in QuPath to create:
  - Airway masks
  - Smooth muscle masks  
  - RBC (red blood cell) masks
- QuPath annotation scripts are provided in `2_preprocessing/qupath_annotation/` for reference

**Required for Modeling Pipeline**:
- Patient metadata file: `1_data/mapping.csv` with de-identified patient information
- Processed WSI and masks from QuPath (see manuscript for data repository location)

**Note**: QuPath annotation steps are performed manually and are not part of the automated pipeline. This documentation focuses on the modeling pipeline that processes the output from QuPath annotations.

#### 1.2 Create Dataset from WSI

Extract patches from whole slide images:

```bash
python 2_preprocessing/patch_splitting/1_create_data_set.py
```

**What it does**:
- Processes WSI images and corresponding masks
- Identifies tissue compartments (Inside/Outside airways)
- Prepares images for patch extraction

**Output**: `1_data/patches_{size}/patches_original/`

#### 1.3 Extract Patches Using Sliding Window

```bash
python 2_preprocessing/patch_splitting/2_sliding_window.py
```

**What it does**:
- Applies sliding window to extract patches from WSI
- Creates patches of specified size (default: 120x120 or 224x224)
- Separates patches into Inside/Outside compartments

**Output**: Patches in `patches_original/Inside/` and `patches_original/Outside/`

#### 1.4 Apply Data Imputation

Apply masking and imputation strategies:

```bash
python 2_preprocessing/patch_splitting/3_data_imputation.py
```

**What it does**:
- Applies tissue threshold filtering (removes patches with <threshold% tissue)
- Implements imputation strategies:
  - **Black**: Replace masked regions with black pixels
  - **Random Image**: Replace with random image patches from dataset
  - **Random Dataset**: Replace with random patches from same class
  - **No Masking**: Keep original patches without masking
- Creates separate datasets for each strategy

**Output**: `patches_cutoff_{strategy}_imputed_{threshold}/`

#### 1.5 Split Data into Train/Val/Test Sets

```bash
python 2_preprocessing/read_and_split_data.py
```

**What it does**:
- Splits data at **patient level** (prevents data leakage)
- Creates train/validation/test splits
- Resizes images to model input size (default: 224x224)
- Optionally balances classes
- Saves split information and summary statistics

**Output**: 
- `{data_path}/train_dir/`, `val_dir/`, `test_dir/`
- `{data_path}/summary_table.csv`

**Configuration**: Edit `CONFIG` dictionary in the script:
- `random_seed`: For reproducibility (default: 424242)
- `processing_methods`: Which imputation strategies to process
- `thresholds`: Tissue thresholds to use
- `partition_type`: 'patient', 'WSI', or 'patch'

### Step 2: Model Training

#### 2.1 Configure Training

Edit configuration files in `3_model/config/`:

- **Single run**: Edit `config_custom.json`
- **Grid search**: Edit `config_grid.json`
- **Default settings**: See `config_default.json`

Key parameters:
```json
{
  "org_size": [120, 120],           // Original patch size
  "size": [224, 224],                // Resized input size
  "data_imputation_type": "random_image",  // Imputation strategy
  "threshold": 0.2,                 // Tissue threshold
  "tf_model": "Inception-ResNet-V2", // Model architecture
  "full_retrain": true,             // Full fine-tuning vs transfer learning
  "freeze_till_block": 5,           // Layers to freeze
  "lr": 0.001,                      // Learning rate
  "batch_size": 64,                 // Batch size
  "epochs": 100,                    // Number of epochs
  "class_imb": "Oversample",        // Class imbalance handling
  "augmentation": "heavy"            // Data augmentation level
}
```

#### 2.2 Run Training

```bash
python 3_model/train_nn_inner_outer_gpu_pytorch.py
```

**What it does**:
- Loads configuration (supports Hydra configs)
- Builds model architecture
- Sets up data loaders with augmentation
- Trains model with validation monitoring
- Saves best models (by AUC and loss)
- Logs metrics to WandB
- Generates training visualizations

**Training Process**:
1. Model initialization (from pre-trained weights or scratch)
2. Training loop with validation
3. Best model checkpointing
4. Final model evaluation on test set
5. Results saving and visualization

**Output**: See [Results & Outputs](#results--outputs) section

### Step 3: Evaluation

#### 3.1 Evaluate Learning Ability

Comprehensive evaluation with GRAD-CAM and Integrated Gradients:

```bash
python 4_evaluation/evaluate_learning_ability.py
```

**What it does**:
- Evaluates model performance on test data
- Generates GRAD-CAM heatmaps for model attention
- Computes Integrated Gradients for attribution analysis
- Tests noise resistance (systematic noise addition)
- Analyzes control images (completely black images)
- Calculates information scores and ratios
- Aggregates results across images and compartments

**Output**: `5_results/learning_evaluation/{run_name}/`

#### 3.2 Visualize Compartment Differences

Statistical analysis of compartment differences:

```bash
python 4_evaluation/visualize_compartment_differences.py
```

**What it does**:
- Generates model predictions on test data
- Performs statistical analysis (bootstrap, PCA, mixed-effects models)
- Creates visualizations (confusion matrices, distributions, Grad-CAM)
- Compares patient groups (e.g., COPD vs normal)

**Configuration**: Requires config file in `4_evaluation/config/`

#### 3.3 Generate Statistical Plots

Create publication-ready plots with statistical tests:

```bash
python 4_evaluation/plot_plus_statistical_test.py
```

**What it does**:
- Creates bar plots with statistical significance indicators
- Performs mixed-effects analysis (accounting for patient-level clustering)
- Generates plots for Integrated Gradients and GRAD-CAM results
- Saves plots in multiple formats (PNG, SVG, PDF)

**Output**: Plots in `{output_dir}/information_plots/`

## Results & Outputs

### Training Results Structure

Training results are saved in `5_results/` with the following structure:

```
5_results/
├── {data_imputation_type}/              # e.g., 'black', 'random_image', 'no_masking'
│   └── threshold_{threshold}/           # e.g., 'threshold_0.2', 'threshold_0.3'
│       └── {timestamp}_{run_id}_patches_{size}/  # Individual training run
│           ├── best_auc_model.pth       # Best model by validation AUC
│           ├── best_loss_model.pth     # Best model by validation loss
│           ├── final_model.pth          # Final model after all epochs
│           ├── {run_id}_config.json    # Training configuration
│           ├── auc_development.png      # AUC curve over epochs
│           ├── loss_development.png     # Loss curve over epochs
│           ├── acc_class0_development.png  # Accuracy for class 0
│           ├── acc_class1_development.png  # Accuracy for class 1
│           ├── *.pkl                   # Pickled data for plots
│           └── wandb/                  # WandB experiment tracking data
│
├── learning_evaluation/                 # Evaluation analysis results
│   └── {evaluation_run_name}/
│       ├── gradcam_analysis_*.csv       # GRAD-CAM results
│       ├── integrated_gradients_analysis_*.csv  # Integrated Gradients results
│       ├── information_plots/           # Visualization plots
│       └── noise_resistance_*.csv      # Noise resistance analysis
│
└── completed_grid_runs/                 # Grid search tracking
    └── completed_grid_runs_{timestamp}.json
```

### Key Output Files

**Model Checkpoints**:
- `best_auc_model.pth`: Best model based on validation AUC
- `best_loss_model.pth`: Best model based on validation loss
- `final_model.pth`: Final model after all training epochs

**Training Metrics**:
- `auc_development.png`: Validation AUC over training epochs
- `loss_development.png`: Training and validation loss curves
- `acc_class0_development.png`: Accuracy for class 0 (e.g., Outside)
- `acc_class1_development.png`: Accuracy for class 1 (e.g., Inside)

**Configuration**:
- `{run_id}_config.json`: Complete training configuration for reproducibility

**Evaluation Results**:
- CSV files with detailed analysis results
- Aggregated statistics per image, WSI, and patient
- Information scores and attribution metrics

### Accessing Results

**From Python**:
```python
import torch
import pandas as pd

# Load trained model
model = torch.load('5_results/random_image/threshold_0.2/{run_folder}/best_auc_model.pth')

# Load evaluation results
results = pd.read_csv('5_results/learning_evaluation/{run_name}/gradcam_analysis_aggregated.csv')
```

**From Command Line**:
```bash
# List all training runs for a configuration
ls -lh 5_results/random_image/threshold_0.2/

# View training configuration
cat 5_results/random_image/threshold_0.2/{run_folder}/*_config.json

# Check WandB logs
wandb sync 5_results/random_image/threshold_0.2/{run_folder}/wandb/
```

## Configuration & Optimization

### Training Optimization

#### Batch Size and Memory

- **Reduce batch size** if running out of GPU memory:
  ```json
  "batch_size": 32  // or 16, 8 for smaller GPUs
  ```
- **Use gradient accumulation** for effective larger batch sizes (implement in training script)

#### Learning Rate

- **Start with default**: 0.001
- **Reduce if training unstable**: 0.0001 or 0.0005
- **Use learning rate scheduling**: Configured via `lr_patience`, `lr_factor` in config

#### Data Augmentation

- **Light**: Minimal augmentation (good for small datasets)
- **Medium**: Moderate augmentation (default)
- **Heavy**: Strong augmentation (good for large datasets, may slow training)

#### Mixed Precision Training

Enabled by default (`"mixed_precision": true`):
- Reduces memory usage
- Speeds up training
- May slightly reduce numerical precision

### Model Architecture Options

- **Inception-ResNet-V2**: Default, good balance of accuracy and speed
- **Other architectures**: Modify `tf_model` in config (requires model weights in `base_model_architectures/`)

### Transfer Learning vs Full Training

- **Transfer Learning** (`"full_retrain": false`):
  - Faster training
  - Less memory usage
  - Freeze early layers (`"freeze_till_block": 5`)
- **Full Fine-tuning** (`"full_retrain": true`):
  - Better for domain-specific adaptation
  - Requires more memory and time

### Grid Search

To run multiple configurations:

1. Edit `3_model/config/config_grid.json`
2. Define parameter ranges:
   ```json
   {
     "threshold": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
     "data_imputation_type": ["black", "random_image"],
     "lr": [0.001, 0.0005]
   }
   ```
3. Run training script (it will iterate through all combinations)
4. Results tracked in `5_results/completed_grid_runs/`

### Evaluation Optimization

- **Batch processing**: Evaluation scripts process images in batches to manage memory
- **GPU utilization**: Ensure GPU is available for GRAD-CAM and Integrated Gradients
- **Parallel processing**: Some scripts support multiprocessing (check script documentation)

### Storage Optimization

- **Delete redundant model files**: Use `3_model/delete_redundant_wandb_models.py` to clean up
- **Archive old results**: Move completed runs to archive storage
- **Use WandB cloud**: Sync logs to WandB cloud instead of storing locally

## Data Availability

The whole slide images (WSIs) and their manual annotations used in this study can be accessed via the repository specified in the manuscript (see manuscript Data Availability section).

**Important Notes**:
- **WSI Location**: See manuscript for exact repository location and access instructions
- **Patient Data**: All patient data has been completely de-identified. Patient identifiers embedded in WSIs have been removed to ensure patient privacy
- **Individual Patches**: Image patches can be made available from the corresponding author upon reasonable request
- **Dataset Size**: 162 WSIs from 75 patients (see manuscript Supplementary Table 1 for patient characteristics)
- **Patient Metadata**: `1_data/mapping.csv` contains de-identified patient metadata
- **Patient Partitions**: `1_data/patient_partitions_424242.xlsx` contains the exact train/val/test splits used in the manuscript (60/20/20 split, patient-level)


**Requirements**:
- GPU computation is necessary for efficient computation
- See [Requirements](#requirements) section for system and software requirements
- See [Installation & Setup](#installation--setup) for detailed setup instructions

## Citation

If you use this code in your research, please cite:

```bibtex
@article{vanbreugel2024,
  title={A deep learning framework for histopathological analysis of pixel-level extracellular matrix variation in standard H\&E-stained images},
  author={van Breugel, Merlijn and de Jong, Esm{\'e}e and Buikema, Henk J. and Petoukhov, Ilya and Nawijn, Martijn C. and Burgess, Janette K. and Timens, Wim},
  journal={[Journal Name]},
  year={2026},
  note={Code available at: [repository URL]}
}
```

## Reproducing Manuscript Results

For complete step-by-step instructions to reproduce the manuscript results, see **[REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)**.

**Quick Checklist**:
1. **Use exact configuration**: See `REPRODUCTION_GUIDE.md` for manuscript-specific configuration details
2. **Use patient partitions**: The file `1_data/patient_partitions_424242.xlsx` contains the exact patient-level splits used in the manuscript
3. **Set random seed**: Ensure `random_seed: 424242` is used throughout (default in `read_and_split_data.py`)
4. **Use correct imputation method**: See `REPRODUCTION_GUIDE.md` for which method was used for manuscript results
5. **Verify hyperparameters**: See `3_model/config/config_default.json` for default hyperparameters (verify against manuscript Section 2.3.2)


## Contact

For questions, issues, or collaboration inquiries:

**Merlijn van Breugel**
- Email: merlijnvanbreugel@gmail.com
- UMCG: m.van.breugel@umcg.nl