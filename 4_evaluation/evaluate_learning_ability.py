"""
Evaluation Framework for Neural Network Learning Analysis
======================================================

This module provides a comprehensive framework for evaluating how neural networks
learn to distinguish between tissue and artifact regions in medical images. It 
includes multiple analysis techniques and visualization methods.

Key Components
-------------
1. Model Evaluation
   - AUC performance testing across different thresholds
   - Collection and aggregation of model results
   - Visualization of performance metrics

2. Noise Resistance Analysis
   - Systematic testing with incrementally added noise (0-95%)
   - Multiple iterations per noise level
   - Separate analysis for different artifact types
   - Visualization of model robustness

3. GRAD-CAM Analysis
   - Heatmap generation for model attention
   - Batch processing capability
   - Analysis of tissue vs. artifact regions
   - Information score calculations and ratios
   - Visualization of attention patterns

4. Integrated Gradients Analysis
   - Attribution calculation for model decisions
   - Batch processing implementation
   - Comparison of tissue vs. artifact attributions
   - Visualization of attribution patterns

5. Control Image Analysis
   - Testing with completely black images
   - Comparison of GRAD-CAM and Integrated Gradients results
   - Statistical analysis of control results

Main Functions
-------------
evaluate_models(data_type, threshold, config, model):
    Evaluates model performance on test data for given configurations.

analyze_noise_resistance(data_type, threshold, config, model_dir, model):
    Tests model robustness against different levels of noise.

perform_gradcam_analysis(config, output_path, model):
    Conducts GRAD-CAM analysis for understanding model attention.

perform_integrated_gradients_analysis(config, output_path, model):
    Performs attribution analysis using Integrated Gradients.

analyze_control_images(model, config, output_path):
    Analyzes model behavior on control (black) images.

Helper Functions
--------------
load_model(): Loads and configures model for gradient computation
load_config(): Loads configuration parameters
read_img(): Reads and processes input images
get_palette(): Extracts color palette from images
calculate_information_scores(): Computes various attention metrics
aggregate_heatmap_scores(): Aggregates results across images
plot_information_score_ratios(): Visualizes attention patterns

Usage
-----
Run as main script:
    python evaluate_learning_ability.py

The script will:
1. Load configuration and model
2. Perform comprehensive evaluation
3. Generate visualizations and results
4. Save analysis outputs to specified directories

Requirements
-----------
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- tqdm
- pytorch-grad-cam
- captum

Notes
-----
- Requires GPU for efficient processing
- Expects specific directory structure for data and models
- Some functions may be memory-intensive for large datasets
"""

# Import packages
import sys
import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import logging
import timm
import itertools 

from sklearn.metrics import confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import json
from datetime import datetime

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import shutil
from PIL import Image

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

import traceback

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    BASE_DIR, DATA_DIR, RESULTS_DIR, MODEL_WEIGHTS_DIR,
    get_imputed_patches_path, get_model_path,
    setup_environment
)

# Setup environment
setup_environment(verbose=False)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path):
    """Load model and ensure it's properly configured for gradient computation"""
    model = torch.load(model_path)
    # Enable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = True
    return model

# Configuration parameters
def load_config():
    """
    Load configuration parameters for model evaluation.
    Status: Working

    This function defines all the necessary parameters for:
    - Image processing (sizes, thresholds)
    - Model architecture and training
    - Evaluation settings (sample sizes, iterations)
    - File paths and directory structure
    
    Returns:
        dict: Configuration parameters organized by category
    """
    config = {
        # Base directory for all operations (from config module)
        'base_dir': str(BASE_DIR),

        # Image dimensions for processing
        'original_size': [120, 120],  # Size of input patches
        'size': [224, 224],     # Required size for Inception/ResNet models
        
        # Evaluation parameters
        'thresholds': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # Tissue pixel thresholds
        'data_types': ['black', 'random_image'], # ['random_image', 'black'],                    # Types of artifacts
        'n_samples': 1000,   # Number of samples per evaluation for making heatmaps
        'n_iterations': 5, #50,   # Number of repetitions for noise analysis

        # Model parameters
        'tf_model': 'Inception-ResNet-V2',
        'batch_size': 64,

        # File paths (relative to base_dir)
        'models_evaluation_selection': str(RESULTS_DIR / 'models_evaluation_selection.json'),
        'grid_run_selection': str(RESULTS_DIR / 'completed_grid_runs/completed_grid_runs_20241118_123457_threshold_selection_new.json'),

        'inf_data_partition': 'val_dir', # 'val_dir' or 'test_dir', or None for all images,
        'same_data_type_threshold': False # If True, only evaluate models with the same data_type and threshold as the model being evaluated
    }
    
    # Add computed paths using lambda functions for dynamic path generation
    config['paths'] = {
        'data': lambda cfg, data_type, threshold: (
            Path(cfg['base_dir']) / 
            "1_data" /
            f"patches_{cfg['original_size'][0]}x{cfg['original_size'][1]}" /
            f"patches_cutoff_{data_type}_imputed_{threshold}"
        ),
        'model': lambda cfg, data_type, threshold: (
            Path(cfg['base_dir']) / 
            "5_results" /
            f"{data_type}" /
            f"threshold_{threshold}"
        ),
        'results': lambda cfg: Path(cfg['base_dir']) / "5_results",
        'data_original': lambda cfg: Path(cfg['base_dir']) / "1_data" / f"patches_{cfg['original_size'][0]}x{cfg['original_size'][1]}" / "patches_original",
        'output': lambda cfg: Path(cfg['base_dir']) / "5_results" / "learning_evaluation"
    }
    
    return config

# Function to read single image
def read_img(img_path, size, resize=True):
    """
    Read and optionally resize an image.
    
    Args:
        img_path (str/Path): Path to the image file
        size (tuple): Target size (width, height)
        resize (bool): Whether to resize the image
        
    Returns:
        numpy.ndarray: Image array in BGR format
    """
    img = cv2.imread(str(img_path))
    if resize:
        img = cv2.resize(img, (size[0], size[1]))
    return img

def read_to_coloured_image(path):
    """
    Load image and ensure it's RGB (3 channels), even for grayscale images.
    
    Args:
        path: Path to the image file
        
    Returns:
        np.ndarray: RGB image with shape (H, W, 3)
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # or however you currently load images
    # Convert grayscale to RGB by stacking the same values 3 times
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)  # Shape becomes (H, W, 3)
    return img

# Function to get all RGB codes from patches
def get_palette(img):
    colors = set()
    palette = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    palette = np.unique(palette, axis=0)
    for color in palette:
        if list(color) != [0,0,0]:
            colors.add(tuple(list(color)))
    return list(colors)

# Function to apply grad-cam (heatmaps visualization)
def grad_cam_individual_img(image, model, target_layer=None, eps=1e-8):
    """
    Generate Grad-CAM visualization for a single image, based on batch processing approach.
    
    Args:
        image: Input image as numpy array (H, W, C) in RGB format
        model: PyTorch model
        target_layer: Optional specific layer to analyze
        eps: Small value to prevent division by zero
        
    Returns:
        tuple: (original heatmap, colored heatmap overlay)
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Find target layer if not specified
    if target_layer is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = [module]
                break
    
    try:
        # Initialize GradCAM once
        cam = GradCAM(model=model, target_layers=target_layer)
        
        # Prepare image tensor (similar to batch processing)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Convert and normalize image
        if isinstance(image, np.ndarray):
            if image.max() > 1:
                image = image / 255.0
            # Convert numpy array to PIL Image
            img_pil = Image.fromarray((image * 255).astype('uint8'))
            img_tensor = transform(img_pil)
        else:
            img_tensor = image
            
        # Add batch dimension
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
            
        # Move to device and ensure gradients
        img_tensor = img_tensor.to(device).requires_grad_(True)
        
        # Generate GradCAM heatmap
        grayscale_cam = cam(input_tensor=img_tensor)
        heatmap_org = grayscale_cam[0]
        
        # Create colored visualization
        if isinstance(image, np.ndarray):
            vis_image = image
        else:
            # Convert tensor to numpy for visualization
            vis_image = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Denormalize
            vis_image = (vis_image * np.array([0.229, 0.224, 0.225]) + 
                        np.array([0.485, 0.456, 0.406]))
            vis_image = np.clip(vis_image, 0, 1)
            
        heatmap_col = show_cam_on_image(vis_image, heatmap_org, use_rgb=True)
        
        return heatmap_org, heatmap_col
        
    except Exception as e:
        logger.error(f"Error in grad_cam: {str(e)}")
        raise
        
    finally:
        # Clean up resources
        if 'cam' in locals():
            del cam
        torch.cuda.empty_cache()


"""# Test AUC per data type and threshold"""
def collect_model_results(config):
    """
    Collect and aggregate model performance results using paths from JSON file.
    
    Args:
        config (dict): Configuration dictionary containing paths and parameters
    
    Returns:
        tuple: DataFrames containing results for random and black artifacts
    """
    results = {dt: [] for dt in config['data_types']}
    json_path = Path(config['base_dir']) / config['models_evaluation_selection']
    
    # Load JSON file containing model paths
    with open(json_path, 'r') as f:
        model_selections = json.load(f)
    
    # Process each model path in the JSON
    for model_base_path, selection in model_selections.items():
        data_type = selection['data_type']
        threshold = selection['threshold']
        folder = selection['folder']
        
        if data_type not in config['data_types']:
            continue
            
        # Construct full model path
        model_path = Path(model_base_path) / folder
        
        try:
            # Load performance results
            perf_results = pd.read_csv(
                model_path / 'evaluation_best_auc_model.csv', 
                index_col=False
            )
            
            # Add data_type and threshold columns to each row in perf_results
            for _, row in perf_results.iterrows():
                result_row = np.append(row.values, [data_type, threshold])
                results[data_type].append(result_row)
            
        except FileNotFoundError:
            logger.warning(f"Could not find results for {model_path}")
            continue
    
    # Convert results to DataFrames with all columns including partition
    result_columns = ['partition', 'loss', 'tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 'auc', 'data_type', 'threshold']
    random_results_df = pd.DataFrame.from_records(results['random_image'], columns=result_columns)
    black_results_df = pd.DataFrame.from_records(results['black'], columns=result_columns)
    
    # Save random results to CSV
    random_results_path = config['output_path'] / 'random_artifacts_results.csv'
    random_results_df.to_csv(random_results_path, index=False)
    logger.info(f"Saved random artifacts results to {random_results_path}")

    # Save black artifacts results to CSV 
    black_results_path = config['output_path'] / 'black_artifacts_results.csv'
    black_results_df.to_csv(black_results_path, index=False)
    logger.info(f"Saved black artifacts results to {black_results_path}")

    return random_results_df, black_results_df
    
def plot_auc_performance(thresholds, random_results_df, black_results_df, output_path):
    """
    Create separate bar plots for AUC performance on validation and test sets for each artifact type.
    
    Args:
        thresholds: List of threshold values
        random_results_df: DataFrame containing random artifacts results
        black_results_df: DataFrame containing black artifacts results
        output_path: Path to save the output plots
    """
    # Set width for bars
    bar_width = 0.35
    
    # Create positions for bars
    positions = np.arange(len(thresholds))
    
    # Create separate plots for each artifact type
    for data_type, df in [('Random', random_results_df), ('Black', black_results_df)]:
        plt.figure(figsize=(10, 6))
        
        # Get validation and test data
        val_data = df[df.partition == 'val'].auc.values
        test_data = df[df.partition == 'test'].auc.values
        
        # Create bars
        plt.bar(positions - bar_width/2, val_data, bar_width, 
                label='Validation', color='#ff7f0e')
        plt.bar(positions + bar_width/2, test_data, bar_width,
                label='Test', color='green')
        
        # Customize plot
        plt.title(f'AUC Performance - {data_type} Artifacts')
        plt.ylabel('AUC')
        plt.xlabel('Threshold % tissue pixels')
        plt.xticks(positions, thresholds)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on top of bars
        for i in positions:
            plt.text(i - bar_width/2, val_data[i] - 0.05, f'{val_data[i]:.3f}', 
                    ha='center', va='top', rotation=90)
            plt.text(i + bar_width/2, test_data[i] - 0.05, f'{test_data[i]:.3f}', 
                    ha='center', va='top', rotation=90)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'AUC_performance_{data_type.lower()}_artifacts.png'), dpi=300)
        plt.close()

def select_model_subfolder(model_path, config, data_type, threshold, use_json=True):
    """
    Select the appropriate subfolder within the model path.

    Args:
        model_path (Path): The base path where model subfolders are located.
        use_json (bool): Whether to use a JSON file for selection.

    Returns:
        Path: The selected subfolder path.
    """
    json_path = Path(config['base_dir']) / config['models_evaluation_selection']

    if use_json:
        with open(json_path, 'r') as f:
            selection_data = json.load(f)
            
        # Get the model info using the path as key
        model_key = str(model_path)  # Convert Path to string to match JSON keys
        
        if model_key not in selection_data:
            raise ValueError(f"No matching folder found in JSON for {model_key}")
            
        model_info = selection_data[model_key]

        # Otherwise use the default folder
        return model_path / model_info['folder']
    else:
        # Find the most recent subfolder based on alphabetical order
        subfolders = [f for f in model_path.iterdir() if f.is_dir()]
        subfolders.sort(reverse=True)  # Sort alphabetically in descending order
        
        # Load existing selections or create new structure
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}
            
        # Add new selection using full path as key
        full_path = str(model_path)  # Convert Path object to string
        data[full_path] = {
            'data_type': data_type,
            'threshold': threshold,
            'folder': subfolders[0].name
        }
        
        # Save updated selections
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

        return subfolders[0] if subfolders else None


def plot_information_score_ratios(combined_agg, thresholds, output_path):
    """
    Plot information score ratios for different compartments and artifact types.
    """
    # Split data by data_type
    random_results_df = combined_agg[combined_agg['data_type'] == 'random_image']
    black_results_df = combined_agg[combined_agg['data_type'] == 'black']
    
    metrics = [
        {
            'column': 'avg_ratio_inf_score_black_vs_tissue',
            'title_suffix': 'black vs tissue',
            'ylabel': 'Avg ratio information score black vs tissue pixels'
        },
        {
            'column': 'avg_ratio_inf_score_black_vs_max_tissue',
            'title_suffix': 'black vs max tissue',
            'ylabel': 'Avg ratio information score black vs max tissue pixels'
        }
    ]
    
    for compartment in random_results_df.compartment.unique():
        for metric in metrics:
            plt.figure()
            
            # Filter data for current compartment and threshold
            random_data = []
            black_data = []
            
            for threshold in thresholds:
                # Get data for each threshold
                random_val = random_results_df[
                    (random_results_df.compartment == compartment) & 
                    (random_results_df.threshold == threshold)
                ][metric['column']].values
                
                black_val = black_results_df[
                    (black_results_df.compartment == compartment) & 
                    (black_results_df.threshold == threshold)
                ][metric['column']].values
                
                random_data.append(random_val[0] if len(random_val) > 0 else 0)
                black_data.append(black_val[0] if len(black_val) > 0 else 0)
            
            # Create plot
            plt.plot(thresholds, random_data, label='Random artifacts')
            plt.plot(thresholds, black_data, label='Black artifacts')
            
            # Set labels and title
            plt.title(f"{compartment} - {metric['title_suffix']}")
            plt.ylabel(metric['ylabel'])
            plt.xlabel('Threshold % tissue pixels')
            plt.legend(loc='upper left')
            
            # Save plot
            filename = (f"{compartment}_full_set_non_standardized_ratio_"
                       f"{metric['title_suffix'].replace(' ', '_')}_"
                       f"information_score_per_threshold.png")
            plt.savefig(output_path / filename)
            plt.close()


def evaluate_models(data_type, threshold, config, model, rerun_eval = True, model_folder=None):
    """
    Evaluate both AUC-optimized and loss-optimized models on the test set.
    
    This function:
    1. Sets up data generators for test data
    2. Loads and evaluates the best AUC model
    3. Loads and evaluates the best loss model
    4. Saves evaluation results to CSV files
    
    Args:
        data_type (str): Type of data ('black' or 'random')
        threshold (float): Threshold value for tissue pixels
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (auc_df, loss_df) DataFrames containing evaluation metrics
    """
    logger.info(f"Evaluating models for {data_type} data (threshold={threshold})")
    
    # Get paths from config
    data_path = config['paths']['data'](config, data_type, threshold)
    model_base_path = config['paths']['model'](config, data_type, threshold)
    
    # Select the appropriate model subfolder
    # TO DO: remove model_folder from config and use only use_json, now redundant I think
    if model_folder is not None:
        model_path = model_base_path / model_folder
    else:
        model_path = select_model_subfolder(model_base_path, config, data_type, threshold, use_json=True)
    
    # Check if evaluation files already exist
    auc_results_path = model_path / 'evaluation_best_auc_model.csv'
    loss_results_path = model_path / 'evaluation_best_loss_model.csv'
    
    # Load existing results if available
    if auc_results_path.exists() and loss_results_path.exists() and not rerun_eval:
        logger.info("Found existing evaluation results - loading from files")
        auc_df = pd.read_csv(auc_results_path)
        loss_df = pd.read_csv(loss_results_path)
        return auc_df, loss_df

    print(f"Selected model subfolder: {model_path}")
    # Setup data transforms
    data_transforms = transforms.Compose([
        transforms.Resize(config['size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Setup datasets and dataloaders for all partitions
    partitions = {
        'train': 'train_dir',
        'val': 'val_dir',
        'test': 'test_dir'
    }
        
    # In evaluate_models function, modify the dataloaders setup:
    dataloaders = {
        partition: DataLoader(
            ImageFolder(data_path / dir_name, transform=data_transforms),
            batch_size=500,
            shuffle=False,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True  # Keep workers alive between batches
        )
        for partition, dir_name in partitions.items()
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Lists to store results for each partition
    auc_results = []
    loss_results = []
    
    # Evaluate both models on all partitions
    for model_type in ['auc', 'loss']:
        logger.info(f"Evaluating {model_type}-optimized model")
        checkpoint = torch.load(model_path / f'best_{model_type}_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Evaluate on each partition
        for partition, dataloader in dataloaders.items():
            logger.info(f"Evaluating on {partition} set")
            metrics = evaluate_model(model, dataloader, device)
            metrics['partition'] = partition
            
            if model_type == 'auc':
                auc_results.append(metrics)
            else:
                loss_results.append(metrics)
            
            logger.info(f"{model_type.upper()} model achieved {metrics['auc']:.4f} AUC on {partition} set")
    
    # Create results DataFrames
    columns = ['partition', 'loss', 'tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 'auc']
    auc_df = pd.DataFrame(auc_results)[columns]
    loss_df = pd.DataFrame(loss_results)[columns]
    
    # Save results
    auc_df.to_csv(model_path / 'evaluation_best_auc_model.csv', index=False)
    loss_df.to_csv(model_path / 'evaluation_best_loss_model.csv', index=False)
    
    logger.info("Evaluation complete for all partitions")
    return auc_df, loss_df, model_path

def evaluate_model(model, dataloader, device):
    """
    Evaluate a PyTorch model on the test set with optimized GPU usage
    """
    model.eval()  # Ensure model is in evaluation mode
    
    # Initialize metrics
    total_loss = 0
    all_labels = []
    all_probs = []
    criterion = nn.BCEWithLogitsLoss().to(device)  # Move criterion to GPU
    
    # Process batches more efficiently
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in dataloader:
            # Move data to device and process entire batch at once
            inputs = inputs.to(device, non_blocking=True)  # Enable async data transfer
            labels = labels.float().to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            
            # Store predictions (keep on GPU until end of batch)
            probs = torch.sigmoid(outputs)
            all_labels.append(labels)
            all_probs.append(probs)
    
    # Concatenate all batches efficiently on GPU first, then move to CPU
    y_true = torch.cat(all_labels).cpu().numpy()
    y_pred = torch.cat(all_probs).cpu().numpy()
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    n_samples = len(dataloader.dataset)
    
    return {
        'loss': total_loss / n_samples,
        'tp': tp, 
        'fp': fp, 
        'tn': tn, 
        'fn': fn,
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'auc': roc_auc_score(y_true, y_pred)
    }

def analyze_gradcam_batch(data_type, threshold, compartment, config, model_dir, model, batch_size=200, inf_data_partition='val_dir', target_layers='middle'):
    """
    Analyze a batch of images using GRAD-CAM efficiently with true batch processing.

    Args:
        inf_data_partition (str): 'val_dir' or 'test_dir', or None for all images
    """
    logger.info(f"Starting batch GRAD-CAM analysis for {data_type}, {compartment}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Enable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Setup data paths
    data_path = config['paths']['data'](config, data_type, threshold)
    if inf_data_partition is not None:  
        data_path = data_path / inf_data_partition

    black_data_path = config['paths']['data'](config, 'black', threshold)
    if inf_data_partition is not None:
        black_data_path = black_data_path / inf_data_partition
    
    transform = transforms.Compose([
        transforms.Resize(config['size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Get image paths for the compartment
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f)) 
              and '_dir' not in f and '_oversample' not in f and compartment in f]
    
    result_per_img = []
    
    # Option 1: Use multiple conv layers for a multi-scale analysis
    def get_last_layers(model):
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)
        
        # Can choose:
        return [conv_layers[-1]]  # Last layer (current approach)
        # return [conv_layers[-3], conv_layers[-2], conv_layers[-1]]  # Last few layers
        # return conv_layers  # All conv layers

    # Option 2: Use layer closest to middle of network
    def get_middle_conv_layer(model):
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)
        
        middle_idx = len(conv_layers) // 2
        return [conv_layers[middle_idx]]
    # The middle layer is default and if often best at more local and fine-grained features

    if target_layers == 'last':
        target_layer = get_last_layers(model)
    elif target_layers == 'middle':
        target_layer = get_middle_conv_layer(model)
    else:
        raise ValueError(f"Invalid target_layers value: {target_layers}")
    
    
    # Initialize GradCAM once
    cam = GradCAM(model=model, target_layers=target_layer)
    
    try:
        for folder in folders:
            logger.info(f'Processing {folder}')
            pathlist = list(Path(data_path / folder).glob('**/*.png'))

            # Calculate number of batches needed to reach n_samples
            total_samples_needed = min(config['n_samples'], len(pathlist))
            print(f"Total samples available for {folder}: {len(pathlist)}")
            num_batches = (total_samples_needed + batch_size - 1) // batch_size  # Ceiling division
            
            # Select random samples for all batches at once
            pathlist_selection = random.sample(pathlist, total_samples_needed)
            
            # Process in batches
            progress_bar = tqdm(
                range(0, num_batches), 
                desc=f"Processing {folder}",
                unit="batch"
            )

            for i in progress_bar:
                # Clear GPU cache at the start of each batch
                torch.cuda.empty_cache()

                batch_paths = pathlist_selection[i:i + batch_size]
                current_batch_size = len(batch_paths)
                
                try:
                    # Prepare batch tensors
                    batch_imgs = torch.zeros((current_batch_size, 3, config['size'][0], config['size'][1]))
                    batch_masks = torch.zeros((current_batch_size, config['size'][0], config['size'][1]))
                    
                    # Load and preprocess all images in batch
                    for j, img_path in enumerate(batch_paths):
                        # Load original image
                        img = cv2.imread(str(img_path))
                        if j == 0:
                            first_batch_img = img

                        if img is None:
                            logger.error(f"Failed to load image: {img_path}")
                            continue
                            
                        # Convert BGR to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Resize image to model size
                        target_size = (config['size'][1], config['size'][0])  # (width, height)
                        
                        # Load corresponding black image and create mask
                        black_path = str(img_path).replace(str(data_path), str(black_data_path))
                        black_img = cv2.imread(black_path)
                        if black_img is None:
                            logger.error(f"Failed to load black image: {black_path}")
                            continue
                            
                        # Resize black image to same size as model input
                        black_img = cv2.resize(black_img, target_size)
                        
                        # Calculate binary mask (resized to match model size)
                        gray = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
                        mask = (gray > 0).astype(np.float32)
                        
                        # Convert numpy array to PIL Image before applying transform
                        img_pil = Image.fromarray((img * 255).astype('uint8'))
                        img_tensor = transform(img_pil)  # This will handle the normalization
                        batch_imgs[j] = img_tensor
                        batch_masks[j] = torch.from_numpy(mask)
                    
                    # Move batch to device
                    batch_imgs = batch_imgs.to(device)
                    batch_masks = batch_masks.to(device)
                    
                    # Generate GradCAM heatmaps for entire batch
                    try:
                        grayscale_cams = cam(input_tensor=batch_imgs)
                        
                        # Convert entire batch of grayscale maps to uint8 (0-255 range)
                        heatmaps_uint8 = (grayscale_cams * 255).astype(np.uint8)
                        
                        # Initialize array for colored heatmaps
                        colored_cams = np.zeros((current_batch_size, 
                                               config['size'][0], 
                                               config['size'][1], 
                                               3), dtype=np.uint8)
                        
                        # Vectorized resize and colormap application
                        for k in range(current_batch_size):
                            # Resize in one step
                            resized = cv2.resize(heatmaps_uint8[k], 
                                               (config['size'][0], config['size'][1]))
                            # Apply colormap (this can't be vectorized due to OpenCV limitations)
                            colored_cams[k] = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
   
                    except Exception as e:
                        logger.error(f"GradCAM failed with error: {str(e)}")
                        raise
                    
                    # Calculate metrics for entire batch
                    for j in range(current_batch_size):               
                        bw = batch_masks[j].cpu().numpy()
                        bw_reverse = 1 - bw
                        nr_tissue_pixels = np.sum(bw)
                        nr_black_pixels = np.sum(bw_reverse)
                        
                        scores = calculate_information_scores(
                            grayscale_cams[j],          # The grayscale GradCAM heatmap for this image
                            colored_cams[j],            # Colored heatmap overlay
                            bw,                         # Binary mask showing tissue pixels (1s) vs background (0s)
                            bw_reverse,                 # Inverse binary mask showing background pixels (1s) vs tissue (0s)
                            nr_tissue_pixels,           # Total count of tissue pixels in the mask
                            nr_black_pixels,            # Total count of background/black pixels in the mask
                            batch_imgs[j].cpu().numpy().transpose(1, 2, 0)  # Original image converted from tensor to numpy array
                        )
                        
                        result = [data_type, threshold, batch_paths[j].parent.name, 
                                batch_paths[j].name] + list(scores.values())
                        result_per_img.append(result)
                    
                        # Save visualization for first image in batch
                        if j == 0:                       
                            save_example_heatmap_visualization(
                                first_batch_img,
                                colored_cams[0],
                                batch_masks[0].cpu().numpy(),
                                grayscale_cams[0],
                                config['output_path'] / "example_heatmaps",
                                batch_paths[0].stem
                            )
                            # Save the scores of a black image, only do this once
                            complete_black_image_cam(cam, device, config)

                    # Clean up batch resources
                    del batch_imgs, batch_masks, grayscale_cams
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    continue

    finally:
        # Clean up resources
        del cam
        torch.cuda.empty_cache()
    
    # Create results DataFrames
    columns = ['data_type', 'threshold', 'compartment', 'filename'] + list(scores.keys())
    results_df = pd.DataFrame(result_per_img, columns=columns)

    # Create data type subfolder if it doesn't exist
    output_dir = config['paths']['output'](config) / data_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results to data type subfolder
    results_df.to_csv(output_dir / f'threshold={threshold}_compartment={compartment}_full_set_heatmap_information_scores_per_img.csv')
    
    # Calculate aggregated results
    agg_results = aggregate_heatmap_results(results_df)
    agg_results.to_csv(output_dir / f'threshold={threshold}_compartment={compartment}_full_set_agg_heatmap_information_scores.csv')

    return results_df, agg_results

def complete_black_image_cam(cam, device, config):
    """
    Analyze GRAD-CAM scores for a completely black image.
    
    Args:
        cam: Initialized GradCAM object
        device: PyTorch device (cuda/cpu)
        config: Configuration dictionary containing image size
        
    Returns:
        dict: GRAD-CAM scores for the black image
    """
    # Create a black image tensor with correct dimensions
    black_img = np.zeros((config['size'][0], config['size'][1], 3), dtype=np.uint8)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    results = []
    n_iterations = 1  # Number of times to repeat analysis for stability
    
    for i in range(n_iterations):
        try:
            # Convert black image to PIL Image first
            black_pil = Image.fromarray(black_img)
            img_tensor = transform(black_pil).unsqueeze(0).to(device)
            
            # Generate GradCAM heatmap
            grayscale_cam = cam(input_tensor=img_tensor)
            heatmap_org = grayscale_cam[0]

            if i==0:
                first_black_img = heatmap_org
                plt.imsave(config['output_path'] / 'control_black_image_heatmap.png', first_black_img, cmap='jet')
            
            # Create colored visualization (ensure matching dimensions)
            heatmap_uint8 = (heatmap_org * 255).astype(np.uint8)
            colored_cam = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            
            # Calculate scores
            scores = calculate_information_scores(
                heatmap_org,                    # The grayscale GradCAM heatmap
                colored_cam,                    # Colored heatmap
                np.zeros_like(heatmap_org),     # All black mask
                np.ones_like(heatmap_org),      # All ones mask (inverse of black)
                0,                              # No tissue pixels
                heatmap_org.size,               # All pixels are black
                black_img / 255.0               # Normalized black image as original
            )
            
            results.append({
                'iteration': i,
                **scores
            })
            
        except Exception as e:
            logger.error(f"Error in control image analysis: {str(e)}\n{traceback.format_exc()}")
            continue
    
    if not results:
        logger.warning("No successful iterations in control image analysis")
        return pd.DataFrame()
        
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(config['output_path'] / 'control_black_image_analysis.csv')
    
    return results_df

def save_example_heatmap_visualization(original_img, colored_cam, mask, grayscale_cam, 
                                                        output_path, filename, empty_output_path=True):
    """
    Create and save a 4-panel visualization showing original image, colored GradCAM,
    binary mask, and grayscale GradCAM.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Create output directory if it doesn't exist
    if empty_output_path:       
        if output_path.exists():
            shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot each panel
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Colored GradCAM
    axes[1].imshow(colored_cam)
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')
    
    # Binary mask
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Binary Mask')
    axes[2].axis('off')
    
    # Grayscale GradCAM
    axes[3].imshow(grayscale_cam, cmap='gray')
    axes[3].set_title('Grayscale GradCAM')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_path / f'gradcam_visualization_{filename}.png'
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()

def aggregate_heatmap_results(results_df):
    """
    Aggregate results per compartment and create a total aggregation.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing per-image results
        
    Returns:
        pd.DataFrame: Aggregated results including per-compartment and total statistics
    """
    # Define aggregation functions without nesting
    agg_funcs = {
        'min_information_score_total': 'min',
        'max_information_score_total': 'max',
        'nr_tissue_pixels': 'sum',
        'information_score_tissue': 'mean',
        'max_tissue': lambda x: (x - results_df.information_score_tissue).sum(),
        'nr_black_pixels': 'sum',
        'information_score_black': 'mean',
        'ratio_inf_score_black_vs_tissue': 'mean',
        'ratio_inf_score_black_vs_max_tissue': 'mean',
        'ratio_black_vs_tissue_attr': ['mean', 'median', 'std'],  # Use list for multiple aggregations
        'ratio_tissue_vs_black_attr': ['mean', 'median', 'std'],  # Use list for multiple aggregations
        'total_attribution': 'mean',
        'perc_tissue_attribution': ['mean', 'median', 'std'],
        'perc_black_attribution': ['mean', 'median', 'std'],
        'perc_tissue_attr_total_attribution': ['mean', 'median', 'std'],
        'perc_black_attr_total_attribution': ['mean', 'median', 'std'],
        'rel_perc_black_attr_total_attribution': ['mean', 'median', 'std']
    }
    
    # Round noise_pixels_perc for grouping
    results_df['noise_pixels_perc_rnd'] = results_df['noise_pixels_perc'].round(1)
    
    # Aggregate by compartment
    compartment_agg = (results_df.groupby(['data_type', 'threshold', 'compartment'])
                      .agg(agg_funcs)
                      .reset_index())
    
    # Aggregate total (across all compartments)
    total_agg = (results_df.groupby(['data_type', 'threshold'])
                 .agg(agg_funcs)
                 .reset_index()
                 .assign(compartment='Total'))
    
    # Combine results
    agg_results = pd.concat([compartment_agg, total_agg], ignore_index=True)
    
    # Flatten multi-level column names
    agg_results.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col 
        for col in agg_results.columns
    ]
    agg_results.columns = [col[:-1] if col.endswith('_') else col for col in agg_results.columns]

    # print(agg_results.columns)
    # Rename columns to match the old naming convention
    column_mapping = {
        'min_information_score_total_min': 'min_information_score_total',
        'max_information_score_total_max': 'max_information_score_total',
        'nr_tissue_pixels_sum': 'tot_nr_tissue_pixels',
        'information_score_tissue_mean': 'avg_information_score_tissue',  # Changed this line
        'max_tissue_<lambda>': 'max_minus_avg_information_score_tissue',  # Note: may need to verify this name
        'nr_black_pixels_sum': 'tot_nr_black_pixels',
        'information_score_black_mean': 'avg_information_score_black',
        'ratio_inf_score_black_vs_tissue_mean': 'avg_ratio_inf_score_black_vs_tissue',
        'ratio_inf_score_black_vs_max_tissue_mean': 'avg_ratio_inf_score_black_vs_max_tissue',
        'total_attribution_mean': 'avg_total_attribution'
    }
    agg_results = agg_results.rename(columns=column_mapping)
    
    # Calculate scaled scores
    agg_results = calculate_scaled_scores(agg_results)
    
    return agg_results

def calculate_information_scores(heatmap_org, heatmap_col, bw_img, bw_img_reverse, 
                               nr_tissue_pixels, nr_black_pixels, img, eps=1e-8):
    """
    Calculate various information and attention scores from GRAD-CAM heatmap data.
    """
    # Basic statistics
    max_total = np.max(heatmap_org)
    max_tissue = np.max(heatmap_org * bw_img)
    max_black = np.max(heatmap_org * bw_img_reverse)
    
    # Information scores with standard deviations
    tissue_values = heatmap_org[bw_img > 0]
    black_values = heatmap_org[bw_img_reverse > 0]
    
    info_score_tissue = np.sum(heatmap_org * bw_img) / (nr_tissue_pixels + eps)
    info_score_black = np.sum(heatmap_org * bw_img_reverse) / (nr_black_pixels + eps)
    
    # Calculate standard deviations
    std_tissue = np.std(tissue_values) if len(tissue_values) > 0 else 0
    std_black = np.std(black_values) if len(black_values) > 0 else 0
    
    # Red pixel analysis
    try:
        lower_red = np.array([50,0,0])
        upper_red = np.array([255,255,10])
        bin_heatmap = cv2.inRange(heatmap_col, lower_red, upper_red)
        nr_red_total = cv2.countNonZero(bin_heatmap)
        
        img_focus = cv2.bitwise_and(img, img, mask=bin_heatmap)
        
        # Check if img_focus is valid before color conversion
        if img_focus.size > 0 and img_focus.shape[-1] == 3:
            gray_img_focus = cv2.cvtColor(img_focus, cv2.COLOR_BGR2GRAY)
            _, bw_img_focus = cv2.threshold(gray_img_focus, 0, 255, cv2.THRESH_BINARY)
            bw_img_focus = np.array(bw_img_focus, np.float32) / 255
        else:
            # If img_focus is invalid, create zero arrays
            bw_img_focus = np.zeros_like(bw_img, dtype=np.float32)
            nr_red_total = 0
    except Exception as e:
        # logger.warning(f"Error in red pixel analysis: {str(e)}. Using fallback values.")
        bw_img_focus = np.zeros_like(bw_img, dtype=np.float32)
        nr_red_total = 0
    
    return {
        'nr_tissue_pixels': nr_tissue_pixels,
        'nr_black_pixels': nr_black_pixels,
        'noise_pixels_perc': nr_black_pixels / (nr_tissue_pixels + nr_black_pixels),
        'max_information_score_total': max_total,
        'max_tissue': max_tissue,
        'max_black': max_black,
        'min_information_score_total': np.min(heatmap_org),
        'avg_information_score_total': np.mean(heatmap_org),
        'med_information_score_total': np.median(heatmap_org),
        'information_score_tissue': info_score_tissue,
        'information_score_black': info_score_black,
        'nr_red_total': nr_red_total,
        'nr_red_tissue_pixels': cv2.countNonZero(bw_img_focus * bw_img),
        'nr_red_black_pixels': nr_red_total - cv2.countNonZero(bw_img_focus * bw_img),
        'information_score_red_tissue': np.sum(heatmap_org * bw_img_focus * bw_img) / (nr_tissue_pixels + eps),
        'information_score_red_black': np.sum(heatmap_org * bw_img_focus * bw_img_reverse) / (nr_black_pixels + eps),
        'ratio_inf_score_black_vs_tissue': info_score_black / (info_score_tissue + eps),
        'ratio_inf_score_black_vs_max_tissue': info_score_black / (max_tissue + eps),
        'ratio_black_vs_tissue_attr': info_score_black / (info_score_tissue + eps),
        'ratio_tissue_vs_black_attr': info_score_tissue / (info_score_black + eps),
        'total_attribution': np.mean(heatmap_org),
        'perc_tissue_attribution': np.mean(heatmap_org * bw_img) / (nr_tissue_pixels + eps),
        'perc_black_attribution': np.mean(heatmap_org * bw_img_reverse) / (nr_black_pixels + eps),
        'perc_tissue_attr_total_attribution': np.sum(heatmap_org * bw_img) / (np.sum(heatmap_org * bw_img) + np.sum(heatmap_org * bw_img_reverse) + eps),
        'perc_black_attr_total_attribution': np.sum(heatmap_org * bw_img_reverse) / (np.sum(heatmap_org * bw_img) + np.sum(heatmap_org * bw_img_reverse)+ eps),
        'rel_perc_black_attr_total_attribution': (np.sum(heatmap_org * bw_img_reverse) / (np.sum(heatmap_org * bw_img) + np.sum(heatmap_org * bw_img_reverse)+ eps)) / (nr_black_pixels / (nr_tissue_pixels + nr_black_pixels) + eps)
        # Final two metrics are important, as they give a more interpretable score
    }

def calculate_scaled_scores(df, eps=1e-8):
    """
    Calculate scaled information scores for a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing raw scores
        eps (float): Small value to prevent division by zero
        
    Returns:
        pd.DataFrame: DataFrame with additional scaled score columns
    """
    # Calculate denominators once
    range_denominator = df['max_information_score_total'] - df['min_information_score_total'] + eps
    
    # Add scaled columns
    df['avg_information_score_tissue_scaled'] = (
        df['avg_information_score_tissue'] - df['min_information_score_total']
    ) / range_denominator
    
    df['avg_information_score_black_scaled'] = (
        df['avg_information_score_black'] - df['min_information_score_total']
    ) / range_denominator
    
    df['max_minus_avg_information_score_tissue_scaled'] = (
        df['max_minus_avg_information_score_tissue'] - df['min_information_score_total']
    ) / range_denominator
    
    return df

def load_aggregated_results(data_type, threshold, org_size):
    """
    Load aggregated results for a specific data type and threshold.
    
    Args:
        data_type (str): Type of data ('random' or 'black')
        threshold (float): Threshold value
        org_size (list): Original image dimensions [height, width]
        
    Returns:
        pd.DataFrame: Aggregated results
    """
    model_path = Path('/content/drive/My Drive/Esmee') / f"{org_size[0]}x{org_size[1]}" / f"Results_{data_type}" / \
                f"threshold={threshold}_Inception-ResNet-V2_Oversample_retrain=True_0_binary_crossentropy_0.005_0_0.6"
    
    results_path = model_path / 'Visualizations' / f'threshold={threshold}_total_full_set_agg_heatmap_information_scores_subset.csv'
    
    return pd.read_csv(results_path).iloc[:, 1:]  # Skip index column


def evaluate_noise_resistance(data_type, threshold, config, model_dir, model, noise_data_partition='val_dir', n_samples_image_noise=50):
    """
    Evaluate model's resistance to noise by testing predictions with incrementally added noise.
    
    This function:
    1. Loads the model and test images
    2. For each image:
        - Applies different levels of noise (0% to 95% in 5% increments)
        - Runs multiple iterations at each noise level
        - Averages predictions across iterations
    3. Saves and returns the results
    
    Args:
        data_type (str): Type of data ('black' or 'random')
        threshold (float): Threshold value for tissue pixels
        config (dict): Configuration dictionary
        model_dir (Path): Path to the model directory
        model (torch.nn.Module): The trained model
        noise_data_partition (str): The data partition to use for noise evaluation
        
    Returns:
        pd.DataFrame: DataFrame containing predictions for different noise levels
    """
    logger.info(f"Starting noise resistance evaluation for {data_type} data (threshold={threshold})")
    
    # Get paths
    # Get paths
    data_path = config['paths']['data'](config, data_type, threshold)
    
    # Add partition path if specified
    if noise_data_partition is not None:
        data_path = data_path / noise_data_partition

    # Get folders for both compartments
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f)) 
              and '_dir' not in f and '_oversample' not in f 
              and any(comp in f for comp in ['Inside', 'Outside'])]
    
    # Collect image paths per compartment
    img_paths = []
    for folder in folders:
        folder_path = data_path / folder
        folder_images = list(folder_path.glob('*.png'))
        
        # Take up to n_samples images from each folder
        if len(folder_images) > n_samples_image_noise:
            folder_images = random.sample(folder_images, n_samples_image_noise)
        
        img_paths.extend(folder_images)

    logger.info(f"Processing {len(img_paths)} images with noise levels from 0% to 95%")

    # Load model and data
    logger.info("Loading model and preparing test data")
    checkpoint = torch.load(model_dir / 'best_auc_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode

    # noise_per_img_df = pd.read_csv(config['paths']['data_original'](config) / f'threshold={threshold}-perc_black_per_img.csv')
    # noise_per_img_df = noise_per_img_df.sample(frac=1)
    
    # # Select test images
    # patches = {
    #     'Outside': noise_per_img_df[noise_per_img_df.compartment == 'Outside'].image[:50],
    #     'Inside': noise_per_img_df[noise_per_img_df.compartment == 'Inside'].image[:50]
    # }
    
    # img_paths = [
    #     data_path / compartment / fname 
    #     for compartment, fnames in patches.items() 
    #     for fname in fnames
    # ]
    
    predictions = []
    for img_path in tqdm(img_paths, desc=f"Processing {data_type} images"):
        try:
            compartment = img_path.parent.name
            img = cv2.imread(str(img_path))
            
            # Get color palette for noise generation
            colors = [[0,0,0]] if data_type == 'black' else get_palette(img)
            img = np.array(img, np.float32) / 255
            
            # Test different noise levels
            for noise_level in range(0, 100, 5):                
                # Average predictions across multiple iterations
                avg_pred = _apply_noise_and_predict(
                    img, model, noise_level, colors, 
                    config['n_iterations'], config['size']
                )
                predictions.append([
                    data_type, threshold, noise_level, img_path, compartment, avg_pred
                ])
                
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue
    
    # Create and save results DataFrame
    logger.info("Saving noise resistance evaluation results")
    columns = ['data_type', 'threshold', 'noise_level', 'img_path', 'compartment', 'avg_pred']
    results_df = pd.DataFrame(predictions, columns=columns)
    results_df.to_csv(
        model_dir / f'predictions_per_noise_level.csv'
    )
    
    logger.info(f"Completed noise resistance evaluation for {data_type} data")
    return results_df

def _apply_noise_and_predict(img, model, noise_level, colors, n_iterations, target_size):
    """Helper function to apply noise and get model predictions"""
    device = next(model.parameters()).device
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert colors to numpy array for vectorized operations
    colors = np.array(colors)
    img_shape = img.shape
    total_pixels = img_shape[0] * img_shape[1]
    
    sum_probs = 0
    with torch.no_grad():
        for i in range(n_iterations):
            # Create noisy image
            # print(f"Now working on iteration {i+1}/{n_iterations}")
            img_new = img.copy()
            
            # Calculate number of pixels to modify
            n_pixels_to_modify = int(total_pixels * noise_level / 100)
            
            # Create mask for noise application
            noise_mask = np.zeros(total_pixels, dtype=bool)
            noise_mask[:n_pixels_to_modify] = True
            np.random.shuffle(noise_mask)
            noise_mask = noise_mask.reshape(img_shape[0], img_shape[1])
            
            # Generate random colors for all noise pixels at once
            random_colors = colors[np.random.randint(0, len(colors), size=n_pixels_to_modify)]
            
            # Apply noise using broadcasting
            img_new[noise_mask] = random_colors
            
            # Process and predict
            img_new = cv2.resize(img_new, tuple(target_size))
            img_tensor = transform(img_new).unsqueeze(0).to(device)
            
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
            sum_probs += prob
            
    return sum_probs / n_iterations

def analyze_and_plot_noise_resistance(config, model):
    """
    Evaluate noise resistance, analyze results and create visualization plots.
    
    Args:
        config (dict): Configuration dictionary containing thresholds and data types
        output_path (Path): Path where output plots should be saved
    """
    # Evaluate noise resistance and collect results
    noise_resistance_results = {}

    model_to_evaluate = Path(config['base_dir']) / config['models_evaluation_selection']
    with open(model_to_evaluate, 'r') as f:
        model_selections = json.load(f)
        
    for model_base_path, selection in model_selections.items():
        data_type = selection['data_type']
        threshold = selection['threshold']
        
        model_dir = Path(model_base_path) / selection['folder']
        
        print(f"\nEvaluating noise resistance for {data_type} data, threshold {threshold}")
        results_df = evaluate_noise_resistance(data_type, threshold, config, model_dir, model)
        noise_resistance_results[(data_type, threshold)] = results_df

    # Create plots for each threshold
    for threshold in config['thresholds']:
        plt.figure(figsize=(10, 6))
        
        for data_type in config['data_types']:
            results = (noise_resistance_results[(data_type, threshold)]
                      .groupby('noise_level')['avg_pred']
                      .mean())
            
            # Calculate average predictions per noise level and compartment
            for compartment in ['Outside', 'Inside']:
                avg_preds = (results[results.compartment == compartment]
                           .groupby('noise_level')['avg_pred']
                           .mean())
                
                label = f"{data_type.capitalize()} artifacts - {compartment}"
                plt.plot(
                    range(0, 100, 5),  # noise levels
                    avg_preds,
                    label=label,
                    marker='o'
                )
        
        plt.xlabel('Percentage of noisy pixels')
        plt.ylabel('Probability of adventitia')
        plt.ylim(0, 1)
        plt.title(f'Model Predictions vs Noise Level (Threshold {threshold})')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(
            config['paths']['output'](config) / f'noise_resistance_analysis_{data_type}_threshold_{threshold}.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

    return noise_resistance_results

            
def perform_gradcam_analysis(config, output_path, model):
    """
    Perform GRAD-CAM analysis for different data types and compartments.
    
    Args:
        config (dict): Configuration dictionary containing data types and parameters
        output_path (Path): Path where results should be saved
        
    Returns:
        tuple: (combined_results, combined_agg) DataFrames containing all results 
        and aggregated results respectively
    """
    logger.info("Starting GRAD-CAM analysis...")
    gradcam_results = []
    
    # Optionally only analyze threshold=0.2 for GRAD-CAM to reduce computation time
    analysis_threshold = False
    
    model_to_evaluate = Path(config['base_dir']) / config['models_evaluation_selection']
    with open(model_to_evaluate, 'r') as f:
        model_selections = json.load(f)
        
    for model_base_path, selection in model_selections.items():
        data_type = selection['data_type']
        threshold = selection['threshold']
        print(f"Processing GRAD-CAM analysis for {data_type} data, threshold {threshold}")
        if not (threshold == analysis_threshold or analysis_threshold is False):
            print(f"Skipping GRAD-CAM analysis for {data_type} data, threshold {threshold}")
            continue
        else:
            model_dir = Path(model_base_path) / selection['folder']
            for compartment in ['Inside', 'Outside']:
                logger.info(f"\nProcessing GRAD-CAM analysis for {data_type} data, {compartment} compartment")
                
                results_df, agg_results = analyze_gradcam_batch(data_type, threshold, compartment, config, model_dir, model, inf_data_partition=config['inf_data_partition'])
                
                gradcam_results.append({
                    'data_type': data_type,
                    'compartment': compartment,
                    'results_df': results_df,
                    'agg_results': agg_results
                })
    # Combine all results
    combined_results = pd.concat([r['results_df'] for r in gradcam_results])
    combined_agg = pd.concat([r['agg_results'] for r in gradcam_results])
    
    # We need to aggregate the results in total over all compartments
    combined_agg = aggregate_heatmap_results(combined_results)

    # Save results
    combined_results.to_csv(output_path / f'gradcam_analysis_all_results.csv')
    combined_agg.to_csv(output_path / f'gradcam_analysis_aggregated.csv')
    
    logger.info("Completed GRAD-CAM analysis")
    return combined_results, combined_agg


def perform_integrated_gradients_analysis(config, output_path, model):
    """
    Perform Integrated Gradients analysis for different data types and compartments.
    
    Args:
        config (dict): Configuration dictionary containing data types and parameters
        output_path (Path): Path where results should be saved
        model: PyTorch model
        
    Returns:
        tuple: (combined_results, combined_agg) DataFrames containing all results 
        and aggregated results respectively
    """
    logger.info("Starting Integrated Gradients analysis...")
    ig_results = []
    
    model_to_evaluate = Path(config['base_dir']) / config['models_evaluation_selection']
    with open(model_to_evaluate, 'r') as f:
        model_selections = json.load(f)
    
    # Setup Integrated Gradients
    ig = IntegratedGradients(model)
    
    for model_base_path, selection in model_selections.items():
        data_type = selection['data_type']
        threshold = selection['threshold']
        print(f"Processing IG analysis for {data_type} data, threshold {threshold}")
        
        model_dir = Path(model_base_path) / selection['folder']
        for compartment in ['Inside', 'Outside']:
            logger.info(f"\nProcessing IG analysis for {data_type} data, {compartment} compartment")
            
            results_df, agg_results = analyze_ig_batch(
                data_type, 
                threshold, 
                compartment, 
                config, 
                model_dir, 
                model,
                ig,
                inf_data_partition=config['inf_data_partition']
            )
            
            ig_results.append({
                'data_type': data_type,
                'compartment': compartment,
                'results_df': results_df,
                'agg_results': agg_results
            })

    # Combine all results
    combined_results = pd.concat([r['results_df'] for r in ig_results])
    combined_agg = pd.concat([r['agg_results'] for r in ig_results])
    
    combined_agg = aggregate_attribution_results(combined_results)
    # Save results
    combined_results.to_csv(output_path / f'integrated_gradients_analysis_all_results.csv')
    combined_agg.to_csv(output_path / f'integrated_gradients_analysis_aggregated.csv')
    
    logger.info("Completed Integrated Gradients analysis")
    return combined_results, combined_agg

def analyze_ig_batch(data_type, threshold, compartment, config, model_dir, model, ig, inf_data_partition='val_dir', batch_size=32, ):
    """
    Analyze a batch of images using Integrated Gradients efficiently with batch processing.
    inf_data_partition: Can run on 'val_dir' or 'test_dir' partition. If None, then use all images. 
    """
    logger.info(f"Starting batch IG analysis for {data_type}, {compartment}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Setup data paths
    data_path = config['paths']['data'](config, data_type, threshold)
    if inf_data_partition is not None:
        data_path = data_path / inf_data_partition
    black_data_path = config['paths']['data'](config, 'black', threshold)
    if inf_data_partition is not None:
        black_data_path = black_data_path / inf_data_partition  
    
    transform = transforms.Compose([
        transforms.Resize(config['size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Get image paths for the compartment
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f)) 
              and '_dir' not in f and '_oversample' not in f and compartment in f]
    
    result_per_img = []
    
    try:
        for folder in folders:
            logger.info(f'Processing {folder}')
            pathlist = list(Path(data_path / folder).glob('**/*.png'))
            
            # Calculate number of batches needed
            total_samples_needed = min(config['n_samples'], len(pathlist))

            print(f"Total samples available for {folder}: {len(pathlist)}")
            pathlist_selection = random.sample(pathlist, total_samples_needed)
            
            progress_bar = tqdm(pathlist_selection, desc=f"Processing {folder}")
            
            for img_path in progress_bar:
                try:
                    # Load and preprocess image
                    img = read_img(img_path, config['size'])
                    if img is None:
                        logger.error(f"Failed to load image: {img_path}")
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Ensure image is float32 and normalized to [0,1]
                    img = img.astype(np.float32) / 255.0
                    
                    # Convert to PIL Image for transforms
                    img_pil = Image.fromarray((img * 255).astype('uint8'))
                    img_tensor = transform(img_pil).unsqueeze(0).to(device)
                    
                    # Load corresponding black image
                    black_path = str(img_path).replace(str(data_path), str(black_data_path))
                    black_img = read_img(black_path, config['size'])
                    if black_img is None:
                        logger.error(f"Failed to load black image: {black_path}")
                        continue
                    
                    # Create mask from black image
                    gray = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
                    mask = (gray > 0).astype(np.float32) # 1 for non-black pixels, 0 for black
                    
                    # Calculate attributions
                    attributions = ig.attribute(img_tensor, n_steps=50)
                    attr_np = attributions.squeeze().cpu().detach().numpy()
                    
                    # At this point, all images should be config['size']:
                    # - img: (H, W, 3)
                    # - mask: (H, W)
                    # - attr_np: (3, H, W)
                    assert img.shape[:2] == tuple(config['size']), f"Image size mismatch: {img.shape[:2]} vs {config['size']}"
                    assert mask.shape == tuple(config['size']), f"Mask size mismatch: {mask.shape} vs {config['size']}"
                    assert attr_np.shape[1:] == tuple(config['size']), f"Attribution size mismatch: {attr_np.shape[1:]} vs {config['size']}"
                    
                    # Calculate attribution scores
                    scores = calculate_attribution_scores(
                        attr_np,
                        mask,
                        1 - mask,  # inverse mask for black regions
                        np.sum(mask),
                        np.sum(1 - mask)
                    )
                    
                    # Save example visualization for first image
                    if len(result_per_img) == 0:
                        save_ig_visualization(
                            img,
                            attr_np,
                            mask,
                            config['output_path'] / "example_attributions",
                            img_path.stem
                        )
                    
                    result = [data_type, threshold, img_path.parent.name, img_path.name] + list(scores.values())
                    result_per_img.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {str(e)}\nTraceback: {traceback.format_exc()}")
                    continue
                
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}\nTraceback: {traceback.format_exc()}")
        raise
    
    # Create results DataFrames
    columns = ['data_type', 'threshold', 'compartment', 'filename'] + list(scores.keys())
    results_df = pd.DataFrame(result_per_img, columns=columns)
    
    # Calculate aggregated results
    agg_results = aggregate_attribution_results(results_df)
    
    return results_df, agg_results

def calculate_attribution_scores(attributions, tissue_mask, black_mask, nr_tissue_pixels, nr_black_pixels, eps=1e-8):
    """
    Calculate various attribution scores for tissue and black regions.
    """
    # Take absolute values of attributions if you want to measure pure magnitude
    abs_attributions = np.abs(attributions)
    
    # Calculate mean attribution for each region
    tissue_attrs = abs_attributions * tissue_mask
    black_attrs = abs_attributions * black_mask
    
    tissue_mean = np.sum(tissue_attrs) / (nr_tissue_pixels + eps)
    black_mean = np.sum(black_attrs) / (nr_black_pixels + eps)
    
    return {
        'nr_tissue_pixels': nr_tissue_pixels,
        'nr_black_pixels': nr_black_pixels,
        'noise_pixels_perc': nr_black_pixels / (nr_tissue_pixels + nr_black_pixels),
        'mean_tissue_attribution': tissue_mean,
        'mean_black_attribution': black_mean,
        'max_tissue_attribution': np.max(tissue_attrs),
        'max_black_attribution': np.max(black_attrs),
        'ratio_black_vs_tissue_attr': black_mean / (tissue_mean + eps),
        'ratio_tissue_vs_black_attr': tissue_mean / (black_mean + eps),
        'total_attribution': np.mean(abs_attributions),
        'perc_tissue_attribution': np.mean(abs_attributions * tissue_mask) / (nr_tissue_pixels + eps),
        'perc_black_attribution': np.mean(abs_attributions * black_mask) / (nr_black_pixels + eps),
        'perc_tissue_attr_total_attribution': np.mean(abs_attributions * tissue_mask) / (np.mean(abs_attributions) + eps),
        'perc_black_attr_total_attribution': np.mean(abs_attributions * black_mask) / (np.mean(abs_attributions) + eps),
        'rel_perc_black_attr_total_attribution': (np.mean(abs_attributions * black_mask) / (np.mean(abs_attributions) + eps)) / (nr_black_pixels / (nr_tissue_pixels + nr_black_pixels) + eps)
    }

def save_ig_visualization(original_img, attributions, mask, output_path, output_name):
    """
    Create and save visualization of Integrated Gradients attributions.
    
    Args:
        original_img: numpy array of shape (H, W, 3)
        attributions: numpy array of shape (3, H, W)
        mask: numpy array of shape (H, W)
        output_path: Path object for output directory
        output_name: Base name for output file
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
       
    attr_mean = np.mean(attributions, axis=0)
    # Ensure attributions match original image size
    if attr_mean.shape != original_img.shape[:2]:
        attr_mean = cv2.resize(attr_mean, (original_img.shape[1], original_img.shape[0]))
    
    # Attribution heatmap (mean across channels)
    max_abs = np.max(np.abs(attr_mean))
    # Resize attr_mean to match original image dimensions
    attr_mean_resized = cv2.resize(attr_mean, (original_img.shape[1], original_img.shape[0]))
    im = axes[1].imshow(attr_mean_resized, cmap='seismic', vmin=-max_abs, vmax=max_abs)
    axes[1].set_title('Attribution Heatmap')
    axes[1].axis('off')
    
    # Add colorbar with matching height
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Tissue mask
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Tissue Mask')
    axes[2].axis('off')
    
    # Custom overlay visualization
    # Normalize attributions to [-1, 1] range
    attr_norm = attr_mean / (max_abs + 1e-8)
    
    # Create RGB overlay
    overlay = np.zeros_like(original_img)
    # Red for positive attributions, Blue for negative
    overlay[..., 0] = np.maximum(0, attr_norm)  # Red channel
    overlay[..., 2] = np.maximum(0, -attr_norm)  # Blue channel
    
    # Blend with original image
    alpha = 0.5
    blended = (1 - alpha) * original_img + alpha * overlay
    
    # Ensure values are in valid range
    blended = np.clip(blended, 0, 1)
    
    axes[3].imshow(blended)
    axes[3].set_title('Attribution Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / f'ig_visualization_{output_name}.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def aggregate_attribution_results(results_df):
    """
    Aggregate attribution results by compartment and create total aggregation.
    """
    # Define aggregation functions without nesting
    agg_funcs = {
        'nr_tissue_pixels': 'sum',
        'nr_black_pixels': 'sum',
        'noise_pixels_perc': 'mean',
        'mean_tissue_attribution': 'mean',
        'mean_black_attribution': 'mean',
        'max_tissue_attribution': 'max',
        'max_black_attribution': 'max',
        'ratio_black_vs_tissue_attr': ['mean', 'median', 'std'],  # Added median and std
        'ratio_tissue_vs_black_attr': ['mean', 'median', 'std'],  # Added median and std
        'total_attribution': 'mean',
        'perc_tissue_attribution': ['mean', 'median', 'std'],
        'perc_black_attribution': ['mean', 'median', 'std'],
        'perc_tissue_attr_total_attribution': ['mean', 'median', 'std'],
        'perc_black_attr_total_attribution': ['mean', 'median', 'std'],
        'rel_perc_black_attr_total_attribution': ['mean', 'median', 'std']
    }
    
    # Aggregate by compartment
    compartment_agg = (results_df.groupby(['data_type', 'threshold', 'compartment'])
                      .agg(agg_funcs)
                      .reset_index())
    
    # Add total aggregation
    total_agg = (results_df.groupby(['data_type', 'threshold'])
                 .agg(agg_funcs)
                 .assign(compartment='Total')
                 .reset_index())
    
    combined_agg = pd.concat([compartment_agg, total_agg], ignore_index=True)
    
    # Flatten column names by adding suffixes
    combined_agg.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col 
        for col in combined_agg.columns
    ]

    # Remove trailing underscores from column names. Not the neatest but works.
    combined_agg.columns = [col[:-1] if col.endswith('_') else col for col in combined_agg.columns]
    
    # Drop duplicate rows
    combined_agg = combined_agg.drop_duplicates()

    # Reset index before saving
    combined_agg = combined_agg.reset_index(drop=True)
    
    return combined_agg


def plot_attribution_ratios(combined_agg, thresholds, output_path):
    """
    Plot attribution score ratios with error bars showing standard deviation.
    """
    random_results_df = combined_agg[combined_agg['data_type'] == 'random_image']
    black_results_df = combined_agg[combined_agg['data_type'] == 'black']
    
    metrics = [
        {
            'mean_column': 'mean_tissue_attribution',
            'std_column': 'std_ratio_tissue_vs_black_attr',
            'title_suffix': 'tissue attribution',
            'ylabel': 'Mean tissue attribution score (± std)'
        },
        {
            'mean_column': 'mean_black_attribution',
            'std_column': 'std_ratio_black_vs_tissue_attr',
            'title_suffix': 'black attribution',
            'ylabel': 'Mean black attribution score (± std)'
        }
    ]
    
    for compartment in random_results_df.compartment.unique():
        for metric in metrics:
            plt.figure()
            
            for df, label in [(random_results_df, 'Random artifacts'),
                            (black_results_df, 'Black artifacts')]:
                # Get data for current compartment
                mask = (df.compartment == compartment)
                means = []
                stds = []
                
                for threshold in thresholds:
                    threshold_mask = mask & (df.threshold == threshold)
                    mean_val = df.loc[threshold_mask, metric['mean_column']].values
                    std_val = df.loc[threshold_mask, metric['std_column']].values
                    
                    means.append(mean_val[0] if len(mean_val) > 0 else 0)
                    stds.append(std_val[0] if len(std_val) > 0 else 0)
                
                # Plot with error bars
                plt.errorbar(thresholds, means, yerr=stds, label=label, capsize=5)
            
            plt.title(f"{compartment} - {metric['title_suffix']}")
            plt.ylabel(metric['ylabel'])
            plt.xlabel('Threshold % tissue pixels')
            plt.legend(loc='upper left')
            
            filename = f"{compartment}_attribution_{metric['title_suffix'].replace(' ', '_')}_per_threshold.png"
            plt.savefig(output_path / filename)
            plt.close()

def transform_grid_run_to_model_selection(config):
    """
    Transform grid run selection JSON into models evaluation selection format.
    
    Args:
        config (dict): Configuration dictionary containing file paths
    """
    base_dir = Path(config['base_dir'])
    grid_run_path = base_dir / config['grid_run_selection']
    model_selection_path = base_dir / config['models_evaluation_selection']
    
    # Create backup of existing model selection if it exists
    if model_selection_path.exists():
        backup_dir = model_selection_path.parent / 'learning_evaluation/previous_model_selection'
        backup_dir.mkdir(exist_ok=True)
        
        # Create timestamped backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f'models_evaluation_selection_{timestamp}.json'
        shutil.copy2(model_selection_path, backup_path)
        logger.info(f"Backed up existing model selection to {backup_path}")
    
    # Load grid run selection
    with open(grid_run_path, 'r') as f:
        grid_runs = json.load(f)

    print(grid_runs)
    
    # Transform into new format
    model_selections = {}
    
    if not config['same_data_type_threshold']:
            if isinstance(grid_runs, dict):
                # If grid_runs is a dictionary
                for key, run_info in grid_runs.items():
                    try:
                        # Extract data type and threshold from the path or run info
                        path_parts = Path(key).parts
                        data_type = next(part for part in path_parts if part in ['black', 'random_image'])
                        threshold = next(float(part) for part in key.split('_') if part.replace('.', '').isdigit())
                        
                        # Create key using full model path
                        full_path = str(Path(key))
                        
                        # Create new format entry
                        model_selections[full_path] = {
                            'data_type': data_type,
                            'threshold': threshold,
                            'folder': next(iter(run_info['runs'].values()))['folder']
                        }
                    except Exception as e:
                        logger.warning(f"Could not process entry {key}: {str(e)}")
                        continue
            else:
                logger.warning("Grid runs structure is not a dictionary")
    else:
        # Handle single data type and threshold with multiple runs
        if isinstance(grid_runs, dict):
            key = next(iter(grid_runs.keys()))  # Get the single key
            run_info = grid_runs[key]
            
            try:
                # Extract data type and threshold from the path or run info
                path_parts = Path(key).parts
                data_type = next(part for part in path_parts if part in ['black', 'random_image'])
                threshold = next(float(part) for part in key.split('_') if part.replace('.', '').isdigit())
                
                # Create base path
                full_path = str(Path(key))
                
                # Add an entry for each run
                for run_id, run_data in run_info['runs'].items():
                    model_key = f"{full_path}_run_{run_id}"
                    model_selections[model_key] = {
                        'data_type': data_type,
                        'threshold': threshold,
                        'folder': run_data['folder'],
                        'run_id': run_id
                    }
                    
            except Exception as e:
                logger.error(f"Could not process grid runs: {str(e)}")
                raise
        else:
            logger.warning("Grid runs structure is not a dictionary")

    # Save new model selection format
    with open(model_selection_path, 'w') as f:
        json.dump(model_selections, f, indent=4)
    
    logger.info(f"Transformed grid run selection into model selection format at {model_selection_path}")
    return model_selections

def main():
    """Main execution function"""
    config = load_config()
    print(config)
    
    # Transform grid run selection if specified
    if 'grid_run_selection' in config:
        model_selection = transform_grid_run_to_model_selection(config)
    
    print("Models that will be evaluated:")
    print(json.dumps(model_selection, indent=4))

    output_path = config['paths']['output'](config)
    if config.get('grid_run_selection'):
        # Extract name without extension from grid run selection path
        grid_run_name = Path(config['grid_run_selection']).stem
        config['output_path'] = output_path / grid_run_name
    config['output_path'].mkdir(parents=True, exist_ok=True)

    # Create a model object that can load in existing weights from trained models
    model_path = get_model_path(config['tf_model'])
    model = torch.load(str(model_path))
    
    # logger.info("Evaluating models for all configurations...")
    # for model_path, selection in model_selection.items():
    #     data_type = selection['data_type']
    #     threshold = selection['threshold']
    #     folder = selection['folder']
    #     run_id = selection.get('run_id')  # Will be None if not present
        
    #     logger.info(f"Evaluating model for {data_type} data (threshold={threshold}, folder={folder})")
    #     evaluate_models(
    #         data_type=data_type,
    #         threshold=threshold,
    #         config=config,
    #         model=model,
    #         rerun_eval=True,
    #         model_folder=folder
    #     )
        
    # # Now collect the results from all evaluations
    # logger.info("Collecting model results...")
    # random_results_df, black_results_df = collect_model_results(config)

    # # Plot AUC performance using the collected results
    # logger.info("Plotting AUC performance...")
    # plot_auc_performance(
    #     thresholds=config['thresholds'],
    #     random_results_df=random_results_df,
    #     black_results_df=black_results_df,
    #     output_path=config['output_path']
    # )
    
    # # Evaluate noise resistance and collect results
    # analyze_and_plot_noise_resistance(config, model)
    
    # Perform GRAD-CAM analysis
    gradcam_combined_results, gradcam_combined_agg = perform_gradcam_analysis(config, config['output_path'], model)
    
    # Load gradcam_combined_results if not yet defined
    if 'gradcam_combined_results' not in locals():
        gradcam_results_path = config['output_path'] / 'gradcam_analysis_aggregated.csv'
        gradcam_combined_agg = pd.read_csv(gradcam_results_path)
    
    # #Plot information score ratios using combined_agg
    # plot_information_score_ratios(
    #     combined_agg=gradcam_combined_agg,
    #     thresholds=config['thresholds'],
    #     output_path=config['output_path']
    # )
    
    #Perform Integrated Gradients analysis
    ig_combined_results, ig_combined_agg = perform_integrated_gradients_analysis(config, config['output_path'], model)
    
    # # Load ig_combined_results if not yet defined
    # if 'ig_combined_results' not in locals():
    #     ig_combined_agg = pd.read_csv(config['output_path'] / 'integrated_gradients_analysis_aggregated.csv')
    
    # Plot information score ratios using combined_agg (if needed)
    # plot_attribution_ratios(
    #     combined_agg=ig_combined_agg,
    #     thresholds=config['thresholds'],
    #     output_path=config['output_path']
    # )
    
    # Analyze control images
    ## TO DO: detele this probably
    # control_results = analyze_control_images(model, config, output_path)
    
    # # Compare control results with regular analysis
    # control_mean = control_results.groupby('method')['mean_score', 'mean_attr'].mean()
    
    # # Add control comparison to the results dictionary
    # results = {
    #     'gradcam_results': {
    #         'combined_results': gradcam_combined_results,
    #         'combined_agg': gradcam_combined_agg,
    #         'control_comparison': control_mean['mean_score']
    #     },
    #     'ig_results': {
    #         'combined_results': ig_combined_results,
    #         'combined_agg': ig_combined_agg,
    #         'control_comparison': control_mean['mean_attr']
    #     }
    # }

### Duplicate functions coming from train_nn_inner_outer_gpu_pytorch.py


if __name__ == "__main__":
    main()

