import sys
import os
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import numpy as np
import shutil
import timm
from tqdm import tqdm
from datetime import datetime
import wandb
from torch.utils.data import WeightedRandomSampler
import uuid
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, List
from torch.amp import autocast, GradScaler
from itertools import product

# Import config module
from config import (
    BASE_DIR, RESULTS_DIR, MODEL_DIR, MODEL_WEIGHTS_DIR, 
    get_imputed_patches_path, get_model_path, get_config_path,
    validate_model_weights, setup_environment
)

# Setup environment
setup_environment(verbose=False)

# Set the environment variable for CUDA device-side assertions
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Import libraries for handling imbalanced datasets
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Import scikit-learn libraries for data preprocessing and evaluation
from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle

# Import image processing libraries
from skimage.transform import rotate
import scipy.ndimage as ndi

from subprocess import check_output

# Replace Tee class and stdout redirection with Python's logging module
import logging
log_file = BASE_DIR / 'terminal_output.txt'
logging.basicConfig(filename=str(log_file), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Print available devices (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Add garbage collection at the beginning of the script
import gc
gc.collect()

# Add at the beginning of the file, after the imports
GRID_EXECUTION = True

# Add this new function
def count_images_in_train_dir(data_path):
    train_dir = os.path.join(data_path, 'train_dir')
    total_images = 0
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                total_images += 1
    return total_images

# Function to summarize model performance and create visualizations
def summarize_model(output_path, model, history):
    
    # Visualize training history for AUC
    plt.figure()
    plt.plot(history['train_auc'])
    plt.plot(history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(output_path, 'auc_development.png'))
    
    # Save AUC data as pickle
    auc_data = {
        'train_auc': history['train_auc'],
        'val_auc': history['val_auc']
    }
    with open(os.path.join(output_path, 'auc_data.pkl'), 'wb') as f:
        pickle.dump(auc_data, f)
    
    # Visualize training for loss
    plt.figure()
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(output_path, 'loss_development.png'))
    
    # Save loss data as pickle
    loss_data = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss']
    }
    with open(os.path.join(output_path, 'loss_data.pkl'), 'wb') as f:
        pickle.dump(loss_data, f)
    
    # Visualize training per class
    plt.figure()
    plt.plot(history['train_acc_class0'])
    plt.plot(history['val_acc_class0'])
    plt.title('Development of classification accuracy for class 0 (Inside)')
    plt.ylabel('% Correct classification')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(output_path, 'acc_class0_development.png'))
    
    # Save class 0 accuracy data as pickle
    acc_class0_data = {
        'train_acc_class0': history['train_acc_class0'],
        'val_acc_class0': history['val_acc_class0']
    }
    with open(os.path.join(output_path, 'acc_class0_data.pkl'), 'wb') as f:
        pickle.dump(acc_class0_data, f)
    
    plt.figure()
    plt.plot(history['train_acc_class1'])
    plt.plot(history['val_acc_class1'])
    plt.title('Development of classification accuracy for class 1 (Outside)')
    plt.ylabel('% Correct classification')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(output_path, 'acc_class1_development.png'))
    
    # Save class 1 accuracy data as pickle
    acc_class1_data = {
        'train_acc_class1': history['train_acc_class1'],
        'val_acc_class1': history['val_acc_class1']
    }
    with open(os.path.join(output_path, 'acc_class1_data.pkl'), 'wb') as f:
        pickle.dump(acc_class1_data, f)
    
    # Test set evaluation will be done in the main function

# Define layer start indices for different model architectures
resnet_blocks = np.array([
    ['ResNet', 0, 0],
    ['ResNet', 1, 7],
    ['ResNet', 2, 39],
    ['ResNet', 3, 81],
    ['ResNet', 4, 143]
])

inceptionV3_blocks = np.array([
    ['Inception', 0, 42],
    ['Inception', 1, 65],
    ['Inception', 2, 88],
    ['Inception', 3, 102],
    ['Inception', 4, 134],
    ['Inception', 5, 166],
    ['Inception', 6, 198],
    ['Inception', 7, 230],
    ['Inception', 8, 250],
])

inception_resnet_V2_blocks = np.array([
    ['Inception', 0, 1],
    ['Inception', 1, 60],  # Inception-ResNet-A block
    ['Inception', 2, 288], # Inception-ResNet-B block
    ['Inception', 3, 595],
    ['Inception', 4, 631], # Inception-ResNet-C block
    ['Inception', 5, 777]
])

efficientnetb6_blocks = np.array([
    ['EfficientNetB6', 0, 0],    # Input and initial convolution
    ['EfficientNetB6', 1, 3],    # First set of MBConv1 blocks
    ['EfficientNetB6', 2, 7],    # MBConv6 blocks (112x112)
    ['EfficientNetB6', 3, 17],   # MBConv6 blocks (56x56)
    ['EfficientNetB6', 4, 34],   # MBConv6 blocks (28x28)
    ['EfficientNetB6', 5, 55],   # MBConv6 blocks (14x14, first part)
    ['EfficientNetB6', 6, 99],   # MBConv6 blocks (14x14, second part)
    ['EfficientNetB6', 7, 175],  # MBConv6 blocks (7x7, first part)
    ['EfficientNetB6', 8, 323],  # MBConv6 blocks (7x7, second part)
    ['EfficientNetB6', 9, 528]   # Final convolution and pooling
])

# Function to build the model
def build_model(config):
    model_name = config['tf_model']
    num_classes = 1  # Since this is binary classification
    full_retrain = config['full_retrain']
    freeze_till_block = config['freeze_till_block']
    final_connected_layer = config['final_connected_layer']  # Read from config

    if model_name == 'InceptionV3':
        model = models.inception_v3(pretrained=True)
        blocks = inceptionV3_blocks
    elif model_name == 'Inception-ResNet-V2':
        model = timm.create_model('inception_resnet_v2', pretrained=True)
        blocks = inception_resnet_V2_blocks
    elif model_name == 'EfficientNetB6':
        model = models.efficientnet_b6(pretrained=True)
        blocks = efficientnetb6_blocks
    elif model_name == 'ResNet':
        model = models.resnet50(pretrained=True)
        blocks = resnet_blocks
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Calculate lay_num_freeze based on freeze_till_block
    lay_num_freeze = int(blocks[freeze_till_block, 2])
    
    
    # Modify the last layer for binary classification
    if model_name == 'Inception-ResNet-V2':
        num_ftrs = model.classif.in_features
        if final_connected_layer == 'fc':
            model.classif = nn.Sequential(
                nn.Linear(num_ftrs, 1024),
                nn.ReLU(),
                nn.Dropout(config['dr']),
                nn.Linear(1024, num_classes)
            )
        else:  # 'classif'
            model.classif = nn.Linear(num_ftrs, num_classes)
    else:
        num_ftrs = model.fc.in_features
        if final_connected_layer == 'fc':
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 1024),
                nn.ReLU(),
                nn.Dropout(config['dr']),
                nn.Linear(1024, num_classes)
            )
        else:  # 'classif'
            model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Print the number of parameters in the last layer
    if model_name == 'Inception-ResNet-V2':
        print(f"Number of parameters in the last layer: {sum(p.numel() for p in model.classif.parameters())}")
    else:
        print(f"Number of parameters in the last layer: {sum(p.numel() for p in model.fc.parameters())}")
    
    # Set up layer freezing based on full_retrain flag and freeze_till_block
    if full_retrain:
        for param in model.parameters():
            param.requires_grad = True
        logging.info("Full retrain: All layers are set to trainable.")
    else:
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False
        logging.info("All layers are initially frozen.")
        
        # Unfreeze layers from the specified block onwards
        for i, (name, param) in enumerate(model.named_parameters()):
            if i >= lay_num_freeze:
                param.requires_grad = True
                logging.info(f"Layer {name} is set to trainable.")
        
        # Ensure the final layer is trainable if freeze_till_block is the last block
        if freeze_till_block == len(blocks) - 1:
            logging.info(f"Freezing till block {freeze_till_block}, so final layer is trainable.")
            # Get the appropriate final layer attribute (classif for Inception-ResNet-V2, fc for others)
            final_layer = model.classif if model_name == 'Inception-ResNet-V2' else model.fc
            
            if final_connected_layer == 'fc':
                # Make both layers in the sequential block trainable
                for param in final_layer[0].parameters():  # Linear(num_ftrs, 1024)
                    param.requires_grad = True
                for param in final_layer[3].parameters():  # Linear(1024, num_classes)
                    param.requires_grad = True
                logging.info("Final FC layers are set to trainable.")
            else:  # 'classif'
                # Make the entire layer trainable
                for param in final_layer.parameters():
                    param.requires_grad = True
                logging.info(f"Final layer ({final_connected_layer}) is set to trainable.")
    
    # Count and print the number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Percentage of trainable parameters: {trainable_params/total_params*100:.2f}%")
    
    # Save the model architecture to a file
    model_path = get_model_path(config['tf_model'])
    if not model_path.exists():
        torch.save(model, str(model_path))
    # Return the model and only the parameters that require gradients
    return model, filter(lambda p: p.requires_grad, model.parameters())

# Add this function to initialize wandb
def init_wandb(config):
    wandb.init(
        project="ImageRecognition",
        config=config,
        name=f"{config['tf_model']}_{config['org_size'][0]}x{config['org_size'][1]}_cutoff_{config['data_imputation_type']}_imputed_{config['threshold']}_aug_{config['augmentation']}",
    )

# Modify the train_model function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, config):
    model.to(device)
    best_val_auc = 0.0
    best_val_loss = float('inf')
    best_epoch_auc = -1
    best_epoch_loss = -1
    history = {
        'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': [],
        'train_acc_class0': [], 'train_acc_class1': [],
        'val_acc_class0': [], 'val_acc_class1': []
    }
    
    # Initialize scaler for mixed precision training if enabled
    scaler = GradScaler() if config['mixed_precision'] else None
    
    lr_scheduler = ReduceLROnPlateau(optimizer,
                                    mode='min',
                                    patience=config['lr_patience'],
                                    verbose=True,
                                    factor=config['lr_factor'],
                                    cooldown=config['lr_cooldown'],
                                    threshold=config['lr_threshold'],
                                    min_lr=config['lr_min'])

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if config['mixed_precision']:
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float().unsqueeze(1))
                
                # Scale loss and perform backward pass
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if config.get('clip_grad', False):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels.float().unsqueeze(1))
                loss.backward()
                
                if config.get('clip_grad', False):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)
        
        val_loss, val_auc = evaluate_model(model, val_loader, criterion, device)
        
        # Save the best AUC model
        if val_auc > best_val_auc and epoch >= config['warmup_epochs'] :
            best_val_auc = val_auc
            best_epoch_auc = epoch
            best_auc_model_path = os.path.join(config['output_path'], 'best_auc_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'val_loss': val_loss
            }, best_auc_model_path)
            wandb.save(best_auc_model_path, base_path=config['output_path'])
        
        # Save the best loss model
        if val_loss < best_val_loss and epoch >= config['warmup_epochs']:
            best_val_loss = val_loss
            best_epoch_loss = epoch
            best_loss_model_path = os.path.join(config['output_path'], 'best_loss_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'val_loss': val_loss
            }, best_loss_model_path)
            wandb.save(best_loss_model_path, base_path=config['output_path'])
        
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        # Calculate per-class accuracies for training (simplified)
        train_preds_binary = (np.array(train_preds) > 0.5).astype(int).reshape(-1)
        train_labels = np.array(train_labels).reshape(-1)
        
        train_acc_class0 = np.sum((train_preds_binary == 0) & (train_labels == 0)) / max(np.sum(train_labels == 0), 1) * 100
        train_acc_class1 = np.sum((train_preds_binary == 1) & (train_labels == 1)) / max(np.sum(train_labels == 1), 1) * 100
        
        # Calculate per-class accuracies for validation (simplified)
        val_preds = []
        val_labels = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_preds_binary = (np.array(val_preds) > 0.5).astype(int).reshape(-1)
        val_labels = np.array(val_labels).reshape(-1)
        
        val_acc_class0 = np.sum((val_preds_binary == 0) & (val_labels == 0)) / max(np.sum(val_labels == 0), 1) * 100
        val_acc_class1 = np.sum((val_preds_binary == 1) & (val_labels == 1)) / max(np.sum(val_labels == 1), 1) * 100
        
        # Store accuracies in history
        history['train_acc_class0'].append(train_acc_class0)
        history['train_acc_class1'].append(train_acc_class1)
        history['val_acc_class0'].append(val_acc_class0)
        history['val_acc_class1'].append(val_acc_class1)
        
        # Add to wandb logging
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "train_acc_class0": train_acc_class0,
            "train_acc_class1": train_acc_class1,
            "val_acc_class0": val_acc_class0,
            "val_acc_class1": val_acc_class1,
        })

        # Step the learning rate scheduler
        lr_scheduler.step(val_loss)
    
    return history, best_epoch_auc, best_epoch_loss

# Function to evaluate the model
def evaluate_model(model, data_loader, criterion, device, use_mixed_precision=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    auc = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc

def create_output_folder(config):
    # Get current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construct the base output path with the new structure
    base_output_path = config['results_path']
    imputation_path = os.path.join(base_output_path, config['data_imputation_type'])
    threshold_path = os.path.join(imputation_path, f"threshold_{config['threshold']}")
    
    # Create a folder name with datetime, run_id, and other naming conventions
    run_id = config.get('run_id', 'default_id')
    folder_name = f"{current_time}_{run_id}_patches_{config['org_size'][0]}x{config['org_size'][1]}"
    
    # Full output path
    output_path = os.path.join(threshold_path, folder_name)
    
    # Create all necessary directories
    os.makedirs(output_path, exist_ok=True)
    
    return output_path

# Add this function to create augmentation transforms
def get_augmentation_transforms(aug_level, size):
    if aug_level == 'none':
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif aug_level == 'light':
        return transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif aug_level == 'medium':
        return transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif aug_level == 'heavy':
        return transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Unsupported augmentation level: {aug_level}")

import random

def handle_class_imbalance(train_dataset, config):
    # Ensure all samples are used while addressing class imbalance
    if config['class_imb'] == 'Oversample':
        class_counts = [0, 0]
        for _, label in train_dataset.samples:
            class_counts[label] += 1
        
        weights = [1.0 / class_counts[label] for _, label in train_dataset.samples]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        
        # Return DataLoader with all samples but using the sampler
        return DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, num_workers=4)
    
    elif config['class_imb'] == 'Undersample':
        # This method will not use all samples, so consider removing it if full usage is required
        class_counts = [0, 0]
        for _, label in train_dataset.samples:
            class_counts[label] += 1
        
        min_count = min(class_counts)
        
        class_indices = [[] for _ in range(len(class_counts))]
        for idx, (_, label) in enumerate(train_dataset.samples):
            class_indices[label].append(idx)
        
        undersampled_indices = []
        for indices in class_indices:
            undersampled_indices.extend(random.sample(indices, min_count))
        
        undersampled_dataset = torch.utils.data.Subset(train_dataset, undersampled_indices)
        
        return DataLoader(undersampled_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    
    else:
        # Return DataLoader with all samples
        return DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_grid_config(tracking_file=None) -> List[Dict[str, Any]]:
    """Load grid search configuration and generate all parameter combinations."""

    grid_config_path = get_config_path('config_grid.json')
    with open(grid_config_path, 'r') as f:
        grid_config = json.load(f)

     # Add logic to track completed runs
    if grid_config['grid_start_file'] is not None:
        # If a tracking file with partly completed runs is specified, use it
        tracking_file = grid_config['grid_start_file']
    else:
        # Create new tracking file with empty dictionary
        tracking_data = {}
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=4)

    if tracking_file is not None and os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)    

    configs_to_run = []
    
    # Define which parameters to exclude from grid search
    exclude_params = {'size', 'org_size', 'grid_start_file', 'results_path'}

    # Get all parameters that are lists (except excluded ones)
    grid_params = {
        key: value for key, value in grid_config.items() 
        if isinstance(value, list) and key not in exclude_params
    }

    # Generate all possible combinations
    param_names = list(grid_params.keys())
    param_values = [grid_params[name] for name in param_names]
    
    for params in product(*param_values):
        # Create a parameter dictionary for this combination
        param_dict = dict(zip(param_names, params))
        
        # Create path key for checking (using threshold and data_imputation_type)
        path_key = os.path.join(str(RESULTS_DIR), 
                               param_dict['data_imputation_type'],
                               f"threshold_{param_dict['threshold']}")
        
        # Skip if already processed
        if tracking_data and path_key in tracking_data and tracking_data[path_key].get("runs", {}):
            print(f"Skipping configuration with params: {param_dict}")
            continue
        
        # If not processed, create full config for this combination
        run_config = grid_config.copy()
        for param_name, param_value in param_dict.items():
            run_config[param_name] = param_value
            
        configs_to_run.append(run_config)
    
    return configs_to_run, tracking_file

def load_custom_config() -> Dict[str, Any]:
    """Load custom configuration from JSON file."""
    custom_config_path = get_config_path('config_custom.json')
    with open(custom_config_path, 'r') as f:
        return json.load(f)

# TO DO: Hydra configuration setup
# - config_path: Should point to directory containing Hydra config files
# - config_name: Base Hydra config that determines whether to run grid search or single config
# - Grid search uses config_threshold_data_type_grid.json to generate parameter combinations
# - Single run uses config_custom.json for fixed parameters
# TODO: Update config_path to be relative to this file's location rather than absolute


def main() -> None:
    # Convert Hydra config to dict for easier handling    
    # Determine configurations to run

    start_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracking_file = str(RESULTS_DIR / 'completed_grid_runs' / f'completed_grid_runs_{start_date}.json')

    # Initialize new tracking file  
    if GRID_EXECUTION:
        configs_to_run, tracking_file = load_grid_config(tracking_file)
    else:
        configs_to_run = [load_custom_config()]

    try:
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        tracking_data = {}

    # Run training for each configuration
    for config in configs_to_run:
    
        # Create a unique identifier for this run
        start_time = datetime.now()
        run_id = str(uuid.uuid4())
        config['run_id'] = run_id
        
        # Set base_dir and results_path from config module if not set in config file
        if config.get('base_dir') is None:
            config['base_dir'] = str(BASE_DIR)
        if config.get('results_path') is None:
            config['results_path'] = str(RESULTS_DIR)
        
        print(f"Running with config: {config}")

        # Set up data transforms
        train_transforms = get_augmentation_transforms(config['augmentation'], config['size'])
        val_transforms = transforms.Compose([
            transforms.Resize(config['size']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Configuration - use config module for path generation
        config['data_path'] = str(get_imputed_patches_path(
            config['data_imputation_type'], 
            config['threshold'], 
            tuple(config['org_size'])
        ))
        
        config['output_path'] = create_output_folder(config)
        
        # Load datasets with different transforms for train and val/test
        train_dataset = ImageFolder(os.path.join(config['data_path'], 'train_dir'), transform=train_transforms)
        val_dataset = ImageFolder(os.path.join(config['data_path'], 'val_dir'), transform=val_transforms)
        test_dataset = ImageFolder(os.path.join(config['data_path'], 'test_dir'), transform=val_transforms)
        
        # Handle class imbalance
        train_loader = handle_class_imbalance(train_dataset, config)
        
        # Create data loaders for validation and test sets (no balancing needed)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        
        # Build model and get trainable parameters
        model, trainable_params = build_model(config)
        
        # Set up loss and optimizer with only trainable parameters
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(trainable_params, lr=config['lr'])
        
        try:
            # Initialize wandb with the unique run_id and set the directory to output_path
            wandb.init(
                project="ImageRecognition",
                config=config,
                name=f"{config['run_id']}_{config['tf_model']}_{config['org_size'][0]}x{config['org_size'][1]}_cutoff_{config['data_imputation_type']}_imputed_{config['threshold']}_aug_{config['augmentation']}",
                dir=config['output_path'],
                reinit=True
            )
            
            # Train model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            history, best_epoch_auc, best_epoch_loss = train_model(model, train_loader, val_loader, criterion, optimizer, config['epochs'], device, config)
            
            # Load the best model state
            checkpoint = torch.load(os.path.join(config['output_path'], 'best_auc_model.pth'))
            model.load_state_dict(checkpoint['model_state_dict'])
            best_val_auc = checkpoint['val_auc']
            best_val_loss = checkpoint['val_loss']
            
            # Evaluate on test set
            test_loss, test_auc = evaluate_model(model, test_loader, criterion, device, 
                                               use_mixed_precision=config['mixed_precision'])
            logging.info(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}')
            
            # Log final test metrics to wandb
            wandb.log({
                "test_loss": test_loss,
                "test_auc": test_auc,
            })
            
            # Generate and save visualizations
            summarize_model(config['output_path'], model, history)

            # Log all relevant model settings
            wandb.run.summary.update({
                "run_id": run_id,
                "model": config['tf_model'],
                "image_size": f"{config['org_size'][0]}x{config['org_size'][1]}",
                "data_imputation": config['data_imputation_type'],
                "threshold": config['threshold'],
                "full_retrain": config['full_retrain'],
                "freeze_till_block": config['freeze_till_block'],
                "learning_rate": config['lr'],
                "dropout_rate": config['dr'],
                "batch_size": config['batch_size'],
                "epochs": config['epochs'],
                "class_imbalance_handling": config['class_imb'],
                "augmentation_level": config['augmentation'],
                "final_test_loss": test_loss,
                "final_test_auc": test_auc,
                "best_model_epoch": best_epoch_auc,
                "best_val_auc": best_val_auc,
                "best_val_loss": best_val_loss,
                "mixed_precision": config['mixed_precision'],
            })

            # Save the entire config dictionary to a file using wandb
            config_file = os.path.join(config['output_path'], f"{run_id}_config.json")
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            wandb.save(config_file, base_path=config['output_path'])

            # Save final model
            final_model_path = os.path.join(config['output_path'], 'final_model.pth')
            torch.save(model.state_dict(), final_model_path)
            wandb.save(final_model_path, base_path=config['output_path'])

            # Delete .pth files in wandb subdirectory
            wandb_dir = os.path.join(config['output_path'], 'wandb')
            if os.path.exists(wandb_dir):
                for file in os.listdir(wandb_dir):
                    if file.endswith('.pth'):
                        file_path = os.path.join(wandb_dir, file)
                        os.remove(file_path)
                        logging.info(f"Deleted {file_path}")

                       # Calculate run duration
            end_time = datetime.now()
            duration = end_time - start_time
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            duration_str = f"{hours:02d}:{minutes:02d}"

            # Create path key directly matching the JSON structure
            path_key = os.path.join(config['results_path'], 
                                  config['data_imputation_type'],
                                  f"threshold_{config['threshold']}")
            
            # Update tracking data
            if path_key not in tracking_data:
                tracking_data[path_key] = {
                    "data_type": config['data_imputation_type'], 
                    "threshold": config['threshold'],
                    "runs": {}
                }
            
            tracking_data[path_key]["runs"][run_id] = {
                "folder": os.path.basename(config['output_path']),
                "completion_time": end_time.strftime("%Y%m%d_%H%M%S"),
                "duration": duration_str
            }

            # Save updated tracking data
            with open(tracking_file, 'w') as f:
                json.dump(tracking_data, f, indent=4)

        except Exception as e:
            logging.error(f"Error during run {run_id}: {str(e)}")
            raise
        finally:
            # Finish the wandb run
            wandb.finish()

if __name__ == "__main__":
    main()
