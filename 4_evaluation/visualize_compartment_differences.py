"""
Compartment Difference Analysis and Visualization Tool
---------------------------------------------------

This script provides a comprehensive suite of tools for analyzing and visualizing 
compartment differences in medical image recognition results, particularly focused 
on comparing tissue types between patient groups (e.g., COPD vs normal patients).

Main Components:
---------------
1. Data Processing and Model Predictions
    - Configuration management and path setup
    - Image loading and preprocessing
    - Model prediction generation with memory-efficient processing
    - Feature extraction from deep learning models

2. Statistical Analysis Pipeline
    - Basic statistical comparisons between groups
    - Bootstrap analysis for robust statistical testing
    - Principal Component Analysis (PCA) for feature dimensionality reduction
    - Mixed effects modeling for hierarchical data analysis
    - Support Vector Machine (SVM) classification analysis

3. Visualization Tools
    - Confusion matrix plotting
    - Distribution visualization
    - CNN filter activation visualization
    - Gradient-weighted Class Activation Mapping (Grad-CAM)
    - PCA and UMAP visualizations

Key Functions:
-------------
Data Processing:
    - load_config(): Loads and extends configuration with computed paths
    - read_img(): Reads and preprocesses single images
    - read_data(): Loads full dataset with labels

Prediction Generation:
    - calculate_pred(): Generates comprehensive prediction DataFrame
    - calculate_pred_and_features(): Memory-efficient prediction generation
    - determine_prototypes(): Identifies exemplar cases

Statistical Analysis:
    - calculate_aggregates(): Patient/WSI/airway level statistics
    - run_bootstrap_analysis(): Bootstrap sampling analysis
    - perform_pca_analysis(): PCA dimensionality reduction
    - perform_mixed_model_analysis(): Mixed effects modeling
    - perform_svm_analysis(): SVM classification

Visualization:
    - plot_cm(): Confusion matrix visualization
    - plot_histogram(): Distribution plotting
    - visualize_filter(): CNN filter visualization
    - grad_cam(): Gradient-weighted Class Activation Mapping

Typical Workflow:
---------------
1. Load configuration and data
2. Generate model predictions and extract features
3. Perform multi-level statistical analyses
4. Create visualizations for interpretation
5. Save results and figures

Dependencies:
------------
- Core: numpy, pandas, scipy, sklearn
- Deep Learning: tensorflow, torch, torchvision
- Visualization: matplotlib, seaborn
- Image Processing: cv2
- Statistical: statsmodels

Usage:
------
The script is typically run as part of a larger image recognition pipeline,
taking model outputs and performing comprehensive analysis of compartment
differences. It requires a configuration file specifying paths and parameters.

Example:
    config = load_config('config.json')
    data, labels = read_data(config['data_path'], config['size'])
    predictions = calculate_pred(model, data, labels)
    analyze_intersection(predictions, 'copd_group', config['output_path'])

Notes:
------
- Designed for medical imaging analysis, particularly tissue compartment comparison
- Includes memory-efficient processing for large datasets
- Provides multiple levels of statistical analysis
- Generates publication-ready visualizations
"""

# Import packages
import cv2
import os, gc, sys, glob
import pandas as pd
import numpy as np
from numpy import expand_dims
from tqdm import tqdm
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import seaborn as sns
import random
import itertools
import csv
import statistics
import plotly.express as px
import umap.umap_ as umap
import hdbscan 

from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.svm import SVC

from subprocess import check_output

import tensorflow as tf
import tempfile

import torch
import torchvision
from torchvision import transforms  # Add this import
from torchvision.datasets import ImageFolder  # Add this import
from torch.utils.data import DataLoader  # Also add this for test_loader

import statsmodels.api as sm
import statsmodels.formula.api as smf

import json

from multiprocessing import Pool, cpu_count
from functools import partial

def load_config(config_path):
    """
    Load and extend configuration with computed paths.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        dict: Configuration with computed paths
    """
    # Load base configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add computed paths
    base_dir = Path(config['base_dir'])
    patch_dir = f"patches_{config['original_size'][0]}x{config['original_size'][1]}"
    
    # Define individual paths instead of a dictionary
    config['data_path'] = base_dir / "1_data" / patch_dir / f"patches_cutoff_{config['data_type']}_imputed_{config['threshold']}"
    config['model_path'] = base_dir / "5_results" / config['data_type'] / f"threshold_{config['threshold']}" / config['model_run_id']
    config['results_path'] = base_dir / "5_results"
    config['data_original_path'] = base_dir / "1_data" / patch_dir / "patches_original"
    config['output_path'] = base_dir / "5_results" / "learning_evaluation"
    
    return config
    
# Function to read single image
def read_img(img_path, size):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Model is trained on RGB colors, but CV2 imports image as BGR
    img = cv2.resize(img, (size[0], size[1]))
    return img

def process_image(img_path, size, folder_name):
    """Process a single image and return its data and label."""
    try:
        img_data = read_img(img_path, size)
        return {
            'data': img_data,
            'label': [img_path.name, folder_name],
            'success': True
        }
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return {
            'success': False
        }

def read_data(data_path, size, debug_run=False, n_workers=None):
    """Load full data and labels using parallel processing."""
    folders = [a for a in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, a))    
              and 'oversample' not in a and 'dir' not in a]
    dirs = [data_path / f for f in folders]

    print(f'Loading data from directories: {", ".join(folders)}')
    
    # Initialize multiprocessing pool
    n_workers = n_workers or max(cpu_count() - 1, 1)
    pool = Pool(processes=n_workers)
    
    data = []
    labels = []
    
    # Process each directory
    for folder_idx, directory in enumerate(dirs):
        # Get all image paths
        img_paths = list(directory.glob('**/*.png'))
        if debug_run:
            img_paths = img_paths[:100]
            
        # Create partial function with fixed size and folder name
        process_func = partial(process_image, size=size, folder_name=folders[folder_idx])
        
        # Process images in parallel with progress tracking
        total = len(img_paths)
        for i, result in enumerate(pool.imap(process_func, img_paths)):
            if result['success']:
                data.append(result['data'])
                labels.append(result['label'])
            
            # Print progress every 1000 images
            if (i + 1) % 1000 == 0:
                print(f'Loading data from dir {directory}: {i+1}/{total}')

    pool.close()
    pool.join()

    if not data:
        raise ValueError("No images were loaded")
    
    data = np.array(data, dtype=np.float32) / 255

    labels = pd.DataFrame(labels, columns=['image', 'label'])
    labels['tiff'] = labels['image'].str.split('_').str[0]
    enc_labels = LabelEncoder().fit_transform(labels['label'])

    return data, labels, enc_labels

# Function to calculate predictions
# All useful information is stored in dataframe, including patient characteristics
def calculate_pred(model, data, labels, data_path):
    """
    Calculate model predictions and create a comprehensive DataFrame with prediction results
    and patient characteristics.
    
    Args:
        model: The trained model to generate predictions
        data: Array of preprocessed image data
        labels: DataFrame containing image labels
        data_path: Path to the data directory containing mapping.csv
    
    Returns:
        DataFrame containing predictions, patient data, and aggregated statistics
    """
    # 1. Generate predictions
    class_names = list(np.unique(labels.label))
    probabilities = model.predict(data)
    predictions = (probabilities[:,0] >= 0.5).astype(int)

    # 2. Create base predictions DataFrame
    tissue_type_mapping = {
        'Inside': 'Submucosa',
        'Outside': 'Adventitia'
    }

    preds_df = pd.DataFrame({
        'image': labels.image,
        'airway': [f"{img_name.split('__')[1].split(')')[0]})" for img_name in labels.image],
        'tiff': labels.tiff,
        'label': labels.label,
        'official_label': labels.label.replace(tissue_type_mapping),
        'enc_label': enc_labels,
        'pred': predictions,
        'pred_label': [class_names[i] for i in predictions],
        'prob_outside': probabilities[:,0]
    })
    
    # Calculate prediction accuracy metrics
    preds_df['prob_correct'] = abs(1 - (1 * preds_df.enc_label + preds_df.prob_outside))
    preds_df['ind_correct_pred'] = (preds_df.enc_label == preds_df.pred).astype(int)

    # 3. Add patient characteristics
    # TO DO: make better parent data path
    patient_data = pd.read_csv(data_path.parent.parent + '/mapping.csv', sep=';')
    preds_with_patient_data = preds_df.merge(patient_data, on='tiff', how='left')

    # Simplify COPD grouping
    preds_with_patient_data['copd_group_full'] = 'COPD'
    preds_with_patient_data.loc[preds_with_patient_data.copd_group == 'Normal', 'copd_group_full'] = 'Normal'

    # 4. Calculate patient-level statistics
    patient_stats = preds_with_patient_data.groupby('patient').agg({
        'tiff': 'nunique',
        'airway': 'nunique',
        'image': 'count',
        'label': lambda x: sum(x == 'Inside')
    }).rename(columns={
        'tiff': 'n_tiffs',
        'airway': 'n_airways', 
        'image': 'n_images',
        'label': 'n_submucosa'
    })

    # Add derived statistics
    patient_stats['n_adventitia'] = patient_stats['n_images'] - patient_stats['n_submucosa']
    patient_stats['perc_submucosa'] = patient_stats['n_submucosa'] / patient_stats['n_images']

    # 5. Create final DataFrame
    final_df = preds_with_patient_data.merge(
        patient_stats,
        left_on='patient',
        right_index=True
    )

    # Remove duplicate images
    final_df = final_df[~final_df.image.str.endswith('_2.png')]

    return final_df

# Function to determine top x true positives, negatives and false positives, negatives
def determine_prototypes(top, output_path, preds_df, data, labels):
    tn_certain = preds_df[(preds_df.enc_label == 0) & (preds_df.ind_correct_pred == 1)].sort_values(by=['prob_outside'], ascending=True)['image'].head(top)
    tp_certain = preds_df[(preds_df.enc_label == 1) & (preds_df.ind_correct_pred == 1)].sort_values(by=['prob_outside'], ascending=False)['image'].head(top)
    fn_certain = preds_df[(preds_df.enc_label == 1) & (preds_df.ind_correct_pred == 0)].sort_values(by=['prob_outside'], ascending=True)['image'].head(top)
    fp_certain = preds_df[(preds_df.enc_label == 0) & (preds_df.ind_correct_pred == 0)].sort_values(by=['prob_outside'], ascending=False)['image'].head(top)

    file_names = list(tn_certain) + list(tp_certain) + list(fn_certain) + list(fp_certain)
    pred_categories = list(np.repeat('tn', len(list(tn_certain)))) + list(np.repeat('tp', len(list(tp_certain)))) + list(np.repeat('fn', len(list(fn_certain)))) + list(np.repeat('fp', len(list(fp_certain))))

    for i in range(len(file_names)):
        file_name = file_names[i]
        img_index = labels[labels['image']==file_name].index.values.astype(int)[0]
        img = data[img_index] * 255
        patient = preds_df.patient[img_index]
        prob = preds_df.prob_outside[img_index]
        category = pred_categories[i]

        try:
            os.makedirs(output_path + '/Visualizations/prototypes/overall/' + category + '/')
        except:
            pass

        try:
            os.makedirs(output_path + '/Visualizations/prototypes/per_patient/' + str(patient) + '/' + category + '/')
        except:
            pass

        cv2.imwrite(output_path + '/Visualizations/prototypes/overall/' + category + '/' + str(prob) + '_' + file_name, img.astype(int))
        cv2.imwrite(output_path + '/Visualizations/prototypes/per_patient/' + str(patient) + '/' + category + '/' + str(prob) + '_' + file_name, img.astype(int))

    return tn_certain, tp_certain, fn_certain, fp_certain


# Function to extract features resulting from the last pooling layer
# prior to final classification layer
def calculate_features():
    model_without_top = Model(inputs=model.inputs, outputs=model.get_layer('global_average_pooling2d_7').output)
    features = model_without_top.predict(data)

    return features

# Function to calculate predictions - avoid GPU memory error
# All useful information is stored in dataframe, including patient characteristics
def calculate_pred_and_features(model, data_path, size, device, data_partitions='all'):
    """
    Calculate model predictions and extract features while avoiding GPU memory errors.
    
    Args:
        model: The trained model to use for predictions
        data_path: Path to the data directory
        size: Tuple of (height, width) for image resizing
        device: torch.device to use for computation
    
    Returns:
        tuple: (predictions_df, predictions_and_features_df)
    """
    # 1. Set up data loader for predictions
    test_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets based on specified partitions
    if data_partitions == 'all':
        # Load dataset directly from Inside/Outside directories
        test_dataset = ImageFolder(
            root=str(data_path) + '_all', # TO DO: make this dynamic and not duplicate folder
            transform=test_transforms,
        )
    else:
        # Load single specified partition
        test_dataset = ImageFolder(os.path.join(data_path, f'{data_partitions}_dir'), transform=test_transforms)

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)


    # test_dataset = ImageFolder(os.path.join(data_path, 'test_dir'), transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Get predictions and basic information
    model.eval()  # Set model to evaluation mode
    probabilities = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Generating predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            probabilities.extend(probs)
    
    probabilities = np.array(probabilities)
    predictions = (probabilities[:,0] >= 0.5).astype(int)
    
    # 2. Create base predictions DataFrame
    tissue_type_mapping = {
        'Inside': 'Submucosa',
        'Outside': 'Adventitia'
    }
    
    preds_df = pd.DataFrame({
        'image': [os.path.basename(test_dataset.samples[i][0]) for i in range(len(test_dataset))],
        'airway': [(os.path.basename(test_dataset.samples[i][0]).split('__')[1]).split(')')[0] + ')' 
                  for i in range(len(test_dataset))],
        'tiff': [os.path.basename(test_dataset.samples[i][0])[:os.path.basename(test_dataset.samples[i][0]).find('_')] 
                for i in range(len(test_dataset))],
        'label': [os.path.basename(os.path.dirname(test_dataset.samples[i][0])) 
                 for i in range(len(test_dataset))],
        'enc_label': test_dataset.targets,
        'pred': predictions,
        'pred_label': [list(test_dataset.classes)[i] for i in predictions],
        'prob_outside': probabilities[:,0]
    })
    
    # Add derived columns
    preds_df['official_label'] = preds_df.label.replace(tissue_type_mapping)
    preds_df['prob_correct'] = abs(1 - (1 * preds_df.enc_label + preds_df.prob_outside))
    preds_df['ind_correct_pred'] = (preds_df.enc_label == preds_df.pred).astype(int)

    # 3. Add patient characteristics
    patient_data = pd.read_csv(data_path.parent.parent / 'mapping.csv', sep=';')
    preds_with_patient_data = preds_df.merge(patient_data, on='tiff', how='left')
    
    # Simplify COPD grouping
    preds_with_patient_data['copd_group_full'] = 'COPD'
    preds_with_patient_data.loc[preds_with_patient_data.copd_group == 'Normal', 'copd_group_full'] = 'Normal'

    # 4. Calculate patient-level statistics
    patient_stats = preds_with_patient_data.groupby('patient').agg({
        'tiff': 'nunique',
        'airway': 'nunique',
        'image': 'count',
        'label': lambda x: sum(x == 'Inside')
    }).rename(columns={
        'tiff': 'n_tiffs',
        'airway': 'n_airways', 
        'image': 'n_images',
        'label': 'n_submucosa'
    })

    # Add derived statistics
    patient_stats['n_adventitia'] = patient_stats['n_images'] - patient_stats['n_submucosa']
    patient_stats['perc_submucosa'] = patient_stats['n_submucosa'] / patient_stats['n_images']

    # 5. Create final predictions DataFrame
    preds_df_final = preds_with_patient_data.merge(
        patient_stats,
        left_on='patient',
        right_index=True
    )

    # Print key characteristics of dataset
    print(f'Dataset contains {preds_df_final.shape[0]} images')
    print(f'Dataset contains {preds_df_final.patient.nunique()} patients')
    print(f'Dataset contains {preds_df_final.airway.nunique()} airways')
    
    # Remove duplicate images
    preds_df_final = preds_df_final[~preds_df_final.image.str.endswith('_2.png')]

    # 6. Extract and add model features
    features = []
    
    # Create a hook to get the features from the last layer before classification
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach().view(output.shape[0], -1)
        return hook

    # You may need to adjust this layer name based on your exact model architecture
    feature_layer = 'global_pool' #'conv2d_7b.bn.act'  # or 'global_pool' depending on your model
    if hasattr(model, feature_layer):
        getattr(model, feature_layer).register_forward_hook(get_activation('features'))
    else:
        # Print available layer names for debugging
        print("Available layers:")
        for name, _ in model.named_modules():
            print(name)
        raise ValueError(f"Could not find layer '{feature_layer}'. Please check the model architecture and use the correct layer name.")

    model.eval()
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Extracting features"):
            inputs = inputs.to(device)
            _ = model(inputs)
            features.extend(activation['features'].cpu().numpy())
    
    features = np.array(features)
    
    # Create features DataFrame
    features_df = pd.DataFrame(
        features, 
        columns=[f'f_{i}' for i in range(features.shape[1])]
    )
    # Fix: Extract just the path from each sample tuple and get the basename
    features_df['image'] = [os.path.basename(f[0]) for f in test_dataset.samples]

    # 7. Combine predictions with features
    preds_and_features_df = preds_df_final.merge(features_df, on='image')

    return preds_df_final, preds_and_features_df

"""## Functions to analyze (visualize) model"""

def plot_cm(obs, preds, output_path):
    """Plot confusion matrix."""
    # Calculate confusion matrix
    cm = confusion_matrix(obs, preds)
    
    # Calculate percentages and counts
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    
    # Create annotation text
    annot = np.empty_like(cm, dtype=str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Extract single elements before formatting
            percentage = float(cm_perc[i, j])
            count = int(cm[i, j])
            sum_val = int(cm_sum[i, 0])
            annot[i, j] = f'{percentage:.1f}%\n{count}/{sum_val}'
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=annot,
        fmt='',
        cmap='Blues',
        xticklabels=['Inside', 'Outside'],
        yticklabels=['Inside', 'Outside']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save plot
    plt.savefig(output_path)
    plt.close()

# Function to plot histogram, with two vertical lines if desired
def plot_histogram(data, xlabel, range, vertical_line_1, vertical_line_2, output_path, font_scale=1.5):
    """
    Create and save a publication-ready histogram plot.
    
    Args:
        data: Data to plot
        xlabel: Label for x-axis
        range: Tuple of (min, max) for x-axis range
        vertical_line_1: Position of first vertical line (or None)
        vertical_line_2: Position of second vertical line (or None)
        output_path: Path to save the figure
        font_scale: Scale factor for font sizes (default=1.5)
    """
    # Set up publication-ready style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.serif': ['Arial'],
        'text.usetex': False,
        'axes.labelsize': 11 * font_scale,
        'axes.titlesize': 12 * font_scale,
        'xtick.labelsize': 10 * font_scale,
        'ytick.labelsize': 10 * font_scale,
        'legend.fontsize': 10 * font_scale
    })

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    plt.hist(data, density=True, range=range, color='silver', alpha=0.7, edgecolor='black')
    
    # Add vertical lines if specified
    if vertical_line_1 is not None:
        plt.axvline(vertical_line_1, color='k', linestyle='dashed', linewidth=1)
    if vertical_line_2 is not None:
        plt.axvline(vertical_line_2, color='k', linestyle='dashed', linewidth=1)
    
    # Customize axes
    plt.xlabel(xlabel, labelpad=10)
    plt.ylabel('Density', labelpad=10)
    
    # Remove grid
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    
    # Save figure
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)  

# Function to get feature maps for specific layer
def visualize_filter(model, file_name, layer, output_path):
    img_index = labels[labels['image']==file_name].index.values.astype(int)[0]
    img = data[img_index]
    image = expand_dims(img, axis=0)
    sub_model = Model(inputs=model.inputs, outputs=model.layers[layer].output)
    feature_maps = sub_model.predict(image)

    fig = plt.figure()

    ix = 1
    for _ in range(4):
        for _ in range(8):
            ax = plt.subplot(4, 8, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0,:,:,ix-1], cmap = 'gray')
            ix += 1

    fig.suptitle('Layer ' + str(layer))

    try:
      os.makedirs(output_path + '/Visualizations/Filters/')
    except:
      pass

    fig.savefig(output_path + '/Visualizations/Filters/Layer=' + str(layer) + '_' + file_name)

    return fig

# Function to apply grad-cam (heatmaps visualization)
def grad_cam(image, model, eps=1e-8):
    gradModel = Model(
          inputs=[model.inputs],
          outputs=[model.get_layer(layerName).output,
            model.output])

    with tf.GradientTape() as tape:
          # cast the image tensor to a float-32 data type, pass the
          # image through the gradient model, and grab the loss
          # associated with the specific class index
          (convOutputs, predictions) = gradModel(np.expand_dims(img, axis=0))
          loss = predictions[:, 0]
    # use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, convOutputs)

    # compute the guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    # the convolution and guided gradients have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]

    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

    (w, h) = (img.shape[0], img.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))
    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")

    heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
    heatmap = cv2.applyColorMap(heatmap, colormap=cv2.COLORMAP_JET)

    return heatmap

# Function to cluster patches using UMAP cluster algorithm
def cluster_patches(output_path, features, file_names, labels):
    """
    Cluster and visualize patches using UMAP dimensionality reduction.
    
    Args:
        output_path (str): Path to save the output visualization
        features (np.ndarray): Feature matrix
        file_names (list): List of file names to process
        labels (pd.DataFrame): DataFrame containing image labels
        
    Returns:
        matplotlib.figure.Figure: The generated plot figure
        
    Raises:
        ValueError: If no valid files are found or if inputs are invalid
    """
    # Validate inputs
    if len(file_names) == 0:
        raise ValueError("No file names provided")
        
    # Get indices for selected files
    index_filenames = labels.image.index[labels.image.isin(file_names)]
    if len(index_filenames) == 0:
        raise ValueError("No matching files found in labels")
    
    # Select relevant features and labels
    features_selected = features[index_filenames]
    labels_selected = labels.label[index_filenames]
    
    # Configure UMAP
    mapping = umap.UMAP(
        n_neighbors=20,
        min_dist=0.001,
        metric='correlation',
        random_state=42  # Added for reproducibility
    ).fit(features_selected)
    
    # Determine color scheme based on unique labels
    unique_labels = np.unique(labels_selected)
    color_key = {
        'Inside': 'red',
        'Outside': 'blue'
    }
    colors = [color_key[label] for label in unique_labels]
    
    # Create and save plot
    plt.figure(figsize=(10, 8))
    p = umap.plot.points(
        mapping, 
        labels=labels_selected, 
        color_key=dict(zip(unique_labels, colors)),
        title='UMAP Clustering of Patches'
    )
    fig = p.get_figure()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def determine_cluster(features, file_names, labels, 
                     n_neighbors=200, min_dist=0.0, 
                     min_samples=100, min_cluster_size=500):
    """
    Perform UMAP dimensionality reduction and HDBSCAN clustering on image features.
    
    Args:
        features (np.ndarray): Feature matrix of shape (n_samples, n_features)
        file_names (list): List of image file names to process
        labels (pd.DataFrame): DataFrame containing image labels
        n_neighbors (int, optional): UMAP n_neighbors parameter. Defaults to 200
        min_dist (float, optional): UMAP min_dist parameter. Defaults to 0.0
        min_samples (int, optional): HDBSCAN min_samples parameter. Defaults to 100
        min_cluster_size (int, optional): HDBSCAN min_cluster_size parameter. Defaults to 500
    
    Returns:
        tuple: (standard_embedding, clustered, labels_pred_hdbscan)
    """
    # Normalize file names and ensure they exist in features index
    labels = labels.copy()
    labels['image'] = labels['image'].apply(lambda x: os.path.basename(str(x)))
    
    # Find common file names between features and labels
    common_files = list(set(features.index) & set(labels['image']))
    if not common_files:
        raise ValueError("No matching files found between features and labels")
    
    # Filter features and labels to only include common files
    features_selected = features.loc[common_files]
    labels_selected = labels[labels['image'].isin(common_files)]
    
    
    try:
        # Standard UMAP for visualization
        standard_embedding = umap.UMAP(
            random_state=42
        ).fit_transform(features_selected)

        # UMAP for clustering
        clusterable_embedding = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='correlation',
            n_components=100,
            random_state=42
        ).fit_transform(features_selected)

        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size
        )
        labels_pred_hdbscan = clusterer.fit_predict(clusterable_embedding)
        
        # Determine which points were assigned to clusters
        clustered = (labels_pred_hdbscan >= 0)
        
        print(f"\nClustering Results:")
        print(f"Number of clusters found: {len(set(labels_pred_hdbscan[labels_pred_hdbscan >= 0]))}")
        print(f"Number of points clustered: {sum(clustered)}")
        print(f"Number of noise points: {sum(~clustered)}")
        
        return standard_embedding, clustered, labels_pred_hdbscan
        
    except Exception as e:
        raise RuntimeError(f"Error during clustering: {str(e)}")

# Function to visualize clusters and quantify clusterability
# With the option to only select a specific subgroup through param file_names_selected
def visualize_and_quantify_cluster(labels, output_path, file_names_full, file_names_selected, standard_embedding, clustered, labels_pred_hdbscan):
    """
    Visualize and quantify clustering results.
    
    Args:
        labels (pd.DataFrame): DataFrame containing image labels
        output_path (Path): Path to save output files
        file_names_full (list): Complete list of file names
        file_names_selected (list): Selected file names to analyze
        standard_embedding (np.ndarray): UMAP embedding coordinates
        clustered (np.ndarray): Boolean array indicating clustered points
        labels_pred_hdbscan (np.ndarray): Cluster labels from HDBSCAN
    """
    # Get indices and labels for selected filenames
    selected_indices = [file_names_full.index(f) for f in file_names_selected]
    labels_selected = labels.loc[selected_indices, 'label'].values  # Use .values to get numpy array
    
    # Get embeddings and classifications for selected files
    standard_embedding_selected = standard_embedding[selected_indices]
    clustered_selected = clustered[selected_indices]
    labels_pred_hdbscan_selected = labels_pred_hdbscan[selected_indices]

    cdict = {
        'Inside': 'red',
        'Outside': 'blue',
        'Submucosa': 'red',
        'Adventitia': 'blue'
    }

    # Original clusters - Single figure with both compartments
    fig, ax = plt.subplots()
    
    # Plot both compartments on the same axes
    for compartment in np.unique(labels_selected):
        ix = labels_selected == compartment
        plot_compartment = 'Submucosa' if compartment == 'Inside' else 'Adventitia'
        ax.scatter(
            standard_embedding_selected[ix, 0], 
            standard_embedding_selected[ix, 1], 
            c=cdict[plot_compartment], 
            label=plot_compartment, 
            s=0.2,
            alpha=0.5  # Added transparency to see overlapping points
        )

    ax.legend(loc='upper right', markerscale=9)
    plt.savefig(str(output_path).replace('_classified', ''))
    plt.close()

    # Classified clusters - Single figure with noise and both compartments
    fig, ax = plt.subplots()
    
    # Plot noise points first
    ax.scatter(
        standard_embedding_selected[~clustered_selected, 0],
        standard_embedding_selected[~clustered_selected, 1],
        c=(0.5, 0.5, 0.5),
        s=0.2,
        alpha=0.5,
        label='Noise'
    )

    # Plot both compartments on the same axes
    for compartment in np.unique(labels_selected):
        ix = (labels_selected == compartment) & clustered_selected
        plot_compartment = 'Submucosa' if compartment == 'Inside' else 'Adventitia'
        ax.scatter(
            standard_embedding_selected[ix, 0], 
            standard_embedding_selected[ix, 1], 
            c=cdict[plot_compartment], 
            label=plot_compartment, 
            s=0.2,
            alpha=0.5  # Added transparency to see overlapping points
        )

    ax.legend(loc='upper right', markerscale=9)
    plt.savefig(output_path)
    plt.close()

    # Calculate metrics
    n_patches = len(labels_selected)
    n_submucosa = np.sum(labels_selected == 'Inside')
    n_adventitia = np.sum(labels_selected == 'Outside')
    n_noise = np.sum(~clustered_selected)
    n_noise_submucosa = np.sum((~clustered_selected) & (labels_selected == 'Inside'))
    n_noise_adventitia = np.sum((~clustered_selected) & (labels_selected == 'Outside'))

    # Calculate homogeneity scores
    homogen_score_full = metrics.homogeneity_score(labels_selected, labels_pred_hdbscan_selected)
    complete_score_full = metrics.completeness_score(labels_selected, labels_pred_hdbscan_selected)
    v_score_full = metrics.v_measure_score(labels_selected, labels_pred_hdbscan_selected)

    # Calculate scores for clustered points only
    clustered_mask = clustered_selected
    homogen_score_clustered = metrics.homogeneity_score(
        labels_selected[clustered_mask], 
        labels_pred_hdbscan_selected[clustered_mask]
    )
    complete_score_clustered = metrics.completeness_score(
        labels_selected[clustered_mask], 
        labels_pred_hdbscan_selected[clustered_mask]
    )
    v_score_clustered = metrics.v_measure_score(
        labels_selected[clustered_mask], 
        labels_pred_hdbscan_selected[clustered_mask]
    )

    return (n_patches, n_submucosa, n_adventitia, 
            homogen_score_full, complete_score_full, v_score_full,
            homogen_score_clustered, complete_score_clustered, v_score_clustered,
            n_noise, n_noise_submucosa, n_noise_adventitia)


"""# 2. Basic analyses per intersection"""
def calculate_aggregates(preds_df, output_path):
    """
    Calculate and save patient-level, WSI-level, and airway-level aggregates from prediction results.
    
    Args:
        preds_df (pd.DataFrame): DataFrame containing model predictions and metadata
        output_path (Path): Path where the aggregate CSV files should be saved
        
    Returns:
        tuple: (aggs_patient, aggs_wsi, aggs_airway) DataFrames containing the aggregated results
    """
    # Define common metrics to calculate
    def get_metrics(group_df):
        return pd.Series({
            'n_patches': len(group_df),
            'n_adventitia': group_df['enc_label'].sum(),
            'n_submucosa': len(group_df) - group_df['enc_label'].sum(),
            'acc': group_df['ind_correct_pred'].mean(),
            'tp': ((group_df['enc_label'] == 1) & (group_df['ind_correct_pred'] == 1)).sum(),
            'tn': ((group_df['enc_label'] == 0) & (group_df['ind_correct_pred'] == 1)).sum(),
            'fp': ((group_df['enc_label'] == 1) & (group_df['ind_correct_pred'] == 0)).sum(),
            'fn': ((group_df['enc_label'] == 0) & (group_df['ind_correct_pred'] == 0)).sum()
        })

    # Patient-level aggregation
    patient_groups = ['patient', 'copd_group', 'smoking_group', 'gold_stage', 'borderline']
    
    # First get the patient characteristics (take first occurrence for each patient)
    patient_chars = preds_df[patient_groups].drop_duplicates('patient')
    
    # Then calculate the metrics
    aggs_patient = (preds_df.groupby(patient_groups)
                   .apply(get_metrics)
                   .reset_index())
    
    # Add number of unique tiffs per patient
    tiff_counts = preds_df.groupby(patient_groups)['tiff'].nunique().reset_index()
    aggs_patient = aggs_patient.merge(tiff_counts, on=patient_groups)
    aggs_patient = aggs_patient.rename(columns={'tiff': 'n_tiffs'})
    
    # WSI-level aggregation
    wsi_groups = patient_groups + ['tiff']
    aggs_wsi = preds_df.groupby(wsi_groups).apply(get_metrics).reset_index()

    # Airway-level aggregation
    airway_groups = patient_groups + ['tiff', 'airway']
    aggs_airway = preds_df.groupby(airway_groups).apply(get_metrics).reset_index()

    # Add percentage calculations
    for df in [aggs_patient, aggs_wsi, aggs_airway]:
        df['perc_submucosa'] = df['n_submucosa'] / df['n_patches']

    # Save results
    vis_path = output_path
    vis_path.mkdir(parents=True, exist_ok=True)
    
    aggs_patient.to_csv(vis_path / 'patient_aggregates.csv', index=False)
    aggs_wsi.to_csv(vis_path / 'wsi_aggregates.csv', index=False)
    aggs_airway.to_csv(vis_path / 'airway_aggregates.csv', index=False)

    print("Columns in aggs_patient:", aggs_patient.columns.tolist())
    print("\nSample of aggs_patient data:")
    print(aggs_patient[['patient', 'gold_stage', 'copd_group']].head())
    
    # Save aggs_patient to csv in output path
    aggs_patient.to_csv(output_path / 'patient_aggregates.csv', index=False)
    return aggs_patient, aggs_wsi, aggs_airway

def filter_predictions(preds_df, intersection):
    """Filter predictions based on intersection type."""
    if intersection == 'smoking_group':
        return preds_df[(preds_df.smoking_group != '?') & 
                       (preds_df.smoking_group != 'ExS < year')]
    elif intersection == 'copd_group':
        return preds_df[(preds_df.smoking_group == 'ExS >= year') & 
                       (preds_df.copd_group != 'Else')]
    return preds_df

def calculate_group_metrics(group_df):
    """Calculate metrics for a group of predictions."""
    metrics = {
        'acc': group_df['ind_correct_pred'].mean(),
        'n_patients': group_df['patient'].nunique(),
        'n_tiffs': group_df['tiff'].nunique(),
        'n_patches': len(group_df),
        'n_outside': group_df['enc_label'].sum(),
        'n_inside': len(group_df) - group_df['enc_label'].sum(),
        'tp': ((group_df['enc_label'] == 1) & (group_df['ind_correct_pred'] == 1)).sum(),
        'tn': ((group_df['enc_label'] == 0) & (group_df['ind_correct_pred'] == 1)).sum(),
        'fp': ((group_df['enc_label'] == 1) & (group_df['ind_correct_pred'] == 0)).sum(),
        'fn': ((group_df['enc_label'] == 0) & (group_df['ind_correct_pred'] == 0)).sum()
    }
    return pd.Series(metrics)

def plot_prediction_counts(preds_filter, intersection, output_path):
    """Plot bar chart of correct/incorrect predictions."""
    count_cor_preds_df = (preds_filter.groupby([intersection, 'ind_correct_pred'])
                         [intersection].count().unstack('ind_correct_pred'))
    
    fig, ax = plt.subplots()
    count_cor_preds_df.plot(kind="bar", ax=ax)
    ax.legend(['Incorrect prediction', 'Correct prediction'])
    fig.savefig(output_path / 'barchart_count_cor_preds.png')
    plt.close(fig)

def plot_violin(data, x, y, title, xlabel, output_path, color="silver", 
                figsize=(10, 6), font_scale=1.2, dpi=300, rotate_xlabels=0):
    """
    Create and save a publication-ready violin plot.
    If x is 'patient', colors will be based on data partitions.
    """
    # Set up publication-ready style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.serif': ['Arial'],
        'text.usetex': False,
        'axes.labelsize': 11 * font_scale,
        'axes.titlesize': 12 * font_scale,
        'xtick.labelsize': 10 * font_scale,
        'ytick.labelsize': 10 * font_scale,
        'legend.fontsize': 10 * font_scale
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if x == 'patient':
        # Load patient partitions
        try:
            partitions_df = pd.read_excel('/workspace/ImageRecognition/1_data/patches_120x120/patient_partitions.xlsx', sheet_name='Partitions')
            partitions_df['patient'] = partitions_df['patient'].astype(int)

            # Create color mapping
            partition_colors = {
                'train': '#4DBBD5FF',  # blue
                'val': '#F39B7FFF',    # orange
                'test': '#8491B4FF'    # purple
            }

            # Sort data by partition (train -> val -> test)
            partition_order = ['train', 'val', 'test']
            data = data.copy()  # Create copy to avoid modifying original

            data['patient'] = data['patient'].astype(str)
            partitions_df['patient'] = partitions_df['patient'].astype(str)
            
            # Merge partition information
            data = data.merge(
                partitions_df[['patient', 'partition']], 
                on='patient',
                how='left'
            )
            
            # Create violin plot for each partition
            for partition in partition_order:
                partition_data = data[data['partition'] == partition]
                if not partition_data.empty:
                    sns.violinplot(
                        x=x, 
                        y=y, 
                        data=partition_data,
                        inner='quartile',
                        color=partition_colors[partition],
                        cut=0,
                        saturation=0.7,
                        width=0.8,
                        ax=ax
                    )
            
            # Add legend
            legend_elements = [plt.Rectangle((0,0),1,1, fc=color, label=part.capitalize()) 
                                        for part, color in partition_colors.items()]
            ax.legend(handles=legend_elements,
                     title="Data Partition",
                     loc='lower right',  # Move to bottom right 
                     frameon=True,       # Show frame
                     framealpha=1,       # Solid background
                     facecolor='white')         # Ensure legend is on top
            
        except Exception as e:
            print(f"Warning: Could not load patient partitions, using default color. Error: {e}")
            vplot = sns.violinplot(
                x=x, 
                y=y, 
                inner='quartile', 
                data=data, 
                color=color,
                cut=0,
                ax=ax,
                saturation=0.7,
                width=0.8
            )
    else:
        # Original behavior for non-patient x-axis
        vplot = sns.violinplot(
            x=x, 
            y=y, 
            inner='quartile', 
            data=data, 
            color=color,
            cut=0,
            order=data[x].sort_values(ascending=True).unique(),
            ax=ax,
            saturation=0.7,
            width=0.8
        )
    
    # Add subtle jitter points
    sns.stripplot(
        x=x,
        y=y,
        data=data,
        color='black',
        alpha=0.5,
        size=0.6,
        jitter=0.1,
        ax=ax,
        order=data[x].sort_values(ascending=True).unique() if x != 'patient' else None
    )
    
    # Enhance violin plot appearance
    for violin in ax.collections:
        if isinstance(violin, matplotlib.collections.PolyCollection):  # Only modify violin plots
            violin.set_alpha(0.8)
            violin.set_edgecolor('black')
            violin.set_linewidth(0.8)
    
    # Customize quartile lines
    for l in ax.lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(1.2)
        l.set_color('black')
        l.set_alpha(0.8)
    
    # Enhance grid
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    
    # Set labels and title
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel('True Class Probability', labelpad=10)
    if title:
        ax.set_title(title, pad=15)
    
    # Rotate x-labels if specified
    if rotate_xlabels:
        plt.xticks(rotation=rotate_xlabels)
    
    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def create_violin_plots(preds_filter, intersection, output_path):
    """Create all violin plots for a given intersection."""
    xlabel = ('Intersection on smoking history' if intersection == 'smoking_group' 
             else 'Intersection on lung function')
    
    # Overall distribution
    plot_violin(preds_filter, intersection, 'prob_correct', None, xlabel,
               output_path / 'violinplot_correct_probs.png', "silver")
    
    # Submucosa distribution
    submucosa_data = preds_filter[preds_filter.label == 'Inside']
    plot_violin(submucosa_data, intersection, 'prob_correct',
               'Correct probability distribution - Submucosa', xlabel,
               output_path / 'violinplot_submucosa_correct_probs.png', "salmon")
    
    # Adventitia distribution
    adventitia_data = preds_filter[preds_filter.label == 'Outside']
    plot_violin(adventitia_data, intersection, 'prob_correct',
               'Correct probability distribution - Adventitia', xlabel,
               output_path / 'violinplot_adventitia_correct_probs.png', "skyblue")

def create_patient_level_plots(group_df, group, output_path, min_patches=10):
    """Create patient-level violin plots for a specific group."""
    filtered_df = group_df[
        (group_df.n_submucosa >= min_patches) & 
        (group_df.n_adventitia >= min_patches)
    ].copy()
    filtered_df['patient'] = filtered_df['patient'].astype(int)
    
    plot_configs = [
        (filtered_df, None, "silver", 'violinplot_'),
        (filtered_df[filtered_df.label == 'Inside'], 
         'Correct probability distribution - Submucosa', "salmon", 
         'violinplot_submucosa_'),
        (filtered_df[filtered_df.label == 'Outside'],
         'Correct probability distribution - Adventitia', "skyblue",
         'violinplot_adventitia_')
    ]
    
    for data, title, color, prefix in plot_configs:
        try:
            plot_violin(
                data, 'patient', 'prob_correct', title, 'Patient',
                output_path / f"{prefix}{str(group.replace('/',''))}_correct_probs.png",
                color
            )
        except Exception as e:
            print(f"Failed to create {prefix} plot for group {group}: {str(e)}")

def create_detailed_violin_plots(preds_filter, intersection, output_path):
    """Create detailed violin plots for different lung function groups."""
    # Add new division categories
    preds_filter = preds_filter.copy()
    preds_filter['new_division'] = 'Else'
    preds_filter.loc[(preds_filter.gold_stage == 4), 'new_division'] = 'COPD stage IV'
    preds_filter.loc[(preds_filter.gold_stage == 3), 'new_division'] = 'COPD stage III'
    preds_filter.loc[(preds_filter.copd_group == 'Normal'), 'new_division'] = 'Normal'
    preds_filter.loc[(preds_filter.borderline == 1), 'new_division'] = 'Borderline'

    # Create output directory if it doesn't exist
    plot_dir = output_path / intersection
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot configurations
    plot_configs = [
        (preds_filter, None, "silver", 'violinplot_correct_probs_detailed'),
        (preds_filter[preds_filter.label == 'Inside'], 
         'Correct probability distribution - Submucosa', "salmon", 
         'violinplot_submucosa_correct_probs_detailed'),
        (preds_filter[preds_filter.label == 'Outside'],
         'Correct probability distribution - Adventitia', "skyblue",
         'violinplot_adventitia_correct_probs_detailed')
    ]
    
    for data, title, color, filename in plot_configs:
        plot_violin(
            data=data,
            x='new_division',
            y='prob_correct',
            title=title,
            xlabel='Lung Function Group',
            output_path=plot_dir / f'{filename}.png',
            color=color
        )

def create_density_plots(preds_df, output_path):
    """Create density plots for lung function groups and compartments."""
    # Set style
    sns.set()
    sns.set_style("whitegrid", {'axes.grid': False})
    
    # Create output directories
    plot_dir = output_path / 'density_plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter data for ex-smokers
    filtered_df = preds_df[
        (preds_df['smoking_group'] == 'ExS >= year') &
        (preds_df['copd_group'] != 'Else')
    ]
    
    # Create density plots per lung function group
    for group in filtered_df['copd_group'].unique():
        group_df = filtered_df[filtered_df['copd_group'] == group]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(
            data=group_df,
            x='prob_correct',
            hue='official_label',
            fill=True,
            cut=0,
            common_norm=False,
            alpha=0.3,
            palette=['salmon', 'cornflowerblue'],
            ax=ax
        )
        
        ax.legend(
            title='Lung compartment',
            loc='upper left',
            labels=['Adventitia', 'Submucosa']
        )
        ax.set_xlabel('True class probability')
        ax.set_ylabel('Probability density')
        ax.set_ylim(0, 5)
        
        fig.savefig(plot_dir / f'density_plot_{group.replace("/","")}.png')
        plt.close(fig)
    
    # Create density plots per lung compartment
    for compartment in ['Inside', 'Outside']:
        comp_df = filtered_df[filtered_df['label'] == compartment]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(
            data=comp_df,
            x='prob_correct',
            hue='copd_group',
            cut=0,
            fill=True,
            common_norm=False,
            alpha=0.3,
            ax=ax
        )
        
        ax.legend(
            title='Lung function',
            loc='upper left',
            labels=comp_df['copd_group'].unique()
        )
        ax.set_xlabel('True class probability')
        ax.set_ylabel('Probability density')
        ax.set_ylim(0, 5)
        
        fig.savefig(plot_dir / f'density_plot_{compartment.replace("/","")}.png')
        plt.close(fig)

def analyze_intersection(preds_df, intersection, output_path, min_patches_per_compartment=10):
    """Analyze and visualize predictions for a given intersection."""
    # Create output directory
    vis_path = output_path / intersection
    vis_path.mkdir(parents=True, exist_ok=True)
    
    # Filter predictions
    preds_filter = filter_predictions(preds_df, intersection)
    
    # Calculate and save aggregates
    aggs = (preds_filter.groupby(intersection)
            .apply(calculate_group_metrics)
            .reset_index())
    aggs.to_csv(vis_path / 'aggregates.csv')
    
    # Create visualizations
    plot_prediction_counts(preds_filter, intersection, vis_path)
    create_violin_plots(preds_filter, intersection, vis_path)
    
    # Create confusion matrices for each group
    for group in preds_filter[intersection].unique():
        group_df = preds_filter[preds_filter[intersection].astype(str) == str(group)]
        plot_cm(group_df.label, group_df.pred_label, 
                vis_path / f'confusion_matrix_{str(group.replace("/",""))}.png')
        
        # For COPD groups, create patient-level plots
        if intersection == 'copd_group':
            create_patient_level_plots(group_df, group, vis_path, 
                                     min_patches_per_compartment)

    # Add new detailed visualizations for COPD group
    if intersection == 'copd_group':
        preds_filter = preds_df[preds_df.smoking_group == 'ExS >= year']
        create_detailed_violin_plots(preds_filter, intersection, output_path)
        create_density_plots(preds_df, output_path)



"""# 3. Hypothesis testing - True class probabilities"""
def get_patient_groups(aggs_patient, min_patches_per_compartment):
    """
    Get COPD and normal patient groups based on filtering criteria.
    
    Args:
        aggs_patient (pd.DataFrame): Aggregated patient data
        min_patches_per_compartment (int): Minimum number of patches required per compartment
    
    Returns:
        tuple: Lists of COPD and normal patient IDs
    """
    base_conditions = {
        'smoking_group': 'ExS >= year',
        'n_adventitia': min_patches_per_compartment,
        'n_submucosa': min_patches_per_compartment
    }
    
    # Get COPD patients
    copd_mask = (
        (aggs_patient.gold_stage > 0) & 
        (aggs_patient.copd_group != 'Normal') &
        (aggs_patient.smoking_group == base_conditions['smoking_group']) &
        (aggs_patient.n_adventitia >= base_conditions['n_adventitia']) &
        (aggs_patient.n_submucosa >= base_conditions['n_submucosa'])
    )
    bootstrap_patients_copd = list(aggs_patient.patient[copd_mask])
    
    # Get normal patients
    normal_mask = (
        (aggs_patient.copd_group == 'Normal') &
        (aggs_patient.smoking_group == base_conditions['smoking_group']) &
        (aggs_patient.n_adventitia >= base_conditions['n_adventitia']) &
        (aggs_patient.n_submucosa >= base_conditions['n_submucosa'])
    )
    bootstrap_patients_normal = list(aggs_patient.patient[normal_mask])
    
    return bootstrap_patients_copd, bootstrap_patients_normal

def print_group_statistics(aggs_patient):
    """Print summary statistics for COPD and healthy patient groups."""
    copd_count = aggs_patient[
        (aggs_patient.copd_group == 'COPD stage III or IV') & 
        (aggs_patient.smoking_group == 'ExS >= year')
    ].shape[0]
    
    healthy_count = aggs_patient[
        (aggs_patient.copd_group == 'Normal') & 
        (aggs_patient.smoking_group == 'ExS >= year')
    ].shape[0]
    
    print(f'Total number of COPD patients: {copd_count}')
    print(f'Total number of healthy patients: {healthy_count}')

def perform_statistical_tests(copd_sample, normal_sample):
    """
    Perform statistical tests comparing COPD and normal samples.
    
    Args:
        copd_sample (array-like): Sample data from COPD patients
        normal_sample (array-like): Sample data from normal patients
        
    Returns:
        dict: Dictionary containing test results including:
            - Sample sizes for each group
            - Distribution test (Mann-Whitney U)
            - Median test (Mood's)
            - Variance test (Levene's)
            - Basic statistics (medians and variances)
    """
    # Basic statistics
    median_copd = statistics.median(copd_sample)
    median_normal = statistics.median(normal_sample)
    var_copd = statistics.variance(copd_sample)
    var_normal = statistics.variance(normal_sample)
    
    # Distribution testing (Mann-Whitney U)
    u_stat, u_p = stats.mannwhitneyu(copd_sample, normal_sample)
    
    # Median testing (Mood's)
    mood_stat, mood_p, mood_med, _ = stats.median_test(copd_sample, normal_sample)
    
    # Variance testing (Levene's)
    levene_stat, levene_p = stats.levene(copd_sample, normal_sample, center='median')
    
    return {
        'n_copd': len(copd_sample),
        'n_normal': len(normal_sample),
        'u_stat': u_stat,
        'u_p': u_p,
        'mood_stat': mood_stat, 
        'mood_p': mood_p,
        'median_copd': median_copd,
        'median_normal': median_normal,
        'levene_stat': levene_stat,
        'levene_p': levene_p,
        'var_copd': var_copd,
        'var_normal': var_normal
    }

def print_test_results(compartment, test_results):
    """Print formatted statistical test results."""
    print(f"{compartment.upper()}\n"
          f"- Distribution: Statistics={test_results['distribution_test']['statistic']}, p={test_results['distribution_test']['p_value']}\n"
          f"- Median: Statistics={test_results['median_test']['statistic']}, p={test_results['median_test']['p_value']}\n"
          f"- Variance: Statistics={test_results['variance_test']['statistic']}, p={test_results['variance_test']['p_value']}")

def perform_bootstrap_iteration(copd_patients, normal_patients, preds_df, patient_size, compartment_size):
    """
    Perform one bootstrap iteration for both lung compartments.
    
    Returns:
        list: List of dictionaries containing test results for each compartment
    """
    results = []
    
    # Randomly select patients
    copd_patients_sel = random.choices(copd_patients, k=patient_size)
    normal_patients_sel = random.choices(normal_patients, k=patient_size)
    
    for lung_compartment in ['Inside', 'Outside']:
        # Get samples for each group
        copd_sample = []
        normal_sample = []
        
        for patient in copd_patients_sel:
            patient_compartment = list(preds_df.prob_correct[
                (preds_df.patient == patient) & 
                (preds_df.label == lung_compartment)
            ])
            copd_sample.extend(random.choices(patient_compartment, k=compartment_size))
            
        for patient in normal_patients_sel:
            patient_compartment = list(preds_df.prob_correct[
                (preds_df.patient == patient) & 
                (preds_df.label == lung_compartment)
            ])
            normal_sample.extend(random.choices(patient_compartment, k=compartment_size))
        
        # Perform statistical tests
        test_results = perform_statistical_tests(copd_sample, normal_sample)
        test_results['lung_compartment'] = lung_compartment
        
        # Add non-parametric Levene test results
        test_results['np_levene_stat'] = 0  # Placeholder
        test_results['np_levene_p'] = 1  # Placeholder
        
        results.append(test_results)
        
    return results

def run_bootstrap_analysis(preds_df, aggs_patient, config):
    """
    Run complete bootstrap analysis comparing COPD and normal patients.
    
    Args:
        preds_df (pd.DataFrame): Predictions DataFrame
        aggs_patient (pd.DataFrame): Aggregated patient data
        config (dict): Configuration parameters
    """
    # Get patient groups
    bootstrap_patients_copd, bootstrap_patients_normal = get_patient_groups(
        aggs_patient, 
        config['min_patches_per_compartment']
    )
    
    # Print initial statistics
    print_group_statistics(aggs_patient)
    
    # Initialize results DataFrame
    group_tests_df = pd.DataFrame(columns=[
        'lung_compartment', 'u_stat', 'u_p', 'mood_stat', 'mood_p',
        'np_levene_stat', 'np_levene_p', 'levene_stat', 'levene_p',
        'median_copd', 'median_normal', 'var_copd', 'var_normal'
    ])
    
    # Initialize empty list to store all results
    all_results = []
    
    # Run iterations with progress bar
    for _ in tqdm(range(config['bootstrap_iterations']), desc="Running bootstrap analysis"):
        results = perform_bootstrap_iteration(
            bootstrap_patients_copd,
            bootstrap_patients_normal,
            preds_df,
            config['patient_size'],
            config['compartment_size']
        )
        all_results.extend(results)
    
    # Convert all results to DataFrame at once (more efficient than concatenating in loop)
    group_tests_df = pd.DataFrame(all_results)
    
    # Add significance indicators
    alpha = 0.05
    group_tests_df['ind_significant_dist_diff'] = (group_tests_df.u_p <= alpha).astype(int)
    group_tests_df['ind_significant_median_diff'] = (group_tests_df.mood_p <= alpha).astype(int)
    group_tests_df['ind_significant_var_diff'] = (group_tests_df.levene_p <= alpha).astype(int)
    group_tests_df['ind_significant_var_diff_np'] = (group_tests_df.np_levene_stat <= alpha).astype(int)
    
    # Save results
    output_path = config['output_path'] / 'group_level_test_results_COPD34_vs_normal.csv'
    group_tests_df.to_csv(output_path)
    
    return group_tests_df


def calculate_significance_values(group_tests_df):
    """
    Calculate significance values for distributions, medians, and variances.
    
    Args:
        group_tests_df (pd.DataFrame): DataFrame containing group test results
        
    Returns:
        dict: Dictionary containing all significance values
    """
    significance_values = {}
    
    # Calculate distribution significance
    for compartment in ['Inside', 'Outside']:
        df_compartment = group_tests_df[group_tests_df.lung_compartment == compartment]
        
        # Distribution significance
        significance_values[f'dist_{compartment.lower()}'] = 1 - (
            df_compartment.ind_significant_dist_diff.sum() / 
            (1.000 * df_compartment.ind_significant_dist_diff.count())
        )
        
        # Median significance
        significance_values[f'median_{compartment.lower()}'] = 1 - (
            df_compartment.ind_significant_median_diff.sum() / 
            (1.000 * df_compartment.ind_significant_median_diff.count())
        )
        
        # Variance significance
        significance_values[f'var_{compartment.lower()}'] = 1 - (
            df_compartment.ind_significant_var_diff.sum() / 
            (1.000 * df_compartment.ind_significant_var_diff.count())
        )
        
        significance_values[f'var_{compartment.lower()}_np'] = 1 - (
            df_compartment.ind_significant_var_diff_np.sum() / 
            (1.000 * df_compartment.ind_significant_var_diff_np.count())
        )
    
    return significance_values

def print_significance_results(significance_values):
    """Print formatted significance test results."""
    print('FOUND SIGNIFICANCE FOR DIFFERENCE IN DISTRIBUTIONS:')
    print(f"- Inside: {significance_values['dist_inside']}")
    print(f"- Outside: {significance_values['dist_outside']}\n")
    
    print('FOUND SIGNIFICANCE FOR DIFFERENCE IN MEDIANS:')
    print(f"- Inside: {significance_values['median_inside']}")
    print(f"- Outside: {significance_values['median_outside']}\n")
    
    print('FOUND SIGNIFICANCE FOR DIFFERENCE IN VARIANCES:')
    print(f"- Inside: {significance_values['var_inside']} - np: {significance_values['var_inside_np']}")
    print(f"- Outside: {significance_values['var_outside']} - np: {significance_values['var_outside_np']}")

    print(significance_values)

def plot_confidence_intervals(group_tests_df, output_path):
    """
    Plot confidence intervals for medians and variances using bootstrapping.
    
    Args:
        group_tests_df (pd.DataFrame): DataFrame containing group test results
        output_path (Path): Path to save visualization outputs
    """
    for lung_compartment in ['Inside', 'Outside']:
        compartment_data = group_tests_df[group_tests_df.lung_compartment == lung_compartment]
        
        # Calculate median bounds
        med_values = pd.concat([
            compartment_data.median_copd,
            compartment_data.median_normal
        ], axis=0)
        med_max = med_values.max()
        med_min = med_values.min()
        
        # Plot COPD median distribution
        lower_bound_median_copd = np.percentile(compartment_data.median_copd, 5)
        upper_bound_median_copd = np.percentile(compartment_data.median_copd, 95)
        plot_histogram(
            compartment_data.median_copd,
            'Median COPD patients',
            (med_min - 0.1, med_max + 0.1),
            lower_bound_median_copd,
            upper_bound_median_copd,
            output_path / f'distribution_median_copd_{lung_compartment}.png'
        )
        
        # Plot normal median distribution
        lower_bound_median_normal = np.percentile(compartment_data.median_normal, 5)
        upper_bound_median_normal = np.percentile(compartment_data.median_normal, 95)
        plot_histogram(
            compartment_data.median_normal,
            'Median normal patients',
            (med_min - 0.1, med_max + 0.1),
            lower_bound_median_normal,
            upper_bound_median_normal,
            output_path / f'distribution_median_normal_{lung_compartment}.png'
        )
        
        # Calculate variance bounds
        var_values = pd.concat([
            compartment_data.var_copd,
            compartment_data.var_normal
        ], axis=0)
        var_max = var_values.max()
        var_min = var_values.min()
        
        # Plot COPD variance distribution
        lower_bound_variance_copd = np.percentile(compartment_data.var_copd, 5)
        upper_bound_variance_copd = np.percentile(compartment_data.var_copd, 95)
        plot_histogram(
            compartment_data.var_copd,
            'Variance COPD patients',
            (var_min - 0.01, var_max + 0.01),
            lower_bound_variance_copd,
            upper_bound_variance_copd,
            output_path / f'distribution_variance_copd_{lung_compartment}.png'
        )
        
        # Plot normal variance distribution
        lower_bound_variance_normal = np.percentile(compartment_data.var_normal, 5)
        upper_bound_variance_normal = np.percentile(compartment_data.var_normal, 95)
        plot_histogram(
            compartment_data.var_normal,
            'Variance normal patients',
            (var_min - 0.01, var_max + 0.01),
            lower_bound_variance_normal,
            upper_bound_variance_normal,
            output_path / f'distribution_variance_normal_{lung_compartment}.png'
        )

def analyze_bootstrap_results(group_tests_df, output_path):
    """
    Analyze bootstrap results and create visualizations.
    
    Args:
        group_tests_df (pd.DataFrame): DataFrame containing group test results
        output_path (Path): Path to save outputs
    """
    # Calculate significance values
    significance_values = calculate_significance_values(group_tests_df)
    
    # Print results
    print_significance_results(significance_values)
    
    # Create confidence interval plots
    plot_confidence_intervals(group_tests_df, output_path)

def analyze_compartment_ratios(data, group_level='patient', min_patches=10, config=None):
    """
    Analyze compartment ratios between COPD and normal patients at different levels.
    
    Args:
        data (pd.DataFrame): Input dataframe containing predictions and metadata
        group_level (str): Level of analysis ('patient', 'tiff', or 'airway')
        min_patches (int): Minimum number of patches required per compartment
        
    Returns:
        dict: Statistical test results
    """
    # Calculate medians at specified group level
    groupby_cols = get_groupby_columns(group_level)
    medians = calculate_group_medians(data, groupby_cols, group_level)

    # Get COPD and normal samples
    copd_data, normal_data = filter_patient_groups(
        medians, 
        min_patches=min_patches
    )
    
    # First aggregate the data to ensure one value per patient/compartment
    copd_data = copd_data.groupby(
        ['patient', 'copd_group', 'smoking_group', 'gold_stage', 'borderline', 'label']
    )['prob_correct'].mean().reset_index()
    
    normal_data = normal_data.groupby(
        ['patient', 'copd_group', 'smoking_group', 'gold_stage', 'borderline', 'label']
    )['prob_correct'].mean().reset_index()

    # Then merge to match the old implementation style
    copd_data = copd_data.merge(
        copd_data[copd_data.label == 'Inside'][['patient', 'prob_correct']], 
        on='patient', 
        suffixes=('_x', '_y')
    )
    copd_data['ratio'] = copd_data.prob_correct_x / copd_data.prob_correct_y
    
    normal_data = normal_data.merge(
        normal_data[normal_data.label == 'Inside'][['patient', 'prob_correct']], 
        on='patient', 
        suffixes=('_x', '_y')
    )
    normal_data['ratio'] = normal_data.prob_correct_x / normal_data.prob_correct_y

    # Perform statistical tests
    test_results = perform_statistical_tests(
        copd_data.ratio.tolist(),
        normal_data.ratio.tolist()
    )
    
    # Create visualization
    create_scatter_plot(
        copd_data, normal_data,
        x_col='prob_correct_x', 
        y_col='prob_correct_y',
        group_level=group_level,
        config=config
    )
    
    return test_results

def create_scatter_plot(copd_data, normal_data, x_col, y_col, group_level, config):
    """Create scatter plot comparing COPD and normal patients."""
    plt.figure()
    plt.plot(copd_data[x_col], copd_data[y_col], 'o', 
            color='black', label='COPD stage III or IV')
    plt.plot(normal_data[x_col], normal_data[y_col], 'o', 
            color='lime', label='Normal')
    plt.xlabel('Adventitia')
    plt.ylabel('Submucosa')
    plt.legend()
    plt.savefig(f'{config["model_path"]}/Visualizations/copd_group/scatter_{group_level}_ratio_ins_out_COPD34_normal.svg')
    plt.savefig(f'{config["model_path"]}/Visualizations/copd_group/scatter_{group_level}_ratio_ins_out_COPD34_normal.png')
    plt.close()

def get_groupby_columns(group_level):
    """Get groupby columns based on analysis level."""
    base_columns = ['patient', 'label', 'copd_group', 'smoking_group', 
                   'gold_stage', 'borderline']
    if group_level == 'tiff':
        return base_columns + ['tiff']
    elif group_level == 'airway':
        return base_columns + ['tiff', 'airway']
    return base_columns

def calculate_group_medians(data, groupby_cols, group_level):
    """Calculate medians at specified group level."""
    # Calculate medians
    medians = data.groupby(groupby_cols)['prob_correct'].median().reset_index()
    
    # Add patch counts based on group level
    if group_level == 'patient':
        counts = data.groupby('patient').agg({
            'label': lambda x: sum(x == 'Inside'),
            'image': 'count'
        }).rename(columns={
            'label': 'n_submucosa',
            'image': 'total_patches'
        })
        counts['n_adventitia'] = counts['total_patches'] - counts['n_submucosa']
        medians = medians.merge(counts, on='patient')
    elif group_level == 'tiff':
        counts = data.groupby('tiff').agg({
            'label': lambda x: sum(x == 'Inside'),
            'image': 'count'
        }).rename(columns={
            'label': 'n_submucosa',
            'image': 'total_patches'
        })
        counts['n_adventitia'] = counts['total_patches'] - counts['n_submucosa']
        medians = medians.merge(counts, on='tiff')
    else:  # airway level
        counts = data.groupby(['tiff', 'airway']).agg({
            'label': lambda x: sum(x == 'Inside'),
            'image': 'count'
        }).rename(columns={
            'label': 'n_submucosa',
            'image': 'total_patches'
        })
        counts['n_adventitia'] = counts['total_patches'] - counts['n_submucosa']
        medians = medians.merge(counts, on=['tiff', 'airway'])
    
    return medians

def filter_patient_groups(medians, min_patches=10):
    """Filter and split data into COPD and normal groups."""
    base_conditions = {
        'smoking_group': 'ExS >= year',
        'n_adventitia': min_patches,
        'n_submucosa': min_patches
    }
    
    copd_data = medians[
        (medians.copd_group == 'COPD stage III or IV') &
        (medians.smoking_group == base_conditions['smoking_group']) &
        (medians.n_adventitia >= base_conditions['n_adventitia']) &
        (medians.n_submucosa >= base_conditions['n_submucosa'])
    ]
    
    normal_data = medians[
        (medians.copd_group == 'Normal') &
        (medians.smoking_group == base_conditions['smoking_group']) &
        (medians.n_adventitia >= base_conditions['n_adventitia']) &
        (medians.n_submucosa >= base_conditions['n_submucosa'])
    ]
    
    return copd_data, normal_data


"""### 3. Patient level"""
def analyze_patient_level_differences(preds_df, aggs_patient, compartment_size, output_path):
    """
    Analyze differences between patients at the individual level.
    
    Args:
        preds_df (pd.DataFrame): DataFrame containing predictions and metadata
        aggs_patient (pd.DataFrame): Aggregated patient-level data
        compartment_size (int): Minimum number of patches required per compartment
        output_path (Path): Path to save results
        
    Returns:
        pd.DataFrame: Summary of grouped patient test results
    """
    # Select patients meeting criteria
    patients_copd_sel = list(aggs_patient.patient[
        (aggs_patient.copd_group == 'COPD stage III or IV') & 
        (aggs_patient.smoking_group == 'ExS >= year') & 
        (aggs_patient.n_adventitia >= compartment_size) & 
        (aggs_patient.n_submucosa >= compartment_size)
    ])
    
    # Generate patient combinations for testing
    patient_combinations = create_patient_combinations(patients_copd_sel)
    
    # Perform pairwise comparisons
    patient_tests_df = perform_pairwise_comparisons(
        patient_combinations, 
        preds_df
    )
    
    # Add significance indicators
    patient_tests_df = add_significance_indicators(patient_tests_df)
    
    # Save results
    save_path = output_path / 'patient_level_tests_results.csv'
    patient_tests_df.to_csv(save_path)
    
    # Create summary
    grouped_results = summarize_results(patient_tests_df)
    
    return grouped_results

def create_patient_combinations(patients):
    """Create DataFrame of all possible patient combinations."""
    combinations = pd.DataFrame(
        list(itertools.product(patients, patients)),
        columns=['patient_1', 'patient_2']
    )
    return combinations[combinations.patient_1 != combinations.patient_2]

def perform_pairwise_comparisons(patient_combinations, preds_df):
    """
    Perform statistical tests for all patient pairs.
    
    Args:
        patient_combinations (pd.DataFrame): DataFrame containing patient pairs to compare
        preds_df (pd.DataFrame): DataFrame containing predictions
        
    Returns:
        pd.DataFrame: Results of all pairwise comparisons
    """
    results = []
    
    for _, row in patient_combinations.iterrows():
        p1, p2 = row['patient_1'], row['patient_2']
        
        for lung_compartment in ['Inside', 'Outside']:
            # Get data for both patients
            p1_data = get_patient_data(preds_df, p1, lung_compartment)
            p2_data = get_patient_data(preds_df, p2, lung_compartment)
            
            # Perform statistical tests
            test_results = perform_patient_statistical_tests(p1_data, p2_data)
            
            # Combine results
            result = {
                'patient_1': p1,
                'patient_2': p2,
                'lung_compartment': lung_compartment,
                **test_results
            }
            results.append(result)
            
            # Print progress
            print(f"{lung_compartment.upper()} - Comparing patient {p1} with patient {p2}")
            print_test_results(test_results)
    
    return pd.DataFrame(results)

def get_patient_data(preds_df, patient, compartment):
    """Extract probability data for a specific patient and compartment."""
    return list(preds_df.prob_correct[
        (preds_df.patient == patient) & 
        (preds_df.label == compartment)
    ])

def perform_patient_statistical_tests(p1_data, p2_data):
    """
    Perform statistical tests comparing two patients' data.
    
    Returns:
        dict: Dictionary containing test results
    """
    # Distribution testing
    u_stat, u_p = stats.mannwhitneyu(p1_data, p2_data)
    
    # Median testing
    median_p1 = statistics.median(p1_data)
    median_p2 = statistics.median(p2_data)
    mood_stat, mood_p, mood_med, _ = stats.median_test(p1_data, p2_data)
    
    # Variance testing
    var_p1 = statistics.variance(p1_data)
    var_p2 = statistics.variance(p2_data)
    levene_stat, levene_p = stats.levene(p1_data, p2_data, center='median')
    
    # Non-parametric Levene test results (placeholder)
    np_levene_stat, np_levene_p = perform_nonparametric_levene(p1_data, p2_data)
    
    return {
        'u_stat': u_stat,
        'u_p': u_p,
        'mood_stat': mood_stat,
        'mood_p': mood_p,
        'np_levene_stat': np_levene_stat,
        'np_levene_p': np_levene_p,
        'levene_stat': levene_stat,
        'levene_p': levene_p,
        'median_p1': median_p1,
        'median_p2': median_p2,
        'var_p1': var_p1,
        'var_p2': var_p2
    }

def perform_nonparametric_levene(p1_data, p2_data):
    """
    Perform non-parametric Levene test via rank transformations.
    
    Returns:
        tuple: (test statistic, p-value)
    """
    # Create DataFrames for both groups
    p1_df = pd.DataFrame({'group': 'p1', 'probs': p1_data})
    p2_df = pd.DataFrame({'group': 'p2', 'probs': p2_data})
    
    # Combine and rank
    combined_df = pd.concat([p1_df, p2_df], ignore_index=True)
    combined_df['rank'] = combined_df['probs'].rank(method='min')
    
    # Placeholder for actual implementation
    return 0, 1

def print_test_results(results):
    """Print formatted test results."""
    print(f"- Distribution: Statistics={results['u_stat']}, p={results['u_p']}")
    print(f"- Median: Statistics={results['mood_stat']}, p={results['mood_p']}")
    print("- Variance:")
    print(f"   parametric: Statistics={results['levene_stat']}, p={results['levene_p']}")
    print(f"   non-parametric: Statistics={results['np_levene_stat']}, p={results['np_levene_p']}\n")

def add_significance_indicators(df, alpha=0.05):
    """Add binary indicators for statistical significance."""
    # Update column names to match those created in perform_pairwise_comparisons
    df['ind_significant_dist_diff_indiv'] = (df.u_p <= alpha).astype(int)
    df['ind_significant_median_diff_indiv'] = (df.mood_p <= alpha).astype(int)
    df['ind_significant_var_diff_indiv'] = (df.levene_p <= alpha).astype(int)
    
    return df

def summarize_results(patient_tests_df):
    """Create summary of test results grouped by lung compartment."""
    return (patient_tests_df[['lung_compartment', 'u_p', 'levene_p']]
            .groupby('lung_compartment')
            .sum()
            .reset_index())


"""# RUN! 4. Hypothesis testing - PCA on features"""
def perform_pca_analysis(output_df, min_patches=10):
    """
    Perform PCA analysis on features and return transformed data.
    
    Args:
        output_df (pd.DataFrame): DataFrame containing features and metadata
        min_patches (int): Minimum number of patches required per compartment
        
    Returns:
        tuple: (output_pca_df, pca_ex_var) - PCA transformed data and explained variance
    """
    # Extract features (columns after index 23 are feature columns)
    features = output_df.iloc[:,23:]
    
    # Perform PCA with 80% explained variance threshold
    pca = PCA(n_components=0.80, svd_solver='full')
    features_PCA = pca.fit_transform(features)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        features_PCA, 
        columns=[f'pc_{i}' for i in range(features_PCA.shape[1])]
    )
    
    # Merge PCA results with original metadata
    output_pca_df = output_df.iloc[:,0:23].merge(pca_df, left_index=True, right_index=True)
    
    # Filter for COPD and normal patients with minimum patch requirements
    output_pca_df = output_pca_df[
        (output_pca_df.smoking_group == 'ExS >= year') & 
        (output_pca_df.copd_group != 'Else') & 
        (output_pca_df.n_adventitia >= min_patches) & 
        (output_pca_df.n_submucosa >= min_patches)
    ]
    
    return output_pca_df, pca.explained_variance_ratio_

def calculate_compartment_distances(output_pca_df):
    """
    Calculate Euclidean distances between inside and outside compartments per patient.
    
    Args:
        output_pca_df (pd.DataFrame): PCA transformed data
        
    Returns:
        pd.DataFrame: DataFrame containing distances between compartments per patient
    """
    # Get PC columns (ensure they are numeric)
    pc_columns = [col for col in output_pca_df.columns if col.startswith('pc_')]
    
    # Verify data types and convert if necessary
    for col in pc_columns:
        if not np.issubdtype(output_pca_df[col].dtype, np.number):
            print(f"Warning: Converting {col} to numeric")
            output_pca_df[col] = pd.to_numeric(output_pca_df[col], errors='coerce')
    
    # Calculate mean PC values per patient and compartment
    pc_means = output_pca_df.groupby(
        ['patient', 'copd_group', 'smoking_group', 'label']
    )[pc_columns].mean().reset_index()
    
    # Calculate distances between compartments
    distances = []
    patients = []
    copd_groups = []
    
    for patient in pc_means.patient.unique():
        # Get data for current patient
        patient_data = pc_means[pc_means.patient == patient]
        
        # Skip if we don't have both inside and outside data
        if len(patient_data) != 2:
            continue
            
        inside_pc = patient_data[patient_data.label == 'Inside'][pc_columns].values
        outside_pc = patient_data[patient_data.label == 'Outside'][pc_columns].values
        
        if inside_pc.size > 0 and outside_pc.size > 0:
            dist = np.sqrt(np.sum((inside_pc - outside_pc)**2))
            distances.append(dist)
            patients.append(patient)
            copd_groups.append(patient_data.iloc[0]['copd_group'])
    
    # Create distance DataFrame
    dist_df = pd.DataFrame({
        'patient': patients,
        'copd_group': copd_groups,
        'pc_distance_inside_outside': distances
    })
    
    return dist_df, pc_means

def analyze_compartment_differences(dist_df):
    """
    Perform statistical tests comparing COPD and normal patients.
    
    Args:
        dist_df (pd.DataFrame): DataFrame containing compartment distances
        
    Returns:
        dict: Dictionary containing test results
    """
    # Get samples for each group
    copd_sample = dist_df.loc[
        dist_df.copd_group == 'COPD stage III or IV', 
        'pc_distance_inside_outside'
    ]
    normal_sample = dist_df.loc[
        dist_df.copd_group == 'Normal', 
        'pc_distance_inside_outside'
    ]
    
    # Perform statistical tests
    u_stat, u_p = stats.mannwhitneyu(copd_sample, normal_sample)
    mood_stat, mood_p, mood_med, _ = stats.median_test(copd_sample, normal_sample)
    levene_stat, levene_p = stats.levene(copd_sample, normal_sample)
    
    return {
        'distribution': {'statistic': u_stat, 'p_value': u_p},
        'median': {'statistic': mood_stat, 'p_value': mood_p},
        'variance': {'statistic': levene_stat, 'p_value': levene_p}
    }

def plot_pc_visualization(pc_means, output_path, group=None, pca_explained_variance=None):
    """
    Create PCA visualization plots.
    
    Args:
        pc_means (pd.DataFrame): DataFrame containing PC means per patient
        output_path (Path): Path to save visualizations
        group (str, optional): Specific lung function group to plot. Defaults to None.
        pca_explained_variance (np.ndarray, optional): PCA explained variance ratios. Defaults to None.
    """
    cdict = {'Submucosa': 'red', 'Adventitia': 'blue'}
    
    # Filter data if group specified
    plot_data = pc_means[pc_means.copd_group == group] if group else pc_means
    
    # Create plot
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Plot points
    for compartment in np.unique(plot_data.label):
        ix = (plot_data.label == compartment)
        comp_label = 'Submucosa' if compartment == 'Inside' else 'Adventitia'
        ax.scatter(
            plot_data.loc[ix, "pc_0"], 
            plot_data.loc[ix, "pc_1"], 
            c=cdict[comp_label], 
            label=comp_label, 
            s=150  # Increased dot size
        )
    
    # Add annotations
    for ix, txt in enumerate(plot_data.patient):
        ax.annotate(
            txt, 
            (plot_data["pc_0"].iloc[ix], plot_data["pc_1"].iloc[ix]), 
            size=18
        )
    
    # Customize plot
    ax.legend(loc='upper right', markerscale=1, fontsize=25)
    
    # Add explained variance to axis labels if provided
    if pca_explained_variance is not None:
        ax.set_xlabel(f'Principal component 1 ({pca_explained_variance[0]:.1%} explained var.)', fontsize=20)
        ax.set_ylabel(f'Principal component 2 ({pca_explained_variance[1]:.1%} explained var.)', fontsize=20)
    else:
        ax.set_xlabel('Principal component 1', fontsize=20)
        ax.set_ylabel('Principal component 2', fontsize=20)
    
    # Remove grid and add border
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    
    # Save plot
    filename = 'average_pc_per_patient_compartment'
    if group:
        filename += f'_{group.replace("/","")}'
        # Save PCs for the group
        plot_data.to_csv(output_path / f'{filename}.csv')
    
    fig.savefig(output_path / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def analyze_pca_features(output_df, output_path):
    """
    Main function to perform PCA analysis on features.
    
    Args:
        output_df (pd.DataFrame): DataFrame containing features and metadata
        output_path (Path): Path to save results and visualizations
    """
    # Perform PCA
    output_pca_df, pca_ex_var = perform_pca_analysis(output_df)
    print(f"PCA Shape: {output_pca_df.shape}")
    print(f"Components with variance < 1%: {np.where(pca_ex_var < 0.01)}")
    
    # Calculate distances between compartments
    dist_df, pc_means = calculate_compartment_distances(output_pca_df)
    print("\nMean distances between compartments by group:")
    print(dist_df.groupby(['copd_group']).mean())
    
    # Perform statistical analysis
    test_results = analyze_compartment_differences(dist_df)
    print("\nStatistical test results:")
    for test, results in test_results.items():
        print(f"- {test.capitalize()}: Statistics={results['statistic']}, p={results['p_value']}")
    
    # Create visualizations
    plot_pc_visualization(pc_means, output_path)
    
    # Create per-group visualizations
    for group in np.unique(pc_means.copd_group):
        print(f"\nProcessing group: {group}")
        plot_pc_visualization(pc_means, output_path, group)



    # Also print the total number of patients with inside and outside patches, for the copd group and the normal group
    print(f"\nTotal number of patients with inside and outside patches, for the copd group and the normal group:")
    print(pc_means.groupby(['copd_group', 'label']).size())
    
    # Print patch counts per group and compartment
    print("\nNumber of patches per group and compartment:")
    print(output_pca_df.groupby(['copd_group', 'label']).size())

    # Get the PCA transformed features from output_pca_df
    feature_columns = [col for col in output_pca_df.columns if col.startswith('pc_')]
    features_PCA = output_pca_df[feature_columns].values

    return output_pca_df, features_PCA

"""## RUN! Mixed model, with fixed and random (due to hierarchy) effects"""
def perform_mixed_model_analysis(output_pca_df, features_PCA):
    """
    Perform mixed model analysis with fixed and random effects.
    
    Args:
        output_pca_df (pd.DataFrame): DataFrame containing PCA results
        features_PCA (np.ndarray): PCA transformed features
        
    Returns:
        statsmodels.regression.mixed_linear_model.MixedLMResults: Fitted mixed model results
    """
    # Select relevant columns for mixed model
    columns = [
        'enc_label', 'patient', 'copd_group', 'prob_correct'
    ] + [f'pc_{i}' for i in range(features_PCA.shape[1])]
    
    data_for_md = output_pca_df[columns]
    
    # Define model formula
    formula = "prob_correct ~ C(copd_group) + enc_label"
    # Uncomment to add PC components
    # for i in range(3):
    #     formula += f" + pc_{i}"
    
    print(f"Mixed Model Formula: {formula}")
    
    # Fit mixed model
    md = smf.mixedlm(formula, data_for_md, groups=data_for_md['patient'])
    mdf = md.fit()
    
    print("\nMixed Model Results:")
    print(mdf.summary())
    
    return mdf

def perform_svm_analysis(features_PCA, output_df, kernel='poly'):
    """
    Perform SVM analysis on PCA features.
    
    Args:
        features_PCA (np.ndarray): PCA transformed features
        output_df (pd.DataFrame): Original DataFrame with labels
        kernel (str, optional): SVM kernel type. Defaults to 'poly'
        
    Returns:
        tuple: (clf, predictions) - Fitted SVM classifier and predictions
    """

    # Fit SVM model
    clf = SVC(kernel=kernel)
    clf.fit(features_PCA, output_df.enc_label)
    predictions = clf.predict(features_PCA)
    
    # Calculate overall metrics
    print("\nOverall SVM Performance:")
    print(f"Accuracy: {metrics.accuracy_score(output_df.enc_label, predictions):.3f}")
    print(f"Precision: {metrics.precision_score(output_df.enc_label, predictions):.3f}")
    print(f"Recall: {metrics.recall_score(output_df.enc_label, predictions):.3f}")
    
    # Calculate metrics per lung function group
    for lung_function in ['COPD stage III or IV', 'Normal']:
        inds = output_df[output_df.copd_group == lung_function].index
        print(f"\nMetrics for {lung_function}:")
        print(f"Accuracy: {metrics.accuracy_score(output_df.enc_label[inds], predictions[inds]):.3f}")
        print(f"Precision: {metrics.precision_score(output_df.enc_label[inds], predictions[inds]):.3f}")
        print(f"Recall: {metrics.recall_score(output_df.enc_label[inds], predictions[inds]):.3f}")
    
    return clf, predictions

def plot_svm_support_vectors(features_PCA, clf, output_path):
    """
    Create and save visualization of SVM support vectors.
    
    Args:
        features_PCA (np.ndarray): PCA transformed features
        clf (sklearn.svm.SVC): Fitted SVM classifier
        output_path (Path): Path to save visualization
    """
    plt.figure(figsize=(10, 8))
    
    # Plot all data points
    plt.scatter(features_PCA[:,0], features_PCA[:,1], alpha=0.5, label='Data points')
    
    # Plot support vectors
    support_vectors = clf.support_vectors_
    plt.scatter(
        support_vectors[:,0], 
        support_vectors[:,1], 
        color='red', 
        alpha=0.8,
        label='Support vectors'
    )
    
    plt.title('SVM Classification with Support Vectors')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    
    # Save plot
    plt.savefig(output_path / 'svm_support_vectors.png')
    plt.close()

def perform_separate_svm_analysis(features_PCA, output_df):
    """
    Perform separate SVM analyses for each lung function group.
    
    Args:
        features_PCA (np.ndarray): PCA transformed features
        output_df (pd.DataFrame): DataFrame containing labels and metadata
        
    Returns:
        dict: Dictionary containing SVM classifiers and predictions for each group
    """
    results = {}
    
    for lung_function in ['COPD stage III or IV', 'Normal']:
        # Get indices for current lung function group
        inds = output_df[output_df.copd_group == lung_function].index
        
        # Train SVM on current group's data
        clf_separate = SVC(kernel='poly')
        clf_separate.fit(features_PCA[inds], output_df.enc_label[inds])
        
        # Generate predictions for all data
        preds_clf_separate = clf_separate.predict(features_PCA)
        
        # Calculate and print metrics
        print(f"\nMetrics for {lung_function} model:")
        print(f"Accuracy: {metrics.accuracy_score(output_df.enc_label[inds], preds_clf_separate[inds]):.3f}")
        print(f"Precision: {metrics.precision_score(output_df.enc_label[inds], preds_clf_separate[inds]):.3f}")
        print(f"Recall: {metrics.recall_score(output_df.enc_label[inds], preds_clf_separate[inds]):.3f}")
        
        # Store results
        results[lung_function] = {
            'classifier': clf_separate,
            'predictions': preds_clf_separate,
            'training_indices': inds
        }
    
    return results

def plot_separate_svm_results(features_PCA, output_df, svm_results, output_path):
    """
    Create visualization of SVM results for separate lung function models.
    
    Args:
        features_PCA (np.ndarray): PCA transformed features
        output_df (pd.DataFrame): Original DataFrame with labels
        svm_results (dict): Results from separate SVM analyses
        output_path (Path): Path to save visualizations
    """
    colors = {
        'COPD stage III or IV': 'salmon',
        'Normal': 'skyblue'
    }
    
    for lung_function, results in svm_results.items():
        plt.figure(figsize=(10, 8))
        
        # Get indices for current group
        inds = results['training_indices']
        
        # Plot training data points
        plt.scatter(
            features_PCA[inds, 0],
            features_PCA[inds, 1],
            alpha=0.5,
            color=colors[lung_function],
            label='Training data'
        )
        
        # Plot support vectors
        support_vectors = results['classifier'].support_vectors_
        plt.scatter(
            support_vectors[:, 0],
            support_vectors[:, 1],
            color='red',
            alpha=0.8,
            label='Support vectors'
        )
        
        plt.title(f'SVM Classification for {lung_function}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        
        # Save plot
        plt.savefig(output_path / f'svm_support_vectors_{lung_function.replace("/", "")}.png')
        plt.close()


"""# 5. Cluster analysis: PCA and UMAP"""
def create_pca_visualizations(features, preds_df, labels, output_path):
    """
    Create PCA visualizations for the full dataset and per intersection group.
    
    Args:
        features (np.ndarray): Feature matrix
        preds_df (pd.DataFrame): DataFrame containing predictions and metadata
        labels (pd.DataFrame): DataFrame containing image labels
        output_path (Path): Path to save visualizations
        
    Returns:
        tuple: (features_PCA, pca) - PCA transformed features and fitted PCA object
    """
    # Create visualization directory
    vis_path = output_path
    vis_path.mkdir(parents=True, exist_ok=True)
    
    # Perform PCA on full dataset
    pca = PCA(n_components=100)
    features_PCA = pca.fit_transform(features)
    
    # Create labels for PCA components
    pca_labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
    
    # Print cumulative explained variance
    print("Cumulative explained variance ratio:")
    print(np.cumsum(pca.explained_variance_ratio_))
    
    # Create full dataset visualization
    create_scatter_matrix(
        features_PCA,
        pca_labels,
        preds_df.label.map({'Inside': 'Submucosa', 'Outside': 'Adventitia'}),
        vis_path / 'PCA_plot_full_data.png'
    )
    
    # Create per-intersection visualizations
    intersection_configs = {
        'smoking_group': lambda df: df[df.smoking_group != '?'],
        'copd_group': lambda df: df[df.smoking_group == 'ExS >= year']
    }
    
    for intersection, filter_func in intersection_configs.items():
        create_intersection_visualizations(
            intersection,
            filter_func(preds_df),
            features,
            labels,
            vis_path,
            pca.explained_variance_ratio_
        )
    
    return features_PCA, pca

def create_scatter_matrix(features_pca, pca_labels, color_mapping, output_path):
    """
    Create and save a scatter matrix visualization.
    
    Args:
        features_pca (np.ndarray): PCA transformed features
        pca_labels (dict): Labels for PCA components
        color_mapping (pd.Series): Mapping for color coding points
        output_path (Path): Path to save visualization
    """
    # Convert features_pca to DataFrame with labeled columns
    df = pd.DataFrame(
        features_pca[:, :5],  # Take first 5 components
        columns=[pca_labels[str(i)] for i in range(5)]  # Use corresponding labels
    )
    
    # Add color mapping as a column to the DataFrame
    # Make sure color_mapping has the same length as features_pca
    if len(color_mapping) != len(df):
        color_mapping = color_mapping[:len(df)]
    df['compartment'] = color_mapping
    
    # Create scatter matrix
    fig = px.scatter_matrix(
        df,
        dimensions=df.columns[:-1],  # Exclude the color column
        color='compartment'  # Use the added column for coloring
    )
    
    # Update layout for font sizes
    fig.update_layout(
        font=dict(size=15),  # 1.5x default font size (default is 10)
        title_font=dict(size=18)
    )
    
    # Update axis labels font size
    for axis in fig.layout:
        if type(fig.layout[axis]) == go.layout.XAxis:
            fig.layout[axis].title.font.size = 15
        if type(fig.layout[axis]) == go.layout.YAxis:
            fig.layout[axis].title.font.size = 15
    
    fig.update_traces(diagonal_visible=False)
    
    # Save visualization with higher DPI
    fig.write_image(str(output_path), width=1980, height=1080, scale=3)  # scale=3 for 300 DPI

def create_intersection_visualizations(intersection, filtered_df, features, labels, output_path, pca_explained_variance):
    """
    Create PCA visualizations for each group within an intersection.
    
    Args:
        intersection (str): Name of the intersection ('smoking_group' or 'copd_group')
        filtered_df (pd.DataFrame): Filtered DataFrame for the intersection
        features (np.ndarray): Feature matrix
        labels (pd.DataFrame): DataFrame containing image labels
        output_path (Path): Base path for saving visualizations
        pca_explained_variance (np.ndarray): PCA explained variance ratios
    """
    # Create intersection directory
    intersection_path = output_path / intersection
    intersection_path.mkdir(exist_ok=True)
    
    for group in filtered_df[intersection].astype(str).unique():
        # Filter data for current group
        group_df = filtered_df[filtered_df[intersection].astype(str) == group]
        
        # Get features for the current group
        # Convert features to numpy array if it's a DataFrame
        features_array = features.values if isinstance(features, pd.DataFrame) else features
        
        # Get indices for the current group's images
        # Use numpy's where to get valid indices
        group_indices = np.where(labels.image.isin(group_df.image))[0]
        
        if len(group_indices) > 100:  # Only process if we have enough samples
            # Perform PCA for group
            pca = PCA(n_components=100)
            features_pca_group = pca.fit_transform(features_array[group_indices])
            
            # Create labels for PCA components
            pca_labels = {
                str(i): f"PC {i+1} ({var:.1f}%)"
                for i, var in enumerate(pca.explained_variance_ratio_ * 100)
            }
            
            # Create and save visualization
            output_file = intersection_path / f'PCA_plot_{group.replace("/","-")}.png'
            create_scatter_matrix(
                features_pca_group,
                pca_labels,
                group_df.label.map({'Inside': 'Submucosa', 'Outside': 'Adventitia'}),
                output_file,
                pca_explained_variance
            )



"""### Generate, visualize and quantify clusters on overall, group, patient and airway level"""
def setup_cluster_analysis(output_path):
    """Initialize cluster analysis by creating output CSV"""
    fieldnames = [
        'group', 'patient', 'airway', 'n_patches', 'n_submucosa', 'n_adventitia',
        'homogeneity_score_full', 'completeness_score_full', 'v_measure_full',
        'homogeneity_score_clustered', 'completeness_score_clustered', 'v_measure_clustered',
        'n_noise', 'n_noise_submucosa', 'n_noise_adventitia',
        'perc_noise_submucosa', 'perc_noise_adventitia'
    ]
    cluster_df = pd.DataFrame(columns=fieldnames)
    cluster_df.to_csv(output_path / 'clusterability.csv', index=False)
    return cluster_df

def analyze_full_dataset(preds_df, features, labels, output_path):
    """Analyze clusters for the full dataset"""
    # Get embeddings and clusters
    standard_embedding, clustered, labels_pred_hdbscan = determine_cluster(
        features, list(preds_df.image), labels
    )
    
    # Analyze full dataset
    metrics = visualize_and_quantify_cluster(
        labels,
        output_path / 'cluster_classified_full_data.png',
        list(preds_df.image),
        list(preds_df.image),
        standard_embedding,
        clustered,
        labels_pred_hdbscan
    )
    
    # Save results
    save_cluster_metrics('Full', '-', '-', metrics, output_path)
    
    return standard_embedding, clustered, labels_pred_hdbscan

def analyze_compartments(labels, preds_df, standard_embedding, clustered, labels_pred_hdbscan, output_path):
    """Analyze clusters for each lung compartment"""
    for compartment in np.unique(preds_df.label):
        visualize_and_quantify_cluster(
            labels, 
            output_path / f'cluster_classified_{compartment}.png',
            list(preds_df.image),
            list(preds_df[preds_df.label == compartment].image),
            standard_embedding,
            clustered,
            labels_pred_hdbscan
        )

def analyze_intersection_groups(labels, preds_df, standard_embedding, clustered, labels_pred_hdbscan, output_path):
    """Analyze clusters for intersection groups"""
    for intersection in ['smoking_group', 'copd_group']:
        preds_filter = filter_predictions(preds_df, intersection)
        analyze_group_clusters(
            labels, 
            preds_filter,
            intersection,
            standard_embedding,
            clustered, 
            labels_pred_hdbscan,
            output_path
        )

def analyze_group_clusters(labels, preds_filter, intersection, standard_embedding, clustered, labels_pred_hdbscan, output_path):
    """
    Analyze clusters for each group in an intersection, including patient and airway level analysis.
    
    Args:
        preds_filter (pd.DataFrame): Filtered predictions DataFrame
        intersection (str): Intersection type ('smoking_group' or 'copd_group')
        standard_embedding (np.ndarray): UMAP embeddings
        clustered (np.ndarray): Boolean array indicating clustered points
        labels_pred_hdbscan (np.ndarray): HDBSCAN cluster labels
        output_path (Path): Path to save outputs
    """
    print(f'Quantifying clusters per {intersection}')
    
    # Create directories
    intersection_path = output_path / intersection
    intersection_path.mkdir(parents=True, exist_ok=True)
    
    for group in np.unique(preds_filter[intersection].astype(str)):
        # Group level analysis
        group_df = preds_filter[preds_filter[intersection].astype(str) == group]
        metrics = visualize_and_quantify_cluster(
            labels,
            intersection_path / f'cluster_classified_{str(group.replace("/","-"))}.png',
            list(preds_filter.image),
            list(group_df.image),
            standard_embedding,
            clustered,
            labels_pred_hdbscan
        )
        save_cluster_metrics(group, '-', '-', metrics, output_path)
        
        # Patient level analysis
        print(f'Quantifying clusters per patient for intersection on {intersection}')
        patient_path = intersection_path / 'per_patient'
        
        for patient in np.unique(group_df.patient):
            patient_df = group_df[group_df.patient == patient]
            patient_dir = patient_path / str(patient)
            patient_dir.mkdir(parents=True, exist_ok=True)
            
            metrics = visualize_and_quantify_cluster(
                labels,
                patient_dir / f'cluster_classified_{patient}.png',
                list(preds_filter.image),
                list(patient_df.image),
                standard_embedding,
                clustered,
                labels_pred_hdbscan
            )
            
            if min(metrics[4], metrics[5]) > 0:  # Check min patches per compartment
                save_cluster_metrics(group, patient, '-', metrics, output_path)
                
                # Airway level analysis
                print(f'Quantifying clusters per airway for patient {patient} and intersection on {intersection}')
                
                for airway in np.unique(patient_df.airway):
                    airway_df = patient_df[patient_df.airway == airway]
                    metrics = visualize_and_quantify_cluster(
                        labels,
                        patient_dir / f'cluster_classified_{patient}_{airway}.png',
                        list(preds_filter.image),
                        list(airway_df.image),
                        standard_embedding,
                        clustered,
                        labels_pred_hdbscan
                    )
                    
                    if min(metrics[4], metrics[5]) > 0:  # Check min patches per compartment
                        save_cluster_metrics(group, patient, str(airway), metrics, output_path)

def save_cluster_metrics(group, patient, airway, metrics, output_path):
    """Save clustering metrics to CSV"""
    values = [
        str(group.replace('/','-')), patient, airway,
        *metrics #,  # Unpack all metrics
        # metrics[12] / metrics[4],  # n_noise_submucosa / n_submucosa
        # metrics[13] / metrics[5]   # n_noise_adventitia / n_adventitia
    ]
    
    with open(output_path / 'clusterability.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(values)

def run_cluster_analysis(preds_df, features, labels, output_path):
    """Main function to run complete cluster analysis"""
    # Setup
    cluster_df = setup_cluster_analysis(output_path)
    
    # Get embeddings from full dataset
    standard_embedding, clustered, labels_pred_hdbscan = analyze_full_dataset(
        preds_df, features, labels, output_path
    )
    
    # Analyze compartments
    analyze_compartments(
        labels, preds_df, standard_embedding, clustered, labels_pred_hdbscan, output_path
    )
    
    # Analyze intersection groups
    analyze_intersection_groups(
        labels, preds_df, standard_embedding, clustered, labels_pred_hdbscan, output_path
    )

    return standard_embedding, clustered, labels_pred_hdbscan


"""
Make UMAP plot with aggregate per patient for inside and outside
"""
def create_patient_embedding_plot(preds_df, standard_embedding, clustered, labels_pred_hdbscan, output_path):
    """
    Create UMAP plot with aggregated embeddings per patient for inside and outside compartments.
    
    Args:
        preds_df (pd.DataFrame): DataFrame containing predictions
        standard_embedding (np.ndarray): UMAP embeddings
        clustered (np.ndarray): Boolean array indicating clustered points
        labels_pred_hdbscan (np.ndarray): HDBSCAN cluster labels
        output_path (Path): Path to save visualization
    """
    # Enrich DataFrame with embedding information
    preds_df_enr = preds_df.copy()
    preds_df_enr['embedding_0'], preds_df_enr['embedding_1'] = standard_embedding.T
    preds_df_enr['embedding_clustered'] = clustered
    preds_df_enr['embedding_labels_pred_hdbscan'] = labels_pred_hdbscan

    # Group by patient and label
    df_patient_grouped = preds_df_enr.groupby(
        ["patient", "label"], 
        as_index=False
    )["embedding_0", "embedding_1"].mean()

    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    cdict = {'Submucosa': 'red', 'Adventitia': 'blue'}

    # Plot points for each compartment
    for compartment in np.unique(df_patient_grouped.label):
        ix = (df_patient_grouped.label == compartment)
        compartment_label = 'Submucosa' if compartment == 'Inside' else 'Adventitia'
        plt.scatter(
            df_patient_grouped.loc[ix, "embedding_0"],
            df_patient_grouped.loc[ix, "embedding_1"],
            c=cdict[compartment_label],
            label=compartment_label,
            s=20
        )

    # Add patient labels
    for ix, txt in enumerate(df_patient_grouped.patient):
        plt.annotate(
            txt,
            (df_patient_grouped.loc[ix, "embedding_0"],
             df_patient_grouped.loc[ix, "embedding_1"]),
            size=8
        )

    plt.legend(loc='upper right', markerscale=1.5)
    fig.savefig(output_path / 'cluster_average_embedding_per_patient.png')
    plt.close(fig)

def get_prototype_filenames(vis_path):
    """Get filenames for true positive prototype patches."""
    file_names = []
    prototype_dir = vis_path / 'overall/tp/'
    
    for img_path in prototype_dir.glob('*.png'):
        if '_occ' not in str(img_path):
            file_name = img_path.name.split('_', 1)[-1]
            file_names.append(file_name)
            
    return file_names

def create_interactive_umap(features, labels, file_names, output_path):
    """Create interactive UMAP visualization for prototype patches."""
    # Get indices and data for selected files
    index_filenames = labels.Image.index[labels.Image.isin(file_names)]
    features_selected = features[index_filenames]
    labels_selected = labels.Label[index_filenames]

    # Create UMAP embedding
    mapping = umap.UMAP(
        n_neighbors=20,
        min_dist=0.001,
        metric='correlation'
    ).fit(features_selected)

    # Create hover data
    hover_data = pd.DataFrame({'index': index_filenames})

    # Create interactive plot
    p = umap.plot.interactive(
        mapping,
        labels=labels_selected,
        hover_data=hover_data,
        point_size=4
    )
    
    return p

def process_cluster_images(indices, data_path, output_path, model, data, labels, cluster_name):
    """Process and save images for a specific cluster."""
    output_dir = output_path / f'clusters_outside/cluster_{cluster_name}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for index in indices:
        # Save original image
        img = data[index] * 255
        file_name = labels.Image[index]
        cv2.imwrite(
            str(output_dir / file_name),
            img.astype("uint8")
        )

        # Generate and save heatmap
        img = read_img(str(data_path / 'Outside' / file_name), size)
        img = np.array(img, np.float32) / 255
        heatmap = grad_cam(img, model)
        cv2.imwrite(
            str(output_dir / f"{file_name.replace('.png','_occ.png')}"),
            heatmap
        )

def prepare_model_for_gradcam(model):
    """Prepare model for gradient computation."""
    layer_idx = -1
    model.layers[layer_idx].activation = keras.activations.linear
    return utils.apply_modifications(model)

def analyze_patient_exclusions(aggs_patient, min_patches_per_compartment):
    """
    Analyze why patients are being excluded from the analysis.
    Also print mean and total number of patches per compartment for included patients.
    """
    total_patients = len(aggs_patient)
    
    # Check each exclusion criterion
    smoking_excluded = sum(aggs_patient.smoking_group != 'ExS >= year')
    patches_excluded = sum(
        (aggs_patient.n_adventitia < min_patches_per_compartment) | 
        (aggs_patient.n_submucosa < min_patches_per_compartment)
    )
    copd_unclear = sum(
        (aggs_patient.gold_stage == 0) & 
        (aggs_patient.copd_group != 'Normal')
    )
    
    print(f"Total patients: {total_patients}")
    print(f"Excluded due to smoking status: {smoking_excluded}")
    print(f"Excluded due to insufficient patches: {patches_excluded}")
    print(f"Excluded due to unclear COPD status: {copd_unclear}")
    
    # Show included patients
    included_mask = (
        (aggs_patient.smoking_group == 'ExS >= year') &
        (aggs_patient.n_adventitia >= min_patches_per_compartment) &
        (aggs_patient.n_submucosa >= min_patches_per_compartment) &
        ((aggs_patient.gold_stage > 0) | (aggs_patient.copd_group == 'Normal'))
    )
    print(f"Included patients: {sum(included_mask)}")
    
    # Calculate patch statistics for included patients
    included_patients = aggs_patient[included_mask]
    
    # Get min/max/mean for both compartments
    min_adventitia = included_patients.n_adventitia.min()
    max_adventitia = included_patients.n_adventitia.max() 
    mean_adventitia = included_patients.n_adventitia.mean()
    
    min_submucosa = included_patients.n_submucosa.min()
    max_submucosa = included_patients.n_submucosa.max()
    mean_submucosa = included_patients.n_submucosa.mean()
    
    total_patches = included_patients.n_submucosa + included_patients.n_adventitia
    min_total = total_patches.min()
    max_total = total_patches.max()
    mean_total = total_patches.mean()
    
    print("\nPatch statistics for included patients:")
    print(f"Adventitia - Min: {min_adventitia}, Max: {max_adventitia}, Mean: {mean_adventitia:.1f}")
    print(f"Submucosa - Min: {min_submucosa}, Max: {max_submucosa}, Mean: {mean_submucosa:.1f}")
    print(f"Total - Min: {min_total}, Max: {max_total}, Mean: {mean_total:.1f}, Total patches: {total_patches.sum()}")
    
    return aggs_patient[included_mask]

def analyze_umap_clusters(model, data, labels, features, output_path, data_path, size):
    """
    Analyze and visualize UMAP clusters.
    
    Args:
        model: Trained model
        data: Image data
        labels: Image labels
        features: Feature vectors
        output_path (Path): Path to save outputs
        data_path (Path): Path to image data
        size: Image size
    """
    # Get prototype filenames
    print('UMAP - per prototype - tp')
    file_names = get_prototype_filenames(output_path)

    # Create interactive UMAP plot
    p = create_interactive_umap(features, labels, file_names, output_path)
    
    # Save interactive plot
    output_notebook()
    show(p)

    # Define cluster indices
    cluster_indices = {
        'bottom_left': [10129, 11892, 5624, 9440, 5522, 11301, 3890, 10394,
                       10919, 10975, 4183, 11863, 12107, 5403, 10755, 8887,
                       7801, 8914, 3863, 4959, 10546, 6191, 8914, 5094, 10065,
                       12646, 11301, 8278],
        'top_left': [10706, 9607, 4179, 4153, 4127, 7560, 6359, 9108, 4414,
                    5012, 8757, 4645, 9829, 7844, 5629, 6217, 10969, 11424,
                    7650, 4197, 4198, 9385, 9437, 4376, 10034, 9330, 11325,
                    10165, 10670, 11142, 8757, 4645, 4210, 9503, 9574, 8123,
                    7844, 9829, 8126, 11139, 5952, 6359]
    }

    # Prepare model for gradient computation
    model = prepare_model_for_gradcam(model)

    # Process images for each cluster
    for cluster_name, indices in cluster_indices.items():
        process_cluster_images(
            np.unique(indices),
            data_path,
            output_path / 'Visualizations',
            model,
            data,
            labels,
            cluster_name
        )
        
def main():

    config = load_config("/workspace/ImageRecognition/4_evaluation/config/config_visualize_compartment_differences.json")

    # Create output path if it doesn't exist
    try:    
        config['output_path'] = config['model_path'] / 'Visualizations'
        os.makedirs(config['output_path'], exist_ok=True)
    except:
        pass

    """### A. If predictions not yet calculated"""

    print('Loading model')
    # Create a model object that can load in existing weights from trained models
    model = torch.load(os.path.join('/workspace/ImageRecognition/3_model/base_model_architectures/', f"{config['tf_model']}_model.pth"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Evaluate best AUC model
    checkpoint = torch.load(config['model_path'] / 'best_auc_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])  # Load state dict into model
    # model.summary()

    # print('Reading data')
    # data, labels, enc_labels = read_data(config['data_path'], config['size'], config['debug_run'])

    print('Calculating predictions')
    preds_df, preds_and_features_df = calculate_pred_and_features(
        model, 
        config['data_path'], 
        config['size'],
        device,
        data_partitions='all'
    )
    
    preds_df.to_csv(config['output_path'] / 'model_predictions.csv')
    preds_and_features_df.to_csv(config['output_path'] / 'model_predictions_with_features.csv')

    print('Generate confusing matrix on full set')
    obs = preds_df.label
    preds = preds_df.pred_label
    plot_cm(obs, preds, config['output_path'] / 'confusion_matrix_auc_model.png')

    # ## B. If predictions and features yet calculated"""

    preds_df = pd.read_csv(config['output_path'] / 'model_predictions.csv', index_col=0)
    output_df = pd.read_csv(config['output_path'] / 'model_predictions_with_features.csv', index_col=0)

    aggs_patient, aggs_wsi, aggs_airway = calculate_aggregates(preds_df, config['output_path'])
    
    # Check who is removed from analysis
    analyze_patient_exclusions(aggs_patient, config['min_patches_per_compartment'])

    # # Run analysis for each intersection type
    intersections = ['smoking_group', 'copd_group', 'copd_group_full']
    for intersection in intersections:
        analyze_intersection(preds_df, intersection, config['output_path'])

    # # Temp
    # #########################################################################################
    # # NAIVE TESTING: TEST FOR SIGNIFICANCE WITHOUT BOOTSTRAPPING

    # # Exclude patients with less than 10 patches for one compartment
    # min_patches_per_compartment = 10
    # #bootstrap_patients_copd = list(aggs_patient.patient[(aggs_patient.copd_group == 'COPD stage III or IV') & (aggs_patient.smoking_group == 'ExS >= year') & (aggs_patient.n_adventitia >= min_patches_per_compartment) & (aggs_patient.n_submucosa >= min_patches_per_compartment)])
    # bootstrap_patients_copd = list(aggs_patient.patient[(aggs_patient.gold_stage > 0) & (aggs_patient.copd_group != 'Normal') & (aggs_patient.smoking_group == 'ExS >= year') & (aggs_patient.n_adventitia >= min_patches_per_compartment) & (aggs_patient.n_submucosa >= min_patches_per_compartment)])
    # bootstrap_patients_normal = list(aggs_patient.patient[(aggs_patient.copd_group == 'Normal') & (aggs_patient.smoking_group == 'ExS >= year') & (aggs_patient.n_adventitia >= min_patches_per_compartment) & (aggs_patient.n_submucosa >= min_patches_per_compartment)])

    # # Check number of patients per group
    # print('Total number of COPD patients: ' + str(aggs_patient[(aggs_patient.copd_group == 'COPD stage III or IV') & (aggs_patient.smoking_group == 'ExS >= year')].shape[0]))
    # print('Total number of healthy patients: ' + str(aggs_patient[(aggs_patient.copd_group == 'Normal') & (aggs_patient.smoking_group == 'ExS >= year')].shape[0]))

    # preds_df_filter = preds_df[preds_df.patient.isin(bootstrap_patients_copd + bootstrap_patients_normal)]

    # # Test per lung compartment
    # for lung_compartment in ['Inside', 'Outside']:
    #     copd_sample = list(preds_df_filter.prob_correct[(preds_df_filter.smoking_group == 'ExS >= year') & (preds_df_filter.copd_group == 'COPD stage III or IV') & (preds_df_filter.label == lung_compartment)])
    #     normal_sample = list(preds_df_filter.prob_correct[(preds_df_filter.smoking_group == 'ExS >= year') & (preds_df_filter.copd_group == 'Normal') & (preds_df_filter.label == lung_compartment)])

    #     # print quantity
    #     print(lung_compartment.upper() + ' - Number of COPD patients: ' + str(len(preds_df_filter.patient[(preds_df_filter.smoking_group == 'ExS >= year') & (preds_df_filter.copd_group == 'COPD stage III or IV') & (preds_df_filter.label == lung_compartment)].unique())) +
    #             ' , number of patches: ' + str(len(copd_sample)))
    #     print(lung_compartment.upper() + ' - Number of normal patients: ' + str(len(preds_df_filter.patient[(preds_df_filter.smoking_group == 'ExS >= year') & (preds_df_filter.copd_group == 'Normal') & (preds_df_filter.label == lung_compartment)].unique())) +
    #             ' , number of patches: ' + str(len(normal_sample)))

    #     # Perform distribution testing
    #     u_stat, u_p = stats.mannwhitneyu(copd_sample, normal_sample)

    #     # Perform median testing
    #     median_copd = statistics.median(copd_sample)
    #     median_var = statistics.median(normal_sample)

    #     mood_stat, mood_p, mood_med, tbl = stats.median_test(copd_sample, normal_sample)

    #     # Perform variance testing
    #     var_copd = statistics.variance(copd_sample)
    #     var_normal = statistics.variance(normal_sample)

    #     # Parametric median Levene test
    #     levene_stat, levene_p = stats.levene(copd_sample, normal_sample, center='median')

    #     print(lung_compartment.upper() + '\n' +
    #         '- Distribution: Statistics=' + str(u_stat) + ', p=' + str(u_p) + '\n' +
    #         '- Median: Statistics=' + str(mood_stat) + ', p=' + str(mood_p) + '\n' +
    #         '- Variance: Statistics=' + str(levene_stat) + ', p=' + str(levene_p))

    #########################################################################################

    # # ## C. Run bootstrap analysis"""
    # # Hypothesis testing - Bootstrap analysis
    # print('Running bootstrap analysis')
    # results_df = run_bootstrap_analysis(preds_df, aggs_patient, config)
    
    # # Analyze bootstrap results
    # analyze_bootstrap_results(results_df, config['output_path'])

    # # Analyze compartment ratios at different levels
    # print('Analyzing compartment ratios at different levels')
    # for level in ['patient', 'tiff', 'airway']:
    #     test_results = analyze_compartment_ratios(
    #         preds_df, 
    #         group_level=level,
    #         min_patches=10,
    #         config=config
    #     )
        
    #     print(f"\nResults for {level}-level analysis:")
    #     print(test_results)

    print('Analyzing patient level differences')
    # Hypothesis testing - Patient level
    grouped_patient_results = analyze_patient_level_differences(
        preds_df,
        aggs_patient,
        config['compartment_size'],
        config['output_path']
    )
    print("\nPatient-level analysis summary:")
    print(grouped_patient_results)

    # Perform PCA analysis
    output_pca_COPD_normal, features_PCA = analyze_pca_features(
        output_df, 
        config['output_path']
    )    

    ## Deepdive on features using SVM and mixed model
    # Perform mixed model analysis
    mixed_model_results = perform_mixed_model_analysis(
        output_pca_COPD_normal, 
        features_PCA
    )
    
    # # Perform SVM analysis
    # svm_clf, svm_predictions = perform_svm_analysis(
    #     features_PCA, 
    #     output_df
    # )
    
    # # Create SVM visualization
    # plot_svm_support_vectors(
    #     features_PCA, 
    #     svm_clf, 
    #     config['output_path']
    # )

    # # Add separate SVM analysis
    # print("\nPerforming separate SVM analyses for each lung function group...")
    # separate_svm_results = perform_separate_svm_analysis(features_PCA, output_df)
    
    # # Create visualizations for separate SVMs
    # plot_separate_svm_results(
    #     features_PCA,
    #     output_df,
    #     separate_svm_results,
    #     config['output_path']
    # )

    # # Get the features from the model predictions
    features = output_df.loc[:, output_df.columns[output_df.columns.str.startswith('f_')]]
    features.index = output_df.image

    features_PCA, pca = create_pca_visualizations(
        features,  # Pass features instead of extracting again
        preds_df,
        labels,
        config['output_path']
    )

    # Run cluster analysis with the extracted features
    standard_embedding, clustered, labels_pred_hdbscan = run_cluster_analysis(
        preds_df,
        features,  # Use the extracted features
        labels,
        config['output_path']
    )

    # # Create patient embedding plot
    # create_patient_embedding_plot(
    #     preds_df,
    #     standard_embedding,
    #     clustered,
    #     labels_pred_hdbscan,
    #     config['output_path']
    # )

    # # Analyze UMAP clusters
    # analyze_umap_clusters(
    #     model,
    #     data,
    #     labels,
    #     features,
    #     config['output_path'],
    #     config['data_path'],
    #     config['size']
    # )
    
    # print('Determining prototypes')
    # tn_certain, tp_certain, fn_certain, fp_certain = determine_prototypes(1000, model_path)

if __name__ == "__main__":
    main()
