"""
Apply data imputation strategies to image patches.

This script implements the core preprocessing step that removes artifacts and
applies different imputation strategies to masked regions. This is critical for
guiding the model to learn from tissue patterns rather than artifacts.

Key Steps:
1. Tissue threshold filtering: Removes patches with insufficient tissue
2. Artifact masking: Identifies and masks artifacts (RBC, whitespace, purple regions)
3. Imputation: Applies different strategies to fill masked regions:
   - Black: Leaves masked regions as black (baseline)
   - Random Image: Fills with random pixels from same image
   - Random Dataset: Fills with random pixels from dataset color palette
   - No Masking: Keeps original without masking

This creates multiple datasets for comparison, as described in the manuscript
(Figure 4 compares black vs. random imputation).

Prerequisites:
- Raw patches from 2_sliding_window.py
- RBC masks corresponding to patches

Output:
- Multiple imputed datasets: patches_cutoff_{method}_imputed_{threshold}/
- Each dataset contains Inside/ and Outside/ subdirectories
"""

import sys
import os
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import random
import pandas as pd
import logging
from tqdm import tqdm
import shutil
import multiprocessing

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    BASE_DIR, DATA_DIR, 
    get_patches_path, get_imputed_patches_path, 
    setup_environment
)

# Setup environment
setup_environment(verbose=False)

# Define color ranges for artifact exclusion (RGB values)
# These ranges identify non-tissue regions that should be masked
COLOR_RANGES = {
    'whitespace': ((212, 200, 224), (238, 234, 250)),  # Background/whitespace regions
    'purple': ((50, 32, 90), (184, 175, 220)),         # Purple-stained regions
    'black': ((48, 30, 42), (124, 102, 103)),          # Black artifacts
    'blue': ((109, 162, 170), (189, 212, 219))         # Blue-stained regions
}

WINDOW_SIZE = [120, 120]

# Generate paths dynamically using config
def get_dirs():
    """Get input directories for patches using config."""
    patches_base = get_patches_path(tuple(WINDOW_SIZE), 'original')
    return [
        str(patches_base / 'Inside'),
        str(patches_base / 'Outside')
    ]

DIRS = get_dirs()

# Add this parameter to control which processing methods to use
PROCESSING_METHODS = ['no_masking'] #['black', 'random_dataset', 'random_image']
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] #[0.2] # [0.2, 0.3, 0.4, 0.5, 0.6, 0.7] #
SAVE_RAW = False

def create_directories(threshold):
    """Create directories for processed images."""
    patches_base = DATA_DIR / f'patches_{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}'
    main_dir = str(patches_base / 'patches_original')
    raw_dir = str(patches_base / f'patches_cutoff_{threshold}')
    black_dir = str(patches_base / f'patches_cutoff_black_imputed_{threshold}')
    random_dataset_dir = str(patches_base / f'patches_cutoff_random_dataset_imputed_{threshold}')
    random_image_dir = str(patches_base / f'patches_cutoff_random_image_imputed_{threshold}')
    no_masking_dir = str(patches_base / f'patches_cutoff_no_masking_{threshold}')
    
    os.makedirs(raw_dir, exist_ok=True)
    if 'black' in PROCESSING_METHODS:
        os.makedirs(black_dir, exist_ok=True)
    if 'random_dataset' in PROCESSING_METHODS:
        os.makedirs(random_dataset_dir, exist_ok=True)
    if 'random_image' in PROCESSING_METHODS:
        os.makedirs(random_image_dir, exist_ok=True)
    if 'no_masking' in PROCESSING_METHODS:
        os.makedirs(no_masking_dir, exist_ok=True)
    
    return main_dir, raw_dir, black_dir, random_dataset_dir, random_image_dir, no_masking_dir

def create_color_masks(img):
    """
    Create binary masks for different artifact color ranges.
    
    Args:
        img: Input image (RGB, numpy array)
    
    Returns:
        Dictionary of binary masks for each color range
    """
    masks = {}
    for color, (lower, upper) in COLOR_RANGES.items():
        masks[color] = cv2.inRange(img, lower, upper)
    return masks

def process_purple_mask(img, mask_purple):
    """
    Refine purple mask using channel differences and maximum channel analysis.
    
    Purple regions in H&E stains can be ambiguous. This function applies
    additional filtering based on red-blue channel differences and maximum
    channel locations to improve purple mask accuracy.
    
    Args:
        img: Input image (RGB)
        mask_purple: Initial purple mask from color range
    
    Returns:
        Refined purple mask (binary)
    """
    max_channels = np.argmax(img, axis=2)
    max_channels = cv2.merge((max_channels,max_channels,max_channels))
    demask_purple = cv2.inRange(max_channels, (2,2,2), (2,2,2))

    diff_red_blue = img[:,:,2] - img[:,:,0]
    diff_red_blue[diff_red_blue < 12] = 0
    diff_red_blue[diff_red_blue >= 12] = 1
    diff_red_blue = cv2.merge((diff_red_blue,diff_red_blue,diff_red_blue))
    demask_purple = cv2.inRange(diff_red_blue, (1,1,1), (1,1,1))

    return np.minimum.reduce([mask_purple, demask_purple])

def process_rbc_mask(file_name):
    """
    Process RBC (red blood cell) mask by expanding mask regions.
    
    Reads RBC mask file and expands mask regions to cover entire rows/columns
    that contain any RBC pixels. This ensures complete removal of RBC artifacts.
    
    Args:
        file_name: Name of the image file (used to locate corresponding RBC mask)
    
    Returns:
        Processed RBC mask (binary, 255 for RBC regions) or None if not found
    """
    try:
        rbc_mask_path = get_patches_path(tuple(WINDOW_SIZE), 'original') / 'RBC_masks' / file_name
        mask_rbc = cv2.imread(str(rbc_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_rbc is None:
            return None
        
        # Vectorized row processing
        row_mask = (mask_rbc == 255).any(axis=1)[:, np.newaxis]
        col_mask = (mask_rbc == 255).any(axis=0)
        mask_rbc = np.where(row_mask & col_mask, 255, mask_rbc)
        
        return mask_rbc
    except:
        return None

def process_image(path, threshold, mode='random_dataset'):
    """
    Process a single image patch: remove artifacts and apply imputation.
    
    This is the core image processing function that:
    1. Creates masks for all artifact types
    2. Removes artifacts from image
    3. Calculates tissue percentage
    4. Filters by tissue threshold
    5. Applies imputation strategy if threshold is met
    
    Args:
        path: Path to input image patch
        threshold: Minimum tissue percentage required (0.0-1.0)
        mode: Imputation mode (not used here, set in main processing)
    
    Returns:
        List containing [percentage_black, cleaned_image, original_image] if
        threshold is met, None otherwise
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    file_name = os.path.basename(path)

    masks = create_color_masks(img)
    masks['purple'] = process_purple_mask(img, masks['purple'])
    mask_rbc = process_rbc_mask(file_name)

    final_mask = np.maximum.reduce([*masks.values(), mask_rbc] if mask_rbc is not None else masks.values())
    img_clean = cv2.bitwise_and(img, img, mask=~final_mask)

    cnt_not_black = cv2.countNonZero(cv2.cvtColor(img_clean, cv2.COLOR_RGB2GRAY))
    perc_not_black = cnt_not_black/(img_clean.shape[0] * img_clean.shape[1])              

    if perc_not_black >= threshold:
        return [os.path.dirname(path).split('/')[-1], file_name, 1-perc_not_black], img_clean, img

    return None

def replace_black_areas(img, color_sampler, mode='dataset'):
    """
    Replace masked (black) regions with imputed pixels.
    
    Implements different imputation strategies:
    - 'black': No imputation, returns original image
    - 'image': Random pixels from same image (random_image method)
    - 'dataset': Random pixels from dataset color palette (random_dataset method)
    
    Args:
        img: Input image with black masked regions
        color_sampler: Array of colors from dataset (for 'dataset' mode)
        mode: Imputation strategy ('black', 'image', or 'dataset')
    
    Returns:
        Image with imputed regions (same shape as input)
    """
    if mode == 'black':
        return img
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img_bw = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    
    if mode == 'dataset':
        if color_sampler is not None and len(color_sampler) > 0:
            # Randomly select colors from the color_sampler for each pixel
            indices = np.random.randint(0, len(color_sampler), size=img.shape[:2])
            rnd_img = color_sampler[indices]  # Use the color_sampler for random colors
        else:
            rnd_img = np.random.randint(0, 256, size=img.shape, dtype=np.uint8)
    elif mode == 'image':
        non_black_pixels = img[np.any(img != [0, 0, 0], axis=-1)]
        if len(non_black_pixels) > 0:
            # Randomly select indices for each pixel
            indices = np.random.randint(0, len(non_black_pixels), size=img.shape[:2])
            # Use advanced indexing to create the random image
            rnd_img = non_black_pixels[indices]
        else:
            # No adjustment if there are no non-black pixels
            return img
    
    rnd_img = cv2.bitwise_and(rnd_img, rnd_img, mask=~img_bw)
    final_img = img + rnd_img
    
    return final_img.astype(np.uint8)

def process_threshold(threshold):
    logging.info(f"Processing threshold: {threshold}")
    main_dir, raw_dir, black_dir, random_dataset_dir, random_image_dir, no_masking_dir = create_directories(threshold)

    # Remove existing directories only for the methods being used
    logging.info(f"Removing existing directories for threshold {threshold}")
    shutil.rmtree(raw_dir, ignore_errors=True)
    os.makedirs(raw_dir, exist_ok=True)
    
    if 'black' in PROCESSING_METHODS:
        shutil.rmtree(black_dir, ignore_errors=True)
        os.makedirs(black_dir, exist_ok=True)
    if 'random_dataset' in PROCESSING_METHODS:
        shutil.rmtree(random_dataset_dir, ignore_errors=True)
        os.makedirs(random_dataset_dir, exist_ok=True)
    if 'random_image' in PROCESSING_METHODS:
        shutil.rmtree(random_image_dir, ignore_errors=True)
        os.makedirs(random_image_dir, exist_ok=True)
    if 'no_masking' in PROCESSING_METHODS:
        shutil.rmtree(no_masking_dir, ignore_errors=True)
        os.makedirs(no_masking_dir, exist_ok=True)

    total_images = sum(len(list(Path(directory).glob('*.png'))) for directory in DIRS)
    logging.info(f"Total patches to process for threshold {threshold}: {total_images}")

    perc_black_list = []
    colors = set()

    if 'random_dataset' in PROCESSING_METHODS: 
        with tqdm(total=total_images, desc=f"Determining colors and black areas for dataset (threshold={threshold})") as pbar:
            for directory in DIRS:
                logging.info(f"Processing directory: {directory}")
                pathlist = Path(directory).glob('*.png')

                for path in list(pathlist):
                    try:
                        result = process_image(path, threshold) # Only returns when threshold is met 
                        if result:
                            # Get colours from the cleaned images, with masked area and no RBC
                            img_clean = result[1]
                            palette = img_clean.reshape(-1, img_clean.shape[2])
                            palette = np.unique(palette, axis=0)
                            colors.update(tuple(color) for color in palette if list(color) != [0,0,0])
                    except Exception as e:
                        logging.error(f"Error processing {path}: {e}")
                    pbar.update(1)

            colors = list(colors)  # Convert to list once for sampling
            color_sampler = np.array(colors)  # Create a color sampler array

    with tqdm(total=total_images, desc=f"Processing images (threshold={threshold})") as pbar:
        for directory in DIRS:
            logging.info(f"Processing directory: {directory}")
            pathlist = Path(directory).glob('*.png')

            # Get the subfolder name (Inside or Outside)
            subfolder = os.path.basename(os.path.dirname(directory))
            # Create output directories for each processing method
            raw_output_dir = os.path.join(raw_dir, subfolder)
            os.makedirs(raw_output_dir, exist_ok=True)

            # Create relevant subdirectories
            if 'black' in PROCESSING_METHODS:
                black_output_dir = os.path.join(black_dir, subfolder)
                os.makedirs(black_output_dir, exist_ok=True)

            if 'random_dataset' in PROCESSING_METHODS:
                random_dataset_output_dir = os.path.join(random_dataset_dir, subfolder)
                os.makedirs(random_dataset_output_dir, exist_ok=True)

            if 'random_image' in PROCESSING_METHODS:
                random_image_output_dir = os.path.join(random_image_dir, subfolder)
                os.makedirs(random_image_output_dir, exist_ok=True)

            if 'no_masking' in PROCESSING_METHODS:
                no_masking_output_dir = os.path.join(no_masking_dir, subfolder)
                os.makedirs(no_masking_output_dir, exist_ok=True)    
            
            # Loop over images to process them
            for path in list(pathlist):
                try:
                    result = process_image(path, threshold)
                    # Only start processing if the threshold is met
                    if result:
                        perc_black_list.append(result[0])
                        img_clean = result[1]   

                        # Save the cleaned image
                        if SAVE_RAW:
                            output_path = os.path.join(raw_dir, subfolder, os.path.basename(path))
                            cv2.imwrite(output_path, cv2.cvtColor(img_clean, cv2.COLOR_RGB2BGR))
                        
                        # Process imputed versions
                        if 'black' in PROCESSING_METHODS:
                            img_black = replace_black_areas(img_clean, None, mode='black')
                            output_path = os.path.join(black_dir, subfolder, os.path.basename(path))
                            cv2.imwrite(output_path, cv2.cvtColor(img_black, cv2.COLOR_RGB2BGR))
                        
                        if 'random_dataset' in PROCESSING_METHODS:
                            img_dataset = replace_black_areas(img_clean, color_sampler, mode='dataset')  # Pass the sampler
                            output_path = os.path.join(random_dataset_dir, subfolder, os.path.basename(path))
                            cv2.imwrite(output_path, cv2.cvtColor(img_dataset, cv2.COLOR_RGB2BGR))
                        
                        if 'random_image' in PROCESSING_METHODS:
                            img_image = replace_black_areas(img_clean, None, mode='image')
                            output_path = os.path.join(random_image_dir, subfolder, os.path.basename(path))
                            cv2.imwrite(output_path, cv2.cvtColor(img_image, cv2.COLOR_RGB2BGR))
                        
                        # Handle 'no imputation' or no masking setup
                        if 'no_masking' in PROCESSING_METHODS:
                            output_path_no_masking = os.path.join(no_masking_dir, subfolder, os.path.basename(path))
                            img_unmasked = result[2]  
                            cv2.imwrite(output_path_no_masking, cv2.cvtColor(img_unmasked, cv2.COLOR_RGB2BGR))

                except Exception as e:
                    logging.error(f"Error processing {path}: {e}")
                pbar.update(1)

    logging.info("Saving percentage black data")
    perc_black_df = pd.DataFrame(perc_black_list, columns=['compartment', 'image', 'perc_black'])
    perc_black_df.to_csv(os.path.join(main_dir, f'threshold={threshold}-perc_black_per_img.csv'))


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    thresholds = THRESHOLDS

    # Use multiprocessing to process thresholds in parallel
    with multiprocessing.Pool(processes=min(len(thresholds), multiprocessing.cpu_count())) as pool:
        pool.map(process_threshold, thresholds)

if __name__ == "__main__":
    # Check if processing methods are provided as command-line arguments
    if len(sys.argv) > 1:
        PROCESSING_METHODS = sys.argv[1:]
    
    # Validate processing methods
    valid_methods = ['black', 'random_dataset', 'random_image', 'no_masking']
    PROCESSING_METHODS = [method for method in PROCESSING_METHODS if method in valid_methods]
    
    if not PROCESSING_METHODS:
        print("No valid processing methods specified. Please use 'black', 'random_dataset', 'random_image', and/or 'no_masking'.")
        sys.exit(1)
    
    print(f"Processing methods to be used: {PROCESSING_METHODS}")
    main()
