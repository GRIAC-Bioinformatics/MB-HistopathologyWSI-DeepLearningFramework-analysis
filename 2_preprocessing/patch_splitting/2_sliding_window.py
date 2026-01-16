"""
Extract image patches from compartment images using sliding window approach.

This script applies a sliding window algorithm to extract fixed-size patches
from compartment-separated images (Inside/Outside). Each patch maintains
spatial coordinates in its filename for traceability back to the original WSI.

Prerequisites:
- Compartment images from 1_create_data_set.py
- RBC masks corresponding to each image

Parameters:
- windowSize: Size of extracted patches in pixels (default: [180, 180] or [120, 120])
- stepSize: Step size for sliding window (default: 180, non-overlapping)

Output:
- Image patches saved with coordinates in filename: {uuid}_{y}_{x}.png
- Corresponding RBC masks for each patch
"""

# Import packages
import os
import numpy as np
from skimage import io
from scipy import ndimage
from PIL import Image
from pathlib import Path
#import matplotlib.pyplot as plt
import imageio
import cv2

def sliding_window(image, mask, stepSize, windowSize, target_dir, target_dir_mask, file_name):
    """
    Extract patches from an image using sliding window approach.
    
    Args:
        image: Input image array (numpy array, RGB)
        mask: Corresponding RBC mask array
        stepSize: Step size for sliding window (pixels)
        windowSize: Size of extracted patches [width, height]
        target_dir: Directory to save image patches
        target_dir_mask: Directory to save mask patches
        file_name: Base filename for patches
    
    Returns:
        Status message string
    """
    coordinates = []
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # Yield the current window
            window = image[y:y + windowSize[1], x:x + windowSize[0], :]
            RBC_mask = mask[y:y + windowSize[1], x:x + windowSize[0], :]
            # If window to small or black pixels found, don't save
            if window.shape[0:2] != (windowSize[0],windowSize[1]): #or len(np.where(window == (0, 0, 0))[0]) > 720:
                continue
            else:
                overlap = False
                for coordinate in coordinates:
                    x_new = coordinate[0]
                    y_new = coordinate[1]
                    if abs(x-x_new) < windowSize[0] and abs(y-y_new) < windowSize[1]:
                        overlap = True
                        break
                if overlap == False:
                    coordinates.append([x,y])
                    window_name = file_name.replace('.png','') + '_' + str(y) + '_' + str(x) + '.png'
                    cv2.imwrite(target_dir + window_name, window)
                    cv2.imwrite(target_dir_mask + window_name, RBC_mask)
                                    
                    
    return 'Done: ' + target_dir + file_name

# Directories with images of in and out
dirs = ['N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Data/Anonymized/Focus/Inside/',
        'N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Data/Anonymized/Focus/Outside/']

windowSize = [180,180]
  
for directory in dirs:
    # List of paths of all images
    pathlist = Path(directory).glob('*.png')

    # Loop over all images 
    for path in pathlist:
        try:
            # Read image and corresponding mask
            print(path)
            file_name = os.path.basename(path)
            img = np.array(cv2.imread(str(path)));
            target_dir = 'N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Data/Anonymized/Patches_' + str(windowSize[0]) + 'x' + str(windowSize[1]) + '/Patches_orig/' + directory.split('/')[-2] + '/';

            mask_rbc = np.array(cv2.imread('N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Data/Anonymized/Focus/RBC_masks/' + file_name))
            target_dir_rbc = 'N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Data/Anonymized/Patches_' + str(windowSize[0]) + 'x' + str(windowSize[1]) + '/Patches_orig/RBC_masks/';
            
            # Check if image is bigger than window size
            if img.shape[0] >= windowSize[0] and img.shape[1]>= windowSize[1]:
                # Different stepsize for inside and outside
                if "Inside" in directory:
                    sliding_window(img, mask_rbc, 180, windowSize, target_dir, target_dir_rbc, file_name)
                else:
                    sliding_window(img, mask_rbc, 180, windowSize, target_dir, target_dir_rbc, file_name)
            else:
                print('Image too small ' + str(target_dir) + '_' + str(file_name))
        except:
            pass
        
