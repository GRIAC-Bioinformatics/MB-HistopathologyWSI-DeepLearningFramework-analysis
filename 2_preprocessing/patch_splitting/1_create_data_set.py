"""
Create dataset from whole slide images (WSI) and QuPath annotations.

This script processes WSI images and annotation masks to identify and separate
tissue compartments (Inside/Outside airways). It performs anonymization by
replacing patient identifiers with UUIDs.

Prerequisites:
- WSI images and annotation masks must be prepared manually in QuPath
- Annotation masks required: airway masks, smooth muscle masks, RBC masks
- Patient metadata CSV file with patient characteristics

Note: QuPath annotation steps are performed manually and are not part of this
automated pipeline. This script processes the output from QuPath annotations.

Output:
- Compartment-separated images (Inside/Outside) with UUID-based anonymization
- RBC masks for each image
- Mapping file linking anonymized identifiers to patient metadata
"""

# Import packages
import os
import numpy as np
import pandas as pd
from skimage import io
from scipy import ndimage
from PIL import Image
from pathlib import Path
import uuid
import cv2
import random

# Read patient chars
# Note: This path is hardcoded and should be updated for your environment
patient_df = pd.read_csv('N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Data/Patient_chars.csv', sep=';')

# Random map for patients
random_p_number = random.sample(range(1, 1000), 100)
patient_df['p_number'] = 0
for i, p in enumerate(np.unique(patient_df['T_nr'])):
    random_num = random_p_number[i]
    patient_df.p_number[patient_df['T_nr'] == p] = random_num


# Random map of tiff num to unique identifier
keys = [str(uuid.uuid4().hex) for i in range(patient_df.shape[0])]
keys_df = pd.DataFrame({'key': keys})

mapping_df = keys_df.join(patient_df[['p_number','copd_group', 'smoking_group', 'gold_stage', 'borderline']])
mapping_df = mapping_df.sort_values(by=['key'], ascending=False)
mapping_df.to_csv('N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Data/mapping_df.csv', index=False)


# List of paths of all images
pathlist_focus = Path('N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Qupath_annotation_def/masks/').glob('**/*).png')
pathlist_wsi = Path('N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Qupath_annotation_def/WSI_and_masks_area/').glob('**/*).png')


# Random map of area num to unique identifier
areas_df = pd.DataFrame(columns={'area','key'})


# Loop over all focus regions 
for path in pathlist_focus:
    # Read image and corresponding masks
    print(str(path));
    img = np.array(Image.open(str(path)));
    index_area_name = str(path).find('_Area')
    mask_aw = np.array(Image.open(str(path).replace(str(path)[index_area_name:],'-mask_airway.png')).resize(img.shape[1::-1], Image.BILINEAR));
    mask_sm = np.array(Image.open(str(path).replace(str(path)[index_area_name:],'-mask_sm.png')).resize(img.shape[1::-1], Image.BILINEAR));
    mask_rbc = np.array(Image.open(str(path).replace(str(path)[index_area_name:],'-mask_rbc.png')).resize(img.shape[1::-1], Image.BILINEAR)); 

    # Number of pixels
    nrows, ncols = img.shape[0:2]
    
    # Only select non-rotated rectangles
    # Check order of occurence airway and smooth muscle to determine location of airspace
    row_sm = min(np.nonzero(mask_sm)[0])
    col_sm = min(np.nonzero(mask_sm)[1])
    row_aw = min(np.nonzero(mask_aw)[0])
    col_aw = min(np.nonzero(mask_aw)[1])
        
    if row_sm > row_aw and col_sm == col_aw:
        ind_outside_bottom_right = 1
        for i in range(ncols):
            last_white = np.where(mask_aw[:,i] == 255)[0]
            if last_white.size: mask_aw[0:last_white[-1],i] = 255
    elif row_sm < row_aw and col_sm == col_aw:
        ind_outside_bottom_right = 0
        for i in range(ncols):
            first_white = np.where(mask_aw[:,i] == 255)[0]
            if first_white.size: mask_aw[first_white[0]:nrows,i] = 255
    elif row_sm == row_aw and col_sm > col_aw:
        ind_outside_bottom_right = 1
        for i in range(nrows):
            last_white = np.where(mask_aw[i,:] == 255)[0]
            if last_white.size: mask_aw[i,0:last_white[-1]] = 255
    else:
        ind_outside_bottom_right = 0
        for i in range(nrows):
            first_white = np.where(mask_aw[i,:] == 255)[0]
            if first_white.size: mask_aw[i,first_white[0]:ncols] = 255
    
    
    # Combine and reverse mask such that only inside and outside are selected
    mask_rev = ~mask_aw * ~mask_sm
    mask_rev = mask_rev.reshape(*mask_rev.shape, 1)
           
    # Connected components
    label_im, nb_labels = ndimage.label(mask_rev)

    # Save with uuid to completely anonymize images
    tiff_num = int(float(os.path.basename(path).split('.')[0]))
    file_name = keys[tiff_num - 1] + '_' + str(path)[index_area_name:].replace('.png','') + '_' + str(uuid.uuid4().hex)

    area_name = (os.path.basename(path).split('Area_')[1]).split(')')[0] + ')'
    if area_name in areas_df.area:
        area_key = areas_df.key[areas_df.area == area_name]
    else:
        area_key = '(' + str(uuid.uuid4().hex) + ')'
        new_row = {'area': area_name, 'key': area_key}
        areas_df.append(new_row, ignore_index=True)

    file_name = file_name.replace(area_name, area_key)

    #file_name = os.path.basename(path)
    
    # Extract images of only inside or outside
    for i in range(nb_labels):
            
        mask_compare = np.full(np.shape(label_im), i+1) 
            
        # check equality test and have the value 1 on the location of each mask
        separate_mask = np.equal(label_im, mask_compare).astype(int) 
            
        # replace 1 with 255 for visualization as rgb image
        separate_mask[separate_mask == 1] = 255 
        separate_mask = separate_mask / 255 * img
            
        # save masked image
        if nb_labels > 1 and ((ind_outside_bottom_right == 1 and i==1) or (ind_outside_bottom_right == 0 and i==0)):
            target_dir = 'N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Data/Anonymized/Focus/Outside/'
        else:
            target_dir = 'N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Data/Anonymized/Focus/Inside/'
        cv2.imwrite(target_dir + file_name + '.png', cv2.cvtColor(separate_mask.astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Save mask of rbc
    cv2.imwrite('N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Data/Anonymized/Focus/RBC_masks/' + file_name + '.png', mask_rbc)

    


