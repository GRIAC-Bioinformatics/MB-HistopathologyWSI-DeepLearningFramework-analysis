"""
Map model prediction accuracy back to whole slide images (WSI).

This visualization script creates color-coded overlays on WSI images showing
model performance per airway. Airways are colored according to accuracy
scores, with annotations showing accuracy, patch count, and submucosa
percentage.

Prerequisites:
- WSI downscaled images
- Airway mask files
- Accuracy results CSV file (generated from model predictions)

Note: This is a visualization/utility script, not part of the core modeling pipeline.
"""

import os
import numpy as np
import pandas as pd
from skimage import io
from scipy import ndimage
from PIL import Image
from pathlib import Path
import uuid
import cv2
import matplotlib

# Accuracy per airway CSV
# Note: Path is hardcoded and should be updated for your environment
acc_per_airway = pd.read_csv('N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Results/DUMMY_results_per_airway.csv', sep=';')

# Define color scale
cmap = matplotlib.cm.get_cmap('RdYlGn')

# List of all downscaled WSIs
tiff_pathlist = Path('N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Qupath_annotation_def/WSI_and_masks_area/').glob('*WSI_downscaled.png')


for path in tiff_pathlist:
    tiff_num = np.int64(os.path.basename(path).split('.')[0])
    tiff = cv2.imread(str(path))
    tiff = np.asarray(tiff, np.float64)
    
    airway_pathlist = Path('N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Qupath_annotation_def/WSI_and_masks_area/').glob(str(tiff_num) + '.*mask.png')

    print('Coloring airways according to accuracy for WSI ' + str(tiff_num))
    added_image = tiff
    airway_count = 1
    for airway_path in airway_pathlist:
        try:
            # Read mask and calculate reverse mask
            airway_name = os.path.basename(airway_path).split('__')[1]
            airway_mask = cv2.imread(str(airway_path))
            
            # Normalize mask to keep intensity between 0 and 1
            airway_mask = np.asarray(airway_mask, np.float64)

            # Get aggregates for airway
            acc_airway = acc_per_airway.accuracy[(acc_per_airway.tiff == tiff_num) & (acc_per_airway.airway == airway_name)]
            n_patches = acc_per_airway.n_patches[(acc_per_airway.tiff == tiff_num) & (acc_per_airway.airway == airway_name)]
            perc_submucosa = acc_per_airway.perc_submucosa[(acc_per_airway.tiff == tiff_num) & (acc_per_airway.airway == airway_name)]
            
            # Extract color based on airway accuracy
            color = cmap(acc_airway)[0] * 255
            color = [color[2], color[1], color[0]] # CV2 works with BGR instead of RGB codes

            # Overlay reversed mask on original tiff and replace black by extracted color
            print('WSI ' + str(tiff_num) + ' - coloring for airway ' + str(airway_count))
            overlay = added_image + airway_mask
            overlay[np.where(overlay[:,:,0] > 255)] = color
            overlay = np.asarray(overlay, np.float64)

            # Get first white
            first_white_row = min(np.nonzero(airway_mask)[0]) 
            first_white_col = min(np.nonzero(airway_mask)[1]) 

            added_image = cv2.addWeighted(overlay,0.4, added_image,0.6,0)

            cv2.putText(added_image, "Accuracy: " + str(int(acc_airway * 100)) + '%',
                (first_white_col, first_white_row + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
            
            cv2.putText(added_image, "# patches: " + str(int(n_patches)),
                (first_white_col, first_white_row + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(added_image, "% submucosa: " + str(int(perc_submucosa * 100)) + "%",
                (first_white_col, first_white_row + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

        except:
            print('WSI - ' + str(tiff_num) + ' No performance result found for airway ' + str(airway_count))

        airway_count += 1

    cv2.imwrite('N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Results/Map_to_WSI/Tiff_' + str(tiff_num) + '.png', added_image)


        

        

        
    

