"""
Rename and copy heatmap image files with data type suffixes.

This utility script renames heatmap images by appending suffixes (_b.png for 'black',
_r.png for 'random') and copies them to a combined output directory. Used for
organizing heatmap visualizations from different data processing methods.

Note: This script contains hardcoded paths and was used for one-time data organization.
Modify paths as needed for your environment.
"""

# Import packages
import sys
sys.path.append('C:/users/esmeedejong/appdata/roaming/python/python310/site-packages/')

import os
import numpy as np
from PIL import Image
from pathlib import Path
import random
import pandas as pd

# Configuration: data type determines suffix ('black' -> '_b.png', 'random' -> '_r.png')
data_type = 'random'

images_path = 'N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/3. Results/Threshold_0.2/Heatmaps_' + data_type + '_v2/heatmaps' 
#images_path = 'N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/1. Data/2a. Anonymized/Patches_120x120/Patches_cutoff_0.2'

dirs = [#images_path + '/Outside',
        images_path + '/Inside']



for directory in dirs:

    if 'Outside' in directory:
        output_path = 'N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/3. Results/Threshold_0.2/Heatmaps_combined/Outside/'

    if 'Inside' in directory:
        output_path = 'N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/3. Results/Threshold_0.2/Heatmaps_combined/Inside/'

    try:
        os.mkdir(output_path)
    except:
        pass
    
    pathlist = Path(directory).glob('*.png')
    for path in pathlist:
        
        try:
            print(str(path))
            img = Image.open(str(path))
            file_name = os.path.basename(path)
            
            file_name_adj = file_name
            if data_type == 'black':
                file_name_adj = file_name.replace('.png', '_b.png')#.split('_',1)[1]
            if data_type == 'random':
                file_name_adj = file_name.replace('.png', '_r.png')#.split('_',1)[1]
    
            img.save(output_path + file_name_adj)
        
        except:
            pass

        
        
        
