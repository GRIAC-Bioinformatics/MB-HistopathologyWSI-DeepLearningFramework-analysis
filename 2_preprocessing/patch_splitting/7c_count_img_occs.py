"""
Count image occurrences and filter images with complete heatmap sets.

This utility script:
1. Counts occurrences of image files (removing occurrence suffixes like '_occ_b.png', '_occ_r.png')
2. Identifies images that appear exactly 3 times (indicating complete heatmap sets)
3. Copies selected images with complete heatmap sets to a separate directory

Used for quality control to ensure all required heatmap variants are present for each image.

Note: This script contains hardcoded paths and was used for one-time data filtering.
Modify paths as needed for your environment.
"""

import os
import pandas as pd
import pandasql as ps
from pathlib import Path
from PIL import Image

images_path = 'N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/3. Results/Threshold_0.2/Heatmaps_combined/'

dirs = [images_path + '/Outside',
        images_path + '/Inside']

all_imgs = []

for dir in dirs:
    pathlist = Path(dir).glob('*.png')
    for path in pathlist:
        print(path)
        filename = os.path.basename(path)
        filename_org = filename.replace('_occ_b.png','.png').replace('_occ_r.png','.png')
        path_org = str(path).replace('_occ_b.png','.png').replace('_occ_r.png','.png')
        
        all_imgs.append(path_org)
        
all_imgs_df = pd.DataFrame(all_imgs, columns=['path'])
all_imgs_df['n'] = 1

group_by_query = ("SELECT path    AS path, "
                          "SUM(n) AS count "
                  "FROM all_imgs_df "
                  "GROUP BY path ")
grouped_imgs = ps.sqldf(group_by_query)
grouped_imgs_3_occs = grouped_imgs[grouped_imgs['count'] == 3]

grouped_imgs_3_occs.to_csv(images_path + '/imgs_with_3_occs.csv')

# Save all imgs with both heatmaps complete in separate directory
for path in grouped_imgs_3_occs['path']:
    print(path)
    img = Image.open(str(path))
    img.save(str(path).replace('_combined', '_combined_selection'))
    
    img = Image.open(str(path).replace('.png', '_occ_b.png'))
    img.save(str(path).replace('.png', '_occ_b.png').replace('_combined', '_combined_selection'))
    
    img = Image.open(str(path).replace('.png', '_occ_r.png'))
    img.save(str(path).replace('.png', '_occ_r.png').replace('_combined', '_combined_selection'))

 
        
        