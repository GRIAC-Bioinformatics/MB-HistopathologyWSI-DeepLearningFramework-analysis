"""
Aggregate patch counts and statistics at patient level.

This utility script aggregates patch-level data to patient level, calculating
statistics such as number of patches per patient, percentage of Inside vs.
Outside patches, and aggregations by patient groups (COPD status, smoking status).

Output is used for dataset characterization and quality control.

Prerequisites:
- Processed patches from preprocessing pipeline
- Patient metadata mapping file (mapping.csv)

Note: This is a utility script for data characterization, not required for model training.
"""

import os
import pandas as pd
import pandasql as ps
from pathlib import Path

# Note: Paths are hardcoded and should be updated for your environment
data_folder = 'N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/1. Data'
imgs_path = data_folder + '/2a. Anonymized/Patches_120x120/Patches_cutoff_clean_0.2'

mapping = pd.read_csv(data_folder + '/mapping.csv', sep=';')

imgs_dirs = [imgs_path + '/Inside',
             imgs_path + '/Outside']


imgs_per_tiff = []
for dir in imgs_dirs:
     pathlist = Path(dir).glob('*.png')
     for path in pathlist:
         print(path)
         
         filename = os.path.basename(path)
         tiff = filename.split('_')[0]
         compartment = dir.split('/')[-1]
         
         combination = [compartment, tiff]
         
         imgs_per_tiff.append(combination)
         
         
imgs_per_tiff_df = pd.DataFrame(imgs_per_tiff, columns = ['compartment', 'tiff'])
agg_query = ("SELECT tiff, "
             "       SUM(CASE WHEN compartment == 'Inside' THEN 1 END) AS n_inside, "
             "       SUM(CASE WHEN compartment == 'Outside' THEN 1 END) AS n_outside "
             "FROM imgs_per_tiff_df "
             "GROUP BY tiff "
             )
agg_imgs_per_tiff_df = ps.sqldf(agg_query)

full_aggs_per_tiff_df = mapping.merge(agg_imgs_per_tiff_df, on='tiff', how='left')

agg_query_patient = ("SELECT copd_group                 AS patient_group_lung, "
                      "      'total'                    AS group_type_smoking, "
                      "       COUNT(DISTINCT patient)    AS n_patients, "
                      "       COUNT(DISTINCT tiff)       AS n_tiffs, "
                      "       SUM(n_inside + n_outside)  AS n_patches, "
                      "       SUM(n_inside) / SUM(n_inside + n_outside) AS perc_inside "
                      "FROM full_aggs_per_tiff_df "
                      "GROUP BY copd_group "
                       "UNION ALL "
                       "SELECT 'total'                    AS patient_group_lung, "
                       "       smoking_group              AS patient_group_smoking, "
                       "       COUNT(DISTINCT patient)    AS n_patients, "
                       "       COUNT(DISTINCT tiff)       AS n_tiffs, "
                       "       SUM(n_inside + n_outside)  AS n_patches, "
                       "       SUM(n_inside) / SUM(n_inside + n_outside) AS perc_inside "
                       "FROM full_aggs_per_tiff_df "
                       "GROUP BY smoking_group "
                       "UNION ALL "
                       "SELECT copd_group                 AS patient_group_lung, "
                       "       smoking_group              AS patient_group_smoking, "
                       "       COUNT(DISTINCT patient)    AS n_patients, "
                       "       COUNT(DISTINCT tiff)       AS n_tiffs, "
                       "       SUM(n_inside + n_outside)  AS n_patches, "
                       "       SUM(n_inside) / SUM(n_inside + n_outside) AS perc_inside "
                       "FROM full_aggs_per_tiff_df "
                       "GROUP BY copd_group, smoking_group "
                      )


aggs_per_patient = ps.sqldf(agg_query_patient)
aggs_per_patient.to_csv(data_folder + '/aggs_per_patient_threshold_0.2_clean.csv')

