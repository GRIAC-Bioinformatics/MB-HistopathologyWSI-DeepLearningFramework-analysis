"""
Check quantity of patches per intersection (compartment).

This script counts and compares the number of patches in raw vs. cleaned datasets
for each lung compartment (Inside/Outside) and per patient. Used for quality control
to verify data processing steps.

Note: Contains hardcoded paths that should be configured for your environment.
"""

# Import packages
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import pandasql as ps
import glob

# Updated directory paths
WINDOW_SIZE = (224, 224)  # Assuming this is the correct window size
DIRS = [
    f'/workspace/ImageRecognition/1_data/patches_{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}/patches_original/Inside/',
    f'/workspace/ImageRecognition/1_data/patches_{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}/patches_original/Outside/'
]

# Patient mapping
mapping = pd.read_csv(r'N:\Werkstudenten\1. Eigen mappen\Esmee\UMCG\Data\mapping.csv',sep=';')

# Number of raw patches per compartment
patches_inside_raw = glob.glob(DIRS[0] + '*.png')
patches_inside_raw = pd.DataFrame([os.path.basename(file).split('_')[0] for file in patches_inside_raw], columns=['tiff'])
patches_inside_raw_agg = patches_inside_raw.tiff.value_counts().reset_index()
patches_inside_raw_agg['lung_compartment'] = 'Inside'


patches_outside_raw = glob.glob(DIRS[1] + '*.png')
patches_outside_raw = pd.DataFrame([os.path.basename(file).split('_')[0] for file in patches_outside_raw], columns=['tiff'])
patches_outside_raw_agg = patches_outside_raw.tiff.value_counts().reset_index()
patches_outside_raw_agg['lung_compartment'] = 'Outside'

combined_df_raw = patches_inside_raw_agg.append(patches_outside_raw_agg, ignore_index=True)
combined_df_raw.columns = ['tiff', 'amount', 'lung_compartment']

# Number of cleaned patches per compartment
patches_inside_clean = glob.glob(DIRS[0].replace('Patches','Patches_clean_cutoff60') + '*.png')
patches_inside_clean = pd.DataFrame([os.path.basename(file).split('_')[0] for file in patches_inside_clean], columns=['tiff'])
patches_inside_clean_agg = patches_inside_clean.tiff.value_counts().reset_index()
patches_inside_clean_agg['lung_compartment'] = 'Inside'


patches_outside_clean = glob.glob(DIRS[1].replace('Patches','Patches_clean_cutoff60') + '*.png')
patches_outside_clean = pd.DataFrame([os.path.basename(file).split('_')[0] for file in patches_outside_clean], columns=['tiff'])
patches_outside_clean_agg = patches_outside_clean.tiff.value_counts().reset_index()
patches_outside_clean_agg['lung_compartment'] = 'Outside'

combined_df_clean = patches_inside_clean_agg.append(patches_outside_clean_agg, ignore_index=True)
combined_df_clean.columns = ['tiff', 'amount', 'lung_compartment']

# Combine raw and clean amounts
combined_df = combined_df_raw.merge(combined_df_clean, how='left', on=['tiff','lung_compartment'])


# Combine patient mapping
combined_df = combined_df.merge(mapping, how='left', on='tiff')
combined_df.columns = ['tiff', 'amount_raw', 'lung_compartment', 'amount_clean', 'patient',
       'copd_group', 'smoking_group', 'gold_stage', 'borderline']


# Aggregate
agg_query = (" SELECT copd_group, "
            "smoking_group, "
            "lung_compartment, "
            "COUNT(DISTINCT patient)                                AS n_patients,  "
            "COUNT(DISTINCT tiff)                                   AS n_tiffs,  "
            "SUM(amount_raw)                                        AS n_patches_raw, "
            "SUM(amount_clean)                                      AS n_patches_clean "
      "FROM combined_df "
      "GROUP BY copd_group, smoking_group, lung_compartment")

data_quantity_aggregations = ps.sqldf(agg_query)

# Store the results in the 5_results folder
data_quantity_aggregations.to_csv('/workspace/ImageRecognition/5_results/data_quantity_aggregations.csv')

