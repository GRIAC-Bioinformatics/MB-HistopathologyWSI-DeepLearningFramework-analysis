"""
Delete redundant model checkpoint files from WandB run directories.

This utility script removes specified model checkpoint files from WandB run directories
to save disk space. It searches through all subdirectories and removes model files
from the 'wandb/latest-run/files' directory.

Note: This script contains hardcoded paths. Modify 'base_path' and 'model_files_to_delete'
as needed for your environment.
"""

import shutil
import os

# Base path to search for WandB runs
base_path = "/workspace/ImageRecognition/5_results/random_image/threshold_0.2"

# List of model checkpoint files to delete from WandB directories
model_files_to_delete = ['best_auc_model.pth', 'best_loss_model.pth', 'final_model.pth']

# Walk through all subdirectories
for root, dirs, files in os.walk(base_path):
    # Check if 'wandb' directory exists in current directory
    if 'wandb' in dirs:
        latest_run_path = os.path.join(root, 'wandb', 'latest-run', 'files')
        if os.path.exists(latest_run_path):
            # Iterate through the files to delete
            for model_file in model_files_to_delete:
                file_path = os.path.join(latest_run_path, model_file)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")