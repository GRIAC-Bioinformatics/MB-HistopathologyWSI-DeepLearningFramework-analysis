"""
Extract heatmap files from a zip archive.

This utility script extracts heatmap image files from a zip archive to a specified
destination directory. Used for unpacking heatmap visualization results.

Note: This script contains hardcoded paths and was used for one-time data extraction.
Modify paths as needed for your environment.
"""

# Import zipfile module
from zipfile import ZipFile

# Extract all files from the zip archive to the destination path
with ZipFile("N:\\Werkstudenten\\1. Eigen mappen\\Esmee\\UMCG\\3. Results\\Threshold_0.2\\Heatmaps_random.zip", 'r') as zObject:
    zObject.extractall(
        path="N:\\Werkstudenten\\1. Eigen mappen\\Esmee\\UMCG\\3. Results\\Threshold_0.2\\Heatmaps_random_v2")