#!/usr/bin/env python3
import json
import os
import subprocess
import sys

def create_submission(existing_id=None, include_files=None):
    """Create Zenodo submission JSON"""
    import glob

    # Find all .zip files in 1_data directory
    zip_files = glob.glob("../*.zip")  # Relative to script location
    all_files = [os.path.basename(f) for f in zip_files]  # Just filenames for upload

    if not all_files:
        print("No .zip files found in 1_data directory!")
        print("Make sure your data files are compressed as .zip archives.")
        sys.exit(1)

    # Filter files if specific files are requested
    if include_files:
        file_list = [f for f in all_files if f in include_files]
        if not file_list:
            print(f"None of the specified files found: {include_files}")
            print(f"Available files: {all_files}")
            sys.exit(1)
    else:
        file_list = all_files

    data = {
        "file_list": file_list,
        "title": "Histopathology Whole Slide Image Dataset: Lung Tissue Analysis",
        "author": "van Breugel, Merlijn",
        "author_affiliation": "University Medical Center Groningen, Department of Pathology and Medical Biology",
        "description_file": "../zenodo_dataset_description.md"
    }
    if existing_id:
        data["existing_zenodo_id"] = existing_id
    with open("zenodo_submission.json", "w") as f:
        json.dump(data, f)

    print(f"Found {len(all_files)} .zip files total")
    print(f"Selected {len(file_list)} .zip files to upload:")
    for f in file_list:
        print(f"  - {f}")

def upload_to_zenodo(sandbox=True):
    """Upload to Zenodo"""
    cmd = [
        "bigzenodo",
        "--submission", "zenodo_submission.json",
        "--accessTokenFile", "secrets/zenodo_token.txt"
    ]
    if sandbox:
        cmd.append("--sandbox")
    else:
        cmd.append("--publish")

    subprocess.run(cmd)

if __name__ == "__main__":
    existing_id = None
    publish = False
    include_files = None

    for arg in sys.argv[1:]:
        if arg == "--publish":
            publish = True
        elif arg.startswith("--existing-id="):
            existing_id = arg.split("=")[1]
        elif arg.startswith("--files="):
            include_files = [f.strip() for f in arg.split("=", 1)[1].split(",")]

    create_submission(existing_id, include_files)
    upload_to_zenodo(sandbox=not publish)