#!/bin/bash
# Upload files to an existing Zenodo draft
# Usage: ./zenodo_upload.sh DEPOSIT_ID [file1.zip file2.zip ...]
# Example: ./zenodo_upload.sh 18267951 wsi.zip

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")"
TOKEN_FILE="$(dirname "$DATA_DIR")/secrets/zenodo_token.txt"
ZENODO_API="https://zenodo.org/api"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: ./zenodo_upload.sh DEPOSIT_ID [file1.zip file2.zip ...]"
    echo "Example: ./zenodo_upload.sh 18267951 wsi.zip"
    exit 1
fi

DEPOSIT_ID="$1"
shift

# Read token
if [ ! -f "$TOKEN_FILE" ]; then
    echo "Error: Token file not found: $TOKEN_FILE"
    exit 1
fi
TOKEN=$(cat "$TOKEN_FILE")

# Get bucket URL
echo "Getting bucket URL for deposit $DEPOSIT_ID..."
BUCKET_URL=$(curl -s -H "Authorization: Bearer $TOKEN" \
    "$ZENODO_API/deposit/depositions/$DEPOSIT_ID" | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['links']['bucket'])")

if [ -z "$BUCKET_URL" ]; then
    echo "Error: Could not get bucket URL. Check your deposit ID and token."
    exit 1
fi

echo "Bucket URL: $BUCKET_URL"
echo ""

# Get files to upload
if [ $# -eq 0 ]; then
    # No files specified, find all .zip files
    FILES=$(find "$DATA_DIR" -maxdepth 1 -name "*.zip" -type f)
else
    # Use specified files
    FILES=""
    for f in "$@"; do
        FILES="$FILES $DATA_DIR/$f"
    done
fi

# Upload each file
for filepath in $FILES; do
    if [ ! -f "$filepath" ]; then
        echo "File not found: $filepath"
        continue
    fi
    
    filename=$(basename "$filepath")
    filesize=$(ls -lh "$filepath" | awk '{print $5}')
    
    echo "Uploading $filename ($filesize)..."
    
    # Upload with progress - curl's native progress to stderr
    curl \
        -H "Authorization: Bearer $TOKEN" \
        -T "$filepath" \
        "$BUCKET_URL/$filename"
    
    echo "✓ $filename uploaded"
    echo ""
done

echo "Done! View your draft: https://zenodo.org/uploads/$DEPOSIT_ID"
