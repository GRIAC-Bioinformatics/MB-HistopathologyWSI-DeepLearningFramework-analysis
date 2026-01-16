#!/bin/bash
#
# Setup script for ImageRecognition project
# 
# Usage:
#   ./setup.sh           # Full setup
#   ./setup.sh --check   # Only check environment
#   ./setup.sh --help    # Show help
#
# This script:
# 1. Creates a virtual environment
# 2. Installs dependencies
# 3. Sets up environment variables
# 4. Validates the setup
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# Default values
VENV_NAME="venv"
PYTHON_CMD="python3"
CHECK_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --venv)
            VENV_NAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --check         Only check environment, don't install"
            echo "  --venv NAME     Virtual environment name (default: venv)"
            echo "  --python CMD    Python command to use (default: python3)"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "ImageRecognition Setup Script"
echo "========================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
echo "Checking Python installation..."
if command_exists "$PYTHON_CMD"; then
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
    echo -e "${GREEN}✅ $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check pip
echo "Checking pip installation..."
if command_exists pip3 || command_exists pip; then
    PIP_CMD=$(command_exists pip3 && echo "pip3" || echo "pip")
    echo -e "${GREEN}✅ pip is available${NC}"
else
    echo -e "${RED}❌ pip not found. Please install pip${NC}"
    exit 1
fi

# Check for CUDA (optional)
echo "Checking CUDA installation..."
if command_exists nvidia-smi; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo -e "${GREEN}✅ CUDA available: $GPU_INFO${NC}"
else
    echo -e "${YELLOW}⚠️  CUDA not detected. Training will be slow on CPU.${NC}"
fi

if [ "$CHECK_ONLY" = true ]; then
    echo ""
    echo "Check-only mode. Skipping installation."
    echo ""
    # Run Python setup script in check mode
    cd "$PROJECT_ROOT"
    $PYTHON_CMD setup.py --check
    exit 0
fi

echo ""
echo "----------------------------------------"
echo "Setting up virtual environment..."
echo "----------------------------------------"

cd "$PROJECT_ROOT"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment: $VENV_NAME"
    $PYTHON_CMD -m venv "$VENV_NAME"
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists: $VENV_NAME${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "----------------------------------------"
echo "Upgrading pip..."
echo "----------------------------------------"
pip install --upgrade pip
echo -e "${GREEN}✅ pip upgraded${NC}"

# Install dependencies
echo ""
echo "----------------------------------------"
echo "Installing dependencies..."
echo "----------------------------------------"
pip install -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Create necessary directories
echo ""
echo "----------------------------------------"
echo "Creating directories..."
echo "----------------------------------------"

mkdir -p "$PROJECT_ROOT/5_results/completed_grid_runs"
mkdir -p "$PROJECT_ROOT/5_results/learning_evaluation"
mkdir -p "$PROJECT_ROOT/3_model/base_model_architectures"
mkdir -p "$PROJECT_ROOT/config"
echo -e "${GREEN}✅ Directories created${NC}"

# Set up environment variable
echo ""
echo "========================================"
echo "ENVIRONMENT VARIABLE SETUP"
echo "========================================"
echo ""
echo "Add this line to your shell configuration (~/.bashrc, ~/.zshrc):"
echo ""
echo "  export IMAGE_RECOGNITION_BASE_DIR=\"$PROJECT_ROOT\""
echo ""

# Create activation script with environment variable
ACTIVATE_SCRIPT="$PROJECT_ROOT/$VENV_NAME/bin/activate"
if ! grep -q "IMAGE_RECOGNITION_BASE_DIR" "$ACTIVATE_SCRIPT"; then
    echo "" >> "$ACTIVATE_SCRIPT"
    echo "# ImageRecognition project configuration" >> "$ACTIVATE_SCRIPT"
    echo "export IMAGE_RECOGNITION_BASE_DIR=\"$PROJECT_ROOT\"" >> "$ACTIVATE_SCRIPT"
    echo -e "${GREEN}✅ Environment variable added to virtual environment activation script${NC}"
fi

# Run Python setup script
echo ""
echo "----------------------------------------"
echo "Validating setup..."
echo "----------------------------------------"
python setup.py --check

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To start using the project:"
echo "  1. Download pre-trained model weights (see README.md)"
echo "  2. Prepare your data (see PIPELINE_GUIDE.md)"
echo "  3. Run training: python 3_model/train_nn_inner_outer_gpu_pytorch.py"
echo ""
