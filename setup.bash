#!/usr/bin/env bash
set -e
ENV_NAME="R5_analysis_env"
PYTHON_VERSION="3.11"
echo "=== Starting Setup for $ENV_NAME ==="
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in your PATH."
    echo "Please install Miniconda or Anaconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
eval "$(conda shell.bash hook)"
if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation..."
else
    echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION and tk..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION" tk
fi
echo "Activating '$ENV_NAME'..."
conda activate "$ENV_NAME"
echo "Upgrading pip..."
python -m pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    python -m pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found in the current directory."
    echo "Installing core dependencies directly based on project imports..."
    python -m pip install numpy tifffile scikit-learn pandas sv-ttk matplotlib matplotlib-scalebar scipy scikit-image sdtfile
fi
echo ""
echo "=== Setup Complete! ==="
echo "To activate the environment and launch the GUI, run:"
echo "  conda activate $ENV_NAME"
echo "  python GUI2.py"
