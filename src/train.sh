#!/usr/bin/env bash
set -euo pipefail

# 1) Create env if missing
ENV_NAME="mysegment"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda env '$ENV_NAME'..."
  conda env create -f environment.yml
else
  echo "Conda env '$ENV_NAME' already exists"
fi

# 2) Activate
echo "Activating '$ENV_NAME'..."
# ensure 'conda' is in PATH for non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 3) (Optional) upgrade pip & install any pip‐only packages
pip install --upgrade pip
pip install albumentations opencv-python

# 4) Run training
echo "Starting training..."
cd src
python train.py