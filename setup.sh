#!/bin/bash
echo "Setting up environment..."

# Upgrade pip to its latest version
pip install --upgrade pip

# Install required Python packages
pip install split-folders tensorflow matplotlib streamlit

echo "Setup completed."
