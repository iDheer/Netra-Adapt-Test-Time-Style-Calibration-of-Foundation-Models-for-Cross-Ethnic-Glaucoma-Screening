#!/bin/bash

# Setup script for RAG-based glaucoma screening on Vast.ai (datasets already uploaded)
# This script assumes datasets are already extracted in /workspace/data/
# Run this after uploading code to Vast.ai instance with pre-downloaded datasets

set -e  # Exit on error

echo "============================================"
echo "RAG Glaucoma Screening - Setup (From Upload)"
echo "============================================"
echo ""

# Check if running on Vast.ai
if [ ! -d "/workspace" ]; then
    echo "ERROR: /workspace directory not found!"
    echo "This script is designed for Vast.ai instances."
    echo "For local setup, use setup_with_download.sh"
    exit 1
fi

cd /workspace

# Verify datasets exist
echo "Checking for datasets..."
if [ ! -d "/workspace/data/AIROGS" ] || [ ! -d "/workspace/data/CHAKSHU" ]; then
    echo "ERROR: Datasets not found in /workspace/data/"
    echo "Please ensure AIROGS and CHAKSHU folders are extracted in /workspace/data/"
    echo ""
    echo "Expected structure:"
    echo "  /workspace/data/AIROGS/train/NRG/"
    echo "  /workspace/data/AIROGS/train/RG/"
    echo "  /workspace/data/AIROGS/test/NRG/"
    echo "  /workspace/data/AIROGS/test/RG/"
    echo "  /workspace/data/CHAKSHU/train/NRG/"
    echo "  /workspace/data/CHAKSHU/train/RG/"
    echo "  /workspace/data/CHAKSHU/train_unlabelled/"
    echo "  /workspace/data/CHAKSHU/test/NRG/"
    echo "  /workspace/data/CHAKSHU/test/RG/"
    exit 1
fi

echo "✓ Datasets found"
echo ""

# Update system packages
echo "Updating system packages..."
apt-get update -qq
apt-get install -y -qq wget unzip > /dev/null
echo "✓ System packages updated"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q -r /workspace/rag_glaucoma_screening/requirements.txt
echo "✓ Python dependencies installed"
echo ""

# Prepare data (create CSVs)
echo "Preparing data (creating CSV files)..."
cd /workspace/rag_glaucoma_screening
python prepare_data.py
echo "✓ Data preparation complete"
echo ""

# Summary
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Build RAG database:"
echo "     cd /workspace/rag_glaucoma_screening"
echo "     python build_rag_database.py"
echo ""
echo "  2. Run evaluation:"
echo "     python evaluate_rag.py"
echo ""
echo "  Or run the full pipeline:"
echo "     python run_rag_pipeline.py"
echo ""
echo "============================================"
