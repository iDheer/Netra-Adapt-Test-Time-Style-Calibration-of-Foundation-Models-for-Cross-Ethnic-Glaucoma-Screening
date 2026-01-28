#!/bin/bash

# =============================================================================
# NETRA-ADAPT: Automated Research Environment Setup for vast.ai
# =============================================================================
# This script handles complete end-to-end setup including:
# - System dependencies installation
# - Kaggle API configuration
# - AIROGS dataset download (Kaggle)
# - Chákṣu dataset download (Figshare)
# - Data preprocessing and CSV generation
#
# Model: DINOv3 ViT-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m)
# =============================================================================

set -e  # Exit on error

# --- CONFIGURATION ---
# IMPORTANT: Replace with your Kaggle credentials
export KAGGLE_USERNAME="deathtrooper"
export KAGGLE_KEY="05daa0dc02c9f962d7e3bcdaeb7e205e"

BASE_DIR="/workspace/Netra_Adapt"
DATA_DIR="$BASE_DIR/data"
RAW_AIROGS="$DATA_DIR/raw_airogs"
RAW_CHAKSU="$DATA_DIR/raw_chaksu"
CODE_DIR="$BASE_DIR/code"

echo "========================================================"
echo "   NETRA-ADAPT: AUTOMATED RESEARCH ENVIRONMENT SETUP"
echo "   Model: DINOv3 ViT-L/16 (HuggingFace)"
echo "   Target: Source-Free Domain Adaptation"
echo "========================================================"
echo ""

# 1. SYSTEM DEPENDENCIES
echo "[1/7] Installing System Dependencies..."
apt-get update -qq
apt-get install -y libgl1-mesa-glx libglib2.0-0 unzip wget curl -qq

# 2. PYTHON PACKAGES
echo "[2/7] Installing Python Packages..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers>=4.36.0 -q  # For DINOv3 from HuggingFace
pip install timm pandas numpy scikit-learn opencv-python matplotlib tqdm pillow kaggle openpyxl -q

# 3. DIRECTORY STRUCTURE
echo "[3/7] Creating Directory Structure..."
mkdir -p "$RAW_AIROGS"
mkdir -p "$RAW_CHAKSU"
mkdir -p "$BASE_DIR/results/Source_AIROGS"
mkdir -p "$BASE_DIR/results/Oracle_Chaksu"
mkdir -p "$BASE_DIR/results/Netra_Adapt"
mkdir -p "$DATA_DIR/processed_csvs"

# 4. KAGGLE API SETUP
echo "[4/7] Configuring Kaggle API..."
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_KEY}"}
EOF
chmod 600 ~/.kaggle/kaggle.json

# Verify Kaggle credentials
if ! kaggle datasets list -s "test" &>/dev/null; then
    echo "[ERROR] Kaggle authentication failed!"
    echo "Please update KAGGLE_USERNAME and KAGGLE_KEY in this script."
    exit 1
fi
echo "  ✓ Kaggle authentication successful"

# 5. DOWNLOAD AIROGS (KAGGLE)
echo "[5/7] Downloading AIROGS Light V2 from Kaggle..."
if [ ! -d "$RAW_AIROGS/RG" ] || [ ! -d "$RAW_AIROGS/NRG" ]; then
    kaggle datasets download -d deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2 -p "$RAW_AIROGS" --unzip
    
    # Normalize structure: find RG/NRG folders wherever they unzipped
    echo "  Normalizing AIROGS folder structure..."
    find "$RAW_AIROGS" -type d -name "RG" ! -path "$RAW_AIROGS/RG" -exec mv {} "$RAW_AIROGS/RG_temp" \; 2>/dev/null || true
    find "$RAW_AIROGS" -type d -name "NRG" ! -path "$RAW_AIROGS/NRG" -exec mv {} "$RAW_AIROGS/NRG_temp" \; 2>/dev/null || true
    [ -d "$RAW_AIROGS/RG_temp" ] && mv "$RAW_AIROGS/RG_temp" "$RAW_AIROGS/RG"
    [ -d "$RAW_AIROGS/NRG_temp" ] && mv "$RAW_AIROGS/NRG_temp" "$RAW_AIROGS/NRG"
    
    # Clean up nested folders
    find "$RAW_AIROGS" -mindepth 1 -maxdepth 1 -type d ! -name "RG" ! -name "NRG" -exec rm -rf {} \;
    
    echo "  ✓ AIROGS downloaded: $(ls "$RAW_AIROGS/RG" 2>/dev/null | wc -l) RG, $(ls "$RAW_AIROGS/NRG" 2>/dev/null | wc -l) NRG images"
else
    echo "  ✓ AIROGS already exists, skipping download"
fi

# 6. DOWNLOAD CHÁKṢU (FIGSHARE)
echo "[6/7] Downloading Chákṣu V2 from Figshare..."
if [ ! -d "$RAW_CHAKSU/Train" ] && [ ! -d "$RAW_CHAKSU/Bosch" ]; then
    # Figshare direct download URL for Chákṣu dataset
    wget -q --show-progress -O "$DATA_DIR/chaksu.zip" "https://figshare.com/ndownloader/articles/20123135/versions/2"
    
    echo "  Extracting Chákṣu..."
    unzip -q "$DATA_DIR/chaksu.zip" -d "$RAW_CHAKSU"
    rm "$DATA_DIR/chaksu.zip"
    
    # The Figshare download may contain nested zips - extract them
    for zipfile in "$RAW_CHAKSU"/*.zip; do
        if [ -f "$zipfile" ]; then
            echo "  Extracting nested: $(basename $zipfile)"
            unzip -q "$zipfile" -d "$RAW_CHAKSU"
            rm "$zipfile"
        fi
    done
    
    # Count images found
    CHAKSU_COUNT=$(find "$RAW_CHAKSU" -type f \( -iname "*.jpg" -o -iname "*.png" \) | wc -l)
    echo "  ✓ Chákṣu downloaded: $CHAKSU_COUNT images found"
else
    echo "  ✓ Chákṣu already exists, skipping download"
fi

# 7. EXECUTE DATA PREPARATION
echo "[7/7] Running Intelligent Data Preparation..."
cd "$CODE_DIR" 2>/dev/null || cd "$(dirname "$0")"

if [ -f "prepare_data.py" ]; then
    python prepare_data.py
else
    echo "[WARNING] prepare_data.py not found in current directory."
    echo "          Please run 'python prepare_data.py' manually after setup."
fi

echo ""
echo "========================================================"
echo "   ✅ SETUP COMPLETE!"
echo "========================================================"
echo ""
echo "   Dataset Summary:"
echo "   - AIROGS: $RAW_AIROGS"
echo "   - Chákṣu: $RAW_CHAKSU"
echo "   - CSVs:   $DATA_DIR/processed_csvs/"
echo ""
echo "   Training Pipeline:"
echo "   1. python train_source.py   # Phase A: Source Training (AIROGS)"
echo "   2. python train_oracle.py   # Phase B: Oracle Baseline (Chákṣu labeled)"
echo "   3. python adapt_target.py   # Phase C: Netra-Adapt SFDA"
echo "   4. python evaluate.py       # Evaluate all models"
echo ""
echo "   Or run all at once:"
echo "   python train_source.py && python adapt_target.py && python evaluate.py"
echo ""
echo "========================================================"