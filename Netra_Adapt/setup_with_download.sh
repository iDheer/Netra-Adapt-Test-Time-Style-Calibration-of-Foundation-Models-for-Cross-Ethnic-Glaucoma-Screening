#!/bin/bash

# --- CONFIG ---
BASE_DIR="/workspace"
DATA_DIR="$BASE_DIR/data"
RAW_AIROGS="$DATA_DIR/raw_airogs"
RAW_CHAKSU="$DATA_DIR/raw_chaksu"

echo "========================================================"
echo "   NETRA-ADAPT: SETUP FROM KAGGLE + CHAKSHU"
echo "========================================================"

# 1. INSTALL DEPENDENCIES
echo "[1/4] Installing System Libraries..."
apt-get update -qq && apt-get install -y libgl1 libglib2.0-0 unzip wget curl -qq
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install timm pandas numpy scikit-learn opencv-python matplotlib tqdm pillow openpyxl -q

# 2. PREPARE DIRECTORIES
mkdir -p "$RAW_AIROGS"
mkdir -p "$RAW_CHAKSU"
mkdir -p "$BASE_DIR/results"
mkdir -p "$DATA_DIR/processed_csvs"

# 3. PROCESS AIROGS (Already downloaded via Kaggle)
echo "[2/4] Processing AIROGS..."

# Check if Kaggle zip needs to be unzipped
KAGGLE_ZIP="$BASE_DIR/glaucoma-dataset-eyepacs-airogs-light-v2.zip"
if [ -f "$KAGGLE_ZIP" ] && [ ! -d "$BASE_DIR/glaucoma_dataset" ]; then
    echo "      Unzipping Kaggle AIROGS dataset..."
    unzip -q "$KAGGLE_ZIP" -d "$BASE_DIR"
fi

KAGGLE_AIROGS="$BASE_DIR/glaucoma_dataset/eyepac-light-v2-512-jpg"

if [ -d "$KAGGLE_AIROGS" ]; then
    echo "      Found Kaggle AIROGS dataset"
    
    # Combine train/test/validation into single RG and NRG folders
    mkdir -p "$RAW_AIROGS/RG"
    mkdir -p "$RAW_AIROGS/NRG"
    
    echo "      Copying RG (glaucoma) images..."
    find "$KAGGLE_AIROGS" -path "*/RG/*.jpg" -exec cp {} "$RAW_AIROGS/RG/" \; 2>/dev/null
    
    echo "      Copying NRG (normal) images..."
    find "$KAGGLE_AIROGS" -path "*/NRG/*.jpg" -exec cp {} "$RAW_AIROGS/NRG/" \; 2>/dev/null
    
    # Count images
    RG_COUNT=$(ls "$RAW_AIROGS/RG" 2>/dev/null | wc -l)
    NRG_COUNT=$(ls "$RAW_AIROGS/NRG" 2>/dev/null | wc -l)
    echo "      ✓ Found $RG_COUNT RG images and $NRG_COUNT NRG images."
else
    echo "[WARNING] Kaggle AIROGS dataset not found at $KAGGLE_AIROGS"
fi

# 4. PROCESS CHAKSU (Train.zip and Test.zip - User Uploaded)
echo "[3/4] Processing Chákṣu..."
TEMP_CHAKSU="$DATA_DIR/temp_chaksu"
mkdir -p "$TEMP_CHAKSU"

# Process Train.zip
if [ -f "$DATA_DIR/Train.zip" ]; then
    echo "      Extracting Train.zip..."
    unzip -q "$DATA_DIR/Train.zip" -d "$TEMP_CHAKSU"
    echo "      ✓ Train set extracted"
else
    echo "[WARNING] Train.zip not found in $DATA_DIR"
fi

# Process Test.zip
if [ -f "$DATA_DIR/Test.zip" ]; then
    echo "      Extracting Test.zip..."
    unzip -q "$DATA_DIR/Test.zip" -d "$TEMP_CHAKSU"
    echo "      ✓ Test set extracted"
else
    echo "[WARNING] Test.zip not found in $DATA_DIR"
fi

# Handle nested zips if any
echo "      Checking for nested zips..."
find "$TEMP_CHAKSU" -name "*.zip" -exec unzip -q {} -d "$TEMP_CHAKSU" \; 2>/dev/null

echo "      Consolidating Chákṣu folders..."
# Flatten the hierarchy: Pull Bosch/Forus/Remidio/Labels from deep subfolders to root

# 1. Move Images
find "$TEMP_CHAKSU" -type d -name "Bosch" -exec cp -r {}/. "$RAW_CHAKSU/Bosch/" \; 2>/dev/null
find "$TEMP_CHAKSU" -type d -name "Forus" -exec cp -r {}/. "$RAW_CHAKSU/Forus/" \; 2>/dev/null
find "$TEMP_CHAKSU" -type d -name "Remidio" -exec cp -r {}/. "$RAW_CHAKSU/Remidio/" \; 2>/dev/null

# 2. Move Labels (Folder starts with 6.0)
find "$TEMP_CHAKSU" -type d -name "6.0_Glaucoma_Decision" -exec cp -r {}/. "$RAW_CHAKSU/6.0_Glaucoma_Decision/" \; 2>/dev/null

# Cleanup
rm -rf "$TEMP_CHAKSU"

# Verify organization
TOTAL_IMAGES=$(find "$RAW_CHAKSU" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
echo "      ✓ Chákṣu Organized. Found $TOTAL_IMAGES images."

# 5. GENERATE CSVs
echo "[4/4] Generating CSVs..."
cd "$BASE_DIR"
if [ -f "prepare_data.py" ]; then
    python prepare_data.py
else
    echo "[WARNING] prepare_data.py missing. Please upload it!"
fi

echo "========================================================"
echo "   SETUP COMPLETE!"
echo "   Run: python train_source.py"
echo "========================================================"
