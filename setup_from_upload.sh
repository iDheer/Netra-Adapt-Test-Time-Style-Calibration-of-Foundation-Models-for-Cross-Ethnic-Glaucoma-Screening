#!/bin/bash

# --- CONFIG ---
BASE_DIR="/workspace/Netra_Adapt"
DATA_DIR="$BASE_DIR/data"
RAW_AIROGS="$DATA_DIR/raw_airogs"
RAW_CHAKSU="$DATA_DIR/raw_chaksu"

echo "========================================================"
echo "   NETRA-ADAPT: OFFLINE SETUP (FROM UPLOADED ZIPS)"
echo "========================================================"

# 1. INSTALL DEPENDENCIES (Still needed for the environment)
echo "[1/4] Installing System Libraries..."
apt-get update -qq && apt-get install -y libgl1 libglib2.0-0 unzip -qq
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install timm pandas numpy scikit-learn opencv-python matplotlib tqdm pillow openpyxl -q

# 2. PREPARE DIRECTORIES
mkdir -p "$RAW_AIROGS"
mkdir -p "$RAW_CHAKSU"
mkdir -p "$BASE_DIR/results"
mkdir -p "$DATA_DIR/processed_csvs"

# 3. PROCESS AIROGS (archive.zip)
echo "[2/4] Processing AIROGS..."
if [ -f "$DATA_DIR/archive.zip" ]; then
    echo "      Unzipping archive.zip..."
    unzip -q "$DATA_DIR/archive.zip" -d "$RAW_AIROGS"
    
    echo "      Organizing AIROGS..."
    # Smart Find: Locates RG/NRG folders wherever they unzipped and moves them to root
    find "$RAW_AIROGS" -type d -name "RG" -exec mv {} "$RAW_AIROGS/" \; 2>/dev/null
    find "$RAW_AIROGS" -type d -name "NRG" -exec mv {} "$RAW_AIROGS/" \; 2>/dev/null
    
    # Cleanup empty folders
    rm -rf "$RAW_AIROGS/eyepac-light-v2-512-jpg"
    
    # Verify count
    COUNT=$(ls "$RAW_AIROGS/RG" 2>/dev/null | wc -l)
    echo "      ✓ Found $COUNT RG images."
else
    echo "[ERROR] archive.zip not found in $DATA_DIR"
fi

# 4. PROCESS CHAKSU (20123135.zip)
echo "[3/4] Processing Chákṣu..."
if [ -f "$DATA_DIR/20123135.zip" ]; then
    echo "      Unzipping 20123135.zip..."
    # Unzip to temp folder to handle complexity
    TEMP_CHAKSU="$DATA_DIR/temp_chaksu"
    mkdir -p "$TEMP_CHAKSU"
    unzip -q "$DATA_DIR/20123135.zip" -d "$TEMP_CHAKSU"
    
    # Handle nested zips (common in this dataset)
    find "$TEMP_CHAKSU" -name "*.zip" -exec unzip -q {} -d "$TEMP_CHAKSU" \;

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
    
    echo "      ✓ Chákṣu Organized."
else
    echo "[ERROR] 20123135.zip not found in $DATA_DIR"
fi

# 5. GENERATE CSVs
echo "[4/4] Generating CSVs..."
cd "$BASE_DIR"
if [ -f "prepare_data.py" ]; then
    python prepare_data.py
else
    echo "[WARNING] prepare_data.py missing. Please upload it!"
fi

echo "========================================================"
echo "   READY! Run: python train_source.py"
echo "========================================================"