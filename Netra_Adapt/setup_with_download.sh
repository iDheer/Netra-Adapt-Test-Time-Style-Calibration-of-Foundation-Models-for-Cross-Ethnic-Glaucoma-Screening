#!/bin/bash

# --- CONFIG ---
BASE_DIR="/workspace/Netra_Adapt"
DATA_DIR="$BASE_DIR/data"
RAW_AIROGS="$DATA_DIR/raw_airogs"
RAW_CHAKSU="$DATA_DIR/raw_chaksu"

echo "========================================================"
echo "   NETRA-ADAPT: SETUP WITH CHAKSHU AUTO-DOWNLOAD"
echo "========================================================"

# 1. INSTALL DEPENDENCIES
echo "[1/5] Installing System Libraries..."
apt-get update -qq && apt-get install -y libgl1 libglib2.0-0 unzip wget curl -qq
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install timm pandas numpy scikit-learn opencv-python matplotlib tqdm pillow openpyxl -q

# 2. PREPARE DIRECTORIES
mkdir -p "$RAW_AIROGS"
mkdir -p "$RAW_CHAKSU"
mkdir -p "$BASE_DIR/results"
mkdir -p "$DATA_DIR/processed_csvs"

# 3. PROCESS AIROGS (archive.zip - User Uploaded)
echo "[2/5] Processing AIROGS..."
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
    echo "[WARNING] archive.zip not found in $DATA_DIR - skipping AIROGS"
fi

# 4. DOWNLOAD CHAKSHU (Figshare - using Python downloader)
echo "[3/5] Downloading Chákṣu Dataset..."

# Check if files are already downloaded
CHAKSU_FILES=$(ls "$DATA_DIR"/*.zip 2>/dev/null | grep -v archive.zip | wc -l)

if [ "$CHAKSU_FILES" -eq 0 ]; then
    echo "      Using Python downloader (Figshare API)..."
    
    # Run Python downloader in data directory
    cd "$DATA_DIR"
    if [ -f "$BASE_DIR/download_chakshu.py" ]; then
        python3 "$BASE_DIR/download_chakshu.py"
        DOWNLOAD_STATUS=$?
    else
        echo "[ERROR] download_chakshu.py not found!"
        echo "      Please upload download_chakshu.py to $BASE_DIR"
        exit 1
    fi
    
    cd "$BASE_DIR"
    
    # Verify download (should have multiple zip files now)
    CHAKSU_FILES=$(ls "$DATA_DIR"/*.zip 2>/dev/null | grep -v archive.zip | wc -l)
    if [ "$CHAKSU_FILES" -gt 0 ]; then
        echo "      ✓ Downloaded $CHAKSU_FILES CHAKSHU file(s)"
    else
        echo "[ERROR] Download failed!"
        echo "      Please download manually from:"
        echo "      https://doi.org/10.6084/m9.figshare.11857698.v2"
        exit 1
    fi
else
    echo "      ✓ Found $CHAKSU_FILES CHAKSHU zip file(s), skipping download."
fi

# 5. PROCESS CHAKSU
echo "[4/5] Processing Chákṣu..."
TEMP_CHAKSU="$DATA_DIR/temp_chaksu"
mkdir -p "$TEMP_CHAKSU"

# Process all CHAKSHU zip files (excluding archive.zip)
CHAKSU_ZIPS=$(ls "$DATA_DIR"/*.zip 2>/dev/null | grep -v archive.zip)

if [ -n "$CHAKSU_ZIPS" ]; then
    for ZIPFILE in $CHAKSU_ZIPS; do
        ZIPNAME=$(basename "$ZIPFILE")
        echo "      Unzipping $ZIPNAME..."
        unzip -q "$ZIPFILE" -d "$TEMP_CHAKSU"
    done
    
    # Handle nested zips (common in this dataset)
    echo "      Checking for nested zips..."
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
    
    # Verify organization
    TOTAL_IMAGES=$(find "$RAW_CHAKSU" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
    echo "      ✓ Chákṣu Organized. Found $TOTAL_IMAGES images."
else
    echo "[ERROR] No CHAKSHU zip files found!"
    exit 1
fi

# 6. GENERATE CSVs
echo "[5/5] Generating CSVs..."
cd "$BASE_DIR"
if [ -f "prepare_data.py" ]; then
    python prepare_data.py
else
    echo "[WARNING] prepare_data.py missing. Please upload it!"
fi

echo "========================================================"
echo "   SETUP COMPLETE!"
echo "   - AIROGS: Uploaded manually (archive.zip)"
echo "   - Chákṣu: Downloaded automatically (~12GB)"
echo "   Run: python train_source.py"
echo "========================================================"
