#!/bin/bash
###############################################################################
#                   COMPLETE SETUP WITH CREDENTIALS                          #
#                                                                             #
#  This script handles EVERYTHING from scratch:                              #
#  - Sets up credentials (Kaggle, HuggingFace)                               #
#  - Downloads datasets (AIROGS from Kaggle, Chákṣu from Figshare)          #
#  - Installs dependencies                                                    #
#  - Prepares data                                                            #
#                                                                             #
#  You only need to edit the credentials section below!                      #
###############################################################################

set -e  # Exit on error

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - EDIT THESE VALUES
# ═══════════════════════════════════════════════════════════════════════════

# Kaggle API Token (from your QUICK_COMMANDS file)
KAGGLE_API_TOKEN="KGAT_0976ca4654d6cf8831b434187fc9660d"

# HuggingFace Token (needed for DINOv3 model download)
# Get from: https://huggingface.co/settings/tokens
HUGGINGFACE_TOKEN="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"

# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR="/workspace"
DATA_DIR="$BASE_DIR/data"
RAW_AIROGS="$DATA_DIR/raw_airogs"
RAW_CHAKSU="$DATA_DIR/raw_chaksu"
CSV_DIR="$DATA_DIR/processed_csvs"

echo "═══════════════════════════════════════════════════════════════════════"
echo "   NETRA-ADAPT: COMPLETE SETUP FROM SCRATCH"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: System Dependencies
# ═══════════════════════════════════════════════════════════════════════════
echo "[1/6] Installing system dependencies..."
apt-get update -qq
apt-get install -y libgl1 libglib2.0-0 unzip wget curl bc -qq
echo "  ✓ System packages installed"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Python Packages
# ═══════════════════════════════════════════════════════════════════════════
echo "[2/6] Installing Python packages..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers timm pandas numpy scikit-learn opencv-python matplotlib tqdm pillow openpyxl seaborn kaggle huggingface-hub -q
echo "  ✓ Python packages installed"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Setup Credentials
# ═══════════════════════════════════════════════════════════════════════════
echo "[3/6] Setting up credentials..."

# Kaggle
export KAGGLE_API_TOKEN="$KAGGLE_API_TOKEN"
echo "  ✓ Kaggle API token set"

# HuggingFace (login for model downloads)
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential
    echo "  ✓ HuggingFace authentication complete"
else
    echo "  ⚠ HuggingFace token not set - model download may fail"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Download AIROGS Dataset (from Kaggle)
# ═══════════════════════════════════════════════════════════════════════════
echo "[4/6] Downloading AIROGS dataset from Kaggle..."
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -f "glaucoma-dataset-eyepacs-airogs-light-v2.zip" ]; then
    echo "  Downloading from Kaggle... (this may take 10-15 minutes)"
    kaggle datasets download -d deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2
    echo "  ✓ Download complete"
else
    echo "  ✓ AIROGS zip already exists"
fi

# Unzip AIROGS
if [ ! -d "glaucoma_dataset" ]; then
    echo "  Extracting AIROGS dataset..."
    unzip -q glaucoma-dataset-eyepacs-airogs-light-v2.zip -d glaucoma_dataset
    echo "  ✓ Extraction complete"
else
    echo "  ✓ AIROGS dataset already extracted"
fi

# Organize AIROGS into raw_airogs/RG and raw_airogs/NRG
mkdir -p "$RAW_AIROGS/RG"
mkdir -p "$RAW_AIROGS/NRG"

AIROGS_PATH="$DATA_DIR/glaucoma_dataset/eyepac-light-v2-512-jpg"
if [ -d "$AIROGS_PATH" ]; then
    echo "  Organizing AIROGS images..."
    find "$AIROGS_PATH" -path "*/RG/*.jpg" -exec cp {} "$RAW_AIROGS/RG/" \; 2>/dev/null
    find "$AIROGS_PATH" -path "*/NRG/*.jpg" -exec cp {} "$RAW_AIROGS/NRG/" \; 2>/dev/null
    
    RG_COUNT=$(ls "$RAW_AIROGS/RG" 2>/dev/null | wc -l)
    NRG_COUNT=$(ls "$RAW_AIROGS/NRG" 2>/dev/null | wc -l)
    echo "  ✓ Organized: $RG_COUNT RG + $NRG_COUNT NRG images"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Download Chákṣu Dataset (from Figshare)
# ═══════════════════════════════════════════════════════════════════════════
echo "[5/6] Downloading Chákṣu dataset from Figshare..."
cd "$DATA_DIR"

# Download Train.zip (8.1 GB)
if [ ! -f "Train.zip" ]; then
    echo "  Downloading Train.zip (8.1 GB)... this will take 20-30 minutes"
    wget -q --show-progress https://figshare.com/ndownloader/files/35923668 -O Train.zip
    echo "  ✓ Train.zip downloaded"
else
    echo "  ✓ Train.zip already exists"
fi

# Download Test.zip (2.6 GB)
if [ ! -f "Test.zip" ]; then
    echo "  Downloading Test.zip (2.6 GB)... this will take 5-10 minutes"
    wget -q --show-progress https://figshare.com/ndownloader/files/35923671 -O Test.zip
    echo "  ✓ Test.zip downloaded"
else
    echo "  ✓ Test.zip already exists"
fi

# Extract Chákṣu datasets
mkdir -p "$RAW_CHAKSU"
TEMP_CHAKSU="$DATA_DIR/temp_chaksu"
mkdir -p "$TEMP_CHAKSU"

echo "  Extracting Chákṣu datasets..."
if [ -f "Train.zip" ]; then
    unzip -q "Train.zip" -d "$TEMP_CHAKSU"
    echo "  ✓ Train.zip extracted"
fi

if [ -f "Test.zip" ]; then
    unzip -q "Test.zip" -d "$TEMP_CHAKSU"
    echo "  ✓ Test.zip extracted"
fi

# Move Train and Test folders preserving structure
if [ -d "$TEMP_CHAKSU/Train" ]; then
    cp -r "$TEMP_CHAKSU/Train" "$RAW_CHAKSU/"
    echo "  ✓ Train folder organized"
fi

if [ -d "$TEMP_CHAKSU/Test" ]; then
    cp -r "$TEMP_CHAKSU/Test" "$RAW_CHAKSU/"
    echo "  ✓ Test folder organized"
fi

# Cleanup temp
rm -rf "$TEMP_CHAKSU"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Prepare Data CSVs
# ═══════════════════════════════════════════════════════════════════════════
echo "[6/6] Preparing data CSVs..."
cd /workspace/Netra_Adapt

if [ -f "prepare_data.py" ]; then
    python prepare_data.py
    echo "  ✓ CSV files generated"
    
    # List generated CSVs
    if [ -d "$CSV_DIR" ]; then
        echo ""
        echo "  Generated CSV files:"
        for csv in "$CSV_DIR"/*.csv; do
            if [ -f "$csv" ]; then
                LINES=$(wc -l < "$csv")
                echo "    - $(basename $csv): $LINES samples"
            fi
        done
    fi
else
    echo "  ⚠ prepare_data.py not found - skip this step manually later"
fi

# ═══════════════════════════════════════════════════════════════════════════
# COMPLETION
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "   SETUP COMPLETE!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "✓ Credentials configured (Kaggle, HuggingFace)"
echo "✓ Datasets downloaded (AIROGS, Chákṣu)"
echo "✓ Data organized and CSVs generated"
echo ""
echo "Next steps:"
echo "  1. Run the full pipeline:  bash run_everything.sh"
echo "  2. Or run individual phases manually"
echo ""
echo "Data structure:"
echo "  $DATA_DIR/"
echo "    ├── raw_airogs/         (AIROGS images)"
echo "    ├── raw_chaksu/         (Chákṣu images)"
echo "    └── processed_csvs/     (Train/test CSVs)"
echo ""
