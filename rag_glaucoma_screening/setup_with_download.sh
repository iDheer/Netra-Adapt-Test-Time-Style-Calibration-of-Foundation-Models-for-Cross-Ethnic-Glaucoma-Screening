#!/bin/bash

# Setup script for RAG-based glaucoma screening with dataset downloads
# Downloads AIROGS and Chákṣu datasets from Kaggle
# Run this on a fresh Vast.ai instance or local machine

set -e  # Exit on error

echo "============================================"
echo "RAG Glaucoma Screening - Setup (With Download)"
echo "============================================"
echo ""

# Determine if running on Vast.ai or locally
if [ -d "/workspace" ]; then
    WORK_DIR="/workspace"
    echo "✓ Detected Vast.ai environment"
else
    WORK_DIR="."
    echo "✓ Running locally"
fi

cd "$WORK_DIR"

# Check for Kaggle API token
echo "Checking Kaggle credentials..."
if [ -z "$KAGGLE_API_TOKEN" ] && [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "ERROR: Kaggle credentials not found!"
    echo ""
    echo "Option 1 - Use API token (recommended for Vast.ai):"
    echo "  export KAGGLE_API_TOKEN=your_token_here"
    echo ""
    echo "Option 2 - Use kaggle.json:"
    echo "  1. Go to https://www.kaggle.com/settings/account"
    echo "  2. Click 'Create New Token' to download kaggle.json"
    echo "  3. Place it at: $HOME/.kaggle/kaggle.json"
    echo "  4. Run: chmod 600 $HOME/.kaggle/kaggle.json"
    echo ""
    exit 1
fi
echo "✓ Kaggle credentials found"
echo ""

# Update system packages
echo "Updating system packages..."
if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y -qq wget unzip > /dev/null
fi
echo "✓ System packages updated"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q -r "$WORK_DIR/rag_glaucoma_screening/requirements.txt"
echo "✓ Python dependencies installed"
echo ""

# Create data directory
mkdir -p "$WORK_DIR/data"
cd "$WORK_DIR/data"

# Download AIROGS dataset
echo "============================================"
echo "Downloading AIROGS Dataset"
echo "============================================"
if [ ! -d "AIROGS" ]; then
    echo "Downloading from Kaggle..."
    kaggle datasets download -d deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2
    
    echo "Extracting..."
    unzip -q glaucoma-dataset-eyepacs-airogs-light-v2.zip -d AIROGS_temp
    
    # Move contents preserving folder structure
    mkdir -p AIROGS
    if [ -d "AIROGS_temp/AIROGS" ]; then
        mv AIROGS_temp/AIROGS/* AIROGS/
    elif [ -d "AIROGS_temp/airogs" ]; then
        mv AIROGS_temp/airogs/* AIROGS/
    else
        mv AIROGS_temp/* AIROGS/
    fi
    
    rm -rf AIROGS_temp glaucoma-dataset-eyepacs-airogs-light-v2.zip
    echo "✓ AIROGS dataset ready"
else
    echo "✓ AIROGS dataset already exists"
fi
echo ""

# Download Chákṣu dataset
echo "============================================"
echo "Downloading Chákṣu Dataset"
echo "=====================Figshare..."
    
    # Create download script
    cat > download_chaksu.py << 'PYEOF'
import urllib.request, json, os

article_id = 20123135
url = f"https://api.figshare.com/v2/articles/{article_id}/files"

print(f"Fetching file list for Article {article_id}...")
try:
    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)
    data = json.loads(response.read().decode())
    
    print(f"Found {len(data)} files. Starting downloads...\n")
    
    for f in data:
        name = f['name']
        download_url = f['download_url']
        size_mb = f['size'] / (1024 * 1024)
        
        print(f"Downloading: {name} ({size_mb:.2f} MB)")
        os.system(f"wget -q --show-progress -O '{name}' '{download_url}'")
        print("Done.\n")
        
except Exception as e:
    print(f"Error: {e}")
PYEOF
    
    python3 download_chaksu.py
    rm download_chaksu.py
    
    echo "Extracting Train.zip..."
    unzip -q Train.zip -d CHAKSHU/
    
    echo "Extracting Test.zip..."
    unzip -q Test.zip -d CHAKSHU/
    
    # Clean up zip files
    rm Train.zip Test.zip
    
    echo "✓ Chákṣu dataset ready"
else
    echo "✓ Chákṣu dataset already exists"
fi
echo ""

# Prepare data (create CSVs)
echo "============================================"
echo "Preparing Data"
echo "============================================"
cd "$WORK_DIR/rag_glaucoma_screening"
python prepare_data.py
echo "✓ Data preparation complete"
echo ""

# Summary
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Dataset locations:"
echo "  AIROGS: $WORK_DIR/data/AIROGS/"
echo "  Chákṣu: $WORK_DIR/data/CHAKSHU/"
echo ""
echo "Next steps:"
echo "  1. Build RAG database:"
echo "     cd $WORK_DIR/rag_glaucoma_screening"
echo "     python build_rag_database.py"
echo ""
echo "  2. Run evaluation:"
echo "     python evaluate_rag.py"
echo ""
echo "  Or run the full pipeline:"
echo "     python run_rag_pipeline.py"
echo ""
echo "============================================"
