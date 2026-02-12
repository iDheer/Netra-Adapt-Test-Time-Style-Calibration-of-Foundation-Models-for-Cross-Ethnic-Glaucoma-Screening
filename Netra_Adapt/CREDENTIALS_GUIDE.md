# Credentials & Authentication Guide

## Overview

The pipeline needs TWO types of credentials:

1. **Kaggle API Token** - For downloading AIROGS dataset
2. **HuggingFace Token** - For downloading DINOv3 model

---

## 🔑 Option 1: Use setup_complete.sh (EASIEST)

Edit `setup_complete.sh` and update these lines (around line 18-22):

```bash
# Kaggle API Token
KAGGLE_API_TOKEN="KGAT_0976ca4654d6cf8831b434187fc9660d"  # Your token

# HuggingFace Token  
HUGGINGFACE_TOKEN="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"  # Your token
```

Then run:
```bash
bash setup_complete.sh
```

This will:
- Set up both credentials
- Download both datasets
- Prepare all CSVs
- Everything ready for training!

---

## 🔑 Option 2: Manual Setup

### Step 1: Kaggle Authentication

```bash
# Method A: Environment variable
export KAGGLE_API_TOKEN="KGAT_0976ca4654d6cf8831b434187fc9660d"

# Method B: Config file
mkdir -p ~/.kaggle
echo '{"token":"KGAT_0976ca4654d6cf8831b434187fc9660d"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Method C: Using kaggle CLI
pip install kaggle
export KAGGLE_API_TOKEN="KGAT_0976ca4654d6cf8831b434187fc9660d"
```

### Step 2: HuggingFace Authentication

```bash
# Method A: Environment variable (pipeline checks this)
export HF_TOKEN="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"

# Method B: Using huggingface-cli (recommended)
pip install huggingface-hub
huggingface-cli login --token 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

# Method C: Add to ~/.bashrc (persistent)
echo 'export HF_TOKEN="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"' >> ~/.bashrc
source ~/.bashrc
```

---

## 🔍 How Credentials Are Used:

### Kaggle (AIROGS Dataset):
- Used by: `setup_complete.sh`, manual Kaggle downloads
- Downloads: `glaucoma-dataset-eyepacs-airogs-light-v2.zip` (~3 GB)
- Alternative: Manually upload zip to `/workspace/data/`

### HuggingFace (DINOv3 Model):
- Used by: `models.py` → `AutoModel.from_pretrained()`
- Downloads: `facebook/dinov3-vitl16-pretrain-lvd1689m` (~1.2 GB)
- Cached at: `~/.cache/huggingface/`
- Only downloaded once, then cached

---

## ✅ Checking If Credentials Work:

### Test Kaggle:
```bash
kaggle datasets list | head -5
# Should show datasets, not authentication error
```

### Test HuggingFace:
```python
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')"
# Should download model, not show auth error
```

---

## 🚨 What If I Skip Credentials?

### If Kaggle not set:
- ❌ Can't download AIROGS automatically
- ✅ Manual workaround: Download zip from Kaggle website, upload to `/workspace/data/`

### If HuggingFace not set:
- ⚠️ DINOv3 download may fail with "Authentication required"
- ✅ Public models usually work without token
- ❌ If it fails, you MUST set HF_TOKEN

---

## 📋 Complete Setup Flow:

### Scenario A: Fresh Setup with Credentials (RECOMMENDED)
```bash
# 1. Edit credentials in setup_complete.sh
vim setup_complete.sh  # Update lines 18-22

# 2. Run complete setup
bash setup_complete.sh

# 3. Run pipeline
bash run_everything.sh
```

### Scenario B: Data Already Downloaded
```bash
# If data already at /workspace/data/, just set HF token:
export HF_TOKEN="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"

# Run pipeline
bash run_everything.sh
```

### Scenario C: Manual Everything
```bash
# 1. Set tokens
export KAGGLE_API_TOKEN="KGAT_..."
export HF_TOKEN="hf_..."

# 2. Download datasets manually (follow QUICK_COMMANDS_INESH.txt)

# 3. Prepare data
python prepare_data.py

# 4. Run pipeline
bash run_everything.sh
```

---

## 🔐 Your Current Tokens (from QUICK_COMMANDS_INESH.txt):

```bash
# Copy-paste these:
export KAGGLE_API_TOKEN="KGAT_0976ca4654d6cf8831b434187fc9660d"
export HF_TOKEN="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"
```

⚠️ **Note**: These tokens are already in your QUICK_COMMANDS file. The new `setup_complete.sh` automatically uses them!

---

## 🎯 TL;DR - Simplest Approach:

```bash
cd /workspace/Netra_Adapt

# Option 1: Full setup (if data not downloaded)
bash setup_complete.sh  # Credentials already in file!
bash run_everything.sh

# Option 2: Data exists, just run pipeline
export HF_TOKEN="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"
bash run_everything.sh
```

Done! 🎉
