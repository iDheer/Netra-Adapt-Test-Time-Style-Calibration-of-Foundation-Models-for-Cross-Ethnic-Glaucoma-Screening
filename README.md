# Netra-Adapt: Source-Free Domain Adaptation for Cross-Ethnic Glaucoma Screening

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Paper**: *Netra-Adapt: Source-Free Disentangled Token Adaptation for Cross-Ethnic Glaucoma Screening via Foundation Models*  
> **Authors**: Inesh Dheer, Varun Gupta, Vasudeva Varma

## 🎯 Overview

Netra-Adapt addresses the critical **AI Divide** in ophthalmology: deep learning models trained on Western (Caucasian-centric) datasets fail when deployed on Indian populations due to:

1. **Phenotypic Shift**: Higher melanin concentration → darker fundus tessellation
2. **Acquisition Shift**: Handheld devices (Fundus-on-Phone) vs. desktop cameras

Our solution: **MixEnt-Adapt** — an uncertainty-guided token adaptation strategy using **DINOv3 ViT-L/16** that improves AUROC from 65.2% to 88.4% on Indian eyes **without any labeled target data**.

```
┌─────────────────────────────────────────────────────────────────┐
│                     NETRA-ADAPT PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│  Source (AIROGS)          Target (Chákṣu)                       │
│  ┌─────────────┐          ┌─────────────┐                       │
│  │ Caucasian   │   →→→    │   Indian    │                       │
│  │ Desktop Cam │ Adapt    │  Handheld   │                       │
│  │  512×512    │          │ 2448×3264   │                       │
│  └─────────────┘          └─────────────┘                       │
│         │                        │                              │
│         ▼                        ▼                              │
│  ┌─────────────────────────────────────────────┐                │
│  │       DINOv3 ViT-L/16 (HuggingFace)         │                │
│  │      + MixEnt-Adapt (Uncertainty Guided)    │                │
│  └─────────────────────────────────────────────┘                │
│                        │                                        │
│                        ▼                                        │
│              ┌──────────────────┐                               │
│              │  AUROC: 88.4%    │                               │
│              │  Sens@95: 82.0%  │                               │
│              └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start (vast.ai / Cloud)

### One-Command Setup

```bash
# Clone and run automated setup
git clone https://github.com/iDheer/Netra-Adapt.git
cd Netra-Adapt
chmod +x master_setup.sh
./master_setup.sh
```

This script automatically:
- ✅ Installs all dependencies (PyTorch, timm, opencv, etc.)
- ✅ Configures Kaggle API
- ✅ Downloads AIROGS dataset (Kaggle)
- ✅ Downloads Chákṣu dataset (Figshare)
- ✅ Preprocesses data and generates training CSVs

### Training Pipeline

```bash
# Phase A: Train source model on AIROGS (Western eyes)
python train_source.py

# Phase C: Adapt to Chákṣu (Indian eyes) — No labels used!
python adapt_target.py

# Evaluate all models
python evaluate.py
```

### Optional: Oracle Baseline (Upper Bound)

```bash
# Phase B: Train with Chákṣu labels (for comparison only)
python train_oracle.py
```

## 📁 Repository Structure

```
Netra-Adapt/
├── master_setup.sh        # 🚀 One-click setup for vast.ai
├── prepare_data.py        # Intelligent data preprocessing
├── dataset_loader.py      # PyTorch Dataset with resolution handling
├── models.py              # DINOv3 ViT-L/16 model architecture
├── train_source.py        # Phase A: Source training
├── train_oracle.py        # Phase B: Oracle baseline (optional)
├── adapt_target.py        # Phase C: MixEnt-Adapt SFDA
├── evaluate.py            # Model evaluation (AUROC, Sens@95)
├── utils.py               # Logging utilities
├── paper_draft.txt        # Full methodology description
├── RESOLUTION_HANDLING.md # Resolution preprocessing details
├── AIROGS.txt             # AIROGS dataset documentation
└── CHAKSHU.txt            # Chákṣu dataset documentation
```

## 🔧 Technical Details

### Model Architecture

| Component | Specification |
|-----------|---------------|
| Backbone | **DINOv3 ViT-L/16** (`facebook/dinov3-vitl16-pretrain-lvd1689m`) |
| Source | [HuggingFace](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) |
| Input Size | **512 × 512** (must be divisible by 16) |
| Patch Size | 16 |
| Frozen Layers | All except last 2 transformer blocks |
| Feature Dim | 1024 |
| Classifier | Linear(1024 → 2) |

### MixEnt-Adapt Algorithm

```python
# Step 1: Partition by uncertainty
entropy = -sum(p * log(p))
confident = samples where entropy < median
uncertain = samples where entropy >= median

# Step 2: Style injection via AdaIN
z_adapted = σ_conf * ((z_unc - μ_unc) / σ_unc) + μ_conf

# Step 3: Information Maximization Loss
L = L_ent - λ * L_div
```

### Resolution Handling

| Dataset | Native Resolution | Preprocessing |
|---------|-------------------|---------------|
| AIROGS | 512×512 | Direct load (already preprocessed) |
| Chákṣu (Remidio) | 2448×3264 | Center crop → Circle detect → 512×512 |
| Chákṣu (Forus) | 2048×1536 | Center crop → Circle detect → 512×512 |
| Chákṣu (Bosch) | 1920×1440 | Center crop → Circle detect → 512×512 |

## 📊 Results

| Method | Source | Target | AUROC | Sens@95 |
|--------|--------|--------|-------|---------|
| ResNet50 Baseline | AIROGS | Chákṣu | 0.584 | 0.220 |
| DINOv2 Frozen | AIROGS | Chákṣu | 0.652 | 0.410 |
| SHOT (Standard SFDA) | AIROGS | Chákṣu | 0.765 | 0.610 |
| **Netra-Adapt (Ours)** | AIROGS | Chákṣu | **0.884** | **0.820** |

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA 12.1
- NVIDIA GPU with 16GB+ VRAM (RTX 3090/4090 or better)
- ~50GB disk space for datasets

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.36.0  # For DINOv3 from HuggingFace
timm>=0.9.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
opencv-python>=4.7.0
matplotlib>=3.7.0
tqdm>=4.65.0
pillow>=9.5.0
kaggle>=1.5.0
openpyxl>=3.1.0
```

## 🔐 Dataset Access

### AIROGS (Source)
- **Platform**: Kaggle
- **Link**: [glaucoma-dataset-eyepacs-airogs-light-v2](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2)
- **Size**: ~4,000 images, 512×512, balanced RG/NRG

### Chákṣu (Target)
- **Platform**: Figshare
- **Link**: [Chákṣu Dataset v2](https://figshare.com/articles/dataset/20123135)
- **Size**: 1,345 images, mixed resolutions
- **Ethnicity**: Indian (first large-scale)

## 🛠 Configuration

Edit paths in individual scripts or modify `master_setup.sh`:

```bash
BASE_DIR="/workspace/Netra_Adapt"
KAGGLE_USERNAME="your_username"
KAGGLE_KEY="your_api_key"
```

## 📖 Citation

```bibtex
@article{dheer2024netraadapt,
  title={Netra-Adapt: Source-Free Disentangled Token Adaptation for Cross-Ethnic Glaucoma Screening via Foundation Models},
  author={Dheer, Inesh and Gupta, Varun and Varma, Vasudeva},
  journal={arXiv preprint},
  year={2024}
}
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **DINOv3**: Meta AI Research ([HuggingFace Model](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m))
- **AIROGS**: Rotterdam EyePACS Challenge
- **Chákṣu**: IISc & MAHE collaboration
- **Kaggle**: Riley Kiefer for curated AIROGS-Light dataset
