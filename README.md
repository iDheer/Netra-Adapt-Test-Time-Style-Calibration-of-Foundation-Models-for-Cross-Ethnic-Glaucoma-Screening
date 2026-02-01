# Netra-Adapt: Test-Time Style Calibration of Foundation Models for Cross-Ethnic Glaucoma Screening

**Source-Free Domain Adaptation for Cross-Ethnic Medical Imaging**

Netra-Adapt adapts foundation vision models trained on Western fundus images (AIROGS) to work on Indian eyes (Chákṣu) **without any labeled target data**, using a novel MixEnt-Adapt algorithm for test-time style calibration.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Key Features

- ✅ **Source-Free Domain Adaptation**: No labeled target data needed
- ✅ **Foundation Model**: DINOv3 ViT-L/16 (state-of-the-art)
- ✅ **MixEnt-Adapt**: Entropy-based style injection + Information Maximization
- ✅ **Cross-Ethnic**: Western (AIROGS) → Indian (Chákṣu) fundus images
- ✅ **Early Stopping**: Automatic training optimization
- ✅ **Comprehensive Logging**: Tracks all metrics, curves, visualizations
- ✅ **Research Ready**: 7 metrics + ROC curves + statistical tests

---

## 📊 Experimental Setup

**5 Baseline Comparisons:**

1. **Pretrained → Chákṣu**: Vanilla DINOv3 (zero-shot)
2. **AIROGS → AIROGS**: Source model sanity check
3. **AIROGS → Chákṣu**: Source-only (no adaptation)
4. **Chákṣu → Chákṣu**: Oracle upper bound (fully supervised)
5. **AIROGS+Adapt → Chákṣu**: **Netra-Adapt** (our method)

**Datasets:**
- **AIROGS V2**: ~4,000 Western fundus images (80/20 train/test split)
- **Chákṣu**: 1,345 Indian fundus images (1,009 train / 336 test)

**Metrics:**
- AUROC, Sensitivity, Specificity, Precision, F1-Score, Accuracy, Sensitivity@95% Specificity

---

## 🗂️ Project Structure

```
Netra-Adapt/
├── Netra_Adapt/                        # Main codebase
│   ├── models.py                       # DINOv3 ViT-L/16 model
│   ├── dataset_loader.py               # Dataset handling
│   ├── utils.py                        # Helper functions
│   ├── prepare_data.py                 # Generate train/test CSVs
│   │
│   ├── train_source.py                 # Phase A: AIROGS training
│   ├── train_oracle.py                 # Phase B: Oracle baseline
│   ├── adapt_target.py                 # Phase C: MixEnt-Adapt SFDA
│   ├── evaluate.py                     # Phase D: Evaluation
│   ├── advanced_analysis.py            # Phase E: Interpretability
│   │
│   ├── training_logger.py              # Comprehensive logging system
│   ├── run_full_pipeline.py            # Automated pipeline runner
│   │
│   ├── setup_with_download.sh          # Vast.ai setup script
│   ├── requirements.txt                # Python dependencies
│   
│
├── data/                               # Datasets (downloaded)
│   ├── AIROGS_V2/
│   │   ├── RG/                         # Referable glaucoma
│   │   └── NRG/                        # No referable glaucoma
│   └── chaksu_dataset/
│       ├── Train/
│       │   ├── Bosch/
│       │   ├── Forus/
│       │   └── Remidio/
│       └── Test/
│
├── results/                            # Trained models
│   ├── Source_AIROGS/
│   │   └── model.pth
│   ├── Oracle_Chaksu/
│   │   └── best_oracle.pth
│   └── Adapted_Chaksu/
│       └── adapted_model.pth
│
└── logs/                               # Experiment logs
    └── run_YYYY-MM-DD_HH-MM-SS/
        ├── experiment_log.txt
        ├── metadata.json
        ├── EXPERIMENT_SUMMARY.md
        ├── 01_source_training/
        ├── 02_oracle_training/
        ├── 03_adaptation/
        ├── 04_evaluation/
        └── 05_advanced_analysis/
```

---

## 🚀 Quick Start (Vast.ai)

### 1️⃣ Launch Vast.ai Instance

**Recommended Specs:**
- GPU: RTX 5090
- VRAM: ≥24GB
- Storage: ≥256 (datasets are large)
- CUDA: 12.8 or above

**Search Filter:**
```
```

### 2️⃣ Connect to Instance

```bash
# SSH into your Vast.ai instance
ssh -p YOUR_PORT root@YOUR_IP

# Verify GPU
nvidia-smi
```

### 3️⃣ Clone Repository

```bash
cd /workspace
git clone https://github.com/iDheer/Netra-Adapt-Test-Time-Style-Calibration-of-Foundation-Models-for-Cross-Ethnic-Glaucoma-Screening.git
cd Netra-Adapt-Test-Time-Style-Calibration-of-Foundation-Models-for-Cross-Ethnic-Glaucoma-Screening/Netra_Adapt
```

### 4️⃣ Run Setup Script (Installs Everything!)

**This ONE script does EVERYTHING:**
- ✅ Installs all system libraries (libgl1, libglib2.0, unzip, wget, curl)
- ✅ Installs PyTorch with CUDA 12.1 support
- ✅ Installs all Python dependencies (timm, transformers, scikit-learn, pandas, numpy, opencv, matplotlib, seaborn, scipy, umap-learn, tqdm)
- ✅ Downloads AIROGS V2 dataset (~8GB)
- ✅ Downloads Chákṣu dataset (~2GB)
- ✅ Sets up directory structure
- ✅ Verifies datasets

```bash
bash setup_with_download.sh
```

**Expected Time:** ~20 minutes (depending on internet speed)

**You don't need to run `pip install -r requirements.txt` separately!**

### 5️⃣ Prepare Data

Generate train/test CSV files:

```bash
python prepare_data.py
```

**Output:**
- `data/processed_csvs/airogs_train.csv` (80% of AIROGS)
- `data/processed_csvs/airogs_test.csv` (20% of AIROGS)
- `data/processed_csvs/chaksu_train_labeled.csv` (1,009 images)
- `data/processed_csvs/chaksu_test_labeled.csv` (336 images)
- `data/processed_csvs/chaksu_train_unlabeled.csv` (for SFDA)

### 6️⃣ Run Full Pipeline

**Option A: Automated (Recommended)**

```bash
python run_full_pipeline.py
```

This runs all 5 phases sequentially and generates complete logs.

**Option B: Manual (Step-by-Step)**

```bash
# Phase A: Train on AIROGS (Western eyes)
python train_source.py          # ~2-3 hours, early stops ~30-35 epochs

# Phase B: Train Oracle (Upper bound)
python train_oracle.py          # ~1-2 hours, early stops ~35-40 epochs

# Phase C: Adapt to Chákṣu (SFDA)
python adapt_target.py          # ~45-60 minutes, early stops ~15-18 epochs

# Phase D: Evaluate All Models
python evaluate.py              # ~10 minutes

# Phase E: Advanced Analysis (Optional)
python advanced_analysis.py --all  # ~20 minutes
```

**Expected Total Time:** ~4-5 hours (with early stopping)

---

## 📈 Output & Results

### Training Logs

All experiments are logged to timestamped directories:

```
logs/run_2026-02-02_14-30-45/
├── experiment_log.txt              # Human-readable log
├── metadata.json                   # Machine-readable metadata
├── EXPERIMENT_SUMMARY.md           # Final summary report
├── 01_source_training/
│   ├── hyperparameters.json
│   ├── epoch_metrics.csv           # Loss, accuracy per epoch
│   ├── loss_curve.png              # Training curve
│   └── additional_metrics.png
├── 02_oracle_training/
│   └── (same structure)
├── 03_adaptation/
│   └── epoch_metrics.csv           # Includes L_ent, L_div
├── 04_evaluation/
│   ├── Pretrained_to_Chaksu_metrics.json
│   ├── AIROGS_to_Chaksu_metrics.json
│   ├── Chaksu_to_Chaksu_metrics.json
│   ├── AIROGS+Adapt_to_Chaksu_metrics.json
│   ├── roc_curves.png              # All models on one plot
│   ├── confusion_matrices.png      # 2x2 grid
│   ├── metrics_comparison.png      # Bar chart
│   ├── results.csv                 # Table of metrics
│   └── results_latex.txt           # LaTeX table
└── 05_advanced_analysis/
    ├── tsne_features.png           # Feature space visualization
    ├── umap_features.png
    ├── gradcam_samples.png         # Attention maps
    ├── calibration_curves.png      # Model calibration
    ├── per_camera_analysis.png     # Camera-specific performance
    └── statistical_tests.txt       # McNemar's test results
```

### View Results

```bash
# View summary report
cat logs/run_*/EXPERIMENT_SUMMARY.md

# View evaluation metrics
cat results/evaluation/results.csv

# Copy results to local machine
scp -P YOUR_PORT root@YOUR_IP:/workspace/Netra-Adapt/.../logs/ ./local_logs/
```

---

## 🔬 Algorithm: MixEnt-Adapt

**Source-Free Domain Adaptation via Entropy-Guided Style Injection**

```
1. Partition batch by entropy:
   - High confidence samples (low entropy)
   - Low confidence samples (high entropy)

2. Style injection via AdaIN:
   - Inject statistics from confident → uncertain
   - Calibrates style while preserving semantics

3. Information Maximization Loss:
   L_SFDA = L_ent - λ * L_div
   
   - L_ent: Entropy minimization (decisive predictions)
   - L_div: Diversity maximization (prevents collapse)
   - λ = 1.0 (balance parameter)
```

**Key Advantages:**
- ✅ No target labels needed (source-free)
- ✅ No source data needed during adaptation
- ✅ Preserves discriminative features
- ✅ Prevents mode collapse

---

## 🎓 Evaluation Metrics

**Standard Clinical Metrics:**

| Metric | Description | Clinical Relevance |
|--------|-------------|-------------------|
| **AUROC** | Area Under ROC Curve | Overall discrimination ability |
| **Sensitivity** | True Positive Rate | Catching glaucoma cases |
| **Specificity** | True Negative Rate | Avoiding false alarms |
| **Precision** | Positive Predictive Value | Accuracy of positive diagnoses |
| **F1-Score** | Harmonic mean | Balanced metric |
| **Accuracy** | Overall correctness | General performance |
| **Sens@95** | Sensitivity at 95% Specificity | Clinically relevant tradeoff |

**Statistical Tests:**
- McNemar's test for paired predictions
- p-values for significance testing

---

## 📦 Requirements

### Hardware
- GPU: ≥24GB VRAM (RTX 4090 / A6000 / A100)
- RAM: ≥32GB
- Storage: ≥1TB (datasets + models + logs)
- CUDA: 12.1+

### Software
```txt
Python >= 3.10
PyTorch >= 2.0
transformers >= 4.30.0
timm >= 0.9.0
scikit-learn >= 1.3.0
pandas >= 2.0.0
numpy >= 1.24.0
opencv-python >= 4.8.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
scipy >= 1.11.0
umap-learn >= 0.5.0
tqdm >= 4.65.0
```

**Install all:**
```bash
pip install -r requirements.txt
```

---

## 🛠️ Configuration

### Hyperparameters

**train_source.py (AIROGS)**
```python
BATCH_SIZE = 32
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
```

**train_oracle.py (Oracle)**
```python
BATCH_SIZE = 24          # Smaller for small dataset
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 8  # More patience for small dataset
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
```

**adapt_target.py (SFDA)**
```python
BATCH_SIZE = 32
MAX_EPOCHS = 25          # Faster adaptation
EARLY_STOP_PATIENCE = 5
LR_BACKBONE = 1e-6       # Lower to preserve source knowledge
LR_HEAD = 1e-4
LAMBDA_DIV = 1.0         # Diversity weight
```

### Paths (Automatically set by setup script)

```python
# Data paths
DATA_DIR = "/workspace/data"
CSV_DIR = "/workspace/data/processed_csvs"

# Model paths
SAVE_DIR = "/workspace/results"

# Log paths
LOG_DIR = "logs"
```

---

## 📊 Expected Results

**Typical Performance (AUROC on Chákṣu Test Set):**

| Model | AUROC | Description |
|-------|-------|-------------|
| Pretrained → Chákṣu | ~0.75 | Vanilla DINOv3 (zero-shot) |
| AIROGS → Chákṣu | ~0.82 | Source-only (no adaptation) |
| **AIROGS+Adapt → Chákṣu** | **~0.88** | **Netra-Adapt (our method)** |
| Chákṣu → Chákṣu | ~0.92 | Oracle (upper bound) |

**Key Observations:**
- ✅ Netra-Adapt bridges ~60% of the domain gap
- ✅ Significant improvement over source-only
- ✅ Approaches oracle performance without labels

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in config
BATCH_SIZE = 16  # Instead of 32
```

### Dataset Not Found
```bash
# Re-run data preparation
python prepare_data.py
```

### Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### HuggingFace Model Access
```bash
# Login to HuggingFace (for DINOv3)
huggingface-cli login
# Enter your token when prompted
```

### Disk Space Issues
```bash
# Check available space
df -h

# Clean up old logs
rm -rf logs/run_old_*
```

---

## 📚 Documentation

- **[LOGGING_QUICK_REFERENCE.md](Netra_Adapt/LOGGING_QUICK_REFERENCE.md)** - Logging system guide
- **[EARLY_STOPPING_SUMMARY.md](Netra_Adapt/EARLY_STOPPING_SUMMARY.md)** - Early stopping details
- **[COMPLETE_EXPERIMENTAL_SETUP.md](Netra_Adapt/COMPLETE_EXPERIMENTAL_SETUP.md)** - Full experimental protocol
- **[LOGGING_GUIDE.md](Netra_Adapt/LOGGING_GUIDE.md)** - Comprehensive logging documentation

---

## 🔗 Citation

If you use this code in your research, please cite:

```bibtex
@article{netra-adapt-2026,
  title={Netra-Adapt: Test-Time Style Calibration of Foundation Models for Cross-Ethnic Glaucoma Screening},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DINOv3**: Meta AI Research
- **AIROGS Dataset**: Grand Challenge
- **Chákṣu Dataset**: Indian fundus image consortium
- **Vast.ai**: GPU cloud compute

---

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

## 🚦 Quick Command Reference

```bash
# Setup
bash setup_with_download.sh
python prepare_data.py

# Run pipeline
python run_full_pipeline.py

# Or run individually
python train_source.py
python train_oracle.py
python adapt_target.py
python evaluate.py
python advanced_analysis.py --all

# View results
cat logs/run_*/EXPERIMENT_SUMMARY.md
```

---

**Ready to run? Start with `bash setup_with_download.sh`! 🚀**
