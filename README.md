# Netra-Adapt: Source-Free Disentangled Token Adaptation for Cross-Ethnic Glaucoma Screening

**Team:** TorchTwins  
**Authors:** Inesh Dheer, Varun Gupta  
**Affiliation:** International Institute of Information Technology, Hyderabad (IIIT-H)

---

## 📋 Project Overview

**Netra-Adapt** is a Source-Free Domain Adaptation (SFDA) framework designed to address **phenotypic bias** in global ophthalmology AI. 

Standard glaucoma screening models trained on Western datasets (e.g., EyePACS AIROGS) exhibit high false-positive rates when deployed in India due to:
1.  **Phenotypic Shift:** Darker retinal pigmentation (fundus tessellation) in Indian eyes mimics pathology features in standard models.
2.  **Acquisition Shift:** The use of handheld "Fundus-on-Phone" devices introduces glare and artifacts absent in tabletop cameras.

**Our Solution:** We leverage **DINOv3 (ViT-Large)** as a frozen foundation model backbone and introduce **"MixEnt-Adapt,"** a novel uncertainty-guided style injection layer. This allows the model to adapt to the Indian context using only unlabeled target data, without requiring access to the original source data or Indian ground-truth labels.

---

## 📂 Repository Structure

* `config.py`: **The Control Center.** Manages hardware profiles (RTX 5090 vs 2080 Ti), dataset selection (Light vs Full), and model hyperparameters.
* `dataset_manager.py`: **Auto-Downloader.** Automatically fetches AIROGS (Kaggle), Chákṣu (Figshare), and handles file extraction.
* `dataset_loader.py`: **Preprocessing Pipeline.** Implements robust circle-cropping to handle "Fundus-on-Phone" image artifacts and dimension mismatches.
* `netra_model.py`: **Model Architecture.** Implements the DINOv3 backbone and the custom `MixEntLayer` logic.
* `main_train.py`: **Training Engine.** Executes the 3-Stage training pipeline (Head Training $\rightarrow$ Adaptation $\rightarrow$ Partial Fine-Tuning).
* `evaluate.py`: **Evaluation Suite.** Calculates AUROC, Accuracy, and **Expected Calibration Error (ECE)**, and generates Attention Heatmaps.
* `baselines.py`: Implementation of comparison methods (SHOT, Tent) for benchmarking.
* `requirements.txt`: List of Python dependencies.

---

## 🛠️ Prerequisites & Setup

### 1. Python Environment
Ensure you have Python 3.8+ installed. Install the dependencies:
```bash
pip install -r requirements.txt




### 2. Hugging Face Access (CRITICAL)

This project uses DINOv3, which is a gated model.

1. Visit [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).
2. Accept the license agreement.
3. Login via terminal using your Access Token:

```bash
huggingface-cli login
```
### 3. Kaggle API

To download the AIROGS dataset automatically:

1. Go to your [Kaggle Account Settings](https://www.kaggle.com/settings) → Create New API Token.
2. Place `kaggle.json` in:
   - **Linux/Mac:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<User>\.kaggle\kaggle.json`

---

## ⚙️ Configuration

Open `config.py` to tailor the project to your hardware:

### Hardware Profiles

**High-End (RTX 3090 / 4090 / 5090):**
```python
GPU_PROFILE = "5090"  # Batch Size 32, BF16 Precision, No Checkpointing
```

**Edge / Lower VRAM (RTX 2080 Ti / 3060):**
```python
GPU_PROFILE = "2080ti"  # Batch Size 4, FP16, Gradient Checkpointing ON
```

### Dataset Mode

- `DATASET_MODE = "light"`: Downloads AIROGS Light v2 (~3.5GB). Recommended for development.
- `DATASET_MODE = "full"`: Downloads Full AIROGS (~60GB). Required for SOTA reproduction.

---

## 🚀 Usage

### 1. Training

Run the main training script. It will check for datasets (and download them if missing) before starting the 3-Stage pipeline.

```bash
python main_train.py
```

**What happens:**
- **Stage 1:** Trains Classifier Head on AIROGS.
- **Stage 2:** Adapts to Chákṣu using unsupervised MixEnt.
- **Stage 3:** Unfreezes last 2 transformer blocks for final refinement.
- **Output:** Saves weights to `./checkpoints/netra_adapt_dinov3.pth`.

### 2. Evaluation

Run the evaluation script to generate metrics and visualizations.

```bash
python evaluate.py
```

**Output:**
- Prints AUROC, Accuracy, and ECE.
- Generates `heatmap_chakshu_sample.png`: Side-by-side comparison of the original image vs. model attention.

---

## 📊 Results Summary

| Method | Source | Target | AUROC | ECE (Calibration) |
|--------|--------|--------|-------|-------------------|
| Baseline (Source-Only) | AIROGS | Chákṣu | 0.652 | 0.245 |
| Netra-Adapt (Ours) | AIROGS | Chákṣu | 0.884 | 0.082 |

---

## 📜 Citation

If you use this code or methodology, please cite:

```bibtex
@article{dheer2026netraadapt,
  title={Netra-Adapt: Source-Free Disentangled Token Adaptation for Cross-Ethnic Glaucoma Screening via Foundation Models},
  author={Dheer, Inesh and Gupta, Varun and Varma, Vasudeva},
  journal={preprint},
  year={2026}
}
```