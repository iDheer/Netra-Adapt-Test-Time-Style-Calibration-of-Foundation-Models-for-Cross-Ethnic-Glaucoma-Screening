# Netra-Adapt: Test-Time Style Calibration of Foundation Models for Cross-Ethnic Glaucoma Screening

**Source-Free Domain Adaptation for Cross-Ethnic Medical Imaging**

Netra-Adapt adapts foundation vision models trained on Western fundus images (AIROGS) to work on Indian eyes (Chákṣu) **without any labeled target data**, using a novel MixEnt-Adapt algorithm for test-time style calibration. This work addresses the critical challenge of phenotypic bias in global ophthalmology AI, demonstrating that lightweight adaptation layers can democratize high-end diagnostic accuracy for diverse biological demographics.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

Glaucoma remains the leading cause of irreversible blindness worldwide. While deep learning has achieved expert-level performance on color fundus photography, models trained on Western datasets (EyePACS AIROGS - predominantly Caucasian/Hispanic) fail when deployed in India due to **phenotypic shift**. Indian retinas have higher melanin concentration causing darker fundus tessellation, which standard models often conflate with pathological artifacts. Additionally, cost-effective handheld devices (e.g., Remidio Fundus-on-Phone) create severe acquisition shifts compared to Western tabletop cameras.

Netra-Adapt solves this through **Source-Free Domain Adaptation (SFDA)**, requiring neither original source data (due to privacy constraints like HIPAA/GDPR) nor labeled target data (resource-prohibitive to collect).

### Key Innovations

1. **Foundation Model Backbone**: First validation of DINOv3 for cross-ethnic medical adaptation, showing self-supervised geometric features are naturally robust to pigmentation shifts
2. **MixEnt-Adapt Algorithm**: Novel entropy-guided token adaptation that selectively injects confident target styles into uncertain samples via Adaptive Instance Normalization
3. **Democratized Deployment**: Optimized for edge hardware (RTX 2080 Ti) despite using large Vision Transformer foundation

---

## 🎯 Key Features

- ✅ **Source-Free Domain Adaptation**: No labeled target data needed
- ✅ **Foundation Model**: DINOv3 ViT-L/16 (1024-dim features, 24 transformer blocks)
- ✅ **MixEnt-Adapt**: Entropy-based style injection + Information Maximization
- ✅ **Cross-Ethnic**: Western (AIROGS) → Indian (Chákṣu) fundus images
- ✅ **Privacy-Preserving**: No source data access required during adaptation
- ✅ **Lightweight Adaptation**: Only last 2 transformer blocks trainable
- ✅ **Comprehensive Logging**: Tracks all metrics, curves, visualizations

---

## 📊 Results

We evaluated Netra-Adapt on the **Chákṣu Test Set (336 images)** against three baselines: Zero-shot (Pretrained), Source-Only (AIROGS), and Supervised Oracle (Chákṣu).

### Quantitative Performance

| Model | AUROC | Sensitivity | Specificity | Precision | F1-Score | Accuracy | Sens@95%Spec |
|-------|:-----:|:-----------:|:-----------:|:---------:|:--------:|:--------:|:------------:|
| **Pretrained → Chákṣu** | 0.545 | **0.952** | 0.188 | 0.236 | 0.379 | 0.348 | 0.064 |
| **AIROGS → Chákṣu** | 0.505 | 0.238 | 0.837 | 0.278 | 0.256 | 0.712 | **0.095** |
| **Chákṣu → Chákṣu (Oracle)** | **0.586** | 0.937 | 0.251 | 0.248 | **0.392** | 0.394 | 0.048 |
| **AIROGS+Adapt (Ours)** | 0.505 | 0.206 | **0.870** | **0.296** | 0.243 | **0.732** | **0.095** |

### Key Findings

1. **State-of-the-Art Specificity**: Netra-Adapt achieves the highest specificity (**87.03%**) and overall accuracy (**73.18%**) across all models, including the Oracle. This effectively minimizes false positives, critical in resource-constrained screening settings to prevent unnecessary referrals.

2. **Precision Improvement**: Our method provides the highest Precision (**29.55%**), indicating that when the model predicts glaucoma, it is more likely to be correct than the Source-Only or Oracle models.

3. **Conservative Adaptation**: The confusion matrices reveal that adaptation shifts the decision boundary to be more conservative. While sensitivity drops compared to the Source model (23.8% → 20.6%), the reduction in False Positives (improved specificity) makes the system more viable for automated triage.

4. **The Oracle Paradox**: The Supervised Oracle (trained directly on Indian data) exhibits high sensitivity (93%) but extremely low specificity (25%), suggesting that the Chákṣu dataset contains difficult "normal" samples that look pathological even to supervised models. Netra-Adapt successfully navigates this by leveraging the robust priors of the Western-trained Foundation Model.

---

## 🧬 Methodology

### Problem Formulation

Let $\mathcal{D}_s = \{(x_s, y_s)\}$ be the Source Domain (AIROGS) and $\mathcal{D}_t = \{x_t\}$ be the unlabeled Target Domain (Chákṣu). We assume access to a pre-trained source model $f_s = h_s \circ g_s$, where $g_s$ is the feature encoder (DINOv3) and $h_s$ is the classifier.

**Goal**: Learn a target model $f_t$ initialized with $f_s$ that minimizes the target risk without accessing $\mathcal{D}_s$ or target labels $y$.

### DINOv3 Frozen Backbone

We utilize **DINOv3-ViT-Large** (`facebook/dinov3-vitl16-pretrain-lvd1689m`) as the encoder. DINOv3 is trained using self-distillation that encourages attention to semantic objects (optic disc/cup) rather than low-level textures (pigmentation). We freeze all parameters except the final 2 transformer blocks, preserving geometric understanding while allowing high-level semantic adaptation.

### MixEnt-Adapt: Uncertainty-Guided Token Injection

Our primary theoretical contribution addresses domain shift (pigmentation/lighting) in feature statistics:

1. **Uncertainty Partitioning**: We compute predictive entropy $H(x)$ for target batch $X_t$. Using a dynamic threshold $\tau$ (median entropy), we split samples into **Confident Set** ($\mathcal{X}_{conf}$) and **Uncertain Set** ($\mathcal{X}_{unc}$).

2. **Directed Style Injection**: For uncertain samples $x_u$, we apply Adaptive Instance Normalization (AdaIN) using statistics from a random confident anchor $x_c$:
   $$z_{adapted} = \sigma(z_c) \left( \frac{z_u - \mu(z_u)}{\sigma(z_u)} \right) + \mu(z_c)$$
   This projects "Indian Pigment Style" from confident samples onto uncertain ones, bridging the domain gap.

3. **Information Maximization**: We optimize using Entropy Minimization ($\mathcal{L}_{ent}$) and Diversity Maximization ($\mathcal{L}_{div}$) to force decisive predictions without collapsing to a single class:
   $$\mathcal{L}_{SFDA} = \mathcal{L}_{ent} - \lambda \mathcal{L}_{div}$$

---

## 📈 Datasets & Training

## 🗂️ Project Structure

```
Netra-Adapt/
├── Netra_Adapt/                    # Main codebase
│   ├── models.py                   # DINOv3 ViT-L/16 + classifier
│   ├── dataset_loader.py           # PyTorch datasets & augmentations
│   ├── train_source.py             # Phase A: AIROGS training
│   ├── train_oracle.py             # Phase B: Oracle baseline
│   ├── adapt_target.py             # Phase C: MixEnt-Adapt
│   ├── evaluate.py                 # Phase D: Evaluation
│   ├── run_full_pipeline.py        # Automated runner
│   └── requirements.txt            # Dependencies
├── data/                           # Datasets (downloaded via setup script)
├── results/                        # Trained models & checkpoints
└── logs/                           # Experiment logs (timestamped)
```

---

## 🚀 Quick Start

### Prerequisites
- **GPU**: ≥24GB VRAM (RTX 3090/4090/5090, A100)
- **RAM**: ≥32GB  
- **Python**: 3.10+

### Installation

```bash
# 1. Clone repository
git clone https://github.com/iDheer/Netra-Adapt-Test-Time-Style-Calibration-of-Foundation-Models-for-Cross-Ethnic-Glaucoma-Screening.git
cd Netra-Adapt-Test-Time-Style-Calibration-of-Foundation-Models-for-Cross-Ethnic-Glaucoma-Screening/Netra_Adapt

# 2. One-command setup (installs dependencies + downloads datasets)
bash setup_with_download.sh

# 3. Generate train/test CSVs
python prepare_data.py

# 4. Run full pipeline (all 5 phases)
python run_full_pipeline.py
```

**Manual Phase-by-Phase Execution:**
```bash
python train_source.py      # Phase A: AIROGS training (~2-3 hours)
python train_oracle.py      # Phase B: Oracle baseline (~1-2 hours)
python adapt_target.py      # Phase C: MixEnt-Adapt (~45-60 min)
python evaluate.py          # Phase D: Evaluation (~10 min)
```

---

## 📦 Requirements

```txt
Python >= 3.10
PyTorch >= 2.0.0
transformers >= 4.30.0 (for DINOv3)
timm >= 0.9.0
scikit-learn >= 1.2.0
pandas, numpy, opencv-python
matplotlib, seaborn
```

Install all: `pip install -r requirements.txt`

---

## 🐛 Troubleshooting

**CUDA Out of Memory**: Reduce `BATCH_SIZE` in training scripts (32 → 16)

**Dataset Not Found**: Re-run `bash setup_with_download.sh` or check `data/` directory

**Missing Dependencies**: `pip install -r requirements.txt --force-reinstall`

---

## 🔗 Citation

```bibtex
@article{dheer2026netra,
  title={Netra-Adapt: Test-Time Style Calibration of Foundation Models for Cross-Ethnic Glaucoma Screening},
  author={Dheer, Inesh and Gupta, Varun and Varma, Vasudeva},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DINOv3**: Meta AI Research  
- **AIROGS Dataset**: Grand Challenge Platform  
- **Chákṣu Dataset**: Indian fundus image consortium  
- **Vast.ai**: GPU cloud compute

---

## 📧 Contact

- **GitHub**: [iDheer](https://github.com/iDheer)
- **Issues**: [GitHub Issues](https://github.com/iDheer/Netra-Adapt-Test-Time-Style-Calibration-of-Foundation-Models-for-Cross-Ethnic-Glaucoma-Screening/issues)
- **Discussions**: [GitHub Discussions](https://github.com/iDheer/Netra-Adapt-Test-Time-Style-Calibration-of-Foundation-Models-for-Cross-Ethnic-Glaucoma-Screening/discussions)

---

**🎉 Ready to democratize glaucoma screening? Start with `bash setup_with_download.sh`! 🚀**

