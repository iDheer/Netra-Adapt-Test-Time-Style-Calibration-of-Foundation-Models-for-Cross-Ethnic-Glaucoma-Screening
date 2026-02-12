# RAG-Based Glaucoma Screening

A Retrieval-Augmented Generation (RAG) approach for cross-ethnic glaucoma screening using DINOv2 features and similarity-based classification.

## 🎯 Overview

This project implements a **zero-shot RAG-based classification system** for glaucoma detection. Instead of fine-tuning models, we:

1. **Extract features** from AIROGS images (Western) using DINOv3-large
2. **Build a vector database** using FAISS for efficient similarity search
3. **Classify Chákṣu test images** (Indian) by retrieving similar Western examples

**Key Advantage**: Pure zero-shot cross-ethnic transfer - no training or adaptation required!

## 🗂️ Project Structure

```
rag_glaucoma_screening/
├── prepare_data.py              # Create CSVs from image folders
├── build_rag_database.py        # Extract features & build FAISS index
├── rag_retrieval.py             # Retrieval & classification logic
├── evaluate_rag.py              # Comprehensive evaluation
├── run_rag_pipeline.py          # Full pipeline orchestrator
├── utils.py                     # Shared utilities
├── requirements.txt             # Python dependencies
├── setup_from_upload.sh         # Setup for Vast.ai (pre-downloaded data)
├── setup_with_download.sh       # Setup with Kaggle downloads
└── README.md                    # This file
```

## 📊 Datasets

### RAG Database
- **AIROGS train**: 7,632 Western fundus images (labeled)
- **AIROGS test**: 1,908 Western fundus images (labeled)

**Total database**: ~9,540 Western images ONLY

**Design choice**: We intentionally use ONLY Western images to test pure zero-shot cross-ethnic generalization!

### Test Set
- **Chákṣu test**: 302 Indian fundus images (labeled)
  - **Goal**: Evaluate cross-ethnic generalization

## 🚀 Quick Start (Vast.ai)

### Prerequisites
1. Vast.ai instance with GPU (NVIDIA RTX 3090 or better recommended)
2. Datasets already uploaded to `/workspace/data/` with structure:
   ```
   /workspace/data/
   ├── AIROGS/
   │   ├── train/NRG/, train/RG/
   │   └── test/NRG/, test/RG/
   └── CHAKSHU/
       ├── train/NRG/, train/RG/
       ├── train_unlabelled/
       └── test/NRG/, test/RG/
   ```

### Run Complete Pipeline

```bash
# Setup environment
cd /workspace/rag_glaucoma_screening
bash setup_from_upload.sh

# Run full pipeline (data prep → build database → evaluate)
python run_rag_pipeline.py
```

**Expected runtime**: ~30-60 minutes depending on GPU

## 📋 Step-by-Step Execution

### 1. Data Preparation
```bash
python prepare_data.py
```
Creates 5 CSV files:
- `airogs_train.csv`, `airogs_test.csv`
- `chaksu_train_labeled.csv`, `chaksu_train_unlabeled.csv`, `chaksu_test_labeled.csv`

### 2. Build RAG Database
```bash
python build_rag_database.py
```
- Extracts DINOv2 features from all images in database
- Builds FAISS index for fast similarity search
- Saves to `/workspace/rag_database/`

**Output files**:
- `faiss_index.bin` - Vector index (dimension: 1024)
- `database_metadata.csv` - Image paths and labels
- `database_stats.json` - Database statistics

### 3. Run Evaluation
```bash
python evaluate_rag.py
```
Tests multiple configurations:
- **k values**: [5, 10, 20, 50] neighbors
- **Aggregation methods**:
  - `majority_vote`: Simple majority voting
  - `weighted_vote`: Weight by similarity (inverse distance)
  - `mean_prob`: Mean label probability

**Output**: `/workspace/evaluation_results/`

## 📈 Evaluation Outputs

### Summary Files
- `summary_table.csv` - All configurations with metrics
- `pipeline_summary.json` - Timing and execution details

### Visualizations
- `auroc_heatmap.png` - AUROC across all configurations
- `accuracy_heatmap.png` - Accuracy heatmap
- `sensitivity_specificity.png` - Trade-off analysis
- `all_metrics_comparison.png` - Line plots for all metrics
- `best_config_metrics.png` - Bar chart of best configuration

### Per-Configuration Results
Each configuration (e.g., `k10_weighted_vote/`) contains:
- `metrics.json` - All metrics
- `roc_curve.png` - ROC curve with AUROC
- `confusion_matrix.png` - Confusion matrix
- `rag_predictions.csv` - Predictions for all test images

## 🔧 Advanced Usage

### Custom Configuration
```bash
python evaluate_rag.py \
    --k-values 15 25 35 \
    --aggregation-methods weighted_vote \
    --output-dir ./custom_results
```

### Single Configuration Test
```bash
python rag_retrieval.py \
    --k 20 \
    --aggregation weighted_vote \
    --test-csv /workspace/data/chaksu_test_labeled.csv \
    --output-dir ./test_results
```

### Build Database Only
```bash
python build_rag_database.py
```

## 📊 Metrics Reported

For each configuration:
- **AUROC**: Area under ROC curve
- **Accuracy**: Overall classification accuracy
- **Sensitivity**: True positive rate (recall for glaucoma)
- **Specificity**: True negative rate
- **Precision**: Per-class precision
- **F1-Score**: Per-class F1 scores
- **Confusion Matrix**: TP, TN, FP, FN

## 🛠️ Technical Details

### Feature Extraction
- **Model**: DINOv3-large (facebook/dinov3-large)
- **Features**: [CLS] token embeddings (1024-dim)
- **Batch size**: 32 images per batch
- **Device**: Auto-detect CUDA/CPU

### FAISS Index
- **Distance metric**: L2 (Euclidean distance)
- **Index type**: Flat (exact search)
- **Can scale to**: Millions of vectors

### Aggregation Methods

1. **Majority Vote**
   ```python
   prob_glaucoma = count(glaucoma_neighbors) / k
   ```

2. **Weighted Vote** (Recommended)
   ```python
   similarity = 1 / (distance + epsilon)
   prob_glaucoma = sum(similarity * label) / sum(similarity)
   ```

3. **Mean Probability**
   ```python
   prob_glaucoma = mean(neighbor_labels)
   ```

## 🔬 Research Motivation

**Why RAG for Medical Imaging?**

1. **Pure zero-shot**: Trained on Western (AIROGS) only, tested on Indian (Chákṣu)
2. **Interpretability**: Can inspect retrieved neighbors for each prediction
3. **Cross-ethnic generalization**: Tests if foundation models capture universal glaucoma features
4. **Robustness**: Cannot overfit to target domain artifacts
5. **Flexibility**: Easy to add new images to database

**Comparison with Fine-tuning**:
- Fine-tuning: Train on source → Adapt on target train → Test on target test
- RAG: Build database from source only → Test directly on target test
- RAG = True zero-shot, Fine-tuning = Has access to target domain during adaptation

## 📦 Dependencies

```
torch==2.1.0
torchvision==0.16.0
transformers==4.36.0
Pillow==10.1.0
numpy==1.24.3
pandas==2.1.3
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
tqdm==4.66.1
faiss-cpu==1.7.4  # Use faiss-gpu for faster search
kaggle==1.5.16
```

## 🚨 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in build_rag_database.py
python build_rag_database.py  # Default batch_size=32
```

### Slow Feature Extraction
- Use GPU instance (Vast.ai with CUDA)
- Install `faiss-gpu` instead of `faiss-cpu`
- Increase batch size if memory allows

### Missing CSV Files
```bash
# Re-run data preparation
python prepare_data.py
```

### Kaggle API Errors
```bash
# Set up credentials
mkdir -p ~/.kaggle
# Place kaggle.json in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## 📝 Local Setup (Alternative)

If running locally instead of Vast.ai:

```bash
# Clone and setup
cd rag_glaucoma_screening
bash setup_with_download.sh  # Downloads datasets from Kaggle

# Run pipeline
python run_rag_pipeline.py
```

**Note**: Kaggle credentials required in `~/.kaggle/kaggle.json`

## 🎯 Expected Results

Based on cross-ethnic evaluation (Western → Indian):

| Configuration | Expected AUROC | Expected Accuracy |
|--------------|----------------|-------------------|
| k=10, weighted_vote | **0.75-0.85** | **0.72-0.80** |
| k=20, weighted_vote | 0.73-0.83 | 0.70-0.78 |
| k=50, majority_vote | 0.70-0.80 | 0.68-0.76 |

**Hypothesis**: RAG should outperform fine-tuned models on cross-ethnic transfer by leveraging diverse training examples rather than memorizing dataset-specific features.

## 🔍 Interpreting Results

### Check Best Configuration
```bash
# View summary table
cat /workspace/evaluation_results/summary_table.csv | column -t -s,

# View best configuration
ls -lh /workspace/evaluation_results/k*_*/metrics.json
```

### Visualize Predictions
```python
import pandas as pd
df = pd.read_csv('/workspace/evaluation_results/k10_weighted_vote/rag_predictions.csv')
print(df.head(20))
```

### Analyze Failure Cases
```python
# Find misclassified images
df = pd.read_csv('/workspace/evaluation_results/k10_weighted_vote/rag_predictions.csv')
errors = df[df['true_label'] != df['predicted_class']]
print(f"Error rate: {len(errors)/len(df)*100:.2f}%")
print(errors)
```

## 📚 Citation

If you use this code, please cite:

```bibtex
@article{netra-rag-2024,
  title={RAG-Based Cross-Ethnic Glaucoma Screening with Foundation Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the authors.

## 🙏 Acknowledgments

- **DINOv2**: Meta AI Research
- **AIROGS Dataset**: Grand Challenge
- **Chákṣu Dataset**: Indian glaucoma screening initiative
- **FAISS**: Meta AI Research (vector similarity search)

---

**Last Updated**: February 2026
