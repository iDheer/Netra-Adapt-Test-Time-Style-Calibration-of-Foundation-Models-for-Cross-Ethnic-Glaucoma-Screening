# RAG-Based Glaucoma Screening - Complete Solution

## 📦 What Was Created

A complete **Retrieval-Augmented Generation (RAG)** system for glaucoma screening that uses similarity-based classification instead of model fine-tuning.

## 🗂️ Folder Structure

```
rag_glaucoma_screening/
├── prepare_data.py              # Creates CSV files from image folders
├── build_rag_database.py        # Extracts DINOv2 features, builds FAISS index
├── rag_retrieval.py             # Retrieval & classification logic
├── evaluate_rag.py              # Comprehensive evaluation (multiple configs)
├── run_rag_pipeline.py          # Full pipeline orchestrator
├── utils.py                     # Utilities (metrics, plotting, JSON)
├── requirements.txt             # Python dependencies
├── setup_from_upload.sh         # Vast.ai setup (datasets pre-uploaded)
├── setup_with_download.sh       # Setup with Kaggle downloads
├── README.md                    # Complete documentation
└── QUICK_START.md               # This file
```

## 🎯 How It Works

### Traditional Approach (Fine-tuning)
1. Train model on source domain (AIROGS - Western)
2. Adapt/fine-tune on target domain (Chákṣu train - Indian)
3. Predict on test set (Chákṣu test - Indian)
❌ **Problem**: Models can overfit to imaging artifacts, has access to target domain

### RAG Approach (Pure Zero-shot)
1. **Build database**: Extract DINOv3 features from AIROGS images ONLY (Western)
2. **Retrieve neighbors**: For each Chákṣu test image, find k most similar AIROGS images
3. **Aggregate labels**: Combine neighbor labels (weighted by similarity)
✅ **Advantage**: True zero-shot cross-ethnic test, interpretable, no target domain access

## 🚀 Running on Vast.ai

### Step 1: Upload to Vast.ai

**Option A: With Datasets Already on Vast.ai**
```bash
# SSH into Vast.ai instance
# Navigate to /workspace
cd /workspace

# Verify datasets exist
ls /workspace/data/AIROGS
ls /workspace/data/CHAKSHU

# Run setup
cd /workspace/rag_glaucoma_screening
bash setup_from_upload.sh
```

**Option B: Download Datasets via Kaggle**
```bash
# Set up Kaggle credentials first
mkdir -p ~/.kaggle
# Upload your kaggle.json to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Run setup with downloads
cd /workspace/rag_glaucoma_screening
bash setup_with_download.sh
```

### Step 2: Run Full Pipeline
```bash
cd /workspace/rag_glaucoma_screening
python run_rag_pipeline.py
```

**This will**:
1. Create CSV files from image folders (~30 seconds)
2. Extract features from all images (~20-40 minutes)
3. Build FAISS index (~30 seconds)
4. Evaluate all configurations (~10-20 minutes)

**Total time**: ~30-60 minutes

### Step 3: Check Results
```bash
# View summary table
cat /workspace/evaluation_results/summary_table.csv

# View best configuration
cd /workspace/evaluation_results
ls -lh k*_*/
```

## 📊 What Gets Evaluated

### Configurations Tested
- **k values**: [5, 10, 20, 50] neighbors
- **Aggregation methods**:
  - `majority_vote`: Simple voting
  - `weighted_vote`: Weight by similarity (recommended)
  - `mean_prob`: Mean of neighbor labels

**Total**: 12 configurations (4 k values × 3 methods)

### Metrics Reported
For each configuration:
- AUROC (Area Under ROC Curve)
- Accuracy
- Sensitivity (Recall for glaucoma class)
- Specificity
- Precision (per class)
- F1-Score (per class)
- Confusion Matrix

## 📈 Expected Results

Based on cross-ethnic evaluation (training on Western + Indian, testing on Indian):

| Configuration | Expected AUROC | Notes |
|--------------|----------------|-------|
| k=10, weighted_vote | **0.75-0.85** | Best balance |
| k=20, weighted_vote | 0.73-0.83 | More robust |
| k=50, weighted_vote | 0.70-0.80 | Smoother predictions |

**Hypothesis**: RAG should handle cross-ethnic transfer better than fine-tuned models because:
1. No overfitting to single dataset's imaging characteristics
2. Explicitly includes diverse ethnic examples in database
3. Classification based on actual similar cases, not learned features

## 📁 Output Structure

```
/workspace/
├── data/                        # Datasets
│   ├── AIROGS/
│   ├── CHAKSHU/
│   ├── airogs_train.csv
│   ├── airogs_test.csv
│   ├── chaksu_train_labeled.csv
│   ├── chaksu_train_unlabeled.csv
│   └── chaksu_test_labeled.csv
│
├── rag_database/                # Feature database
│   ├── faiss_index.bin         # Vector index (1024-dim)
│   ├── database_metadata.csv   # Image paths & labels
│   └── database_stats.json     # Statistics
│
└── evaluation_results/          # Evaluation outputs
    ├── summary_table.csv        # All configs compared
    ├── pipeline_summary.json    # Timing info
    ├── auroc_heatmap.png        # AUROC heatmap
    ├── accuracy_heatmap.png
    ├── sensitivity_specificity.png
    ├── all_metrics_comparison.png
    ├── best_config_metrics.png
    │
    ├── k5_majority_vote/        # Per-config results
    │   ├── metrics.json
    │   ├── roc_curve.png
    │   ├── confusion_matrix.png
    │   └── rag_predictions.csv
    │
    ├── k10_weighted_vote/       # Best config (likely)
    │   └── ...
    │
    └── ... (12 total folders)
```

## 🔍 Analyzing Results

### 1. View Overall Summary
```bash
cd /workspace/evaluation_results
cat summary_table.csv | column -t -s,
```

### 2. Check Best Configuration
```python
import pandas as pd
df = pd.read_csv('/workspace/evaluation_results/summary_table.csv')
best = df.loc[df['auroc'].idxmax()]
print(f"Best config: {best['config_name']}")
print(f"AUROC: {best['auroc']:.4f}")
print(f"Accuracy: {best['accuracy']:.4f}")
```

### 3. Analyze Predictions
```python
# Load predictions from best config
df_pred = pd.read_csv('/workspace/evaluation_results/k10_weighted_vote/rag_predictions.csv')

# Check errors
errors = df_pred[df_pred['true_label'] != df_pred['predicted_class']]
print(f"Error rate: {len(errors)/len(df_pred)*100:.2f}%")

# Analyze confidence
print(f"Mean probability: {df_pred['predicted_probability'].mean():.3f}")
print(f"Std probability: {df_pred['predicted_probability'].std():.3f}")
```

### 4. View Visualizations
```bash
# Download from Vast.ai or view directly
cd /workspace/evaluation_results
ls *.png

# Best config visualizations
ls k10_weighted_vote/*.png
```

## 🎨 Key Visualizations

1. **auroc_heatmap.png**: AUROC for all k values × aggregation methods
2. **accuracy_heatmap.png**: Accuracy heatmap
3. **sensitivity_specificity.png**: Trade-off curves for each method
4. **all_metrics_comparison.png**: Line plots showing how metrics change with k
5. **best_config_metrics.png**: Bar chart of all metrics for best configuration

Each configuration folder also has:
- **roc_curve.png**: ROC curve with AUROC score
- **confusion_matrix.png**: 2×2 confusion matrix

## 🛠️ Advanced Usage

### Run Single Configuration
```bash
python rag_retrieval.py \
    --k 20 \
    --aggregation weighted_vote \
    --test-csv /workspace/data/chaksu_test_labeled.csv \
    --output-dir /workspace/custom_results
```

### Test Custom k Values
```bash
python evaluate_rag.py \
    --k-values 15 25 35 \
    --aggregation-methods weighted_vote \
    --output-dir /workspace/custom_eval
```

### Rebuild Database Only
```bash
python build_rag_database.py
```

## 🔬 Technical Details

### Feature Extraction
- **Model**: DINOv3-large (facebook/dinov3-large)
- **Feature dim**: 1024 (from [CLS] token)
- **Batch size**: 32
- **Device**: Auto-detect CUDA/CPU

### Database Contents
- AIROGS train: 7,632 images (Western)
- AIROGS test: 1,908 images (Western)
- **Total**: ~9,540 Western images ONLY

**Note**: Chákṣu train is intentionally excluded to test pure zero-shot transfer!

### Classification
For each test image:
1. Extract features using DINOv2
2. Search FAISS index for k nearest neighbors
3. Aggregate neighbor labels (weighted by inverse distance)
4. Output probability of glaucoma

## 📊 Comparison with Fine-tuning

| Aspect | Fine-tuning | RAG (AIROGS-only) |
|--------|-------------|-------------------|
| Training time | Hours | None |
| Target domain access | Yes (Chákṣu train) | No |
| Risk of overfitting | High | None (zero-shot) |
| Interpretability | Black box | Can inspect neighbors |
| Adding new data | Must retrain | Just add to database |
| Cross-ethnic test | Semi-supervised | Pure zero-shot |

## 🚨 Troubleshooting

### Out of Memory
```bash
# Reduce batch size in build_rag_database.py
# Edit line: batch_size=32 → batch_size=16
```

### Slow Feature Extraction
- Use GPU instance (Vast.ai with CUDA)
- Install faiss-gpu instead of faiss-cpu
- Increase batch size if GPU memory allows

### CSV Files Not Found
```bash
python prepare_data.py
```

## 📞 Next Steps

1. **Run the pipeline** on Vast.ai
2. **Check summary_table.csv** for best configuration
3. **Compare with fine-tuning results** from main project
4. **Analyze retrieval patterns**: Which dataset's images are being retrieved most?
5. **Investigate failures**: When does RAG fail? What neighbors are retrieved?

## 💡 Research Questions to Explore

1. **Does RAG outperform fine-tuning** on cross-ethnic transfer?
2. **Which k value is optimal** for medical imaging?
3. **Does weighted voting help** compared to simple majority?
4. **What's the ethnic composition** of retrieved neighbors?
5. **Can we visualize** the feature space (t-SNE/UMAP)?

## 📚 Key Files to Review

- [README.md](README.md) - Complete documentation
- [build_rag_database.py](build_rag_database.py) - Feature extraction logic
- [rag_retrieval.py](rag_retrieval.py) - Classification algorithm
- [evaluate_rag.py](evaluate_rag.py) - Evaluation framework

---

**Created**: February 2026  
**Status**: ✅ Ready to run on Vast.ai
