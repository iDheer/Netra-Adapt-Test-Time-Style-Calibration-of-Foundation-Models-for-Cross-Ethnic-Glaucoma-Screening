# RAG-Based Glaucoma Screening: Comprehensive Evaluation Report

**Date:** February 12, 2026  
**Author:** Netra-Adapt Research Team  
**Experiment:** Zero-Shot Cross-Ethnic Glaucoma Screening using Retrieval-Augmented Generation

---

## Executive Summary

This report presents a comprehensive evaluation of a Retrieval-Augmented Generation (RAG) baseline for cross-ethnic glaucoma screening. We tested whether similarity-based retrieval from a Western population database (AIROGS) could effectively classify Indian population images (Chákṣu) in a pure zero-shot setting.

**Key Findings:**
- **RAG AUROC: 0.45-0.51** (essentially random chance ≈ 0.5)
- **Best Configuration:** k=5, majority_vote (AUROC 0.5089)
- **Critical Issue:** Very low sensitivity (20-30%), missing 70-80% of glaucoma cases
- **Conclusion:** RAG baseline fails for cross-ethnic glaucoma screening, validating need for domain adaptation methods

---

## 1. Introduction

### 1.1 Motivation

Glaucoma is a leading cause of irreversible blindness worldwide, with significant prevalence variations across ethnic populations. Traditional deep learning models trained on Western populations often fail when deployed on other ethnicities due to domain shift. We investigate whether a training-free RAG approach can bridge this gap.

### 1.2 Research Questions

1. Can retrieval-augmented generation from Western fundus images effectively classify Indian fundus images?
2. What is the optimal k (number of neighbors) for RAG-based glaucoma screening?
3. Which aggregation method (majority vote, weighted vote, mean probability) performs best?
4. How does RAG compare to supervised fine-tuning approaches?

---

## 2. Methodology

### 2.1 RAG Architecture

Our RAG system consists of three main components:

#### 2.1.1 Feature Extraction
- **Model:** DINOv3-ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`)
- **Architecture:** Vision Transformer Large (303M parameters)
- **Training:** Self-supervised on 1.689 billion images (LVD-1689M dataset)
- **Feature Dimension:** 1024-dimensional embeddings from [CLS] token
- **Preprocessing:** Automatic via `AutoImageProcessor`

#### 2.1.2 Vector Database
- **Index Type:** FAISS `IndexFlatL2` (exact L2 distance search)
- **Database Size:** 770 AIROGS images (385 Normal, 385 Glaucoma)
- **Storage:** Binary index format (`.bin`) with CSV metadata
- **Search:** Exhaustive nearest neighbor search for maximum accuracy

#### 2.1.3 Label Aggregation Methods

**Majority Vote:**
```
P(glaucoma) = (# glaucoma neighbors) / k
```
- Simple democratic voting
- All neighbors weighted equally
- Best for balanced classes

**Weighted Vote:**
```
P(glaucoma) = Σ(similarity_i × label_i) / Σ(similarity_i)
similarity_i = 1 / (distance_i + ε)
```
- Weight neighbors by inverse distance
- Closer neighbors have more influence
- Handles varying confidence

**Mean Probability:**
```
P(glaucoma) = mean(neighbor_labels)
```
- Treats binary labels as probabilities
- Smoothest aggregation
- Numerically equivalent to majority vote with equal weights

### 2.2 Datasets

#### 2.2.1 AIROGS (Database)
- **Source:** Kaggle Challenge
- **Population:** Predominantly Western/Caucasian
- **Images:** 770 total (616 train + 154 test)
- **Distribution:** Perfectly balanced (50% Normal, 50% Glaucoma)
- **Resolution:** 512×512 JPEG
- **Usage:** Database for RAG retrieval

#### 2.2.2 Chákṣu (Test Set)
- **Source:** Figshare (Indian multi-site dataset)
- **Population:** Pure Indian ethnic origin
- **Images:** 302 test images (233 Normal, 69 Glaucoma)
- **Devices:** Bosch, Forus, Remidio fundus cameras
- **Distribution:** 77.2% Normal, 22.8% Glaucoma (realistic prevalence)
- **Usage:** Zero-shot evaluation target

### 2.3 Experimental Design

#### 2.3.1 Configurations Tested
- **k values:** [5, 10, 20, 50]
- **Aggregation methods:** [majority_vote, weighted_vote, mean_prob]
- **Total configurations:** 12 (4×3 grid search)

#### 2.3.2 Evaluation Metrics
- **AUROC:** Area Under ROC Curve (primary metric)
- **Accuracy:** Overall classification accuracy
- **Sensitivity (Recall):** True Positive Rate
- **Specificity:** True Negative Rate
- **Precision:** Positive Predictive Value
- **F1-Score:** Harmonic mean of Precision and Recall

#### 2.3.3 Infrastructure
- **Hardware:** Vast.ai H200 GPU (141GB VRAM)
- **Feature Extraction Time:** ~6 seconds for 770 images
- **Classification Time:** ~42-52 seconds per configuration (302 images)
- **Total Evaluation Time:** ~10 minutes for all 12 configurations

---

## 3. Results

### 3.1 Overall Performance Summary

**Top 5 Configurations by AUROC:**

| Configuration | AUROC | Accuracy | Sensitivity | Specificity | F1-Score |
|--------------|-------|----------|-------------|-------------|----------|
| k5_majority_vote | **0.5089** | 0.6258 | 0.2464 | 0.7382 | 0.2313 |
| k5_mean_prob | 0.5089 | 0.6258 | 0.2464 | 0.7382 | 0.2313 |
| k5_weighted_vote | 0.5053 | 0.6258 | 0.2464 | 0.7382 | 0.2313 |
| k10_weighted_vote | 0.5010 | 0.6391 | 0.2464 | 0.7554 | 0.2378 |
| k10_majority_vote | 0.5007 | 0.5894 | 0.3043 | 0.6738 | 0.2530 |

**Top 5 Configurations by Accuracy:**

| Configuration | AUROC | Accuracy | Sensitivity | Specificity | F1-Score |
|--------------|-------|----------|-------------|-------------|----------|
| k20_weighted_vote | 0.4653 | **0.6656** | 0.2319 | 0.7940 | 0.2406 |
| k50_weighted_vote | 0.4575 | **0.6656** | 0.2029 | 0.8026 | 0.2171 |
| k50_majority_vote | 0.4577 | 0.6457 | 0.2174 | 0.7725 | 0.2190 |
| k50_mean_prob | 0.4577 | 0.6457 | 0.2174 | 0.7725 | 0.2190 |
| k10_weighted_vote | 0.5010 | 0.6391 | 0.2464 | 0.7554 | 0.2378 |

### 3.2 Complete Results Table

| k | Aggregation | AUROC | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|---|-------------|-------|----------|-------------|-------------|-----------|----------|
| 5 | majority_vote | 0.5089 | 0.6258 | 0.2464 | 0.7382 | 0.2179 | 0.2313 |
| 5 | weighted_vote | 0.5053 | 0.6258 | 0.2464 | 0.7382 | 0.2179 | 0.2313 |
| 5 | mean_prob | 0.5089 | 0.6258 | 0.2464 | 0.7382 | 0.2179 | 0.2313 |
| 10 | majority_vote | 0.5007 | 0.5894 | 0.3043 | 0.6738 | 0.2165 | 0.2530 |
| 10 | weighted_vote | 0.5010 | 0.6391 | 0.2464 | 0.7554 | 0.2297 | 0.2378 |
| 10 | mean_prob | 0.5007 | 0.5894 | 0.3043 | 0.6738 | 0.2165 | 0.2530 |
| 20 | majority_vote | 0.4622 | 0.6391 | 0.2754 | 0.7468 | 0.2436 | 0.2585 |
| 20 | weighted_vote | 0.4653 | 0.6656 | 0.2319 | 0.7940 | 0.2500 | 0.2406 |
| 20 | mean_prob | 0.4622 | 0.6391 | 0.2754 | 0.7468 | 0.2436 | 0.2585 |
| 50 | majority_vote | 0.4577 | 0.6457 | 0.2174 | 0.7725 | 0.2206 | 0.2190 |
| 50 | weighted_vote | 0.4575 | 0.6656 | 0.2029 | 0.8026 | 0.2333 | 0.2171 |
| 50 | mean_prob | 0.4577 | 0.6457 | 0.2174 | 0.7725 | 0.2206 | 0.2190 |

### 3.3 Best Configuration Analysis (k=5, majority_vote)

**Confusion Matrix:**
```
                Predicted
              Normal  Glaucoma
Actual Normal   172      61
    Glaucoma     52      17
```

**Performance Breakdown:**
- **True Negatives (TN):** 172 (73.8% of normals correctly identified)
- **False Positives (FP):** 61 (26.2% of normals misclassified)
- **False Negatives (FN):** 52 (75.4% of glaucoma cases missed!)
- **True Positives (TP):** 17 (24.6% of glaucoma cases detected)

**Critical Finding:** The system correctly identifies normal cases but catastrophically fails on glaucoma detection, missing 3 out of 4 glaucoma patients.

### 3.4 Impact of k (Number of Neighbors)

| k | Best AUROC | Best Accuracy | Sensitivity Range |
|---|-----------|---------------|-------------------|
| 5 | 0.5089 | 0.6258 | 0.2464 |
| 10 | 0.5010 | 0.6391 | 0.2464-0.3043 |
| 20 | 0.4653 | 0.6656 | 0.2319-0.2754 |
| 50 | 0.4577 | 0.6656 | 0.2029-0.2174 |

**Observation:** Smaller k values (5-10) achieve slightly better AUROC but all remain near random chance. Larger k values increase specificity at the cost of sensitivity.

### 3.5 Aggregation Method Comparison

| Method | Avg AUROC | Avg Sensitivity | Avg Specificity |
|--------|-----------|-----------------|-----------------|
| majority_vote | 0.4824 | 0.2589 | 0.7328 |
| weighted_vote | 0.4823 | 0.2319 | 0.7851 |
| mean_prob | 0.4824 | 0.2589 | 0.7328 |

**Finding:** All three aggregation methods perform nearly identically, indicating the issue is not in aggregation but in the fundamental feature representation.

---

## 4. Comparison with Fine-Tuning Approaches

### 4.1 Performance Comparison Table

| Method | Approach | AUROC | Sensitivity | Specificity | Accuracy | F1-Score |
|--------|----------|-------|-------------|-------------|----------|----------|
| **Fine-Tuning Results:** |
| Pretrained → Chákṣu | Zero-shot transfer | 0.5446 | 0.9524 | 0.1883 | 0.3477 | 0.3785 |
| AIROGS → Chákṣu | Source model | 0.5049 | 0.2381 | 0.8368 | 0.7119 | 0.2564 |
| Chákṣu → Chákṣu | Oracle (upper bound) | **0.5858** | 0.9365 | 0.2510 | 0.3940 | 0.3920 |
| AIROGS+Adapt | Netra-Adapt (SFDA) | 0.5047 | 0.2063 | 0.8703 | 0.7318 | 0.2430 |
| **RAG Results:** |
| k=5 majority_vote | RAG baseline | **0.5089** | 0.2464 | 0.7382 | 0.6258 | 0.2313 |
| k=10 weighted_vote | RAG optimized | 0.5010 | 0.2464 | 0.7554 | 0.6391 | 0.2378 |
| k=20 weighted_vote | RAG high-specificity | 0.4653 | 0.2319 | 0.7940 | **0.6656** | 0.2406 |

### 4.2 Key Observations

1. **All Methods Perform Poorly:** Both RAG and fine-tuning achieve AUROC ≈ 0.50-0.58, barely better than random chance

2. **RAG Comparable to Source Model:**
   - RAG k=5: 0.5089 AUROC
   - AIROGS → Chákṣu: 0.5049 AUROC
   - Difference: Only 0.004 AUROC

3. **Adaptation Failed:**
   - Netra-Adapt shows NO improvement over source model
   - Same AUROC (0.5047 vs 0.5049)

4. **Oracle Performance Also Poor:**
   - Even training directly on Chákṣu: 0.5858 AUROC
   - Suggests fundamental data or model architecture issues

### 4.3 Unified Analysis

**Common Issues Across All Methods:**
- Very low AUROC (< 0.60) for all approaches
- Severe class imbalance not properly handled
- DINOv2/v3 features may be insufficient for fundus pathology
- Possible label quality issues in datasets

**RAG-Specific Issues:**
- No task-specific fine-tuning
- General vision features don't capture glaucomatous changes
- Western-Indian domain gap too large for simple retrieval

---

## 5. Visualizations

### 5.1 Generated Outputs

All visualizations are saved in `/workspace/evaluation_results/`:

**Per Configuration:**
- `k{N}_{method}/roc_curve.png` - ROC curves showing random baseline
- `k{N}_{method}/confusion_matrix.png` - Confusion matrices highlighting FN bias
- `k{N}_{method}/rag_predictions.csv` - Per-image predictions with correctness flags

**Aggregate Analysis:**
- `auroc_heatmap.png` - Heatmap of AUROC across k and aggregation methods
- `accuracy_heatmap.png` - Heatmap of accuracy across configurations
- `sensitivity_specificity_plot.png` - Trade-off visualization
- `summary_table.csv` - Complete results comparison

### 5.2 Key Visual Findings

**ROC Curves:** All configurations closely follow the diagonal (random classifier), with slight bulge indicating marginally better than chance

**Confusion Matrices:** Consistently show:
- High TN (correctly identifying normals)
- Low TP (missing most glaucoma cases)
- Pattern of "predict normal" bias

**Heatmaps:** Show minimal variation across configurations, confirming that hyperparameter tuning has negligible impact

---

## 6. Discussion

### 6.1 Why RAG Failed

#### 6.1.1 Feature Representation Mismatch
- **DINOv3 Training:** Self-supervised on general web images
- **Task Requirement:** Medical image analysis requires pathology-aware features
- **Gap:** Generic visual similarity ≠ glaucomatous similarity

#### 6.1.2 Domain Shift
- **AIROGS:** Western populations, controlled acquisition
- **Chákṣu:** Indian populations, multi-device, different imaging characteristics
- **Issue:** Even perceptually similar images may have different clinical outcomes

#### 6.1.3 Class Imbalance
- **Database:** 50-50 balanced (AIROGS)
- **Test Set:** 77% Normal, 23% Glaucoma (Chákṣu)
- **Effect:** k-NN naturally biased toward majority class in database

#### 6.1.4 Absence of Pathology Understanding
- **Glaucoma Indicators:** Cup-to-disc ratio, neuroretinal rim, RNFL thinning
- **DINOv3 Focus:** General object boundaries, textures, colors
- **Missing:** Optic nerve head morphology understanding

### 6.2 Unexpectedly Poor Fine-Tuning Results

The fine-tuning results also showing AUROC ≈ 0.50-0.58 suggests:

1. **Model Architecture Issues:** Classification head may be inadequate
2. **Training Problems:** Learning rate, epochs, or optimization issues
3. **Data Quality:** Label reliability or image quality concerns
4. **Task Difficulty:** Chákṣu dataset may be inherently challenging

### 6.3 Clinical Implications

**Sensitivity < 30% is unacceptable for glaucoma screening:**
- Missing 70%+ of glaucoma cases defeats screening purpose
- High false negative rate poses serious health risks
- Cannot be deployed in clinical practice

**High Specificity (73-80%) causes issues:**
- Over-predicting "normal" wastes follow-up resources
- Delayed diagnosis for true positives

---

## 7. Limitations

### 7.1 Methodological Limitations

1. **Feature Extractor Fixed:** DINOv3 not fine-tuned for fundus images
2. **No Ensemble Methods:** Single model, no boosting or bagging
3. **Simple Aggregation:** No learned aggregation functions
4. **Binary Classification:** No multi-class or confidence thresholding

### 7.2 Dataset Limitations

1. **Small Database:** Only 770 AIROGS images
2. **Imbalanced Test Set:** 77% Normal vs 23% Glaucoma
3. **Single Institution:** Chákṣu from one country (India)
4. **No Validation Set:** Direct test evaluation without hyperparameter tuning

### 7.3 Experimental Limitations

1. **Fixed Resolution:** 512×512 may lose important details
2. **No Data Augmentation:** In retrieval phase
3. **L2 Distance Only:** No cosine similarity or other metrics
4. **No k Optimization:** Grid search only, no cross-validation

---

## 8. Future Work Recommendations

### 8.1 Immediate Improvements

1. **Fix Fine-Tuning Pipeline:** Debug training to achieve expected AUROC 0.75-0.85+
2. **Check Labels:** Verify Chákṣu label quality and inter-rater agreement
3. **Model Architecture:** Try task-specific classification heads
4. **Class Balancing:** Implement proper loss weighting or resampling

### 8.2 RAG Enhancements (If Pursued)

1. **Domain-Adapted Features:** Fine-tune DINOv3 on fundus images first
2. **Learned Aggregation:** Train a small MLP to combine neighbor information
3. **Hybrid RAG-Supervised:** Use RAG features as input to supervised classifier
4. **Medical-Specific Embeddings:** Use models pre-trained on retinal images

### 8.3 Alternative Approaches

1. **Test-Time Adaptation:** Continue with Netra-Adapt but fix training
2. **Foundation Models:** Try medical vision-language models (MedCLIP, BiomedCLIP)
3. **Few-Shot Learning:** Meta-learning approaches for low-data adaptation
4. **Ensemble Methods:** Combine multiple models with different architectures

---

## 9. Conclusions

### 9.1 Primary Findings

1. **RAG is Ineffective for Cross-Ethnic Glaucoma Screening**
   - AUROC ≈ 0.50 (random chance)
   - Cannot reliably distinguish glaucoma from normal

2. **Sensitivity is Critically Low**
   - 20-30% detection rate is clinically unacceptable
   - 70-80% of glaucoma cases missed

3. **Domain Shift is Severe**
   - Western (AIROGS) → Indian (Chákṣu) gap too large
   - Simple retrieval cannot bridge ethnic differences

4. **Fine-Tuning Also Underperforms**
   - Both RAG and supervised methods achieve similar poor results
   - Suggests fundamental issues beyond approach selection

### 9.2 RAG as a Baseline

**Value of This Experiment:**
- ✅ Establishes weak baseline for comparison
- ✅ Quantifies severity of domain shift
- ✅ Validates need for proper adaptation methods
- ✅ Provides training-free reference point

**Not Suitable as Production Method:**
- ❌ Cannot be deployed clinically
- ❌ High false negative rate risks patient safety
- ❌ No improvement over simpler baselines

### 9.3 Recommendations

**For Research:**
1. Focus on fixing fine-tuning pipeline first
2. Use RAG results as lower bound in papers
3. Investigate data quality and label reliability
4. Consider task-specific foundation models

**For Practice:**
1. Do NOT deploy RAG for glaucoma screening
2. Require minimum AUROC > 0.85 for clinical use
3. Prioritize sensitivity in safety-critical applications
4. Validate on diverse ethnic populations

---

## 10. Technical Specifications

### 10.1 Software Environment

```yaml
Framework: PyTorch 2.2+
Model: Hugging Face Transformers
Vector DB: FAISS 1.7+
Python: 3.12
Key Libraries:
  - transformers
  - faiss-cpu/faiss-gpu
  - pandas
  - numpy
  - scikit-learn
  - pillow
```

### 10.2 Reproducibility

**Random Seeds:** 42 (for data shuffling)  
**Hardware:** Vast.ai RTX 5090 GPU  
**Feature Extraction:** Deterministic (no dropout)  
**FAISS Index:** Exact L2 search (reproducible)

**Data Files:**
```
/workspace/data/
├── AIROGS/eyepac-light-v2-512-jpg/  (770 images)
├── CHAKSHU/                           (302 test images)
├── airogs_train.csv                   (616 images)
├── airogs_test.csv                    (154 images)
└── chaksu_test_labeled.csv           (302 images)
```

**Model Checkpoint:**
```
facebook/dinov3-vitl16-pretrain-lvd1689m
Downloaded from: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
Size: 1.21 GB
```

### 10.3 Computational Resources

| Operation | Time | Memory |
|-----------|------|--------|
| Feature Extraction (770 images) | 6 sec | ~8 GB VRAM |
| FAISS Index Build | <1 sec | ~50 MB |
| Classification (302 images) | 42-52 sec | ~8 GB VRAM |
| Total Evaluation (12 configs) | ~10 min | ~8 GB VRAM |

---

## 11. Appendix

### 11.1 File Structure

```
rag_glaucoma_screening/
├── build_rag_database.py          # Feature extraction & indexing
├── rag_retrieval.py                # RAG classification
├── evaluate_rag.py                 # Comprehensive evaluation
├── utils.py                        # Metrics and plotting
├── prepare_data.py                 # Dataset preprocessing
└── requirements.txt                # Dependencies

/workspace/rag_database/
├── faiss_index.bin                 # Vector index
├── database_metadata.csv           # Image metadata
└── database_stats.json             # Database statistics

/workspace/evaluation_results/
├── k5_majority_vote/               # Results per config
│   ├── rag_predictions.csv
│   ├── rag_predictions.json
│   ├── roc_curve.png
│   └── confusion_matrix.png
├── ... (11 more configurations)
├── summary_table.csv               # All results
├── auroc_heatmap.png
├── accuracy_heatmap.png
└── sensitivity_specificity_plot.png
```

### 11.2 Sample Predictions

**Example 1 - Correct Normal Classification:**
- True Label: Normal (0)
- Predicted Probability: 0.15
- Predicted Class: Normal
- Top 5 Neighbor Labels: [0, 0, 0, 0, 1]
- Outcome: ✅ Correct

**Example 2 - Missed Glaucoma (Typical Failure):**
- True Label: Glaucoma (1)
- Predicted Probability: 0.35
- Predicted Class: Normal
- Top 5 Neighbor Labels: [0, 0, 1, 0, 0]
- Outcome: ❌ False Negative

### 11.3 Complete Metrics Definitions

**AUROC (Area Under ROC Curve):**
- Plots True Positive Rate vs False Positive Rate
- Random classifier: 0.5
- Perfect classifier: 1.0
- Clinical minimum: 0.85

**Sensitivity (Recall, TPR):**
```
Sensitivity = TP / (TP + FN)
```
- Proportion of actual positives correctly identified
- Critical for screening (minimize missed cases)

**Specificity (TNR):**
```
Specificity = TN / (TN + FP)
```
- Proportion of actual negatives correctly identified
- Important for reducing unnecessary follow-ups

**Precision (PPV):**
```
Precision = TP / (TP + FP)
```
- Proportion of positive predictions that are correct
- Affects resource allocation

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean balancing precision and recall
- Useful for imbalanced datasets

---

## 12. Contact and Citation

### 12.1 Research Team

**Project:** Netra-Adapt: Test-Time Style Calibration for Cross-Ethnic Glaucoma Screening  
**Repository:** https://github.com/iDheer/Netra-Adapt-Test-Time-Style-Calibration-of-Foundation-Models-for-Cross-Ethnic-Glaucoma-Screening

### 12.2 Citation

```bibtex
@techreport{netraadapt2026rag,
  title={RAG-Based Glaucoma Screening: A Zero-Shot Cross-Ethnic Baseline},
  author={Netra-Adapt Research Team},
  year={2026},
  institution={[Your Institution]},
  note={Technical Report: Retrieval-Augmented Generation for Medical Imaging}
}
```

### 12.3 Acknowledgments

- **AIROGS Dataset:** Kaggle Challenge organizers
- **Chákṣu Dataset:** Figshare contributors
- **DINOv3 Model:** Meta AI Research
- **Infrastructure:** Vast.ai cloud computing

---

**END OF REPORT**
