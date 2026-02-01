"""
evaluate.py - Comprehensive Model Evaluation for Netra-Adapt

Evaluates all trained models on the labeled Chákṣu test set:
- Phase A: Source model (baseline from AIROGS)
- Phase B: Oracle model (upper bound with Chákṣu labels)
- Phase C: Netra-Adapt (source-free adapted model)

Metrics (Research Paper Standard):
- AUROC: Area Under ROC Curve
- Sensitivity/Recall: True Positive Rate
- Specificity: True Negative Rate
- Precision: Positive Predictive Value
- F1-Score: Harmonic mean of Precision/Recall
- Sens@95: Sensitivity at 95% Specificity (clinically relevant)

Visualizations:
- ROC Curves (all models on same plot)
- Confusion Matrices
- Comparison Bar Charts
- Results saved to CSV
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, 
                             precision_recall_fscore_support, accuracy_score)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models import NetraModel
from dataset_loader import GlaucomaDataset, get_transforms

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_CSV = "/workspace/data/processed_csvs/chaksu_labeled.csv"
RESULTS_DIR = "/workspace/results/evaluation"

# Model paths
MODELS = {
    "Source (AIROGS)": "/workspace/results/Source_AIROGS/model.pth",
    "Oracle (Supervised)": "/workspace/results/Oracle_Chaksu/oracle_model.pth",
    "Netra-Adapt (SFDA)": "/workspace/results/Netra_Adapt/adapted_model.pth",
}

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate(model_path, name):
    """Evaluate a single model on the test set with comprehensive metrics."""
    
    if not os.path.exists(model_path):
        print(f"  [SKIP] Model not found: {model_path}")
        return None
    
    # Load model
    model = NetraModel(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Load test data
    dataset = GlaucomaDataset(TEST_CSV, transform=get_transforms(is_training=False))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  Evaluating", leave=False):
            images = images.to(DEVICE)
            
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of glaucoma
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.float().cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Filter out any invalid labels (label=-1)
    valid_mask = all_labels >= 0
    all_labels = all_labels[valid_mask]
    all_probs = all_probs[valid_mask]
    
    if len(all_labels) == 0:
        print(f"  [ERROR] No valid test samples")
        return None
    
    # Calculate comprehensive metrics
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        print(f"  [ERROR] Cannot compute AUROC (possibly single class)")
        return None
    
    # Get predictions at optimal threshold (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    predictions = (all_probs >= optimal_threshold).astype(int)
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = accuracy_score(all_labels, predictions)
    
    # Sensitivity at 95% Specificity
    valid_indices = np.where(fpr <= 0.05)[0]
    sens_at_95 = tpr[valid_indices[-1]] if len(valid_indices) > 0 else 0.0
    
    # Store for ROC curve plotting
    metrics = {
        'auroc': auroc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'accuracy': accuracy,
        'sens_at_95': sens_at_95,
        'confusion_matrix': confusion_matrix(all_labels, predictions),
        'fpr': fpr,
        'tpr': tpr,
        'predictions': predictions,
        'labels': all_labels,
        'probs': all_probs
    }
    
    return metrics


def plot_roc_curves(all_results):
    """Plot ROC curves for all models on same figure"""
    plt.figure(figsize=(10, 8))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for (name, metrics), color in zip(all_results.items(), colors):
        if metrics and 'fpr' in metrics:
            plt.plot(metrics['fpr'], metrics['tpr'], 
                    label=f"{name} (AUROC={metrics['auroc']:.3f})",
                    linewidth=2.5, color=color)
    
    # Diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random (AUROC=0.500)')
    
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold')
    plt.title('ROC Curves: Cross-Ethnic Glaucoma Screening', fontsize=15, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    
    plt.savefig(f"{RESULTS_DIR}/roc_curves.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{RESULTS_DIR}/roc_curves.pdf", bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved ROC curves to {RESULTS_DIR}/roc_curves.png")


def plot_confusion_matrices(all_results):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, metrics) in enumerate(all_results.items()):
        if metrics and 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Glaucoma'],
                       yticklabels=['Normal', 'Glaucoma'],
                       ax=axes[idx], cbar=True, square=True,
                       annot_kws={'fontsize': 14, 'fontweight': 'bold'})
            
            axes[idx].set_title(name, fontsize=13, fontweight='bold', pad=10)
            axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{RESULTS_DIR}/confusion_matrices.pdf", bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrices to {RESULTS_DIR}/confusion_matrices.png")


def plot_metrics_comparison(all_results):
    """Plot bar chart comparing all metrics across models"""
    metrics_to_plot = ['auroc', 'sensitivity', 'specificity', 'precision', 'f1', 'sens_at_95']
    metric_names = ['AUROC', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'Sens@95%Spec']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx]
        
        models = list(all_results.keys())
        values = [all_results[m][metric] if all_results[m] else 0 for m in models]
        
        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel(name, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    plt.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{RESULTS_DIR}/metrics_comparison.pdf", bbox_inches='tight')
    plt.close()
    print(f"✓ Saved metrics comparison to {RESULTS_DIR}/metrics_comparison.png")


def save_results_table(all_results):
    """Save all metrics to CSV for LaTeX tables"""
    rows = []
    
    for name, metrics in all_results.items():
        if metrics:
            rows.append({
                'Model': name,
                'AUROC': f"{metrics['auroc']:.4f}",
                'Sensitivity': f"{metrics['sensitivity']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Sens@95%Spec': f"{metrics['sens_at_95']:.4f}"
            })
    
    df = pd.DataFrame(rows)
    csv_path = f"{RESULTS_DIR}/results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved results table to {csv_path}")
    
    # Also save LaTeX-friendly version
    latex_path = f"{RESULTS_DIR}/results_table.tex"
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))
    print(f"✓ Saved LaTeX table to {latex_path}")


def main():
    print("\n" + "="*70)
    print("   NETRA-ADAPT: COMPREHENSIVE EVALUATION")
    print("   Test Set: Chákṣu (Indian Eyes, n=228)")
    print("="*70)
    
    # Validate test data exists
    if not os.path.exists(TEST_CSV):
        print(f"\n[ERROR] Test CSV not found: {TEST_CSV}")
        print("        Run prepare_data.py first!")
        return
    
    all_results = {}
    
    for name, path in MODELS.items():
        print(f"\n{name}")
        print("-" * 70)
        
        metrics = evaluate(path, name)
        
        if metrics:
            all_results[name] = metrics
            
            print(f"  AUROC:        {metrics['auroc']:.4f} ({metrics['auroc']*100:.1f}%)")
            print(f"  Sensitivity:  {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.1f}%)")
            print(f"  Specificity:  {metrics['specificity']:.4f} ({metrics['specificity']*100:.1f}%)")
            print(f"  Precision:    {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
            print(f"  F1-Score:     {metrics['f1']:.4f}")
            print(f"  Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
            print(f"  Sens@95Spec:  {metrics['sens_at_95']:.4f} ({metrics['sens_at_95']*100:.1f}%)")
        else:
            all_results[name] = None
    
    # Generate all visualizations
    print("\n" + "="*70)
    print("   GENERATING VISUALIZATIONS...")
    print("="*70)
    
    plot_roc_curves(all_results)
    plot_confusion_matrices(all_results)
    plot_metrics_comparison(all_results)
    save_results_table(all_results)
    
    print("\n" + "="*70)
    print("   EVALUATION COMPLETE")
    print(f"   All results saved to: {RESULTS_DIR}/")
    print("="*70)


if __name__ == "__main__":
    main()
