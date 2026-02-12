"""
Utility functions for RAG-based glaucoma screening
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)


def get_project_root():
    """Get the project root directory"""
    return os.path.dirname(os.path.abspath(__file__))


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)
    return directory


def save_json(data, filepath):
    """Save dictionary as JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved JSON to {filepath}")


def load_json(filepath):
    """Load JSON file as dictionary"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_roc_curve(y_true, y_scores, save_path, title="ROC Curve"):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUROC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ROC curve to {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path, title="Confusion Matrix"):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Glaucoma'],
                yticklabels=['Normal', 'Glaucoma'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix to {save_path}")


def calculate_metrics(y_true, y_scores, threshold=0.5):
    """Calculate comprehensive classification metrics"""
    y_pred = (y_scores >= threshold).astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_scores)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )
    
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    metrics = {
        'accuracy': float(accuracy),
        'auroc': float(auroc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision_normal': float(precision[0]),
        'precision_glaucoma': float(precision[1]),
        'recall_normal': float(recall[0]),
        'recall_glaucoma': float(recall[1]),
        'f1_normal': float(f1[0]),
        'f1_glaucoma': float(f1[1]),
        'support_normal': int(support[0]),
        'support_glaucoma': int(support[1]),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'threshold': float(threshold)
    }
    
    return metrics


def print_metrics(metrics, title="Metrics"):
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    print(f"AUROC:          {metrics['auroc']:.4f}")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Sensitivity:    {metrics['sensitivity']:.4f}")
    print(f"Specificity:    {metrics['specificity']:.4f}")
    print(f"\nPer-Class Precision:")
    print(f"  Normal:       {metrics['precision_normal']:.4f}")
    print(f"  Glaucoma:     {metrics['precision_glaucoma']:.4f}")
    print(f"\nPer-Class Recall:")
    print(f"  Normal:       {metrics['recall_normal']:.4f}")
    print(f"  Glaucoma:     {metrics['recall_glaucoma']:.4f}")
    print(f"\nPer-Class F1-Score:")
    print(f"  Normal:       {metrics['f1_normal']:.4f}")
    print(f"  Glaucoma:     {metrics['f1_glaucoma']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['true_negatives']:>4}  FP: {metrics['false_positives']:>4}")
    print(f"  FN: {metrics['false_negatives']:>4}  TP: {metrics['true_positives']:>4}")
    print(f"{'='*60}\n")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
