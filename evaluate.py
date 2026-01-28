"""
evaluate.py - Model Evaluation for Netra-Adapt

Evaluates all trained models on the labeled Chákṣu test set:
- Phase A: Source model (baseline from AIROGS)
- Phase B: Oracle model (upper bound with Chákṣu labels)
- Phase C: Netra-Adapt (source-free adapted model)

Metrics:
- AUROC: Area Under ROC Curve
- Sens@95: Sensitivity at 95% Specificity (clinically relevant)
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from models import NetraModel
from dataset_loader import GlaucomaDataset, get_transforms

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_CSV = "/workspace/Netra_Adapt/data/processed_csvs/chaksu_labeled.csv"

# Model paths
MODELS = {
    "Phase A: Source Baseline (AIROGS)": "/workspace/Netra_Adapt/results/Source_AIROGS/model.pth",
    "Phase B: Oracle (Chákṣu Supervised)": "/workspace/Netra_Adapt/results/Oracle_Chaksu/oracle_model.pth",
    "Phase C: Netra-Adapt (SFDA)": "/workspace/Netra_Adapt/results/Netra_Adapt/adapted_model.pth",
}


def evaluate(model_path, name):
    """Evaluate a single model on the test set."""
    
    if not os.path.exists(model_path):
        print(f"  [SKIP] Model not found: {model_path}")
        return None, None
    
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
        print(f"  [ERROR] No valid labels found!")
        return None, None
    
    # Calculate metrics
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        print(f"  [ERROR] Cannot compute AUROC (possibly single class)")
        return None, None
    
    # Sensitivity at 95% Specificity
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    # Find the threshold where specificity >= 95% (FPR <= 5%)
    valid_indices = np.where(fpr <= 0.05)[0]
    sens_at_95 = tpr[valid_indices[-1]] if len(valid_indices) > 0 else 0.0
    
    return auroc, sens_at_95


def main():
    print("\n" + "="*70)
    print("   NETRA-ADAPT EVALUATION RESULTS")
    print("   Test Set: Chákṣu (Indian Eyes)")
    print("="*70)
    
    # Validate test data exists
    if not os.path.exists(TEST_CSV):
        print(f"\n[ERROR] Test CSV not found: {TEST_CSV}")
        print("        Run prepare_data.py first!")
        return
    
    results = []
    
    for name, path in MODELS.items():
        print(f"\n{name}")
        print("-" * 50)
        
        auroc, sens_at_95 = evaluate(path, name)
        
        if auroc is not None:
            print(f"  AUROC:    {auroc:.4f} ({auroc*100:.1f}%)")
            print(f"  Sens@95:  {sens_at_95:.4f} ({sens_at_95*100:.1f}%)")
            results.append({
                "name": name,
                "auroc": auroc,
                "sens_at_95": sens_at_95
            })
    
    # Summary table
    if len(results) > 1:
        print("\n" + "="*70)
        print("   SUMMARY TABLE")
        print("="*70)
        print(f"{'Model':<45} {'AUROC':>10} {'Sens@95':>10}")
        print("-"*70)
        for r in results:
            print(f"{r['name']:<45} {r['auroc']:>10.4f} {r['sens_at_95']:>10.4f}")
        print("="*70)
        
        # Improvement over baseline
        if len(results) >= 2:
            baseline = results[0]['auroc']
            adapted = results[-1]['auroc']
            improvement = (adapted - baseline) * 100
            print(f"\n📈 Improvement from adaptation: +{improvement:.1f}% AUROC")


if __name__ == "__main__":
    main()