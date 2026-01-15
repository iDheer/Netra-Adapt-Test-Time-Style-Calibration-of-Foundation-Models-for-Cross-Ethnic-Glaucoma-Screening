"""Evaluation suite for Netra-Adapt.

Calculates AUROC, Accuracy, and Expected Calibration Error (ECE)
for glaucoma screening models.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from config import cfg
from dataset_manager import DatasetManager
from dataset_loader import NetraDataset
from torch.utils.data import DataLoader
from netra_model import NetraDinoV3
import os
from tqdm import tqdm


def compute_ece(probs, targets, n_bins=10):
    """
    Expected Calibration Error (ECE)
    Critical for Medical AI: Checks if '90% confident' actually means '90% accuracy'.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        # Samples in this bin
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(targets[in_bin] == (probs[in_bin] > 0.5))
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def evaluate_model(model_path, method_name):
    # Load Model
    model = NetraDinoV3().to(cfg.DEVICE)
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        print(f"⚠️ Weights for {method_name} not found. Skipping.")
        return
        
    # Get Data
    dm = DatasetManager()
    _, df_chakshu = dm.get_dfs()
    # Locate images (simplified logic)
    chakshu_root = os.path.join(cfg.DATA_ROOT, "chakshu/images")
    if not os.path.exists(chakshu_root): # Fallback search
         for r,d,f in os.walk(cfg.DATA_ROOT):
             if "images" in d: chakshu_root = os.path.join(r, "images"); break

    ds = NetraDataset(df_chakshu, chakshu_root, mode='eval', dataset_type='chakshu')
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for img, lbl in tqdm(loader, desc=f"Eval {method_name}"):
            img = img.to(cfg.DEVICE)
            logits = model(img)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(lbl.numpy())
            
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Metrics
    auc = roc_auc_score(all_targets, all_probs)
    acc = accuracy_score(all_targets, all_probs > 0.5)
    ece = compute_ece(all_probs, all_targets)
    
    print(f"\n📊 {method_name} Results:")
    print(f"   AUROC: {auc:.4f}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   ECE (Lower is better): {ece:.4f}")
    
    return auc, ece

if __name__ == "__main__":
    # 1. Source Only (Baseline)
    evaluate_model(f"{cfg.OUTPUT_DIR}/netra_adapt_dinov3_stage1.pth", "Baseline (Source-Only)")
    
    # 2. Netra-Adapt (Ours)
    evaluate_model(f"{cfg.OUTPUT_DIR}/netra_adapt_dinov3.pth", "Netra-Adapt (Ours)")
    
    # Note: To test TENT/SHOT, you would need to run a training loop using 'baselines.py' 
    # and save those checkpoints first.