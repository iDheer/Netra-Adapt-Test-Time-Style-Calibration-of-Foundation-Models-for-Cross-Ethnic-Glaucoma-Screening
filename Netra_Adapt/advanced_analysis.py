"""
advanced_analysis.py - Advanced Research Visualizations for Netra-Adapt

Provides deep analysis and publication-ready visualizations:
1. Feature Space Visualization (t-SNE/UMAP)
2. Grad-CAM Heatmaps (Model Interpretability)
3. Per-Camera Performance Analysis
4. Calibration Curves
5. Error Analysis
6. Statistical Significance Tests

Usage:
    python advanced_analysis.py --model_path /path/to/model.pth --name "Model Name"
    python advanced_analysis.py --all  # Analyze all three models
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from sklearn.calibration import calibration_curve
import cv2

from models import NetraModel
from dataset_loader import GlaucomaDataset, get_transforms

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_CSV = "/workspace/data/processed_csvs/chaksu_labeled.csv"
RESULTS_DIR = "/workspace/results/advanced_analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability"""
    
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        
        # Hook to the last transformer block
        target_layer = model.backbone.layer[-1]
        target_layer.register_forward_hook(self.save_feature_maps)
        target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_feature_maps(self, module, input, output):
        self.feature_maps = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, image, target_class):
        """Generate CAM for given image and target class"""
        self.model.eval()
        
        # Forward pass
        logits = self.model(image)
        
        # Backward pass
        self.model.zero_grad()
        class_loss = logits[0, target_class]
        class_loss.backward()
        
        # Generate CAM
        gradients = self.gradients[0].mean(dim=0)  # Average over patches
        feature_maps = self.feature_maps[0]
        
        cam = (gradients.unsqueeze(0) * feature_maps).sum(dim=1)
        cam = F.relu(cam)  # Only positive contributions
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy()


def extract_features(model, dataloader):
    """Extract features from penultimate layer for visualization"""
    model.eval()
    all_features = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(DEVICE)
            
            # Get features from penultimate layer
            features = model.backbone(images)
            features = features.mean(dim=1)  # Average pool over patches
            
            # Get predictions
            logits = model.head(features)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.vstack(all_features), np.array(all_labels), np.array(all_probs)


def plot_tsne_umap(features, labels, probs, name):
    """Visualize feature space using t-SNE and UMAP"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # t-SNE
    print(f"  Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_tsne = tsne.fit_transform(features)
    
    # UMAP
    print(f"  Computing UMAP...")
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    features_umap = reducer.fit_transform(features)
    
    # Plot t-SNE
    scatter1 = axes[0].scatter(features_tsne[:, 0], features_tsne[:, 1],
                              c=probs, cmap='RdYlGn_r', s=50, alpha=0.7,
                              edgecolors='black', linewidth=0.5)
    axes[0].set_title('t-SNE Projection', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE Component 1', fontsize=11)
    axes[0].set_ylabel('t-SNE Component 2', fontsize=11)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Glaucoma Probability', fontsize=10)
    
    # Plot UMAP
    scatter2 = axes[1].scatter(features_umap[:, 0], features_umap[:, 1],
                              c=probs, cmap='RdYlGn_r', s=50, alpha=0.7,
                              edgecolors='black', linewidth=0.5)
    axes[1].set_title('UMAP Projection', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('UMAP Component 1', fontsize=11)
    axes[1].set_ylabel('UMAP Component 2', fontsize=11)
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Glaucoma Probability', fontsize=10)
    
    plt.suptitle(f'Feature Space: {name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = f"{RESULTS_DIR}/feature_space_{name.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def plot_calibration_curve(labels, probs, name):
    """Plot calibration curve to assess probability calibration"""
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels, probs, n_bins=10, strategy='uniform'
    )
    
    plt.figure(figsize=(8, 8))
    
    # Plot calibration curve
    plt.plot(mean_predicted_value, fraction_of_positives, 
             marker='o', linewidth=2, label=name, color='#e74c3c', markersize=8)
    
    # Perfect calibration reference
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    plt.xlabel('Mean Predicted Probability', fontsize=13, fontweight='bold')
    plt.ylabel('Fraction of Positives', fontsize=13, fontweight='bold')
    plt.title(f'Calibration Curve: {name}', fontsize=15, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.tight_layout()
    
    save_path = f"{RESULTS_DIR}/calibration_{name.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved calibration curve to {save_path}")


def per_camera_analysis(csv_path, probs, labels):
    """Analyze performance per camera type (Bosch/Forus/Remidio)"""
    
    df = pd.read_csv(csv_path)
    df = df[df['label'] >= 0].reset_index(drop=True)  # Remove unlabeled
    df['probability'] = probs
    df['prediction'] = (probs >= 0.5).astype(int)
    df['correct'] = (df['prediction'] == labels).astype(int)
    
    # Extract camera type from path
    df['camera'] = df['image_path'].apply(lambda x: 
        'Bosch' if 'Bosch' in x else 
        'Forus' if 'Forus' in x else 
        'Remidio' if 'Remidio' in x else 'Unknown'
    )
    
    # Calculate per-camera metrics
    camera_stats = df.groupby('camera').agg({
        'correct': 'mean',
        'probability': 'mean',
        'label': lambda x: (x == 1).sum() / len(x)  # Positive rate
    }).reset_index()
    
    camera_stats.columns = ['Camera', 'Accuracy', 'Avg_Prob', 'Positive_Rate']
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cameras = camera_stats['Camera'].tolist()
    
    # Accuracy
    axes[0].bar(cameras, camera_stats['Accuracy'], color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8)
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Per-Camera Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Average Probability
    axes[1].bar(cameras, camera_stats['Avg_Prob'], color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8)
    axes[1].set_ylabel('Avg Glaucoma Prob', fontsize=12, fontweight='bold')
    axes[1].set_title('Per-Camera Avg Probability', fontsize=13, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)
    
    # Positive Rate
    axes[2].bar(cameras, camera_stats['Positive_Rate'], color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8)
    axes[2].set_ylabel('Glaucoma Rate', fontsize=12, fontweight='bold')
    axes[2].set_title('Per-Camera Glaucoma Rate', fontsize=13, fontweight='bold')
    axes[2].set_ylim([0, 1])
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{RESULTS_DIR}/per_camera_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved per-camera analysis to {save_path}")
    
    return camera_stats


def statistical_tests(results_dict):
    """Perform statistical significance tests between models"""
    
    print("\n" + "="*70)
    print("   STATISTICAL SIGNIFICANCE TESTS")
    print("="*70)
    
    models = list(results_dict.keys())
    
    # McNemar's test (paired test for classification)
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1, model2 = models[i], models[j]
            
            preds1 = results_dict[model1]['predictions']
            preds2 = results_dict[model2]['predictions']
            labels = results_dict[model1]['labels']
            
            # Build contingency table
            correct1 = (preds1 == labels).astype(int)
            correct2 = (preds2 == labels).astype(int)
            
            n01 = np.sum((correct1 == 0) & (correct2 == 1))
            n10 = np.sum((correct1 == 1) & (correct2 == 0))
            
            # McNemar's test
            if n01 + n10 > 0:
                statistic = (abs(n01 - n10) - 1)**2 / (n01 + n10)
                p_value = 1 - stats.chi2.cdf(statistic, df=1)
                
                print(f"\n{model1} vs {model2}")
                print(f"  McNemar statistic: {statistic:.4f}")
                print(f"  p-value: {p_value:.4f}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
            else:
                print(f"\n{model1} vs {model2}: No disagreements found")


def analyze_model(model_path, name):
    """Run all advanced analyses for a single model"""
    
    print(f"\n{'='*70}")
    print(f"   ADVANCED ANALYSIS: {name}")
    print(f"{'='*70}")
    
    # Load model
    model = NetraModel(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Load data
    dataset = GlaucomaDataset(TEST_CSV, transform=get_transforms(is_training=False))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Extract features
    print("\n1. Feature Extraction...")
    features, labels, probs = extract_features(model, loader)
    
    # Remove invalid labels
    valid_mask = labels >= 0
    features = features[valid_mask]
    labels = labels[valid_mask]
    probs = probs[valid_mask]
    predictions = (probs >= 0.5).astype(int)
    
    # t-SNE/UMAP visualization
    print("\n2. Feature Space Visualization...")
    plot_tsne_umap(features, labels, probs, name)
    
    # Calibration curve
    print("\n3. Calibration Analysis...")
    plot_calibration_curve(labels, probs, name)
    
    # Per-camera analysis
    print("\n4. Per-Camera Performance...")
    camera_stats = per_camera_analysis(TEST_CSV, probs, labels)
    print(camera_stats.to_string(index=False))
    
    return {
        'name': name,
        'predictions': predictions,
        'labels': labels,
        'probs': probs,
        'features': features
    }


def main():
    parser = argparse.ArgumentParser(description='Advanced Analysis for Netra-Adapt')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--name', type=str, help='Model name for plots')
    parser.add_argument('--all', action='store_true', help='Analyze all three models')
    args = parser.parse_args()
    
    if args.all:
        # Analyze all models
        models = {
            "Source (AIROGS)": "/workspace/results/Source_AIROGS/model.pth",
            "Oracle (Supervised)": "/workspace/results/Oracle_Chaksu/oracle_model.pth",
            "Netra-Adapt (SFDA)": "/workspace/results/Netra_Adapt/adapted_model.pth",
        }
        
        results = {}
        for name, path in models.items():
            if os.path.exists(path):
                results[name] = analyze_model(path, name)
        
        # Statistical tests
        if len(results) > 1:
            statistical_tests(results)
    
    elif args.model_path and args.name:
        analyze_model(args.model_path, args.name)
    
    else:
        print("Usage:")
        print("  python advanced_analysis.py --all")
        print("  python advanced_analysis.py --model_path /path/to/model.pth --name 'Model Name'")
    
    print(f"\n{'='*70}")
    print(f"   ANALYSIS COMPLETE")
    print(f"   Results saved to: {RESULTS_DIR}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
