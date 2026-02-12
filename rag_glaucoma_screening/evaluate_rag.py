"""
Comprehensive evaluation of RAG-based glaucoma screening
Tests multiple k values and aggregation methods
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rag_retrieval import RAGClassifier, evaluate_rag_classifier
from utils import (
    ensure_dir, save_json, calculate_metrics, print_metrics,
    plot_roc_curve, plot_confusion_matrix, NumpyEncoder
)


def evaluate_multiple_configurations(index_path, metadata_path, test_csv, 
                                     k_values=[5, 10, 20, 50], 
                                     aggregation_methods=['majority_vote', 'weighted_vote', 'mean_prob'],
                                     output_dir='./evaluation_results'):
    """
    Evaluate RAG classifier with multiple configurations
    
    Args:
        index_path: Path to FAISS index
        metadata_path: Path to metadata CSV
        test_csv: Path to test CSV
        k_values: List of k values to test
        aggregation_methods: List of aggregation methods to test
        output_dir: Output directory for results
    """
    print("\n" + "="*80)
    print("RAG-Based Glaucoma Screening - Comprehensive Evaluation")
    print("="*80)
    
    ensure_dir(output_dir)
    
    all_results = []
    
    # Test each configuration
    for k in k_values:
        for aggregation in aggregation_methods:
            config_name = f"k{k}_{aggregation}"
            print(f"\n{'='*80}")
            print(f"Testing Configuration: k={k}, aggregation={aggregation}")
            print(f"{'='*80}")
            
            # Create classifier with this configuration
            classifier = RAGClassifier(
                index_path=index_path,
                metadata_path=metadata_path,
                k=k,
                aggregation=aggregation
            )
            
            # Evaluate
            config_output_dir = os.path.join(output_dir, config_name)
            results = evaluate_rag_classifier(
                classifier,
                test_csv=test_csv,
                output_dir=config_output_dir
            )
            
            # Calculate metrics
            y_true = np.array(results['true_labels'])
            y_scores = np.array(results['predicted_probabilities'])
            
            metrics = calculate_metrics(y_true, y_scores, threshold=0.5)
            metrics['k'] = k
            metrics['aggregation_method'] = aggregation
            metrics['config_name'] = config_name
            
            # Print metrics
            print_metrics(metrics, title=f"Results: {config_name}")
            
            # Generate visualizations
            plot_roc_curve(
                y_true, y_scores,
                save_path=os.path.join(config_output_dir, 'roc_curve.png'),
                title=f"ROC Curve - {config_name}"
            )
            
            y_pred = (y_scores >= 0.5).astype(int)
            plot_confusion_matrix(
                y_true, y_pred,
                save_path=os.path.join(config_output_dir, 'confusion_matrix.png'),
                title=f"Confusion Matrix - {config_name}"
            )
            
            # Save metrics
            metrics_path = os.path.join(config_output_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                import json
                json.dump(metrics, f, indent=2, cls=NumpyEncoder)
            
            all_results.append(metrics)
    
    # Create comparison visualizations
    create_comparison_plots(all_results, output_dir)
    
    # Save summary table
    df_results = pd.DataFrame(all_results)
    summary_path = os.path.join(output_dir, 'summary_table.csv')
    df_results.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary table to {summary_path}")
    
    # Print final summary
    print("\n" + "="*80)
    print("Evaluation Complete - Summary")
    print("="*80)
    print("\nTop 5 Configurations by AUROC:")
    top_configs = df_results.nlargest(5, 'auroc')[['config_name', 'auroc', 'accuracy', 'sensitivity', 'specificity']]
    print(top_configs.to_string(index=False))
    
    print("\nTop 5 Configurations by Accuracy:")
    top_configs = df_results.nlargest(5, 'accuracy')[['config_name', 'auroc', 'accuracy', 'sensitivity', 'specificity']]
    print(top_configs.to_string(index=False))
    
    print(f"\nAll results saved to: {output_dir}")
    print("="*80 + "\n")
    
    return all_results


def create_comparison_plots(all_results, output_dir):
    """Create comparison plots across configurations"""
    df = pd.DataFrame(all_results)
    
    # 1. AUROC comparison heatmap
    plt.figure(figsize=(12, 6))
    pivot_auroc = df.pivot(index='k', columns='aggregation_method', values='auroc')
    sns.heatmap(pivot_auroc, annot=True, fmt='.4f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, cbar_kws={'label': 'AUROC'})
    plt.title('AUROC Comparison Across Configurations', fontsize=14, fontweight='bold')
    plt.xlabel('Aggregation Method')
    plt.ylabel('Number of Neighbors (k)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auroc_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved AUROC heatmap")
    
    # 2. Accuracy comparison heatmap
    plt.figure(figsize=(12, 6))
    pivot_acc = df.pivot(index='k', columns='aggregation_method', values='accuracy')
    sns.heatmap(pivot_acc, annot=True, fmt='.4f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, cbar_kws={'label': 'Accuracy'})
    plt.title('Accuracy Comparison Across Configurations', fontsize=14, fontweight='bold')
    plt.xlabel('Aggregation Method')
    plt.ylabel('Number of Neighbors (k)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved accuracy heatmap")
    
    # 3. Sensitivity-Specificity trade-off
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    aggregation_methods = df['aggregation_method'].unique()
    
    for idx, agg_method in enumerate(aggregation_methods):
        df_subset = df[df['aggregation_method'] == agg_method].sort_values('k')
        
        axes[idx].plot(df_subset['k'], df_subset['sensitivity'], 
                      marker='o', label='Sensitivity', linewidth=2)
        axes[idx].plot(df_subset['k'], df_subset['specificity'], 
                      marker='s', label='Specificity', linewidth=2)
        axes[idx].set_xlabel('Number of Neighbors (k)', fontsize=12)
        axes[idx].set_ylabel('Score', fontsize=12)
        axes[idx].set_title(f'{agg_method}', fontsize=13, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
        axes[idx].set_ylim([0, 1.05])
    
    plt.suptitle('Sensitivity-Specificity Trade-off', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_specificity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved sensitivity-specificity plot")
    
    # 4. Metrics comparison bar chart (best configuration)
    best_config = df.loc[df['auroc'].idxmax()]
    metrics_to_plot = ['auroc', 'accuracy', 'sensitivity', 'specificity', 
                       'precision_glaucoma', 'f1_glaucoma']
    values = [best_config[m] for m in metrics_to_plot]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(metrics_to_plot)), values, color='steelblue', alpha=0.8)
    plt.xticks(range(len(metrics_to_plot)), 
               ['AUROC', 'Accuracy', 'Sensitivity', 'Specificity', 
                'Precision', 'F1-Score'], rotation=45, ha='right')
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Best Configuration: {best_config["config_name"]}', 
              fontsize=14, fontweight='bold')
    plt.ylim([0, 1.05])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_config_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved best configuration metrics")
    
    # 5. All metrics line plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics_groups = [
        ('auroc', 'AUROC'),
        ('accuracy', 'Accuracy'),
        ('sensitivity', 'Sensitivity (Recall)'),
        ('specificity', 'Specificity')
    ]
    
    for idx, (metric, title) in enumerate(metrics_groups):
        ax = axes[idx // 2, idx % 2]
        
        for agg_method in aggregation_methods:
            df_subset = df[df['aggregation_method'] == agg_method].sort_values('k')
            ax.plot(df_subset['k'], df_subset[metric], 
                   marker='o', label=agg_method, linewidth=2)
        
        ax.set_xlabel('Number of Neighbors (k)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    plt.suptitle('Performance Metrics Across Configurations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved all metrics comparison")


def main():
    """Main evaluation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RAG-based glaucoma screening')
    parser.add_argument('--index', type=str, default='./rag_database/faiss_index.bin',
                       help='Path to FAISS index')
    parser.add_argument('--metadata', type=str, default='./rag_database/database_metadata.csv',
                       help='Path to metadata CSV')
    parser.add_argument('--test-csv', type=str, default='./data/chaksu_test_labeled.csv',
                       help='Path to test CSV')
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10, 20, 50],
                       help='List of k values to test')
    parser.add_argument('--aggregation-methods', type=str, nargs='+',
                       default=['majority_vote', 'weighted_vote', 'mean_prob'],
                       help='List of aggregation methods to test')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check for Vast.ai environment
    if os.path.exists('/workspace'):
        args.index = '/workspace/rag_database/faiss_index.bin'
        args.metadata = '/workspace/rag_database/database_metadata.csv'
        args.test_csv = '/workspace/data/chaksu_test_labeled.csv'
        args.output_dir = '/workspace/evaluation_results'
    
    # Run comprehensive evaluation
    results = evaluate_multiple_configurations(
        index_path=args.index,
        metadata_path=args.metadata,
        test_csv=args.test_csv,
        k_values=args.k_values,
        aggregation_methods=args.aggregation_methods,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
