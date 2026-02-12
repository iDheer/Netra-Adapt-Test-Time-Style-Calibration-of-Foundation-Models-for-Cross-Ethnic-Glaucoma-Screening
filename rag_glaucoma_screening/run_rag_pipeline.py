"""
RAG-based Glaucoma Screening - Full Pipeline Orchestrator
Runs the complete workflow: data prep → database build → evaluation
"""
import os
import sys
import time
import subprocess
from datetime import datetime
from utils import ensure_dir, save_json


def run_command(command, description):
    """Run a shell command and handle errors"""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Command: {command}")
    print()
    
    start_time = time.time()
    result = subprocess.run(command, shell=True)
    elapsed_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n✗ ERROR: {description} failed!")
        sys.exit(1)
    
    print(f"\n✓ {description} completed in {elapsed_time:.2f} seconds")
    return elapsed_time


def main():
    """Run the complete RAG pipeline"""
    pipeline_start = time.time()
    
    print("\n" + "="*80)
    print("RAG-BASED GLAUCOMA SCREENING - FULL PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Determine environment
    if os.path.exists('/workspace'):
        work_dir = '/workspace/rag_glaucoma_screening'
        env_name = 'Vast.ai'
    else:
        work_dir = os.path.dirname(os.path.abspath(__file__))
        env_name = 'Local'
    
    print(f"Environment: {env_name}")
    print(f"Working directory: {work_dir}")
    
    os.chdir(work_dir)
    
    # Track timing for each step
    timings = {}
    
    # Step 1: Prepare data
    timings['data_preparation'] = run_command(
        "python prepare_data.py",
        "Data Preparation (Creating CSVs)"
    )
    
    # Step 2: Build RAG database
    timings['database_build'] = run_command(
        "python build_rag_database.py",
        "Building RAG Database (Feature Extraction)"
    )
    
    # Step 3: Run comprehensive evaluation
    timings['evaluation'] = run_command(
        "python evaluate_rag.py",
        "Comprehensive Evaluation (Multiple Configurations)"
    )
    
    # Calculate total time
    total_time = time.time() - pipeline_start
    
    # Generate pipeline summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nTotal pipeline time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    print("\nStep-by-step timing:")
    print(f"  1. Data Preparation:    {timings['data_preparation']/60:.2f} min")
    print(f"  2. Database Build:      {timings['database_build']/60:.2f} min")
    print(f"  3. Evaluation:          {timings['evaluation']/60:.2f} min")
    
    # Save pipeline summary
    summary = {
        'pipeline': 'RAG-based Glaucoma Screening',
        'environment': env_name,
        'start_time': datetime.fromtimestamp(pipeline_start).strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'step_timings': {
            'data_preparation_seconds': timings['data_preparation'],
            'database_build_seconds': timings['database_build'],
            'evaluation_seconds': timings['evaluation']
        }
    }
    
    # Determine output directory
    if os.path.exists('/workspace'):
        output_dir = '/workspace/evaluation_results'
    else:
        output_dir = './evaluation_results'
    
    ensure_dir(output_dir)
    summary_path = os.path.join(output_dir, 'pipeline_summary.json')
    save_json(summary, summary_path)
    
    print(f"\n✓ Pipeline summary saved to: {summary_path}")
    
    # Print results location
    print("\n" + "="*80)
    print("RESULTS LOCATION")
    print("="*80)
    
    if os.path.exists('/workspace'):
        print("\nRAG Database:")
        print("  /workspace/rag_database/")
        print("    - faiss_index.bin")
        print("    - database_metadata.csv")
        print("    - database_stats.json")
        print("\nEvaluation Results:")
        print("  /workspace/evaluation_results/")
        print("    - summary_table.csv")
        print("    - auroc_heatmap.png")
        print("    - accuracy_heatmap.png")
        print("    - sensitivity_specificity.png")
        print("    - all_metrics_comparison.png")
        print("    - best_config_metrics.png")
        print("\nPer-configuration results:")
        print("  /workspace/evaluation_results/k{k}_{aggregation}/")
        print("    - metrics.json")
        print("    - roc_curve.png")
        print("    - confusion_matrix.png")
        print("    - rag_predictions.csv")
    else:
        print("\nRAG Database:")
        print("  ./rag_database/")
        print("\nEvaluation Results:")
        print("  ./evaluation_results/")
    
    print("\n" + "="*80)
    print("Check the summary_table.csv for a comparison of all configurations!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
