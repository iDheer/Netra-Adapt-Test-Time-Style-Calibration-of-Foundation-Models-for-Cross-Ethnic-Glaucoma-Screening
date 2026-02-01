"""
Wrapper Script: Run Full Netra-Adapt Pipeline with Comprehensive Logging

This script runs the complete experimental pipeline and generates a final summary report.
All metrics, visualizations, and hyperparameters are automatically logged.

Usage:
    python run_full_pipeline.py
    
Output:
    logs/run_YYYY-MM-DD_HH-MM-SS/
    ├── experiment_log.txt
    ├── metadata.json
    ├── EXPERIMENT_SUMMARY.md
    ├── 01_source_training/
    ├── 02_oracle_training/
    ├── 03_adaptation/
    ├── 04_evaluation/
    └── 05_advanced_analysis/
"""

import sys
import subprocess
import time
from datetime import datetime
from training_logger import get_logger, reset_logger

def run_script(script_name, description):
    """Run a Python script and track execution time."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        elapsed_time = time.time() - start_time
        print(f"\n✅ {description} completed in {elapsed_time/60:.1f} minutes")
        return True, elapsed_time
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed_time/60:.1f} minutes")
        print(f"   Error: {e}")
        return False, elapsed_time

def main():
    print("\n" + "="*80)
    print("   NETRA-ADAPT: FULL EXPERIMENTAL PIPELINE")
    print("   Cross-Ethnic Glaucoma Screening with Test-Time Adaptation")
    print("="*80)
    print(f"\nExperiment Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize logger
    reset_logger()  # Start fresh
    exp_logger = get_logger()
    
    print(f"\n📊 Logging to: {exp_logger.run_dir}")
    print(f"   Run ID: {exp_logger.timestamp}\n")
    
    total_start_time = time.time()
    results = {}
    
    # Step 1: Prepare Data
    success, elapsed = run_script("prepare_data.py", "Data Preparation")
    results["prepare_data"] = {"success": success, "time": elapsed}
    if not success:
        print("\n❌ Pipeline aborted due to data preparation failure")
        return
    
    # Step 2: Train Source Model (AIROGS)
    success, elapsed = run_script("train_source.py", "Phase A: Source Training (AIROGS)")
    results["train_source"] = {"success": success, "time": elapsed}
    if not success:
        print("\n❌ Pipeline aborted due to source training failure")
        return
    
    # Step 3: Train Oracle Model (Chákṣu Supervised)
    success, elapsed = run_script("train_oracle.py", "Phase B: Oracle Training (Chákṣu)")
    results["train_oracle"] = {"success": success, "time": elapsed}
    # Continue even if oracle fails (it's just a baseline)
    
    # Step 4: Adapt to Target Domain (SFDA)
    success, elapsed = run_script("adapt_target.py", "Phase C: Source-Free Domain Adaptation")
    results["adapt_target"] = {"success": success, "time": elapsed}
    if not success:
        print("\n❌ Pipeline aborted due to adaptation failure")
        return
    
    # Step 5: Evaluate All Models
    success, elapsed = run_script("evaluate.py", "Phase D: Model Evaluation")
    results["evaluate"] = {"success": success, "time": elapsed}
    if not success:
        print("\n⚠️ Evaluation failed, but continuing to summary")
    
    # Step 6: Advanced Analysis (optional, don't fail if it errors)
    print(f"\n{'='*80}")
    print("Running: Phase E: Advanced Analysis (Optional)")
    print("Script: advanced_analysis.py")
    print(f"{'='*80}\n")
    try:
        success, elapsed = run_script("advanced_analysis.py", "Phase E: Advanced Analysis")
        results["advanced_analysis"] = {"success": success, "time": elapsed}
    except Exception as e:
        print(f"\n⚠️ Advanced analysis failed (optional): {e}")
        results["advanced_analysis"] = {"success": False, "time": 0}
    
    # Generate Final Summary
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("   PIPELINE EXECUTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal Execution Time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"\nPhase Results:")
    for phase, result in results.items():
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"  {phase:20s}: {status:12s} ({result['time']/60:.1f} min)")
    
    # Generate comprehensive summary report
    print(f"\n{'='*80}")
    print("Generating Experiment Summary...")
    print(f"{'='*80}\n")
    
    summary_path = exp_logger.generate_summary_report()
    
    print(f"\n{'='*80}")
    print("   🎉 EXPERIMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📊 All results saved to: {exp_logger.run_dir}")
    print(f"\n📄 Summary Report: {summary_path}")
    print(f"\nYou can now:")
    print(f"  1. View experiment_log.txt for detailed logs")
    print(f"  2. Open EXPERIMENT_SUMMARY.md for a comprehensive report")
    print(f"  3. Check individual phase directories for detailed outputs")
    print(f"  4. Use metadata.json for automated analysis\n")

if __name__ == "__main__":
    main()
