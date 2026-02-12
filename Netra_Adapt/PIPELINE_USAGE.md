# Pipeline Execution Scripts

## Quick Start

Run the complete pipeline with a single command:

```bash
cd /workspace/Netra_Adapt

# Option 1: Full featured with progress tracking and results summary
bash run_everything.sh

# Option 2: Simple version (minimal output)
bash run_simple.sh

# Option 3: Python wrapper (with advanced logging)
python run_full_pipeline.py
```

## What Each Script Does

### `run_everything.sh` ⭐ RECOMMENDED
**Full-featured bash script with:**
- ✅ Colored output and progress bars
- ✅ Prerequisite checks (CUDA, data, packages)
- ✅ Automatic data preparation if needed
- ✅ Real-time progress tracking with time estimates
- ✅ Final results summary with AUROC comparison
- ✅ Comprehensive error handling
- ✅ Lists all output files and next steps

**Usage:**
```bash
bash run_everything.sh
```

**Time:** ~4-5 hours total (depends on GPU)

---

### `run_simple.sh`
**Minimal bash script for quick execution:**
- Basic progress messages
- No fancy formatting
- Still runs full pipeline
- Good for automation/scripting

**Usage:**
```bash
bash run_simple.sh
```

---

### `run_full_pipeline.py`
**Python wrapper with advanced logging:**
- Structured experiment logging
- JSON metadata export
- Markdown summary generation
- Phase-by-phase timing

**Usage:**
```bash
python run_full_pipeline.py
```

---

## Individual Phase Execution

If you want to run phases separately:

```bash
# Phase 0: Data preparation (if needed)
python prepare_data.py

# Phase A: Source training (AIROGS)
python train_source.py

# Phase B: Oracle training (Chákṣu)
python train_oracle.py

# Phase C: Test-Time Adaptation
python adapt_target.py

# Phase D: Evaluation
python evaluate.py
```

---

## Expected Timeline

| Phase | Script | Time | Output |
|-------|--------|------|--------|
| **Data Prep** | `prepare_data.py` | ~5 min | CSV files |
| **Source Training** | `train_source.py` | ~2-3 hours | Source model |
| **Oracle Training** | `train_oracle.py` | ~1-2 hours | Oracle model |
| **Adaptation** | `adapt_target.py` | ~30-60 min | Adapted model |
| **Evaluation** | `evaluate.py` | ~10 min | Results + figures |
| **TOTAL** | - | **~4-5 hours** | Full results |

*Times are approximate and depend on GPU (tested on RTX 4090)*

---

## Output Structure

After running, you'll have:

```
/workspace/results/
├── Source_AIROGS/
│   ├── model.pth              # Source model weights
│   ├── best_model.pth
│   └── log.csv                # Training loss per epoch
├── Oracle_Chaksu/
│   ├── oracle_model.pth       # Oracle weights
│   └── log.csv
├── Netra_Adapt/
│   ├── adapted_model.pth      # Adapted model (YOUR METHOD)
│   └── log.csv
└── evaluation/
    ├── results_table.csv      # Main results table
    ├── results_table.tex      # LaTeX formatted
    ├── roc_curves.png         # ROC curves comparison
    ├── confusion_matrices.png # Confusion matrices
    └── metrics_comparison.png # Bar charts
```

---

## Troubleshooting

### Script won't run
```bash
# Make executable
chmod +x run_everything.sh
chmod +x run_simple.sh

# Then run
bash run_everything.sh
```

### CUDA out of memory
Edit batch sizes in training scripts:
- `train_source.py`: Line 21, reduce `BATCH_SIZE` from 48 to 32
- `train_oracle.py`: Line 21, reduce from 32 to 24
- `adapt_target.py`: Line 22, reduce from 48 to 32

### Training taking too long
You can reduce epochs (but may affect results):
- `train_source.py`: Line 22, reduce `MAX_EPOCHS` from 60 to 40
- `train_oracle.py`: Line 22, reduce from 80 to 60
- `adapt_target.py`: Line 23, reduce from 40 to 30

### Pipeline fails midway
Resume from where it failed:
```bash
# If source training already completed, skip it:
# python train_oracle.py
# python adapt_target.py
# python evaluate.py
```

The scripts check if models already exist, so you can safely re-run.

---

## For Paper/Publication

After pipeline completes, you'll need:

1. **Main results:** `results/evaluation/results_table.csv`
   - Copy AUROC values for all 4 models
   - Report: AIROGS→Chákṣu vs AIROGS+Adapt→Chákṣu

2. **Figures for paper:**
   - `roc_curves.png` - ROC comparison (Figure 3)
   - `confusion_matrices.png` - Error analysis (Figure 4)
   - `metrics_comparison.png` - Bar chart (Figure 5)

3. **LaTeX table:** `results/evaluation/results_table.tex`
   - Ready to paste into paper

4. **Training curves:** 
   - Plot the `log.csv` files from each phase
   - Show convergence behavior

---

## Expected Results (Second Run)

After all improvements, you should see:

| Model | AUROC | What it means |
|-------|-------|---------------|
| Pretrained → Chákṣu | 0.50-0.55 | Random baseline |
| **AIROGS → Chákṣu** | **0.60-0.70** | Some transfer, but poor |
| **AIROGS+Adapt → Chákṣu** | **0.75-0.85** | **Strong improvement ✓** |
| Chákṣu → Chákṣu | 0.80-0.90 | Upper bound (oracle) |

Key metric: **AIROGS+Adapt should be 0.10-0.20 higher than AIROGS alone**

This demonstrates your MixEnt-Adapt method works! 🎉
