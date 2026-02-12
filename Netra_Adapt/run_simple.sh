#!/bin/bash
# Quick and simple pipeline runner - no fancy output

set -e
cd /workspace/Netra_Adapt

echo "=== Netra-Adapt Pipeline ==="
echo "Starting at $(date)"
echo ""

# Check if CSVs exist, if not prepare data
if [ ! -f "/workspace/data/processed_csvs/airogs_train.csv" ]; then
    echo "[1/5] Preparing data..."
    python prepare_data.py
fi

echo "[2/5] Training source model (AIROGS)..."
python train_source.py

echo "[3/5] Training oracle model (Chákṣu)..."
python train_oracle.py

echo "[4/5] Running adaptation (MixEnt)..."
python adapt_target.py

echo "[5/5] Evaluating all models..."
python evaluate.py

echo ""
echo "=== Pipeline Complete ==="
echo "Results: /workspace/results/evaluation/"
echo "Finished at $(date)"
