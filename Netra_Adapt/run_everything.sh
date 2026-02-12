#!/bin/bash

###############################################################################
#                        NETRA-ADAPT: FULL PIPELINE                          #
#                                                                             #
#  Runs the complete training and evaluation pipeline from start to finish   #
#  Usage: bash run_everything.sh                                             #
#                                                                             #
#  This script will:                                                          #
#  1. Check prerequisites (data, packages)                                    #
#  2. Prepare data CSVs (if needed)                                           #
#  3. Phase A: Train source model on AIROGS                                   #
#  4. Phase B: Train oracle model on Chákṣu (upper bound)                     #
#  5. Phase C: Test-Time Adaptation with MixEnt                               #
#  6. Phase D: Comprehensive evaluation                                       #
#  7. Display final results                                                   #
#                                                                             #
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Directories
BASE_DIR="/workspace"
DATA_DIR="$BASE_DIR/data"
CSV_DIR="$DATA_DIR/processed_csvs"
RESULTS_DIR="$BASE_DIR/results"

# Timing
TOTAL_START=$(date +%s)

# Function to print section headers
print_header() {
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}\n"
}

# Function to print progress
print_progress() {
    echo -e "${GREEN}[✓]${NC} $1"
}

# Function to print warnings
print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Function to print errors
print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Function to print phase info
print_phase() {
    echo -e "\n${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# Function to display elapsed time
elapsed_time() {
    local END=$(date +%s)
    local DIFF=$((END - $1))
    local HOURS=$((DIFF / 3600))
    local MINUTES=$(((DIFF % 3600) / 60))
    local SECONDS=$((DIFF % 60))
    
    if [ $HOURS -gt 0 ]; then
        echo "${HOURS}h ${MINUTES}m ${SECONDS}s"
    elif [ $MINUTES -gt 0 ]; then
        echo "${MINUTES}m ${SECONDS}s"
    else
        echo "${SECONDS}s"
    fi
}

###############################################################################
#                            MAIN PIPELINE                                    #
###############################################################################

clear
print_header "NETRA-ADAPT: FULL PIPELINE EXECUTION"
echo -e "${BOLD}Date:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "${BOLD}Workspace:${NC} $BASE_DIR"
echo ""

# Set HuggingFace token if not already set (needed for DINOv3 model download)
if [ -z "$HF_TOKEN" ]; then
    # Try to read from huggingface CLI cache
    if [ -f ~/.huggingface/token ]; then
        export HF_TOKEN=$(cat ~/.huggingface/token)
        print_progress "HuggingFace token loaded from cache"
    else
        print_warning "HuggingFace token not set - model download may require authentication"
        echo "  To set token: export HF_TOKEN='your_token_here'"
    fi
fi

# ═════════════════════════════════════════════════════════════════════════
# STEP 0: Prerequisites Check
# ═════════════════════════════════════════════════════════════════════════
print_phase "STEP 0/5: Checking Prerequisites"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    print_progress "Python 3 found: $(python3 --version)"
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
    print_progress "Python found: $(python --version)"
else
    print_error "Python not found! Please install Python 3.8+"
    exit 1
fi

# Check CUDA
if $PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_NAME=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    print_progress "CUDA available: $GPU_NAME"
else
    print_warning "CUDA not available - will use CPU (slower)"
fi

# Check if we're in the right directory
if [ ! -f "train_source.py" ]; then
    print_error "Not in Netra_Adapt directory! Please cd to the correct location."
    exit 1
fi
print_progress "Working directory confirmed"

# Create output directories
mkdir -p "$RESULTS_DIR/Source_AIROGS"
mkdir -p "$RESULTS_DIR/Oracle_Chaksu"
mkdir -p "$RESULTS_DIR/Netra_Adapt"
mkdir -p "$RESULTS_DIR/evaluation"
print_progress "Output directories ready"


# ═════════════════════════════════════════════════════════════════════════
# STEP 1: Data Preparation
# ═════════════════════════════════════════════════════════════════════════
print_phase "STEP 1/5: Data Preparation"

REQUIRED_CSVS=(
    "$CSV_DIR/airogs_train.csv"
    "$CSV_DIR/airogs_test.csv"
    "$CSV_DIR/chaksu_train_labeled.csv"
    "$CSV_DIR/chaksu_test_labeled.csv"
    "$CSV_DIR/chaksu_train_unlabeled.csv"
)

ALL_EXIST=true
for csv in "${REQUIRED_CSVS[@]}"; do
    if [ ! -f "$csv" ]; then
        ALL_EXIST=false
        break
    fi
done

if [ "$ALL_EXIST" = true ]; then
    print_progress "All CSV files already exist - skipping data preparation"
    for csv in "${REQUIRED_CSVS[@]}"; do
        LINES=$(wc -l < "$csv")
        echo "  - $(basename $csv): $LINES samples"
    done
else
    print_warning "CSV files missing - running prepare_data.py..."
    PREP_START=$(date +%s)
    if $PYTHON_CMD prepare_data.py; then
        print_progress "Data preparation completed in $(elapsed_time $PREP_START)"
    else
        print_error "Data preparation failed!"
        exit 1
    fi
fi


# ═════════════════════════════════════════════════════════════════════════
# STEP 2: Phase A - Source Training (AIROGS)
# ═════════════════════════════════════════════════════════════════════════
print_phase "STEP 2/5: Phase A - Source Training on AIROGS"
echo "Training DINOv3 ViT-L/16 on Western fundus images (AIROGS dataset)"
echo "This creates the baseline model that will be adapted to Indian eyes"
echo ""
echo "Expected time: ~2-3 hours"
echo "Configuration:"
echo "  - Batch size: 48"
echo "  - Max epochs: 60"
echo "  - Early stopping patience: 8"
echo "  - Augmentation: Enhanced (rotation, color, affine)"
echo ""

SOURCE_START=$(date +%s)
if $PYTHON_CMD train_source.py; then
    SOURCE_TIME=$(elapsed_time $SOURCE_START)
    print_progress "Source training completed in $SOURCE_TIME"
    
    # Check if model was saved
    if [ -f "$RESULTS_DIR/Source_AIROGS/model.pth" ]; then
        MODEL_SIZE=$(du -h "$RESULTS_DIR/Source_AIROGS/model.pth" | cut -f1)
        print_progress "Model saved: $MODEL_SIZE"
    fi
else
    print_error "Source training failed!"
    exit 1
fi


# ═════════════════════════════════════════════════════════════════════════
# STEP 3: Phase B - Oracle Training (Chákṣu)
# ═════════════════════════════════════════════════════════════════════════
print_phase "STEP 3/5: Phase B - Oracle Training on Chákṣu"
echo "Training on labeled Chákṣu data to establish upper bound performance"
echo "This is NOT part of the main method - just for comparison"
echo ""
echo "Expected time: ~1-2 hours"
echo "Configuration:"
echo "  - Batch size: 32"
echo "  - Max epochs: 80"
echo "  - Early stopping patience: 12"
echo "  - Weight decay: 0.01 (reduced from 0.05)"
echo "  - Learning rates: 2x higher than source training"
echo ""

ORACLE_START=$(date +%s)
if $PYTHON_CMD train_oracle.py; then
    ORACLE_TIME=$(elapsed_time $ORACLE_START)
    print_progress "Oracle training completed in $ORACLE_TIME"
    
    if [ -f "$RESULTS_DIR/Oracle_Chaksu/oracle_model.pth" ]; then
        MODEL_SIZE=$(du -h "$RESULTS_DIR/Oracle_Chaksu/oracle_model.pth" | cut -f1)
        print_progress "Oracle model saved: $MODEL_SIZE"
    fi
else
    print_error "Oracle training failed!"
    exit 1
fi


# ═════════════════════════════════════════════════════════════════════════
# STEP 4: Phase C - Test-Time Adaptation (MixEnt-Adapt)
# ═════════════════════════════════════════════════════════════════════════
print_phase "STEP 4/5: Phase C - Test-Time Adaptation with MixEnt"
echo "Adapting source model to Chákṣu test set WITHOUT using labels"
echo "This is the core contribution: MixEnt uncertainty-guided adaptation"
echo ""
echo "Key Strategy:"
echo "  ✓ Adapts on TEST set (labels ignored during training)"
echo "  ✓ Pure entropy minimization (no diversity penalty)"
echo "  ✓ MixEnt: Confident predictions teach uncertain ones"
echo "  ✓ AdaIN style injection for pigmentation calibration"
echo ""
echo "Expected time: ~30-60 minutes"
echo "Configuration:"
echo "  - Batch size: 48 (larger for better MixEnt statistics)"
echo "  - Max epochs: 40"
echo "  - Optimizer: AdamW"
echo "  - Learning rates: 5x higher than before"
echo ""

ADAPT_START=$(date +%s)
if $PYTHON_CMD adapt_target.py; then
    ADAPT_TIME=$(elapsed_time $ADAPT_START)
    print_progress "Adaptation completed in $ADAPT_TIME"
    
    if [ -f "$RESULTS_DIR/Netra_Adapt/adapted_model.pth" ]; then
        MODEL_SIZE=$(du -h "$RESULTS_DIR/Netra_Adapt/adapted_model.pth" | cut -f1)
        print_progress "Adapted model saved: $MODEL_SIZE"
    fi
else
    print_error "Adaptation failed!"
    exit 1
fi


# ═════════════════════════════════════════════════════════════════════════
# STEP 5: Phase D - Comprehensive Evaluation
# ═════════════════════════════════════════════════════════════════════════
print_phase "STEP 5/5: Phase D - Comprehensive Evaluation"
echo "Evaluating all models on Chákṣu test set (with labels)"
echo "This measures how well each approach performs on Indian eyes"
echo ""
echo "Models being evaluated:"
echo "  1. Pretrained → Chákṣu       (Vanilla DINOv3, no fine-tuning)"
echo "  2. AIROGS → Chákṣu           (Source-only baseline)"
echo "  3. AIROGS+Adapt → Chákṣu     (MixEnt-Adapt - OUR METHOD)"
echo "  4. Chákṣu → Chákṣu           (Oracle - upper bound)"
echo ""
echo "Metrics computed:"
echo "  • AUROC, Sensitivity, Specificity, Precision, F1-Score"
echo "  • Sensitivity @ 95% Specificity (clinically relevant)"
echo "  • ROC curves, confusion matrices, comparison charts"
echo ""

EVAL_START=$(date +%s)
if $PYTHON_CMD evaluate.py; then
    EVAL_TIME=$(elapsed_time $EVAL_START)
    print_progress "Evaluation completed in $EVAL_TIME"
else
    print_error "Evaluation failed!"
    exit 1
fi


# ═════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════
TOTAL_TIME=$(elapsed_time $TOTAL_START)

clear
print_header "PIPELINE EXECUTION COMPLETE!"

echo -e "${BOLD}Execution Time Summary:${NC}"
echo "  • Source Training:    $SOURCE_TIME"
echo "  • Oracle Training:    $ORACLE_TIME"
echo "  • Adaptation:         $ADAPT_TIME"
echo "  • Evaluation:         $EVAL_TIME"
echo "  ${BOLD}• Total Time:         $TOTAL_TIME${NC}"
echo ""

# Display results if CSV exists
if [ -f "$RESULTS_DIR/evaluation/results_table.csv" ]; then
    print_header "FINAL RESULTS"
    echo -e "${BOLD}Performance on Chákṣu Test Set (AUROC):${NC}\n"
    
    # Parse and display results
    tail -n +2 "$RESULTS_DIR/evaluation/results_table.csv" | while IFS=',' read -r model auroc sensitivity specificity precision f1 accuracy sens_at_95; do
        # Color code based on AUROC
        if (( $(echo "$auroc > 0.75" | bc -l) )); then
            COLOR=$GREEN
        elif (( $(echo "$auroc > 0.60" | bc -l) )); then
            COLOR=$YELLOW
        else
            COLOR=$RED
        fi
        
        printf "  %-30s ${COLOR}${BOLD}AUROC: %.4f${NC}  (Sens: %.3f, Spec: %.3f, F1: %.3f)\n" \
            "$model" "$auroc" "$sensitivity" "$specificity" "$f1"
    done
    
    echo ""
    echo -e "${BOLD}Key Findings:${NC}"
    
    # Extract specific values (simplified)
    SOURCE_AUROC=$(awk -F',' 'NR==2 {print $2}' "$RESULTS_DIR/evaluation/results_table.csv")
    ADAPTED_AUROC=$(awk -F',' 'NR==5 {print $2}' "$RESULTS_DIR/evaluation/results_table.csv")
    ORACLE_AUROC=$(awk -F',' 'NR==4 {print $2}' "$RESULTS_DIR/evaluation/results_table.csv")
    
    echo "  • Source model (AIROGS→Chákṣu):  $SOURCE_AUROC"
    echo "  • Adapted model (MixEnt):        $ADAPTED_AUROC"
    echo "  • Oracle (upper bound):          $ORACLE_AUROC"
    echo ""
    
    # Check if adaptation helped
    IMPROVEMENT=$(echo "$ADAPTED_AUROC - $SOURCE_AUROC" | bc -l)
    if (( $(echo "$IMPROVEMENT > 0.1" | bc -l) )); then
        print_progress "MixEnt-Adapt shows STRONG improvement (+$(printf "%.3f" $IMPROVEMENT))! ✓"
    elif (( $(echo "$IMPROVEMENT > 0.05" | bc -l) )); then
        print_progress "MixEnt-Adapt shows moderate improvement (+$(printf "%.3f" $IMPROVEMENT))"
    elif (( $(echo "$IMPROVEMENT > 0" | bc -l) )); then
        print_warning "MixEnt-Adapt shows slight improvement (+$(printf "%.3f" $IMPROVEMENT))"
    else
        print_warning "MixEnt-Adapt did not improve performance (change: $(printf "%.3f" $IMPROVEMENT))"
    fi
fi

echo ""
print_header "OUTPUT FILES"
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "📊 Evaluation Results:"
echo "  • $RESULTS_DIR/evaluation/results_table.csv"
echo "  • $RESULTS_DIR/evaluation/results_table.tex"
echo "  • $RESULTS_DIR/evaluation/roc_curves.png"
echo "  • $RESULTS_DIR/evaluation/confusion_matrices.png"
echo "  • $RESULTS_DIR/evaluation/metrics_comparison.png"
echo ""
echo "🔬 Model Weights:"
echo "  • $RESULTS_DIR/Source_AIROGS/model.pth"
echo "  • $RESULTS_DIR/Oracle_Chaksu/oracle_model.pth"
echo "  • $RESULTS_DIR/Netra_Adapt/adapted_model.pth"
echo ""
echo "📝 Training Logs:"
echo "  • $RESULTS_DIR/Source_AIROGS/log.csv"
echo "  • $RESULTS_DIR/Oracle_Chaksu/log.csv"
echo "  • $RESULTS_DIR/Netra_Adapt/log.csv"
echo ""

print_header "NEXT STEPS"
echo "1. Review visualizations in $RESULTS_DIR/evaluation/"
echo "2. Check training curves in the .csv log files"
echo "3. Analyze confusion matrices to understand error patterns"
echo "4. Compare ROC curves across all models"
echo ""
echo "For paper writing:"
echo "  • Use results_table.tex for LaTeX tables"
echo "  • Use the PNG figures for paper figures"
echo "  • Report AUROC, Sensitivity@95%Specificity for clinical relevance"
echo ""

print_header "PIPELINE FINISHED SUCCESSFULLY! 🎉"
echo -e "${GREEN}${BOLD}All training and evaluation completed!${NC}"
echo ""
