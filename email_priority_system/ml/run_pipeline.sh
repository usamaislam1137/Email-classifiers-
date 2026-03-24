#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh
# Full Email Priority Classification ML Pipeline
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/pipeline.log"
VENV_DIR="$SCRIPT_DIR/venv"

# -- Colours -------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log()  { echo -e "${BLUE}[PIPELINE]${NC} $*" | tee -a "$LOG_FILE"; }
ok()   { echo -e "${GREEN}[OK]${NC}      $*" | tee -a "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[WARN]${NC}    $*" | tee -a "$LOG_FILE"; }
err()  { echo -e "${RED}[ERROR]${NC}   $*" | tee -a "$LOG_FILE"; exit 1; }

# -- Argument parsing ----------------------------------------------------------
SKIP_DOWNLOAD=false
SKIP_PREPROCESS=false
SKIP_FEATURES=false
SKIP_TRAIN=false
SKIP_EVALUATE=false
SKIP_FALLBACK=false
START_API=true
NO_BERT=false
MAX_EMAILS=50000

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --skip-download     Skip dataset download step
  --skip-preprocess   Skip email preprocessing step
  --skip-features     Skip feature engineering step
  --skip-train        Skip model training step
  --skip-evaluate     Skip model evaluation step
  --skip-fallback     Skip fallback check step
  --no-start-api      Do not start the Flask API after pipeline completes
  --no-bert           Skip DistilBERT training (faster)
  --max-emails N      Maximum emails to preprocess (default: 50000)
  -h, --help          Show this help message
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download)   SKIP_DOWNLOAD=true ;;
    --skip-preprocess) SKIP_PREPROCESS=true ;;
    --skip-features)   SKIP_FEATURES=true ;;
    --skip-train)      SKIP_TRAIN=true ;;
    --skip-evaluate)   SKIP_EVALUATE=true ;;
    --skip-fallback)   SKIP_FALLBACK=true ;;
    --no-start-api)    START_API=false ;;
    --no-bert)         NO_BERT=true ;;
    --max-emails)      MAX_EMAILS="$2"; shift ;;
    -h|--help)         usage ;;
    *) warn "Unknown option: $1" ;;
  esac
  shift
done

# -- Environment setup ---------------------------------------------------------
log "=============================================="
log "  Email Priority Classification Pipeline"
log "  Started: $(date)"
log "=============================================="

cd "$SCRIPT_DIR"

# Check Python
PYTHON=$(command -v python3 || command -v python)
if [[ -z "$PYTHON" ]]; then
  err "Python 3 not found. Please install Python 3.10+."
fi
PYTHON_VER=$("$PYTHON" --version 2>&1)
log "Python: $PYTHON_VER"

# Activate or create virtualenv
if [[ -d "$VENV_DIR" ]]; then
  log "Activating existing virtualenv at $VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  log "Creating virtualenv at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  log "Installing dependencies from requirements.txt ..."
  pip install --upgrade pip --quiet
  pip install -r requirements.txt --quiet
  ok "Dependencies installed."
fi

PYTHON="python"  # now using venv python

# -- Step 1: Download dataset --------------------------------------------------
if [[ "$SKIP_DOWNLOAD" == "false" ]]; then
  log "--- Step 1/6: Downloading Enron Dataset ---"
  if "$PYTHON" download_dataset.py --verify 2>/dev/null; then
    ok "Enron dataset already present."
  else
    "$PYTHON" download_dataset.py --enron || {
      warn "Enron download failed. Attempting to continue with existing data."
    }
  fi
else
  warn "Skipping dataset download."
fi

# -- Step 2: Preprocess --------------------------------------------------------
if [[ "$SKIP_PREPROCESS" == "false" ]]; then
  log "--- Step 2/6: Preprocessing emails ---"
  if [[ -f "data/processed/processed_emails.csv" ]]; then
    ROWS=$(wc -l < "data/processed/processed_emails.csv")
    warn "processed_emails.csv already exists ($ROWS rows). Re-processing ..."
  fi
  "$PYTHON" preprocess.py --max "$MAX_EMAILS" || err "Preprocessing failed."
  ok "Preprocessing complete."
else
  warn "Skipping preprocessing."
fi

# -- Step 3: Feature engineering -----------------------------------------------
if [[ "$SKIP_FEATURES" == "false" ]]; then
  log "--- Step 3/6: Feature engineering ---"
  BERT_FLAG=""
  if [[ "$NO_BERT" == "true" ]]; then
    BERT_FLAG="--no-bert"
    warn "BERT embeddings disabled."
  fi
  "$PYTHON" feature_engineering.py $BERT_FLAG || err "Feature engineering failed."
  ok "Feature engineering complete."
else
  warn "Skipping feature engineering."
fi

# -- Step 4: Train models ------------------------------------------------------
if [[ "$SKIP_TRAIN" == "false" ]]; then
  log "--- Step 4/6: Training models ---"
  BERT_FLAG=""
  if [[ "$NO_BERT" == "true" ]]; then
    BERT_FLAG="--no-bert"
  fi
  "$PYTHON" train_models.py $BERT_FLAG || err "Model training failed."
  ok "Model training complete."
else
  warn "Skipping model training."
fi

# -- Step 5: Evaluate models ---------------------------------------------------
if [[ "$SKIP_EVALUATE" == "false" ]]; then
  log "--- Step 5/6: Evaluating models ---"
  "$PYTHON" evaluate_models.py || err "Model evaluation failed."
  ok "Model evaluation complete."
else
  warn "Skipping model evaluation."
fi

# -- Step 6: Fallback check ----------------------------------------------------
if [[ "$SKIP_FALLBACK" == "false" ]]; then
  log "--- Step 6/6: Checking fallback threshold ---"
  if "$PYTHON" fallback_model.py --check-only 2>/dev/null; then
    ok "Performance is above threshold. Primary model will be used."
  else
    warn "Performance below threshold. Applying fallback model ..."
    "$PYTHON" fallback_model.py || warn "Fallback model setup encountered issues."
    ok "Fallback model applied."
  fi
else
  warn "Skipping fallback check."
fi

# -- Summary -------------------------------------------------------------------
log "=============================================="
log "  Pipeline Complete: $(date)"
if [[ -f "models/best_model.txt" ]]; then
  BEST_MODEL=$(cat models/best_model.txt)
  ok "Best model: $BEST_MODEL"
fi
if [[ -f "models/evaluation_report.json" ]]; then
  ACCURACY=$("$PYTHON" -c "
import json
with open('models/evaluation_report.json') as f:
    r = json.load(f)
best = r.get('best_model','unknown')
m = r.get('models', {}).get(best, {})
print(f'accuracy={m.get(\"accuracy\",0):.4f}  macro_f1={m.get(\"macro_f1\",0):.4f}')
" 2>/dev/null || echo "N/A")
  log "Best model performance: $ACCURACY"
fi
log "=============================================="

# -- Start Flask API -----------------------------------------------------------
if [[ "$START_API" == "true" ]]; then
  log "Starting Flask API on port 5000 ..."
  log "Press Ctrl+C to stop."
  exec "$PYTHON" flask_api.py
fi
