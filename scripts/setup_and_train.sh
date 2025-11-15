#!/bin/bash
#
# AI Text Detection - One-Click Training Script
# This script sets up the environment and trains the complete AI detection system
#

set -e  # Exit on error

echo "============================================================"
echo "AI Text Detection - Setup and Training"
echo "============================================================"
echo ""

# Configuration
DATA_DIR="data"
PROCESSED_DIR="$DATA_DIR/processed"
FEATURES_DIR="$DATA_DIR/features"
MODELS_DIR="models"
RAW_DATA="$DATA_DIR/raw.jsonl"

# Check if running from repository root
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the repository root directory"
    exit 1
fi

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
echo "------------------------------------------------------------"
pip install -q -r requirements.txt
echo "Dependencies installed successfully!"
echo ""

# Step 2: Check raw data
echo "Step 2: Checking raw data..."
echo "------------------------------------------------------------"
if [ ! -f "$RAW_DATA" ]; then
    echo "Error: Raw data file not found: $RAW_DATA"
    echo "Please ensure data/raw.jsonl exists with training data"
    exit 1
fi

NUM_SAMPLES=$(wc -l < "$RAW_DATA")
echo "Found $NUM_SAMPLES samples in $RAW_DATA"
echo "Note: This is demo data. For production, replace with real datasets."
echo ""

# Step 3: Preprocess data
echo "Step 3: Preprocessing data..."
echo "------------------------------------------------------------"
mkdir -p "$PROCESSED_DIR"
python src/preprocess.py "$RAW_DATA" "$PROCESSED_DIR"
echo ""

# Step 4: Extract features
echo "Step 4: Extracting features (this may take a few minutes)..."
echo "------------------------------------------------------------"
echo "Downloading models: gpt2-medium, bert-base-chinese..."
mkdir -p "$FEATURES_DIR"
python src/extract_features.py "$PROCESSED_DIR/texts.txt" "$FEATURES_DIR"
echo ""

# Step 5: Train models
echo "Step 5: Training LightGBM and Transformer models..."
echo "------------------------------------------------------------"
mkdir -p "$MODELS_DIR"
python src/train.py "$FEATURES_DIR" "$MODELS_DIR"
echo ""

# Step 6: Verify outputs
echo "Step 6: Verifying outputs..."
echo "------------------------------------------------------------"
if [ -f "$MODELS_DIR/lgb.txt" ] && \
   [ -f "$MODELS_DIR/transformer/model.pt" ] && \
   [ -f "$MODELS_DIR/ensemble.json" ] && \
   [ -f "$MODELS_DIR/metrics.json" ]; then
    echo "✓ All model files generated successfully!"
else
    echo "✗ Some model files are missing"
    exit 1
fi

echo ""
echo "Model Metrics:"
cat "$MODELS_DIR/metrics.json"
echo ""

echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo ""
echo "Models saved to: $MODELS_DIR/"
echo ""
echo "Next steps:"
echo "  1. Test inference: bash scripts/infer.sh \"Your test text\""
echo "  2. Start API service: make serve"
echo "  3. Build Docker image: make docker-build"
echo ""
echo "For production use:"
echo "  - Replace data/raw.jsonl with real AI/Human text datasets"
echo "  - Increase training samples (recommended: 10,000+)"
echo "  - Adjust hyperparameters in src/train.py"
echo "  - Enable GPU for faster training/inference"
echo ""
