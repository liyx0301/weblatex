#!/bin/bash
#
# AI Text Detection - Inference Script
# Quick command-line interface for text detection
#

set -e

# Check if text argument provided
if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/infer.sh <text> [models_dir]"
    echo ""
    echo "Examples:"
    echo "  bash scripts/infer.sh \"Your text here\""
    echo "  bash scripts/infer.sh \"人工智能技术发展迅速\" models/"
    echo ""
    exit 1
fi

TEXT="$1"
MODELS_DIR="${2:-models}"

# Check if models exist
if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Models directory '$MODELS_DIR' not found."
    echo "Please run training first: bash scripts/setup_and_train.sh"
    exit 1
fi

# Run inference
python src/infer_cli.py "$TEXT" "$MODELS_DIR"
