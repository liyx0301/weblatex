#!/bin/bash
#
# Integration test script for AI Text Detection System
# Tests key components without running full training
#

set -e

echo "================================================"
echo "AI Text Detection - Integration Tests"
echo "================================================"
echo ""

# Test 1: Check required files exist
echo "Test 1: Checking required files..."
for file in requirements.txt data/raw.jsonl scripts/setup_and_train.sh scripts/infer.sh src/preprocess.py src/extract_features.py src/train.py src/infer_cli.py service/app/main.py service/Dockerfile Makefile; do
    if [ ! -f "$file" ]; then
        echo "✗ Missing file: $file"
        exit 1
    fi
done
echo "✓ All required files present"
echo ""

# Test 2: Validate data structure
echo "Test 2: Validating data structure..."
python3 - <<'EOF'
import json

with open('data/raw.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
    assert len(data) > 0, "No data found"
    assert all('text' in d and 'label' in d for d in data), "Missing fields"
    assert all(d['label'] in ['AI', 'Human'] for d in data), "Invalid labels"
print("✓ Data structure valid")
EOF
echo ""

# Test 3: Test Python imports
echo "Test 3: Testing Python module imports..."
python3 -c "from src.preprocess import load_raw_data, preprocess_data; print('✓ preprocess module OK')"
python3 -c "from src.extract_features import FeatureExtractor; print('✓ extract_features module OK')"
python3 -c "from src.train import TransformerClassifier; print('✓ train module OK')"
python3 -c "from service.app.main import app; print('✓ FastAPI app OK')"
echo ""

# Test 4: Test preprocessing
echo "Test 4: Testing preprocessing..."
python3 src/preprocess.py data/raw.jsonl data/processed_test
if [ -f data/processed_test/texts.txt ] && [ -f data/processed_test/labels.txt ]; then
    echo "✓ Preprocessing successful"
    rm -rf data/processed_test
else
    echo "✗ Preprocessing failed"
    exit 1
fi
echo ""

# Test 5: Validate shell scripts
echo "Test 5: Validating shell script syntax..."
bash -n scripts/setup_and_train.sh && echo "✓ setup_and_train.sh syntax OK"
bash -n scripts/infer.sh && echo "✓ infer.sh syntax OK"
echo ""

# Test 6: Validate YAML files
echo "Test 6: Validating YAML files..."
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ghcr-docker.yml')); print('✓ GitHub Actions workflow valid')"
python3 -c "import yaml; yaml.safe_load(open('docker-compose.yml')); print('✓ docker-compose.yml valid')"
echo ""

# Test 7: Test Makefile
echo "Test 7: Testing Makefile..."
make help > /dev/null && echo "✓ Makefile help works"
echo ""

echo "================================================"
echo "✓ All integration tests passed!"
echo "================================================"
echo ""
echo "Note: Full training requires downloading models (~1-2GB)"
echo "Run 'bash scripts/setup_and_train.sh' to train the system"
