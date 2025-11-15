.PHONY: all install train serve infer docker-build docker-run clean help

# Default target
all: install train

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Run complete training pipeline
train:
	@echo "Running training pipeline..."
	bash scripts/setup_and_train.sh

# Start FastAPI service
serve:
	@echo "Starting AI Text Detection service..."
	@echo "Service will be available at http://localhost:8000"
	@echo "API docs at http://localhost:8000/docs"
	uvicorn service.app.main:app --host 0.0.0.0 --port 8000 --reload

# Run inference (requires TEXT variable)
infer:
	@if [ -z "$(TEXT)" ]; then \
		echo "Usage: make infer TEXT=\"your text here\""; \
		echo "Example: make infer TEXT=\"人工智能技术发展迅速\""; \
		exit 1; \
	fi
	bash scripts/infer.sh "$(TEXT)"

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -t ai-text-detection:latest -f service/Dockerfile .

# Run Docker container
docker-run:
	@echo "Running Docker container..."
	@echo "Service will be available at http://localhost:8000"
	docker run -p 8000:8000 --name ai-text-detection ai-text-detection:latest

# Stop and remove Docker container
docker-stop:
	@echo "Stopping Docker container..."
	docker stop ai-text-detection || true
	docker rm ai-text-detection || true

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf data/processed data/features models/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Display help
help:
	@echo "AI Text Detection - Makefile Commands"
	@echo "======================================"
	@echo ""
	@echo "Training:"
	@echo "  make all          - Install dependencies and run training"
	@echo "  make install      - Install Python dependencies"
	@echo "  make train        - Run complete training pipeline"
	@echo ""
	@echo "Inference:"
	@echo "  make serve        - Start FastAPI service on port 8000"
	@echo "  make infer TEXT=\"...\" - Run CLI inference on text"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make docker-stop  - Stop and remove Docker container"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        - Remove generated files"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make all"
	@echo "  make infer TEXT=\"机器学习是人工智能的分支\""
	@echo "  make serve"
