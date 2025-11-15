"""
FastAPI service for AI text detection.
Provides REST API endpoints for inference and health checks.
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from infer_cli import AITextDetector


app = FastAPI(
    title="AI Text Detection API",
    description="REST API for detecting AI-generated text",
    version="1.0.0"
)

# Global detector instance
detector: Optional[AITextDetector] = None


class InferenceRequest(BaseModel):
    """Request model for inference."""
    text: str = Field(..., description="Text to analyze", min_length=1)


class InferenceResponse(BaseModel):
    """Response model for inference."""
    text: str
    prob_ai: float
    prob_human: float
    label: str
    perplexity: float
    lgb_prob: float
    trans_prob: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    models_loaded: bool
    version: str


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global detector
    
    # Get models directory from environment or use default
    models_dir = os.getenv('MODELS_DIR', 'models')
    
    if not Path(models_dir).exists():
        print(f"Warning: Models directory '{models_dir}' not found.")
        print("Service will start but inference will not work until models are available.")
        detector = None
    else:
        print(f"Loading models from {models_dir}...")
        detector = AITextDetector(models_dir)
        print("Models loaded successfully!")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status and model availability
    """
    return HealthResponse(
        status="healthy",
        models_loaded=detector is not None,
        version="1.0.0"
    )


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    Perform AI text detection inference.
    
    Args:
        request: Text to analyze
        
    Returns:
        Detection results with probabilities
    """
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please ensure models are trained and available."
        )
    
    try:
        result = detector.predict(request.text)
        return InferenceResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        API information
    """
    return {
        "name": "AI Text Detection API",
        "version": "1.0.0",
        "endpoints": {
            "POST /infer": "Detect if text is AI-generated",
            "GET /health": "Check service health",
            "GET /docs": "Interactive API documentation"
        },
        "models_loaded": detector is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
