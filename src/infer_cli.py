"""
Inference CLI for AI text detection.
"""
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict
import sys

# Import from training module
from train import TransformerClassifier
from extract_features import FeatureExtractor


class AITextDetector:
    """AI Text Detection Inference Engine."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize detector with trained models.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load LightGBM
        import lightgbm as lgb
        self.lgb_model = lgb.Booster(model_file=str(self.models_dir / 'lgb.txt'))
        
        # Load feature metadata
        with open(self.models_dir / 'lgb.txt.features.json', 'r') as f:
            metadata = json.load(f)
            self.num_features = metadata['num_features']
        
        # Load Transformer
        self.trans_model = TransformerClassifier(self.num_features).to(self.device)
        self.trans_model.load_state_dict(
            torch.load(self.models_dir / 'transformer' / 'model.pt', map_location=self.device)
        )
        self.trans_model.eval()
        
        # Load calibrators
        with open(self.models_dir / 'calibrators.pkl', 'rb') as f:
            self.calibrators = pickle.load(f)
        
        # Load ensemble weights
        with open(self.models_dir / 'ensemble.json', 'r') as f:
            self.ensemble_weights = json.load(f)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
        print(f"Models loaded from {models_dir}")
    
    def predict(self, text: str) -> Dict:
        """
        Predict if text is AI-generated.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        perplexity = self.feature_extractor.calculate_perplexity(text)
        embeddings = self.feature_extractor.get_embeddings(text)
        
        # Combine features
        features = np.concatenate([[perplexity], embeddings]).reshape(1, -1)
        
        # Get predictions from both models
        lgb_pred = self.lgb_model.predict(features)[0]
        
        with torch.no_grad():
            trans_pred = self.trans_model(
                torch.FloatTensor(features).to(self.device)
            ).cpu().numpy()[0][0]
        
        # Ensemble prediction
        prob_ai = (self.ensemble_weights['lgb_weight'] * lgb_pred + 
                   self.ensemble_weights['trans_weight'] * trans_pred)
        
        label = 'AI' if prob_ai >= 0.5 else 'Human'
        
        return {
            'text': text,
            'prob_ai': float(prob_ai),
            'prob_human': float(1 - prob_ai),
            'label': label,
            'perplexity': float(perplexity),
            'lgb_prob': float(lgb_pred),
            'trans_prob': float(trans_pred)
        }


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python infer_cli.py <text> [models_dir]")
        print("\nExample:")
        print('  python infer_cli.py "Your text here"')
        print('  python infer_cli.py "Your text here" models/')
        sys.exit(1)
    
    text = sys.argv[1]
    models_dir = sys.argv[2] if len(sys.argv) > 2 else "models"
    
    # Check if models exist
    if not Path(models_dir).exists():
        print(f"Error: Models directory '{models_dir}' not found.")
        print("Please run training first: bash scripts/setup_and_train.sh")
        sys.exit(1)
    
    print(f"Analyzing text: {text[:100]}...")
    print()
    
    detector = AITextDetector(models_dir)
    result = detector.predict(text)
    
    print("=" * 60)
    print("AI Text Detection Result")
    print("=" * 60)
    print(f"Text: {result['text'][:100]}...")
    print(f"\nPrediction: {result['label']}")
    print(f"AI Probability: {result['prob_ai']:.4f}")
    print(f"Human Probability: {result['prob_human']:.4f}")
    print(f"\nPerplexity: {result['perplexity']:.2f}")
    print(f"LightGBM Probability: {result['lgb_prob']:.4f}")
    print(f"Transformer Probability: {result['trans_prob']:.4f}")
    print("=" * 60)
    
    # Also output JSON for programmatic use
    print("\nJSON Output:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
