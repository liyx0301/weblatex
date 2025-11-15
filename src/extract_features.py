"""
Feature extraction module for AI text detection.
Extracts perplexity and transformer embeddings from text.
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    BertModel,
    BertTokenizer
)
from tqdm import tqdm


class FeatureExtractor:
    """Extract features for AI text detection."""
    
    def __init__(
        self, 
        perplexity_model: str = "gpt2-medium",
        embedding_model: str = "bert-base-chinese",
        device: str = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            perplexity_model: Model name for perplexity calculation
            embedding_model: Model name for embeddings
            device: Device to use (cuda/cpu), auto-detected if None
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load perplexity model
        print(f"Loading perplexity model: {perplexity_model}")
        self.ppl_tokenizer = GPT2Tokenizer.from_pretrained(perplexity_model)
        self.ppl_model = GPT2LMHeadModel.from_pretrained(perplexity_model).to(self.device)
        self.ppl_model.eval()
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.emb_tokenizer = BertTokenizer.from_pretrained(embedding_model)
        self.emb_model = BertModel.from_pretrained(embedding_model).to(self.device)
        self.emb_model.eval()
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of text using GPT-2.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score
        """
        encodings = self.ppl_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = encodings['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.ppl_model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get BERT embeddings for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (768-dim for BERT-base)
        """
        encodings = self.emb_tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=True
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.emb_model(input_ids, attention_mask=attention_mask)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings[0]
    
    def extract_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract all features from a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with 'perplexity' and 'embeddings' arrays
        """
        perplexities = []
        embeddings = []
        
        print(f"Extracting features from {len(texts)} texts...")
        for text in tqdm(texts, desc="Processing"):
            ppl = self.calculate_perplexity(text)
            emb = self.get_embeddings(text)
            
            perplexities.append(ppl)
            embeddings.append(emb)
        
        return {
            'perplexity': np.array(perplexities),
            'embeddings': np.array(embeddings)
        }
    
    def save_features(self, features: Dict[str, np.ndarray], output_dir: str):
        """
        Save extracted features to disk.
        
        Args:
            features: Dictionary of feature arrays
            output_dir: Directory to save features
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / 'perplexity.npy', features['perplexity'])
        np.save(output_path / 'embeddings.npy', features['embeddings'])
        
        # Save feature metadata
        metadata = {
            'num_samples': len(features['perplexity']),
            'embedding_dim': features['embeddings'].shape[1],
            'feature_names': ['perplexity'] + [f'emb_{i}' for i in range(features['embeddings'].shape[1])]
        }
        
        with open(output_path / 'features.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Features saved to {output_dir}")


def load_texts(filepath: str) -> List[str]:
    """Load texts from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python extract_features.py <texts_file> <output_dir>")
        sys.exit(1)
    
    texts_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    print(f"Loading texts from {texts_file}...")
    texts = load_texts(texts_file)
    print(f"Loaded {len(texts)} texts")
    
    extractor = FeatureExtractor()
    features = extractor.extract_features(texts)
    extractor.save_features(features, output_dir)
    
    print("Feature extraction complete!")
