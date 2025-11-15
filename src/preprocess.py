"""
Data preprocessing module for AI text detection.
Loads raw JSONL data and prepares it for feature extraction.
"""
import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Tuple


def load_raw_data(filepath: str) -> List[Dict[str, str]]:
    """
    Load raw data from JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of dictionaries containing 'text' and 'label'
    """
    data = []
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            data.append(obj)
    return data


def preprocess_data(data: List[Dict[str, str]]) -> Tuple[List[str], List[int]]:
    """
    Preprocess data into texts and binary labels.
    
    Args:
        data: List of dictionaries with 'text' and 'label' keys
        
    Returns:
        Tuple of (texts, labels) where labels are 1 for AI, 0 for Human
    """
    texts = []
    labels = []
    
    for item in data:
        texts.append(item['text'])
        # Convert label: AI=1, Human=0
        label = 1 if item['label'].upper() == 'AI' else 0
        labels.append(label)
    
    return texts, labels


def save_processed_data(texts: List[str], labels: List[int], output_dir: str):
    """
    Save processed data to files.
    
    Args:
        texts: List of text strings
        labels: List of binary labels
        output_dir: Directory to save processed files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save texts
    with open(output_path / 'texts.txt', 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    
    # Save labels
    with open(output_path / 'labels.txt', 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(str(label) + '\n')
    
    print(f"Saved {len(texts)} samples to {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <input_jsonl> <output_dir>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    print(f"Loading data from {input_file}...")
    data = load_raw_data(input_file)
    print(f"Loaded {len(data)} samples")
    
    print("Preprocessing data...")
    texts, labels = preprocess_data(data)
    
    print(f"Saving processed data to {output_dir}...")
    save_processed_data(texts, labels, output_dir)
    
    print("Preprocessing complete!")
