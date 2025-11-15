"""
Training module for AI text detection.
Trains LightGBM and Transformer models with calibration and ensemble.
"""
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import lightgbm as lgb
from tqdm import tqdm


class TransformerClassifier(nn.Module):
    """Simple transformer-based classifier."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=num_layers
        )
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x shape: (batch, features)
        x = self.fc1(x)
        x = self.dropout(x)
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        x = self.transformer(x)
        x = x.squeeze(1)  # (batch, hidden_dim)
        x = self.fc2(x)
        return torch.sigmoid(x)


def load_features_and_labels(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features and labels from preprocessed data.
    
    Args:
        data_dir: Directory containing preprocessed data
        
    Returns:
        Tuple of (features, labels)
    """
    data_path = Path(data_dir)
    
    # Load features
    perplexity = np.load(data_path / 'perplexity.npy').reshape(-1, 1)
    embeddings = np.load(data_path / 'embeddings.npy')
    
    # Combine features
    features = np.concatenate([perplexity, embeddings], axis=1)
    
    # Load labels
    with open(data_path.parent / 'processed' / 'labels.txt', 'r') as f:
        labels = np.array([int(line.strip()) for line in f])
    
    return features, labels


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray) -> lgb.Booster:
    """
    Train LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Trained LightGBM model
    """
    print("Training LightGBM model...")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
    )
    
    return model


def train_transformer(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      epochs: int = 3, batch_size: int = 4,
                      device: str = None) -> TransformerClassifier:
    """
    Train Transformer classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to use
        
    Returns:
        Trained transformer model
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Transformer model on {device}...")
    
    input_dim = X_train.shape[1]
    model = TransformerClassifier(input_dim).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Mini-batch training
        for i in range(0, len(X_train_t), batch_size):
            batch_X = X_train_t[i:i+batch_size]
            batch_y = y_train_t[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss/len(X_train):.4f}, Val Loss: {val_loss.item():.4f}")
    
    return model


def calibrate_models(lgb_model: lgb.Booster, X_cal: np.ndarray, y_cal: np.ndarray) -> Dict:
    """
    Calibrate model probabilities.
    
    Args:
        lgb_model: LightGBM model
        X_cal: Calibration features
        y_cal: Calibration labels
        
    Returns:
        Dictionary of calibrated models
    """
    print("Calibrating models...")
    
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.base import BaseEstimator, ClassifierMixin
    
    class LGBWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, model):
            self.model = model
        
        def fit(self, X, y):
            return self
        
        def predict_proba(self, X):
            probs = self.model.predict(X)
            return np.column_stack([1 - probs, probs])
    
    lgb_wrapper = LGBWrapper(lgb_model)
    calibrator = CalibratedClassifierCV(lgb_wrapper, cv='prefit', method='isotonic')
    calibrator.fit(X_cal, y_cal)
    
    return {'lgb_calibrator': calibrator}


def find_ensemble_weights(lgb_preds: np.ndarray, trans_preds: np.ndarray, 
                          y_true: np.ndarray) -> Dict[str, float]:
    """
    Find optimal ensemble weights.
    
    Args:
        lgb_preds: LightGBM predictions
        trans_preds: Transformer predictions
        y_true: True labels
        
    Returns:
        Dictionary with optimal weights
    """
    print("Searching for optimal ensemble weights...")
    
    best_score = 0
    best_weight = 0.5
    
    for w in np.arange(0, 1.01, 0.1):
        ensemble_preds = w * lgb_preds + (1 - w) * trans_preds
        score = roc_auc_score(y_true, ensemble_preds)
        if score > best_score:
            best_score = score
            best_weight = w
    
    print(f"Best weight: {best_weight:.2f}, AUC: {best_score:.4f}")
    return {'lgb_weight': best_weight, 'trans_weight': 1 - best_weight}


def evaluate_model(y_true: np.ndarray, y_pred_probs: np.ndarray, 
                   threshold: float = 0.5) -> Dict:
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred_probs: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    auc = roc_auc_score(y_true, y_pred_probs)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc)
    }
    
    return metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python train.py <data_dir> <output_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Load data
    print("Loading features and labels...")
    X, y = load_features_and_labels(data_dir)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train models
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
    trans_model = train_transformer(X_train, y_train, X_val, y_val)
    
    # Save models
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    lgb_model.save_model(str(output_path / 'lgb.txt'))
    
    trans_path = output_path / 'transformer'
    trans_path.mkdir(exist_ok=True)
    torch.save(trans_model.state_dict(), trans_path / 'model.pt')
    
    # Save feature metadata
    with open(output_path / 'lgb.txt.features.json', 'w') as f:
        json.dump({'num_features': X.shape[1]}, f)
    
    # Calibration
    calibrators = calibrate_models(lgb_model, X_val, y_val)
    with open(output_path / 'calibrators.pkl', 'wb') as f:
        pickle.dump(calibrators, f)
    
    # Ensemble weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trans_model.eval()
    with torch.no_grad():
        lgb_preds = lgb_model.predict(X_test)
        trans_preds = trans_model(torch.FloatTensor(X_test).to(device)).cpu().numpy().flatten()
    
    ensemble_weights = find_ensemble_weights(lgb_preds, trans_preds, y_test)
    with open(output_path / 'ensemble.json', 'w') as f:
        json.dump(ensemble_weights, f, indent=2)
    
    # Final evaluation
    ensemble_preds = (ensemble_weights['lgb_weight'] * lgb_preds + 
                      ensemble_weights['trans_weight'] * trans_preds)
    metrics = evaluate_model(y_test, ensemble_preds)
    
    print("\nFinal Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nModels saved to {output_dir}")
