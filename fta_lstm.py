"""
FTA-LSTM: Frequency-Time Attention LSTM
========================================
Implementation for airline sentiment classification.

Architecture:
1. Embedding Layer (GloVe/Word2Vec)
2. Frequency Attention (word importance)
3. Bidirectional LSTM
4. Time Attention (temporal patterns)
5. Dense Classification Head

Usage:
    from fta_lstm import FTALTM, train_model, evaluate_model
    model = FTALTM(vocab_size, embedding_dim, hidden_dim, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from collections import Counter
import pickle
import re
from tqdm import tqdm


# ============================================================
# TEXT PREPROCESSING
# ============================================================

class TextPreprocessor:
    """Tokenize and numericalize text for LSTM input."""
    
    def __init__(self, max_vocab: int = 30000, max_length: int = 256):
        self.max_vocab = max_vocab
        self.max_length = max_length
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_freq = Counter()
        self.fitted = False
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters but keep sentiment punctuation
        text = re.sub(r'[^\w\s!?.,\'-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        return self.clean_text(text).split()
    
    def fit(self, texts: List[str]) -> 'TextPreprocessor':
        """Build vocabulary from texts."""
        print("Building vocabulary...")
        
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = self.tokenize(text)
            self.word_freq.update(tokens)
        
        # Keep top words
        most_common = self.word_freq.most_common(self.max_vocab - 2)
        
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Vocabulary size: {len(self.word2idx):,}")
        self.fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Convert texts to padded sequences."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        sequences = []
        for text in texts:
            tokens = self.tokenize(text)[:self.max_length]
            indices = [self.word2idx.get(t, 1) for t in tokens]  # 1 = <UNK>
            
            # Pad or truncate
            if len(indices) < self.max_length:
                indices = indices + [0] * (self.max_length - len(indices))
            
            sequences.append(indices)
        
        return np.array(sequences)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)
    
    def save(self, path: str):
        """Save preprocessor to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'max_vocab': self.max_vocab,
                'max_length': self.max_length
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'TextPreprocessor':
        """Load preprocessor from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls(data['max_vocab'], data['max_length'])
        preprocessor.word2idx = data['word2idx']
        preprocessor.idx2word = data['idx2word']
        preprocessor.fitted = True
        return preprocessor


# ============================================================
# DATASET
# ============================================================

class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment classification."""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ============================================================
# ATTENTION MECHANISMS
# ============================================================

class FrequencyAttention(nn.Module):
    """
    Frequency Attention: Learn importance of each word/token.
    
    Computes attention weights for each position in the sequence
    based on the embedding itself (which words matter for sentiment).
    """
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1)
        )
    
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (batch, seq_len, embedding_dim)
            mask: (batch, seq_len) - 1 for real tokens, 0 for padding
        
        Returns:
            weighted_embeddings: (batch, seq_len, embedding_dim)
            attention_weights: (batch, seq_len)
        """
        # Compute attention scores
        attn_scores = self.attention(embeddings).squeeze(-1)  # (batch, seq_len)
        
        # Mask padding tokens
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, seq_len)
        
        # Apply attention (element-wise weighting)
        weighted = embeddings * attn_weights.unsqueeze(-1)
        
        return weighted, attn_weights


class TimeAttention(nn.Module):
    """
    Time Attention: Learn importance of each time step (LSTM output).
    
    After LSTM processing, determines which positions in the sequence
    contribute most to the final classification.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, lstm_output: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len)
        
        Returns:
            context_vector: (batch, hidden_dim) - weighted sum of LSTM outputs
            attention_weights: (batch, seq_len)
        """
        attn_scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, seq_len)
        
        # Weighted sum to get context vector
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden_dim)
        
        return context, attn_weights


# ============================================================
# FTA-LSTM MODEL
# ============================================================

class FTALTM(nn.Module):
    """
    Frequency-Time Attention LSTM for Sentiment Classification.
    
    Architecture:
        Input -> Embedding -> FreqAttention -> BiLSTM -> TimeAttention -> Dense -> Output
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_classes: int = 3,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[np.ndarray] = None
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.FloatTensor(pretrained_embeddings))
            self.embedding.weight.requires_grad = True  # Fine-tune embeddings
        
        # Frequency attention (on embeddings)
        self.freq_attention = FrequencyAttention(embedding_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Time attention (on LSTM output)
        self.time_attention = TimeAttention(hidden_dim * 2)  # *2 for bidirectional
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: (batch, seq_len) - input token indices
            return_attention: if True, also return attention weights
        
        Returns:
            logits: (batch, num_classes)
            attention_dict: (optional) dict with 'freq' and 'time' attention weights
        """
        # Create mask for padding tokens
        mask = (x != 0).float()  # (batch, seq_len)
        
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # Frequency attention
        freq_weighted, freq_attn = self.freq_attention(embedded, mask)
        
        # LSTM
        lstm_out, _ = self.lstm(freq_weighted)  # (batch, seq_len, hidden_dim*2)
        
        # Time attention
        context, time_attn = self.time_attention(lstm_out, mask)  # (batch, hidden_dim*2)
        
        # Classification
        logits = self.classifier(context)  # (batch, num_classes)
        
        if return_attention:
            return logits, {'freq': freq_attn, 'time': time_attn}
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


# ============================================================
# TRAINING UTILITIES
# ============================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model on data."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            logits = model(sequences)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(data_loader), np.array(all_preds), np.array(all_labels)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 0.001,
    device: torch.device = None,
    patience: int = 5
) -> Dict:
    """
    Full training loop with early stopping.
    
    Returns:
        dict with training history and best metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': []
    }
    
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate on train set (sample)
        _, train_preds, train_labels = evaluate(model, train_loader, criterion, device)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        # Evaluate on validation set
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    history['best_val_f1'] = best_val_f1
    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = None,
    label_names: List[str] = None
) -> Dict:
    """
    Full evaluation with all metrics.
    
    Returns:
        dict with accuracy, f1, precision, recall, confusion matrix
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            
            logits = model(sequences)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'precision_macro': precision_score(all_labels, all_preds, average='macro'),
        'recall_macro': recall_score(all_labels, all_preds, average='macro'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'predictions': all_preds,
        'probabilities': all_probs
    }
    
    # ROC-AUC (one-vs-rest)
    try:
        results['roc_auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except:
        results['roc_auc'] = None
    
    # Classification report
    if label_names:
        results['classification_report'] = classification_report(
            all_labels, all_preds, target_names=label_names
        )
    
    return results


# ============================================================
# ATTENTION VISUALIZATION
# ============================================================

def get_attention_weights(
    model: nn.Module,
    sequences: torch.Tensor,
    device: torch.device = None
) -> Dict[str, np.ndarray]:
    """Extract attention weights for interpretability."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    sequences = sequences.to(device)
    
    with torch.no_grad():
        _, attention = model(sequences, return_attention=True)
    
    return {
        'freq_attention': attention['freq'].cpu().numpy(),
        'time_attention': attention['time'].cpu().numpy()
    }


def visualize_attention(
    text: str,
    preprocessor: TextPreprocessor,
    model: nn.Module,
    device: torch.device = None
) -> Dict:
    """
    Visualize attention weights for a single text.
    
    Returns:
        dict with tokens, freq_weights, time_weights, prediction
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tokenize and convert
    tokens = preprocessor.tokenize(text)[:preprocessor.max_length]
    sequence = preprocessor.transform([text])
    sequence_tensor = torch.LongTensor(sequence).to(device)
    
    # Get prediction and attention
    model.eval()
    with torch.no_grad():
        logits, attention = model(sequence_tensor, return_attention=True)
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1).item()
    
    # Get weights for actual tokens (not padding)
    freq_weights = attention['freq'][0].cpu().numpy()[:len(tokens)]
    time_weights = attention['time'][0].cpu().numpy()[:len(tokens)]
    
    return {
        'tokens': tokens,
        'freq_weights': freq_weights,
        'time_weights': time_weights,
        'prediction': pred,
        'probabilities': probs[0].cpu().numpy()
    }


# ============================================================
# CROSS-VALIDATION
# ============================================================

def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    vocab_size: int,
    n_folds: int = 5,
    epochs: int = 15,
    batch_size: int = 64,
    **model_kwargs
) -> Dict:
    """
    K-fold cross-validation for FTA-LSTM.
    
    Returns:
        dict with fold results and aggregate metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*50}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create datasets
        train_dataset = SentimentDataset(X_train, y_train)
        val_dataset = SentimentDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        model = FTALTM(vocab_size=vocab_size, **model_kwargs)
        
        # Train
        history = train_model(model, train_loader, val_loader, epochs=epochs, device=device)
        
        # Evaluate
        results = evaluate_model(model, val_loader, device=device)
        results['history'] = history
        fold_results.append(results)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1 (macro): {results['f1_macro']:.4f}")
        print(f"  ROC-AUC: {results['roc_auc']:.4f}" if results['roc_auc'] else "  ROC-AUC: N/A")
    
    # Aggregate results
    aggregate = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
        'f1_macro': np.mean([r['f1_macro'] for r in fold_results]),
        'f1_macro_std': np.std([r['f1_macro'] for r in fold_results]),
        'roc_auc': np.mean([r['roc_auc'] for r in fold_results if r['roc_auc']]),
        'fold_results': fold_results
    }
    
    print(f"\n{'='*50}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {aggregate['accuracy']:.4f} ± {aggregate['accuracy_std']:.4f}")
    print(f"F1 (macro): {aggregate['f1_macro']:.4f} ± {aggregate['f1_macro_std']:.4f}")
    print(f"ROC-AUC: {aggregate['roc_auc']:.4f}")
    
    return aggregate


# ============================================================
# MAIN EXAMPLE
# ============================================================

if __name__ == "__main__":
    # Example usage
    print("FTA-LSTM Model for Sentiment Classification")
    print("="*50)
    
    # Load data (example with synthetic data)
    import sqlite3
    
    DB_PATH = "airline_sentiment.db"
    
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("""
            SELECT p.full_text, s.true_label
            FROM posts p
            JOIN sentiment_labels s ON p.post_id = s.post_id
        """, conn)
        conn.close()
        
        print(f"Loaded {len(df):,} samples")
        
        # Encode labels
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        df['label'] = df['true_label'].map(label_map)
        
        # Preprocess
        preprocessor = TextPreprocessor(max_vocab=20000, max_length=128)
        X = preprocessor.fit_transform(df['full_text'].tolist())
        y = df['label'].values
        
        print(f"Vocabulary size: {len(preprocessor.word2idx):,}")
        print(f"Sequence shape: {X.shape}")
        
        # Run cross-validation
        results = cross_validate(
            X, y,
            vocab_size=len(preprocessor.word2idx),
            n_folds=5,
            epochs=10,
            batch_size=64,
            embedding_dim=200,
            hidden_dim=128,
            num_classes=3,
            dropout=0.3
        )
        
    except Exception as e:
        print(f"Error: {e}")
        print("Run generate_dataset.py first to create the database.")
