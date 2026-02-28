"""
Bipartite Graph Neural Network for Sentiment Classification
============================================================
Models user-post relationships for sentiment prediction.

Graph Structure:
    - User nodes: authors with behavior features
    - Post nodes: posts with text embeddings
    - Edges: authored, commented_on, similar_to

Uses GraphSAGE/GAT for message passing.

Requirements:
    pip install torch-geometric torch-scatter torch-sparse

Usage:
    from bipartite_gnn import BipartiteGNN, build_graph, train_model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import sqlite3
from tqdm import tqdm
import warnings

# Try to import torch_geometric
try:
    import torch_geometric
    from torch_geometric.data import Data, HeteroData
    from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear
    from torch_geometric.loader import NeighborLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    warnings.warn("torch_geometric not installed. Install with: pip install torch-geometric")


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

class GraphBuilder:
    """Build bipartite graph from database."""
    
    def __init__(self, db_path: str = "airline_sentiment.db"):
        self.db_path = db_path
        self.user_to_idx = {}
        self.post_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_post = {}
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load posts, users, and edges from database."""
        conn = sqlite3.connect(self.db_path)
        
        # Load posts with labels
        posts_df = pd.read_sql_query("""
            SELECT 
                p.post_id,
                p.author,
                p.full_text,
                p.score,
                p.num_comments,
                s.true_label,
                s.vader_score
            FROM posts p
            JOIN sentiment_labels s ON p.post_id = s.post_id
        """, conn)
        
        # Load users
        users_df = pd.read_sql_query("""
            SELECT 
                author,
                post_count,
                comment_count,
                total_score,
                avg_score,
                sentiment_tendency
            FROM users
            WHERE post_count > 0 OR comment_count > 0
        """, conn)
        
        # Load edges
        edges_df = pd.read_sql_query("""
            SELECT source_id, target_id, edge_type, weight
            FROM graph_edges
        """, conn)
        
        conn.close()
        
        print(f"Loaded {len(posts_df):,} posts, {len(users_df):,} users, {len(edges_df):,} edges")
        
        return posts_df, users_df, edges_df
    
    def build_index_mappings(self, posts_df: pd.DataFrame, users_df: pd.DataFrame):
        """Create mappings between IDs and indices."""
        # User mappings
        for idx, user in enumerate(users_df['author'].unique()):
            self.user_to_idx[user] = idx
            self.idx_to_user[idx] = user
        
        # Post mappings
        for idx, post_id in enumerate(posts_df['post_id'].unique()):
            self.post_to_idx[post_id] = idx
            self.idx_to_post[idx] = post_id
    
    def create_text_embeddings(
        self, 
        posts_df: pd.DataFrame, 
        method: str = 'tfidf',
        dim: int = 128
    ) -> np.ndarray:
        """
        Create text embeddings for posts.
        
        Methods:
            - 'tfidf': TF-IDF + SVD dimensionality reduction
            - 'mean_word2vec': Average word embeddings (requires gensim)
            - 'random': Random embeddings (for testing)
        """
        if method == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(posts_df['full_text'].fillna(''))
            
            svd = TruncatedSVD(n_components=dim, random_state=42)
            embeddings = svd.fit_transform(tfidf_matrix)
            
            print(f"Created TF-IDF embeddings: {embeddings.shape}")
            return embeddings
        
        elif method == 'random':
            # Random embeddings for testing
            embeddings = np.random.randn(len(posts_df), dim).astype(np.float32)
            return embeddings
        
        else:
            raise ValueError(f"Unknown embedding method: {method}")
    
    def create_user_features(self, users_df: pd.DataFrame) -> np.ndarray:
        """Create feature vectors for user nodes."""
        feature_cols = ['post_count', 'comment_count', 'total_score', 'avg_score', 'sentiment_tendency']
        
        # Fill NaN and scale
        features = users_df[feature_cols].fillna(0).values
        
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        print(f"Created user features: {features.shape}")
        return features.astype(np.float32)
    
    def build_heterogeneous_graph(
        self, 
        posts_df: pd.DataFrame,
        users_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        embedding_dim: int = 128
    ) -> 'HeteroData':
        """
        Build PyTorch Geometric HeteroData graph.
        
        Node types:
            - 'user': author nodes
            - 'post': post nodes
        
        Edge types:
            - ('user', 'authored', 'post')
            - ('user', 'commented', 'post')
            - ('post', 'rev_authored', 'user')  # reverse edges
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for graph construction")
        
        self.build_index_mappings(posts_df, users_df)
        
        data = HeteroData()
        
        # User node features
        user_features = self.create_user_features(users_df)
        data['user'].x = torch.FloatTensor(user_features)
        data['user'].num_nodes = len(users_df)
        
        # Post node features (text embeddings)
        post_embeddings = self.create_text_embeddings(posts_df, method='tfidf', dim=embedding_dim)
        data['post'].x = torch.FloatTensor(post_embeddings)
        data['post'].num_nodes = len(posts_df)
        
        # Post labels
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        labels = posts_df['true_label'].map(label_map).values
        data['post'].y = torch.LongTensor(labels)
        
        # Build edge indices
        authored_src, authored_dst = [], []
        commented_src, commented_dst = [], []
        
        for _, row in edges_df.iterrows():
            if row['edge_type'] == 'authored':
                if row['source_id'] in self.user_to_idx and row['target_id'] in self.post_to_idx:
                    authored_src.append(self.user_to_idx[row['source_id']])
                    authored_dst.append(self.post_to_idx[row['target_id']])
            
            elif row['edge_type'] == 'commented':
                if row['source_id'] in self.user_to_idx and row['target_id'] in self.post_to_idx:
                    commented_src.append(self.user_to_idx[row['source_id']])
                    commented_dst.append(self.post_to_idx[row['target_id']])
        
        # Add edges (and reverse edges for message passing)
        if authored_src:
            data['user', 'authored', 'post'].edge_index = torch.LongTensor([authored_src, authored_dst])
            data['post', 'rev_authored', 'user'].edge_index = torch.LongTensor([authored_dst, authored_src])
        
        if commented_src:
            data['user', 'commented', 'post'].edge_index = torch.LongTensor([commented_src, commented_dst])
            data['post', 'rev_commented', 'user'].edge_index = torch.LongTensor([commented_dst, commented_src])
        
        print(f"Built heterogeneous graph:")
        print(f"  User nodes: {data['user'].num_nodes:,}")
        print(f"  Post nodes: {data['post'].num_nodes:,}")
        print(f"  Authored edges: {len(authored_src):,}")
        print(f"  Commented edges: {len(commented_src):,}")
        
        return data
    
    def build_homogeneous_graph(
        self,
        posts_df: pd.DataFrame,
        users_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        embedding_dim: int = 128
    ) -> 'Data':
        """
        Build simplified homogeneous graph (posts only with similarity edges).
        
        Easier to work with for initial experiments.
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for graph construction")
        
        self.build_index_mappings(posts_df, users_df)
        
        # Create post embeddings
        embeddings = self.create_text_embeddings(posts_df, method='tfidf', dim=embedding_dim)
        
        # Create similarity edges (top-k similar posts)
        print("Computing similarity edges...")
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Compute similarities in batches to save memory
        k = 10  # top-k neighbors
        edge_src, edge_dst = [], []
        
        batch_size = 1000
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Building similarity graph"):
            batch = embeddings[i:i+batch_size]
            similarities = cosine_similarity(batch, embeddings)
            
            for j, sim_row in enumerate(similarities):
                post_idx = i + j
                # Get top-k most similar (excluding self)
                sim_row[post_idx] = -1  # Exclude self
                top_k_indices = np.argpartition(sim_row, -k)[-k:]
                
                for neighbor_idx in top_k_indices:
                    if sim_row[neighbor_idx] > 0.3:  # Similarity threshold
                        edge_src.append(post_idx)
                        edge_dst.append(neighbor_idx)
        
        # Build graph
        data = Data()
        data.x = torch.FloatTensor(embeddings)
        
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        data.y = torch.LongTensor(posts_df['true_label'].map(label_map).values)
        
        data.edge_index = torch.LongTensor([edge_src, edge_dst])
        
        print(f"Built homogeneous graph:")
        print(f"  Nodes: {data.x.shape[0]:,}")
        print(f"  Edges: {data.edge_index.shape[1]:,}")
        
        return data


# ============================================================
# GNN MODELS
# ============================================================

class HomogeneousGNN(nn.Module):
    """
    Simple GNN for homogeneous post similarity graph.
    
    Uses GraphSAGE convolutions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.classifier(x)
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings before classification."""
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return x


class HeterogeneousGNN(nn.Module):
    """
    Heterogeneous GNN for user-post bipartite graph.
    
    Separate convolutions for different edge types.
    """
    
    def __init__(
        self,
        user_input_dim: int,
        post_input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Initial projections
        self.user_lin = nn.Linear(user_input_dim, hidden_dim)
        self.post_lin = nn.Linear(post_input_dim, hidden_dim)
        
        # Heterogeneous convolutions
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'authored', 'post'): SAGEConv(hidden_dim, hidden_dim),
                ('user', 'commented', 'post'): SAGEConv(hidden_dim, hidden_dim),
                ('post', 'rev_authored', 'user'): SAGEConv(hidden_dim, hidden_dim),
                ('post', 'rev_commented', 'user'): SAGEConv(hidden_dim, hidden_dim),
            }, aggr='mean')
            self.convs.append(conv)
        
        # Classifier (for post nodes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = dropout
    
    def forward(self, x_dict: Dict, edge_index_dict: Dict) -> torch.Tensor:
        """Forward pass."""
        # Initial projection
        x_dict = {
            'user': F.relu(self.user_lin(x_dict['user'])),
            'post': F.relu(self.post_lin(x_dict['post']))
        }
        
        # Message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                     for key, x in x_dict.items()}
        
        # Classify post nodes
        return self.classifier(x_dict['post'])


# ============================================================
# TRAINING UTILITIES
# ============================================================

def train_homogeneous(
    model: HomogeneousGNN,
    data: 'Data',
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.01,
    patience: int = 10,
    device: torch.device = None
) -> Dict:
    """Train homogeneous GNN with node-level split."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            
            train_pred = out[train_mask].argmax(dim=1)
            train_acc = (train_pred == data.y[train_mask]).float().mean().item()
            
            val_pred = out[val_mask].argmax(dim=1)
            val_acc = (val_pred == data.y[val_mask]).float().mean().item()
            val_loss = criterion(out[val_mask], data.y[val_mask]).item()
        
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    history['best_val_acc'] = best_val_acc
    return history


def evaluate_gnn(
    model: nn.Module,
    data: 'Data',
    test_mask: torch.Tensor,
    device: torch.device = None
) -> Dict:
    """Evaluate GNN on test set."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    data = data.to(device)
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out[test_mask], dim=1)
        preds = out[test_mask].argmax(dim=1)
        labels = data.y[test_mask]
    
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    probs_np = probs.cpu().numpy()
    
    results = {
        'accuracy': accuracy_score(labels_np, preds_np),
        'f1_macro': f1_score(labels_np, preds_np, average='macro'),
        'f1_weighted': f1_score(labels_np, preds_np, average='weighted'),
        'precision_macro': precision_score(labels_np, preds_np, average='macro'),
        'recall_macro': recall_score(labels_np, preds_np, average='macro'),
        'confusion_matrix': confusion_matrix(labels_np, preds_np),
        'predictions': preds_np,
        'probabilities': probs_np
    }
    
    try:
        results['roc_auc'] = roc_auc_score(labels_np, probs_np, multi_class='ovr')
    except:
        results['roc_auc'] = None
    
    return results


# ============================================================
# CROSS-VALIDATION
# ============================================================

def cross_validate_gnn(
    data: 'Data',
    n_folds: int = 5,
    epochs: int = 100,
    hidden_dim: int = 128,
    num_layers: int = 2,
    **model_kwargs
) -> Dict:
    """K-fold cross-validation for GNN."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    labels = data.y.cpu().numpy()
    indices = np.arange(len(labels))
    
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(indices, labels)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*50}")
        
        # Create masks
        train_mask = torch.zeros(len(labels), dtype=torch.bool)
        test_mask = torch.zeros(len(labels), dtype=torch.bool)
        
        # Use 80% of train for training, 20% for validation
        val_size = int(len(train_idx) * 0.2)
        val_idx = train_idx[:val_size]
        train_idx = train_idx[val_size:]
        
        train_mask[train_idx] = True
        val_mask = torch.zeros(len(labels), dtype=torch.bool)
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        # Initialize model
        model = HomogeneousGNN(
            input_dim=data.x.shape[1],
            hidden_dim=hidden_dim,
            output_dim=3,
            num_layers=num_layers,
            **model_kwargs
        )
        
        # Train
        history = train_homogeneous(
            model, data, train_mask, val_mask,
            epochs=epochs, device=device
        )
        
        # Evaluate
        results = evaluate_gnn(model, data, test_mask, device=device)
        results['history'] = history
        fold_results.append(results)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1 (macro): {results['f1_macro']:.4f}")
    
    # Aggregate
    aggregate = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
        'f1_macro': np.mean([r['f1_macro'] for r in fold_results]),
        'f1_macro_std': np.std([r['f1_macro'] for r in fold_results]),
        'fold_results': fold_results
    }
    
    print(f"\n{'='*50}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {aggregate['accuracy']:.4f} ± {aggregate['accuracy_std']:.4f}")
    print(f"F1 (macro): {aggregate['f1_macro']:.4f} ± {aggregate['f1_macro_std']:.4f}")
    
    return aggregate


# ============================================================
# MAIN EXAMPLE
# ============================================================

if __name__ == "__main__":
    print("Bipartite Graph Neural Network for Sentiment Classification")
    print("="*60)
    
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("\n❌ torch_geometric not installed!")
        print("Install with:")
        print("  pip install torch-geometric torch-scatter torch-sparse")
        print("\nOr run the homogeneous version without graph structure.")
        exit(1)
    
    # Build graph
    builder = GraphBuilder("airline_sentiment.db")
    
    try:
        posts_df, users_df, edges_df = builder.load_data()
        
        # Build homogeneous graph (easier to start with)
        print("\nBuilding graph...")
        data = builder.build_homogeneous_graph(posts_df, users_df, edges_df, embedding_dim=128)
        
        # Run cross-validation
        results = cross_validate_gnn(
            data,
            n_folds=5,
            epochs=100,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3
        )
        
    except FileNotFoundError:
        print("\n❌ Database not found!")
        print("Run generate_dataset.py first to create the database.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
