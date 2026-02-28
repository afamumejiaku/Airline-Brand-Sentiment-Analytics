"""
Model Comparison: FTA-LSTM vs Bipartite GNN
============================================
Comprehensive comparison of both architectures for sentiment classification.

Usage: python compare_models.py
"""

import numpy as np
import pandas as pd
import sqlite3
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Import our models
from fta_lstm import (
    FTALTM, TextPreprocessor, SentimentDataset,
    train_model as train_lstm, evaluate_model as eval_lstm,
    cross_validate as cv_lstm
)

try:
    from bipartite_gnn import (
        GraphBuilder, HomogeneousGNN,
        cross_validate_gnn, evaluate_gnn
    )
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    print("⚠ torch_geometric not available. GNN comparison will be skipped.")

from torch.utils.data import DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "airline_sentiment.db"
RESULTS_PATH = "comparison_results.json"

CONFIG = {
    'random_seed': 42,
    'test_size': 0.15,
    'val_size': 0.15,
    'n_folds': 5,
    
    # FTA-LSTM params
    'lstm': {
        'max_vocab': 20000,
        'max_length': 128,
        'embedding_dim': 200,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'epochs': 15,
        'batch_size': 64,
        'learning_rate': 0.001
    },
    
    # GNN params
    'gnn': {
        'embedding_dim': 128,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'epochs': 100,
        'learning_rate': 0.01
    }
}


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load data from database."""
    conn = sqlite3.connect(DB_PATH)
    
    df = pd.read_sql_query("""
        SELECT 
            p.post_id,
            p.author,
            p.full_text,
            p.score,
            p.created_date,
            pa.airline,
            s.true_label as sentiment,
            s.aspect
        FROM posts p
        JOIN post_airlines pa ON p.post_id = pa.post_id
        JOIN sentiment_labels s ON p.post_id = s.post_id
    """, conn)
    
    conn.close()
    
    print(f"Loaded {len(df):,} samples")
    print(f"\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    
    return df


def prepare_lstm_data(df, config):
    """Prepare data for FTA-LSTM."""
    print("\n" + "="*50)
    print("PREPARING DATA FOR FTA-LSTM")
    print("="*50)
    
    # Encode labels
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    y = df['sentiment'].map(label_map).values
    
    # Preprocess text
    preprocessor = TextPreprocessor(
        max_vocab=config['max_vocab'],
        max_length=config['max_length']
    )
    X = preprocessor.fit_transform(df['full_text'].tolist())
    
    print(f"Vocabulary size: {len(preprocessor.word2idx):,}")
    print(f"Sequence shape: {X.shape}")
    
    return X, y, preprocessor


def prepare_gnn_data(config):
    """Prepare data for Bipartite GNN."""
    if not GNN_AVAILABLE:
        return None
    
    print("\n" + "="*50)
    print("PREPARING DATA FOR BIPARTITE GNN")
    print("="*50)
    
    builder = GraphBuilder(DB_PATH)
    posts_df, users_df, edges_df = builder.load_data()
    
    # Build homogeneous graph
    data = builder.build_homogeneous_graph(
        posts_df, users_df, edges_df,
        embedding_dim=config['embedding_dim']
    )
    
    return data


# ============================================================
# MODEL TRAINING & EVALUATION
# ============================================================

def run_lstm_experiment(X, y, preprocessor, config):
    """Run FTA-LSTM cross-validation experiment."""
    print("\n" + "="*60)
    print("FTA-LSTM EXPERIMENT")
    print("="*60)
    
    results = cv_lstm(
        X, y,
        vocab_size=len(preprocessor.word2idx),
        n_folds=CONFIG['n_folds'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=3,
        dropout=config['dropout']
    )
    
    return results


def run_gnn_experiment(data, config):
    """Run Bipartite GNN cross-validation experiment."""
    if not GNN_AVAILABLE or data is None:
        print("\n⚠ Skipping GNN experiment (torch_geometric not available)")
        return None
    
    print("\n" + "="*60)
    print("BIPARTITE GNN EXPERIMENT")
    print("="*60)
    
    results = cross_validate_gnn(
        data,
        n_folds=CONFIG['n_folds'],
        epochs=config['epochs'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    return results


# ============================================================
# BASELINE MODELS
# ============================================================

def run_baselines(df):
    """Run baseline models for comparison."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    print("\n" + "="*60)
    print("BASELINE MODELS")
    print("="*60)
    
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    y = df['sentiment'].map(label_map).values
    
    results = {}
    
    # VADER baseline
    print("\n1. VADER (lexicon-based)...")
    vader = SentimentIntensityAnalyzer()
    vader_preds = []
    for text in df['full_text']:
        score = vader.polarity_scores(str(text))['compound']
        if score >= 0.05:
            vader_preds.append(2)
        elif score <= -0.05:
            vader_preds.append(0)
        else:
            vader_preds.append(1)
    
    from sklearn.metrics import accuracy_score, f1_score
    results['vader'] = {
        'accuracy': accuracy_score(y, vader_preds),
        'f1_macro': f1_score(y, vader_preds, average='macro')
    }
    print(f"   Accuracy: {results['vader']['accuracy']:.4f}")
    print(f"   F1 (macro): {results['vader']['f1_macro']:.4f}")
    
    # TF-IDF + Logistic Regression
    print("\n2. TF-IDF + Logistic Regression...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df['full_text'].fillna(''))
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X_tfidf, y, cv=5, scoring='f1_macro')
    lr_acc = cross_val_score(lr, X_tfidf, y, cv=5, scoring='accuracy')
    
    results['tfidf_lr'] = {
        'accuracy': lr_acc.mean(),
        'accuracy_std': lr_acc.std(),
        'f1_macro': lr_scores.mean(),
        'f1_macro_std': lr_scores.std()
    }
    print(f"   Accuracy: {results['tfidf_lr']['accuracy']:.4f} ± {results['tfidf_lr']['accuracy_std']:.4f}")
    print(f"   F1 (macro): {results['tfidf_lr']['f1_macro']:.4f} ± {results['tfidf_lr']['f1_macro_std']:.4f}")
    
    # TF-IDF + Random Forest
    print("\n3. TF-IDF + Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf, X_tfidf, y, cv=5, scoring='f1_macro')
    rf_acc = cross_val_score(rf, X_tfidf, y, cv=5, scoring='accuracy')
    
    results['tfidf_rf'] = {
        'accuracy': rf_acc.mean(),
        'accuracy_std': rf_acc.std(),
        'f1_macro': rf_scores.mean(),
        'f1_macro_std': rf_scores.std()
    }
    print(f"   Accuracy: {results['tfidf_rf']['accuracy']:.4f} ± {results['tfidf_rf']['accuracy_std']:.4f}")
    print(f"   F1 (macro): {results['tfidf_rf']['f1_macro']:.4f} ± {results['tfidf_rf']['f1_macro_std']:.4f}")
    
    return results


# ============================================================
# STATISTICAL TESTS
# ============================================================

def statistical_comparison(lstm_results, gnn_results, baseline_results):
    """Perform statistical tests comparing models."""
    from scipy import stats
    
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON")
    print("="*60)
    
    # Extract F1 scores from each fold
    lstm_f1s = [r['f1_macro'] for r in lstm_results['fold_results']]
    
    comparisons = {}
    
    # Compare LSTM vs baselines
    print("\nFTA-LSTM vs TF-IDF + LR:")
    # Using paired t-test (simplified - ideally use same folds)
    t_stat, p_value = stats.ttest_1samp(lstm_f1s, baseline_results['tfidf_lr']['f1_macro'])
    comparisons['lstm_vs_lr'] = {'t_stat': t_stat, 'p_value': p_value}
    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    print(f"  {'✓ Significant' if p_value < 0.05 else '✗ Not significant'} at α=0.05")
    
    if gnn_results:
        gnn_f1s = [r['f1_macro'] for r in gnn_results['fold_results']]
        
        print("\nFTA-LSTM vs Bipartite GNN:")
        t_stat, p_value = stats.ttest_rel(lstm_f1s, gnn_f1s)
        comparisons['lstm_vs_gnn'] = {'t_stat': t_stat, 'p_value': p_value}
        print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
        print(f"  {'✓ Significant' if p_value < 0.05 else '✗ Not significant'} at α=0.05")
    
    return comparisons


# ============================================================
# VISUALIZATION
# ============================================================

def plot_results(baseline_results, lstm_results, gnn_results):
    """Create comparison visualizations."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Collect all results
    models = []
    accuracies = []
    f1_scores = []
    errors = []
    
    # Baselines
    models.extend(['VADER', 'TF-IDF + LR', 'TF-IDF + RF'])
    accuracies.extend([
        baseline_results['vader']['accuracy'],
        baseline_results['tfidf_lr']['accuracy'],
        baseline_results['tfidf_rf']['accuracy']
    ])
    f1_scores.extend([
        baseline_results['vader']['f1_macro'],
        baseline_results['tfidf_lr']['f1_macro'],
        baseline_results['tfidf_rf']['f1_macro']
    ])
    errors.extend([0, baseline_results['tfidf_lr']['f1_macro_std'], 
                   baseline_results['tfidf_rf']['f1_macro_std']])
    
    # FTA-LSTM
    models.append('FTA-LSTM')
    accuracies.append(lstm_results['accuracy'])
    f1_scores.append(lstm_results['f1_macro'])
    errors.append(lstm_results['f1_macro_std'])
    
    # GNN
    if gnn_results:
        models.append('Bipartite GNN')
        accuracies.append(gnn_results['accuracy'])
        f1_scores.append(gnn_results['f1_macro'])
        errors.append(gnn_results['f1_macro_std'])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    ax1 = axes[0]
    x = np.arange(len(models))
    bars1 = ax1.bar(x, accuracies, color=['#95a5a6', '#3498db', '#3498db', '#e74c3c', '#27ae60'][:len(models)])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=max(f1_scores), color='r', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # F1 Score comparison with error bars
    ax2 = axes[1]
    bars2 = ax2.bar(x, f1_scores, yerr=errors, capsize=5,
                    color=['#95a5a6', '#3498db', '#3498db', '#e74c3c', '#27ae60'][:len(models)])
    ax2.set_ylabel('F1 Score (Macro)')
    ax2.set_title('Model F1 Score Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    
    for bar, f1 in zip(bars2, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: model_comparison.png")
    
    # Confusion matrix for best model
    if lstm_results.get('fold_results'):
        best_fold = max(lstm_results['fold_results'], key=lambda x: x['f1_macro'])
        cm = best_fold['confusion_matrix']
        
        fig2, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('FTA-LSTM Confusion Matrix (Best Fold)')
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: confusion_matrix.png")
    
    plt.close('all')


# ============================================================
# RESULTS SUMMARY
# ============================================================

def print_summary(baseline_results, lstm_results, gnn_results):
    """Print final results summary."""
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    print("\n{:<25} {:>12} {:>12} {:>15}".format(
        "Model", "Accuracy", "F1 (Macro)", "Std Dev"
    ))
    print("-"*70)
    
    # Baselines
    print("{:<25} {:>12.4f} {:>12.4f} {:>15}".format(
        "VADER (baseline)", 
        baseline_results['vader']['accuracy'],
        baseline_results['vader']['f1_macro'],
        "-"
    ))
    print("{:<25} {:>12.4f} {:>12.4f} {:>15.4f}".format(
        "TF-IDF + Log. Regression",
        baseline_results['tfidf_lr']['accuracy'],
        baseline_results['tfidf_lr']['f1_macro'],
        baseline_results['tfidf_lr']['f1_macro_std']
    ))
    print("{:<25} {:>12.4f} {:>12.4f} {:>15.4f}".format(
        "TF-IDF + Random Forest",
        baseline_results['tfidf_rf']['accuracy'],
        baseline_results['tfidf_rf']['f1_macro'],
        baseline_results['tfidf_rf']['f1_macro_std']
    ))
    
    # Deep learning models
    print("-"*70)
    print("{:<25} {:>12.4f} {:>12.4f} {:>15.4f}".format(
        "FTA-LSTM",
        lstm_results['accuracy'],
        lstm_results['f1_macro'],
        lstm_results['f1_macro_std']
    ))
    
    if gnn_results:
        print("{:<25} {:>12.4f} {:>12.4f} {:>15.4f}".format(
            "Bipartite GNN",
            gnn_results['accuracy'],
            gnn_results['f1_macro'],
            gnn_results['f1_macro_std']
        ))
    
    print("-"*70)
    
    # Winner
    all_f1s = {
        'VADER': baseline_results['vader']['f1_macro'],
        'TF-IDF + LR': baseline_results['tfidf_lr']['f1_macro'],
        'TF-IDF + RF': baseline_results['tfidf_rf']['f1_macro'],
        'FTA-LSTM': lstm_results['f1_macro']
    }
    if gnn_results:
        all_f1s['Bipartite GNN'] = gnn_results['f1_macro']
    
    winner = max(all_f1s, key=all_f1s.get)
    print(f"\n🏆 Best Model: {winner} (F1 = {all_f1s[winner]:.4f})")
    
    # Key insights
    print("\n📊 KEY INSIGHTS:")
    print(f"  • FTA-LSTM improvement over best baseline: {(lstm_results['f1_macro'] - baseline_results['tfidf_rf']['f1_macro'])*100:.1f}%")
    if gnn_results:
        print(f"  • FTA-LSTM vs GNN difference: {(lstm_results['f1_macro'] - gnn_results['f1_macro'])*100:.1f}%")


def save_results(baseline_results, lstm_results, gnn_results, stats):
    """Save all results to JSON."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'baselines': baseline_results,
        'fta_lstm': {
            'accuracy': lstm_results['accuracy'],
            'accuracy_std': lstm_results['accuracy_std'],
            'f1_macro': lstm_results['f1_macro'],
            'f1_macro_std': lstm_results['f1_macro_std']
        },
        'statistical_tests': stats
    }
    
    if gnn_results:
        results['bipartite_gnn'] = {
            'accuracy': gnn_results['accuracy'],
            'accuracy_std': gnn_results['accuracy_std'],
            'f1_macro': gnn_results['f1_macro'],
            'f1_macro_std': gnn_results['f1_macro_std']
        }
    
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {RESULTS_PATH}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*70)
    print("MODEL COMPARISON: FTA-LSTM vs BIPARTITE GNN")
    print("Airline Sentiment Classification")
    print("="*70)
    
    # Set seeds
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
    # Load data
    print("\n📥 Loading data...")
    df = load_data()
    
    # Run baselines
    baseline_results = run_baselines(df)
    
    # Prepare and run FTA-LSTM
    X, y, preprocessor = prepare_lstm_data(df, CONFIG['lstm'])
    lstm_results = run_lstm_experiment(X, y, preprocessor, CONFIG['lstm'])
    
    # Prepare and run GNN
    gnn_data = prepare_gnn_data(CONFIG['gnn'])
    gnn_results = run_gnn_experiment(gnn_data, CONFIG['gnn'])
    
    # Statistical comparison
    stats = statistical_comparison(lstm_results, gnn_results, baseline_results)
    
    # Visualize
    plot_results(baseline_results, lstm_results, gnn_results)
    
    # Summary
    print_summary(baseline_results, lstm_results, gnn_results)
    
    # Save
    save_results(baseline_results, lstm_results, gnn_results, stats)
    
    print("\n" + "="*70)
    print("✓ COMPARISON COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
