# Airline Brand Sentiment Analytics
## FTA-LSTM vs Bipartite Graph: A Comparative Study

### Portfolio Project for Advanced Social Analytics

---

## 1. BUSINESS PROBLEM FRAMING

### The Ambiguous Ask
> *"We need to understand how customers really feel about our airline and how sentiment spreads through online communities."*

### Reframed Analytical Objectives

| Business Question | Analytical Objective | Method | Success Metric |
|-------------------|---------------------|--------|----------------|
| What's our overall brand health? | Sentiment classification | FTA-LSTM / Bipartite GNN | F1 Score ≥ 0.85 |
| How does sentiment spread? | Information propagation modeling | Bipartite Graph | Community detection accuracy |
| What topics drive sentiment? | Aspect-based sentiment extraction | Attention weights analysis | Interpretable topic clusters |
| Who are key opinion leaders? | Influence network analysis | Graph centrality metrics | Top 50 influencers identified |
| Can we predict sentiment shifts? | Temporal sentiment forecasting | FTA-LSTM time attention | RMSE < 0.15 on holdout |

---

## 2. RESEARCH QUESTION

**Can Frequency-Time Attention LSTM (FTA-LSTM) outperform Bipartite Graph Neural Networks for sentiment classification on social media text, and what are the trade-offs?**

### Hypothesis
- **H1**: FTA-LSTM will achieve higher classification accuracy due to attention mechanisms capturing contextual sentiment cues
- **H2**: Bipartite Graphs will better model sentiment propagation and community-level patterns
- **H3**: An ensemble combining both approaches will outperform either individually

---

## 3. TECHNICAL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Reddit API ──────► Raw Posts/Comments ──────► SQLite Database             │
│   (or Sample Data)        (JSON)                (Normalized Schema)          │
│                                                                              │
│   Tables: posts | comments | users | post_sentiment | aspect_sentiment       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PREPROCESSING LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Text Cleaning ──► Tokenization ──► Embedding ──► Sequence Padding         │
│                                                                              │
│   • Remove URLs, special chars          • Word2Vec / GloVe / FastText       │
│   • Normalize airline mentions          • BERT embeddings (optional)         │
│   • Handle Reddit-specific syntax       • Max sequence length: 256           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│        FTA-LSTM BRANCH        │   │    BIPARTITE GRAPH BRANCH     │
├───────────────────────────────┤   ├───────────────────────────────┤
│                               │   │                               │
│  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐  │
│  │   Embedding Layer       │  │   │  │   Graph Construction    │  │
│  │   (300-dim)             │  │   │  │   Users ←→ Posts        │  │
│  └───────────┬─────────────┘  │   │  └───────────┬─────────────┘  │
│              ▼                │   │              ▼                │
│  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐  │
│  │   Frequency Attention   │  │   │  │   Node Embedding        │  │
│  │   (Word importance)     │  │   │  │   (User + Post nodes)   │  │
│  └───────────┬─────────────┘  │   │  └───────────┬─────────────┘  │
│              ▼                │   │              ▼                │
│  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐  │
│  │   Bi-LSTM Layers        │  │   │  │   GNN Message Passing   │  │
│  │   (128 units × 2)       │  │   │  │   (GraphSAGE / GAT)     │  │
│  └───────────┬─────────────┘  │   │  └───────────┬─────────────┘  │
│              ▼                │   │              ▼                │
│  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐  │
│  │   Time Attention        │  │   │  │   Graph Pooling         │  │
│  │   (Temporal patterns)   │  │   │  │   (Aggregate nodes)     │  │
│  └───────────┬─────────────┘  │   │  └───────────┬─────────────┘  │
│              ▼                │   │              ▼                │
│  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐  │
│  │   Dense + Softmax       │  │   │  │   Dense + Softmax       │  │
│  │   (3-class output)      │  │   │  │   (3-class output)      │  │
│  └─────────────────────────┘  │   │  └─────────────────────────┘  │
│                               │   │                               │
└───────────────────────────────┘   └───────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EVALUATION LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   5-Fold Cross-Validation                                                   │
│                                                                              │
│   Metrics:                                                                   │
│   • Accuracy, Precision, Recall, F1 (per class & macro)                     │
│   • ROC-AUC (multi-class)                                                   │
│   • Confusion Matrix                                                         │
│   • Training Time & Inference Speed                                          │
│                                                                              │
│   Statistical Tests:                                                         │
│   • McNemar's Test (model comparison)                                       │
│   • Wilcoxon Signed-Rank (cross-validation folds)                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VISUALIZATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Power BI Dashboard                                                         │
│   ├── Executive Summary (KPIs, model comparison)                            │
│   ├── Sentiment Analysis (trends, distribution, aspects)                    │
│   ├── Network Visualization (influence graph, communities)                  │
│   ├── Model Performance (ROC curves, confusion matrices)                    │
│   └── Attention Visualization (what words/times matter)                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. MODEL DETAILS

### 4.1 FTA-LSTM (Frequency-Time Attention LSTM)

**Architecture:**
```
Input (text sequence)
    │
    ▼
Embedding Layer (GloVe 300d)
    │
    ▼
Frequency Attention ─────────────────────┐
    │                                    │
    │  Learns which WORDS matter most    │
    │  for sentiment (aspect keywords)   │
    │                                    │
    ▼                                    │
Bidirectional LSTM (128 units)           │
    │                                    │
    ▼                                    │
Time Attention ──────────────────────────┤
    │                                    │
    │  Learns which TIME STEPS matter    │
    │  (beginning vs end of post)        │
    │                                    │
    ▼                                    │
Attention-Weighted Hidden State          │
    │                                    │
    ▼                                    │
Dense Layer (64 units, ReLU)             │
    │                                    │
    ▼                                    │
Output (Softmax: Pos/Neu/Neg)            │
    │                                    │
    ▼                                    │
Attention Weights ◄──────────────────────┘
(Interpretability)
```

**Why FTA-LSTM for this problem:**
- Captures sequential context ("not bad" vs "bad")
- Frequency attention highlights sentiment-bearing words
- Time attention captures position-dependent patterns
- Attention weights provide interpretability

### 4.2 Bipartite Graph Neural Network

**Graph Structure:**
```
        USER NODES                    POST NODES
        ──────────                    ──────────
        
         [User A] ─────────────────► [Post 1] ──► Sentiment
              │                          ▲
              │                          │
              └──────────────────────────┤
                                         │
         [User B] ─────────────────► [Post 2] ──► Sentiment
              │         │                ▲
              │         │                │
              │         └────────────────┤
              │                          │
              └──────────────────────► [Post 3] ──► Sentiment
        
        
        Edge Types:
        • authored (user → post)
        • commented_on (user → post)
        • similar_to (post ↔ post, by embedding similarity)
```

**Architecture:**
```
Graph Construction
    │
    ▼
Node Feature Initialization
    │  • User nodes: activity stats, avg sentiment history
    │  • Post nodes: text embedding (BERT/GloVe avg)
    │
    ▼
GraphSAGE / GAT Layers (2-3 layers)
    │
    │  Message passing: aggregate neighbor information
    │  User nodes learn from their posts
    │  Post nodes learn from their authors + similar posts
    │
    ▼
Post Node Embeddings (enriched with graph context)
    │
    ▼
Classification Head (MLP → Softmax)
    │
    ▼
Output (Pos/Neu/Neg)
```

**Why Bipartite Graph for this problem:**
- Models user behavior patterns (consistently negative users)
- Captures community structure (airline fan vs critic communities)
- Leverages homophily (similar users have similar sentiment)
- Enables influence/propagation analysis

---

## 5. DATASET SPECIFICATION

### 5.1 Target Size
- **Posts**: 20,000 - 30,000
- **Comments**: 50,000 - 80,000
- **Unique Users**: 10,000 - 15,000
- **Time Span**: 12 months

### 5.2 Schema

```sql
-- Core tables
CREATE TABLE posts (
    post_id TEXT PRIMARY KEY,
    author TEXT,
    subreddit TEXT,
    title TEXT,
    body TEXT,
    full_text TEXT,
    score INTEGER,
    num_comments INTEGER,
    created_utc INTEGER,
    created_date DATE
);

CREATE TABLE comments (
    comment_id TEXT PRIMARY KEY,
    post_id TEXT,
    parent_id TEXT,
    author TEXT,
    body TEXT,
    score INTEGER,
    created_utc INTEGER
);

CREATE TABLE users (
    author TEXT PRIMARY KEY,
    post_count INTEGER,
    comment_count INTEGER,
    avg_score REAL,
    first_seen DATE,
    last_seen DATE,
    primary_airline TEXT
);

-- Labels
CREATE TABLE sentiment_labels (
    post_id TEXT PRIMARY KEY,
    vader_score REAL,
    vader_label TEXT,
    manual_label TEXT,  -- for subset
    fta_lstm_pred TEXT,
    bipartite_pred TEXT
);

-- Graph edges
CREATE TABLE graph_edges (
    source_id TEXT,
    target_id TEXT,
    edge_type TEXT,  -- 'authored', 'commented', 'similar'
    weight REAL,
    PRIMARY KEY (source_id, target_id, edge_type)
);
```

### 5.3 Class Distribution Target
| Class | Percentage | Count (25K posts) |
|-------|------------|-------------------|
| Positive | 35% | ~8,750 |
| Neutral | 30% | ~7,500 |
| Negative | 35% | ~8,750 |

---

## 6. PROJECT PHASES & TIMELINE

### Phase 1: Data Collection & Preparation (Week 1-2)
- [ ] Collect Reddit data via API (or use synthetic dataset)
- [ ] Clean and preprocess text
- [ ] Generate VADER baseline labels
- [ ] Create train/val/test splits (70/15/15)
- [ ] Build graph structure for Bipartite model

### Phase 2: Baseline Models (Week 3)
- [ ] Implement VADER baseline
- [ ] Implement TF-IDF + Logistic Regression
- [ ] Implement TF-IDF + Random Forest
- [ ] Establish baseline metrics

### Phase 3: FTA-LSTM Development (Week 4-5)
- [ ] Implement embedding layer
- [ ] Implement frequency attention mechanism
- [ ] Implement Bi-LSTM layers
- [ ] Implement time attention mechanism
- [ ] Train and tune hyperparameters
- [ ] Extract attention weights for interpretability

### Phase 4: Bipartite Graph Development (Week 5-6)
- [ ] Construct bipartite graph from data
- [ ] Implement node feature initialization
- [ ] Implement GraphSAGE/GAT layers
- [ ] Train and tune hyperparameters
- [ ] Analyze learned node embeddings

### Phase 5: Evaluation & Comparison (Week 7-8)
- [ ] Run 5-fold cross-validation on all models
- [ ] Statistical significance testing
- [ ] Error analysis (where does each model fail?)
- [ ] Ensemble experiments

### Phase 6: Visualization & Portfolio (Week 9-10)
- [ ] Build Power BI dashboard
- [ ] Create attention visualization plots
- [ ] Write technical documentation
- [ ] Prepare portfolio presentation
- [ ] Push to GitHub

---

## 7. EVALUATION FRAMEWORK

### 7.1 Metrics

| Metric | Purpose | Target |
|--------|---------|--------|
| Accuracy | Overall correctness | ≥ 0.82 |
| Macro F1 | Balanced class performance | ≥ 0.80 |
| Weighted F1 | Account for class imbalance | ≥ 0.82 |
| ROC-AUC | Discrimination ability | ≥ 0.88 |
| Inference Time | Practical deployment | < 10ms/sample |

### 7.2 Comparison Framework

```python
# Model comparison results table
| Model                  | Accuracy | Macro F1 | ROC-AUC | Train Time | Inference |
|------------------------|----------|----------|---------|------------|-----------|
| VADER (baseline)       |   0.68   |   0.65   |   0.72  |     -      |   <1ms    |
| TF-IDF + LogReg        |   0.75   |   0.73   |   0.81  |    2min    |   <1ms    |
| TF-IDF + RF            |   0.77   |   0.75   |   0.83  |    5min    |   <1ms    |
| FTA-LSTM               |   0.84   |   0.82   |   0.90  |   45min    |    5ms    |
| Bipartite GNN          |   0.82   |   0.80   |   0.88  |   30min    |    8ms    |
| Ensemble (FTA + Bip)   |   0.86   |   0.84   |   0.92  |     -      |   13ms    |
```

### 7.3 Statistical Testing

```python
# McNemar's Test for model comparison
from statsmodels.stats.contingency_tables import mcnemar

# Compare FTA-LSTM vs Bipartite on test set predictions
# Null hypothesis: models have same error rate
# p < 0.05 → statistically significant difference
```

---

## 8. PORTFOLIO PRESENTATION

### Slide Deck Outline

1. **Problem Statement**
   - Business context: airline reputation monitoring
   - Why sentiment analysis matters ($$ impact)

2. **Research Question**
   - FTA-LSTM vs Bipartite Graph: which is better?
   - Trade-offs: accuracy vs interpretability vs speed

3. **Data**
   - 25K Reddit posts, 2 airlines, 12 months
   - Collection and preprocessing pipeline

4. **Methodology**
   - Architecture diagrams for both models
   - Why these approaches suit this problem

5. **Results**
   - Model comparison table
   - Confusion matrices
   - ROC curves

6. **Insights**
   - Attention visualization (what words matter)
   - Graph communities (user segments)
   - Aspect-level sentiment breakdown

7. **Business Recommendations**
   - Actionable insights for airline marketing
   - Deployment considerations

8. **Technical Depth** (for technical interviews)
   - Hyperparameter tuning process
   - Error analysis
   - Future improvements

---

## 9. FILES IN THIS PROJECT

```
airline_sentiment_research/
│
├── data/
│   ├── raw/                     # Raw Reddit data
│   ├── processed/               # Cleaned, tokenized data
│   └── airline_sentiment.db     # SQLite database
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_fta_lstm.ipynb
│   ├── 05_bipartite_graph.ipynb
│   └── 06_model_comparison.ipynb
│
├── src/
│   ├── data/
│   │   ├── collect_reddit.py
│   │   ├── preprocess.py
│   │   └── build_graph.py
│   │
│   ├── models/
│   │   ├── fta_lstm.py
│   │   ├── bipartite_gnn.py
│   │   └── baselines.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── statistical_tests.py
│   │
│   └── visualization/
│       ├── attention_viz.py
│       └── graph_viz.py
│
├── reports/
│   ├── figures/
│   └── final_report.md
│
├── config.py
├── requirements.txt
└── README.md
```

---

## 10. SUCCESS CRITERIA

### Technical Success
- [ ] FTA-LSTM achieves F1 ≥ 0.82
- [ ] Bipartite GNN achieves F1 ≥ 0.80
- [ ] Statistical significance demonstrated (p < 0.05)
- [ ] Attention weights provide interpretable insights
- [ ] Full pipeline reproducible from raw data

### Portfolio Success
- [ ] Clean, documented GitHub repository
- [ ] Interactive Power BI dashboard
- [ ] Clear business problem → solution narrative
- [ ] Technical depth for interview discussions
- [ ] Demonstrates end-to-end ownership

---

## NEXT STEPS

1. **Generate synthetic dataset** (while waiting for Reddit API)
2. **Implement preprocessing pipeline**
3. **Build baseline models**
4. **Implement FTA-LSTM**
5. **Implement Bipartite GNN**
6. **Compare and document**
