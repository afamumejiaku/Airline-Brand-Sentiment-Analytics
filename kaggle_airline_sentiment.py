"""
Kaggle Twitter US Airline Sentiment Dataset
============================================
Downloads and processes the classic airline sentiment dataset (14,640 tweets).

Dataset: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

Setup:
1. Create Kaggle account at kaggle.com
2. Go to Account Settings → API → Create New Token
3. This downloads kaggle.json
4. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:/Users/<user>/.kaggle/ (Windows)

Or manually download:
1. Go to https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
2. Click "Download" button
3. Extract Tweets.csv to this folder

Usage: python kaggle_airline_sentiment.py
"""

import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import re
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "airline_sentiment.db"
KAGGLE_DATASET = "crowdflower/twitter-airline-sentiment"
CSV_FILENAME = "Tweets.csv"

# Map Kaggle airlines to our focus airlines
AIRLINE_MAPPING = {
    'American': 'American Airlines',
    'Southwest': 'Southwest Airlines',
    'United': 'United Airlines',
    'Delta': 'Delta Air Lines',
    'US Airways': 'US Airways',
    'Virgin America': 'Virgin America'
}

# Focus on American and Southwest (can include others)
FOCUS_AIRLINES = ['American', 'Southwest']  # Set to None for all airlines


# ============================================================
# KAGGLE DOWNLOAD
# ============================================================

def download_from_kaggle() -> bool:
    """Download dataset using Kaggle API."""
    try:
        import kaggle
        print("📥 Downloading from Kaggle...")
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=".",
            unzip=True
        )
        print("✓ Download complete")
        return True
    except ImportError:
        print("⚠ Kaggle package not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"⚠ Kaggle download failed: {e}")
        print("\nManual download instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment")
        print("2. Click 'Download' (requires free Kaggle account)")
        print("3. Extract Tweets.csv to this folder")
        return False


def check_local_file() -> bool:
    """Check if CSV file exists locally."""
    if os.path.exists(CSV_FILENAME):
        print(f"✓ Found local file: {CSV_FILENAME}")
        return True
    return False


# ============================================================
# DATA PROCESSING
# ============================================================

def clean_tweet(text: str) -> str:
    """Clean tweet text."""
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove @mentions but keep the text context
    text = re.sub(r'@\w+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove RT prefix
    text = re.sub(r'^RT\s+', '', text)
    
    # Remove special characters but keep sentiment punctuation
    text = re.sub(r'[^\w\s!?.,\'-]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def process_kaggle_data(df: pd.DataFrame, focus_airlines: list = None) -> pd.DataFrame:
    """Process and clean the Kaggle dataset."""
    print("\n🔧 Processing data...")
    
    # Filter to focus airlines if specified
    if focus_airlines:
        df = df[df['airline'].isin(focus_airlines)].copy()
        print(f"   Filtered to {focus_airlines}: {len(df):,} tweets")
    
    # Rename columns for consistency
    df = df.rename(columns={
        'tweet_id': 'post_id',
        'airline_sentiment': 'sentiment',
        'airline_sentiment_confidence': 'confidence',
        'negativereason': 'negative_reason',
        'negativereason_confidence': 'negative_reason_confidence',
        'tweet_created': 'created_at',
        'tweet_location': 'location'
    })
    
    # Map airline names
    df['airline_full'] = df['airline'].map(AIRLINE_MAPPING)
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_tweet)
    
    # Convert sentiment to standard format
    df['sentiment'] = df['sentiment'].str.lower()
    
    # Parse datetime
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['created_date'] = df['created_at'].dt.strftime('%Y-%m-%d')
    
    # Add unique post_id if missing
    if 'post_id' not in df.columns or df['post_id'].isna().any():
        df['post_id'] = [f"tweet_{i:06d}" for i in range(len(df))]
    else:
        df['post_id'] = df['post_id'].astype(str)
    
    # Create author field (anonymized in original data)
    if 'name' in df.columns:
        df['author'] = df['name'].fillna('anonymous')
    else:
        df['author'] = 'twitter_user'
    
    # Filter out empty tweets
    df = df[df['cleaned_text'].str.len() > 10]
    
    print(f"   Final dataset: {len(df):,} tweets")
    
    return df


# ============================================================
# DATABASE OPERATIONS
# ============================================================

def create_database() -> sqlite3.Connection:
    """Create SQLite database with schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.executescript("""
    -- Drop existing tables
    DROP TABLE IF EXISTS posts;
    DROP TABLE IF EXISTS post_airlines;
    DROP TABLE IF EXISTS sentiment_labels;
    DROP TABLE IF EXISTS aspect_mentions;
    DROP TABLE IF EXISTS users;
    DROP TABLE IF EXISTS graph_edges;
    DROP TABLE IF EXISTS data_sources;
    
    -- Posts table
    CREATE TABLE posts (
        post_id TEXT PRIMARY KEY,
        source TEXT,  -- 'kaggle_twitter', 'news', 'skytrax', etc.
        author TEXT,
        title TEXT,
        body TEXT,
        full_text TEXT,
        score INTEGER DEFAULT 0,
        num_comments INTEGER DEFAULT 0,
        created_utc INTEGER,
        created_date TEXT,
        location TEXT,
        permalink TEXT
    );
    
    -- Airline mentions
    CREATE TABLE post_airlines (
        post_id TEXT,
        airline TEXT,
        PRIMARY KEY (post_id, airline)
    );
    
    -- Sentiment labels
    CREATE TABLE sentiment_labels (
        post_id TEXT PRIMARY KEY,
        sentiment TEXT,
        confidence REAL,
        negative_reason TEXT,
        aspect TEXT
    );
    
    -- Aspect mentions
    CREATE TABLE aspect_mentions (
        post_id TEXT,
        aspect TEXT,
        sentiment TEXT,
        PRIMARY KEY (post_id, aspect)
    );
    
    -- Users (for graph construction)
    CREATE TABLE users (
        author TEXT PRIMARY KEY,
        post_count INTEGER DEFAULT 0,
        total_score INTEGER DEFAULT 0,
        avg_sentiment REAL,
        primary_airline TEXT,
        first_seen TEXT,
        last_seen TEXT
    );
    
    -- Graph edges
    CREATE TABLE graph_edges (
        source_id TEXT,
        target_id TEXT,
        edge_type TEXT,
        weight REAL DEFAULT 1.0,
        PRIMARY KEY (source_id, target_id, edge_type)
    );
    
    -- Data sources tracking
    CREATE TABLE data_sources (
        source_name TEXT PRIMARY KEY,
        record_count INTEGER,
        collected_at TEXT
    );
    
    -- Indexes
    CREATE INDEX idx_posts_source ON posts(source);
    CREATE INDEX idx_posts_date ON posts(created_date);
    CREATE INDEX idx_sentiment ON sentiment_labels(sentiment);
    CREATE INDEX idx_airlines ON post_airlines(airline);
    """)
    
    conn.commit()
    print(f"✓ Database created: {DB_PATH}")
    return conn


def save_to_database(conn: sqlite3.Connection, df: pd.DataFrame, source: str):
    """Save processed data to database."""
    cursor = conn.cursor()
    
    posts_data = []
    airline_data = []
    sentiment_data = []
    
    for _, row in df.iterrows():
        # Posts
        created_utc = int(row['created_at'].timestamp()) if pd.notna(row['created_at']) else 0
        
        posts_data.append((
            row['post_id'],
            source,
            row['author'],
            None,  # title (tweets don't have titles)
            row['text'],
            row['cleaned_text'],
            row.get('retweet_count', 0) or 0,
            0,  # num_comments
            created_utc,
            row['created_date'],
            row.get('location', None),
            None  # permalink
        ))
        
        # Airlines
        airline_data.append((row['post_id'], row['airline_full']))
        
        # Sentiment
        sentiment_data.append((
            row['post_id'],
            row['sentiment'],
            row.get('confidence', 1.0),
            row.get('negative_reason', None),
            None  # aspect (to be extracted later)
        ))
    
    # Insert posts
    cursor.executemany("""
        INSERT OR REPLACE INTO posts 
        (post_id, source, author, title, body, full_text, score, 
         num_comments, created_utc, created_date, location, permalink)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, posts_data)
    
    # Insert airlines
    cursor.executemany("""
        INSERT OR REPLACE INTO post_airlines (post_id, airline)
        VALUES (?, ?)
    """, airline_data)
    
    # Insert sentiment
    cursor.executemany("""
        INSERT OR REPLACE INTO sentiment_labels 
        (post_id, sentiment, confidence, negative_reason, aspect)
        VALUES (?, ?, ?, ?, ?)
    """, sentiment_data)
    
    # Track data source
    cursor.execute("""
        INSERT OR REPLACE INTO data_sources (source_name, record_count, collected_at)
        VALUES (?, ?, ?)
    """, (source, len(df), datetime.now().isoformat()))
    
    conn.commit()
    print(f"✓ Saved {len(df):,} records from {source}")


def build_user_table(conn: sqlite3.Connection):
    """Aggregate user statistics."""
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO users (author, post_count, total_score, avg_sentiment, 
                                      primary_airline, first_seen, last_seen)
        SELECT 
            p.author,
            COUNT(*) as post_count,
            SUM(p.score) as total_score,
            AVG(CASE 
                WHEN s.sentiment = 'positive' THEN 1 
                WHEN s.sentiment = 'negative' THEN -1 
                ELSE 0 
            END) as avg_sentiment,
            (SELECT pa.airline FROM post_airlines pa 
             WHERE pa.post_id IN (SELECT post_id FROM posts WHERE author = p.author)
             GROUP BY pa.airline ORDER BY COUNT(*) DESC LIMIT 1) as primary_airline,
            MIN(p.created_date) as first_seen,
            MAX(p.created_date) as last_seen
        FROM posts p
        LEFT JOIN sentiment_labels s ON p.post_id = s.post_id
        GROUP BY p.author
    """)
    
    conn.commit()
    print("✓ Built user statistics table")


def build_graph_edges(conn: sqlite3.Connection):
    """Build graph edges for GNN."""
    cursor = conn.cursor()
    
    # User -> Post edges (authored)
    cursor.execute("""
        INSERT OR REPLACE INTO graph_edges (source_id, target_id, edge_type, weight)
        SELECT author, post_id, 'authored', 1.0
        FROM posts
        WHERE author IS NOT NULL
    """)
    
    conn.commit()
    print("✓ Built graph edges")


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(conn: sqlite3.Connection):
    """Print dataset statistics."""
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    # Total records by source
    print("\n📊 Records by Source:")
    cursor.execute("""
        SELECT source, COUNT(*) FROM posts GROUP BY source
    """)
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]:,}")
    
    # Total records
    cursor.execute("SELECT COUNT(*) FROM posts")
    print(f"\n📊 Total records: {cursor.fetchone()[0]:,}")
    
    # By airline
    print("\n📊 By Airline:")
    cursor.execute("""
        SELECT airline, COUNT(*) FROM post_airlines GROUP BY airline ORDER BY COUNT(*) DESC
    """)
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]:,}")
    
    # By sentiment
    print("\n📊 By Sentiment:")
    cursor.execute("""
        SELECT sentiment, COUNT(*), 
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM sentiment_labels), 1)
        FROM sentiment_labels 
        GROUP BY sentiment
    """)
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]:,} ({row[2]}%)")
    
    # Negative reasons (for negative tweets)
    print("\n📊 Top Negative Reasons:")
    cursor.execute("""
        SELECT negative_reason, COUNT(*) 
        FROM sentiment_labels 
        WHERE negative_reason IS NOT NULL AND negative_reason != ''
        GROUP BY negative_reason 
        ORDER BY COUNT(*) DESC 
        LIMIT 10
    """)
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]:,}")
    
    # Date range
    print("\n📊 Date Range:")
    cursor.execute("SELECT MIN(created_date), MAX(created_date) FROM posts")
    row = cursor.fetchone()
    print(f"   {row[0]} to {row[1]}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("KAGGLE TWITTER AIRLINE SENTIMENT DATASET")
    print("="*60)
    
    # Check for local file or download
    if not check_local_file():
        print(f"\n⚠ {CSV_FILENAME} not found locally")
        if not download_from_kaggle():
            print("\n❌ Cannot proceed without data file.")
            print("Please download manually from Kaggle.")
            return
    
    # Load CSV
    print(f"\n📂 Loading {CSV_FILENAME}...")
    df = pd.read_csv(CSV_FILENAME)
    print(f"   Loaded {len(df):,} tweets")
    
    # Show raw data info
    print(f"\n📊 Raw data columns: {list(df.columns)}")
    print(f"📊 Airlines in dataset: {df['airline'].unique().tolist()}")
    
    # Process data
    df_processed = process_kaggle_data(df, focus_airlines=FOCUS_AIRLINES)
    
    # Create database and save
    conn = create_database()
    save_to_database(conn, df_processed, source='kaggle_twitter')
    
    # Build derived tables
    build_user_table(conn)
    build_graph_edges(conn)
    
    # Print statistics
    print_statistics(conn)
    
    conn.close()
    
    print("\n" + "="*60)
    print("✓ KAGGLE DATA IMPORT COMPLETE")
    print("="*60)
    print(f"\nDatabase: {DB_PATH}")
    print("\nNext steps:")
    print("  1. Run news_api_collector.py to add news articles")
    print("  2. Run skytrax_scraper.py to add airline reviews")
    print("  3. Run compare_models.py to train and evaluate models")


if __name__ == "__main__":
    main()
