"""
Master Data Collection Pipeline
================================
Orchestrates data collection from all sources:
1. Kaggle Twitter Airline Sentiment (14K+ tweets)
2. NewsAPI articles
3. Skytrax reviews
4. Google News (via web scraping)

Usage: python collect_all_data.py [--newsapi_key YOUR_KEY]
"""

import sqlite3
import pandas as pd
from datetime import datetime
import argparse
import os
import sys

DB_PATH = "airline_sentiment.db"


def print_header(title: str):
    """Print section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def collect_kaggle_data():
    """Step 1: Import Kaggle Twitter dataset."""
    print_header("STEP 1: KAGGLE TWITTER AIRLINE SENTIMENT")
    
    try:
        from kaggle_airline_sentiment import main as kaggle_main
        kaggle_main()
        return True
    except FileNotFoundError:
        print("\n⚠ Tweets.csv not found!")
        print("\nTo get the data:")
        print("1. Go to: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment")
        print("2. Download and extract Tweets.csv to this folder")
        print("3. Re-run this script")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def collect_news_data(api_key: str = None):
    """Step 2: Collect news articles."""
    print_header("STEP 2: NEWS API ARTICLES")
    
    if not api_key:
        print("\n⚠ No NewsAPI key provided")
        print("\nTo get news articles:")
        print("1. Create free account at: https://newsapi.org/")
        print("2. Get your API key")
        print("3. Run: python collect_all_data.py --newsapi_key YOUR_KEY")
        return False
    
    try:
        from news_api_collector import collect_news
        collect_news(api_key, days_back=30)
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def collect_review_data():
    """Step 3: Collect airline reviews."""
    print_header("STEP 3: AIRLINE REVIEWS (SKYTRAX)")
    
    try:
        from airline_review_scraper import main as scraper_main
        scraper_main()
        return True
    except Exception as e:
        print(f"\n⚠ Review scraping error: {e}")
        print("This is expected if the site blocks scraping.")
        print("\nAlternative: Download from Kaggle")
        print("https://www.kaggle.com/datasets/efehandanisman/skytrax-airline-reviews")
        return False


def collect_google_news():
    """Step 4: Collect Google News (basic scraping)."""
    print_header("STEP 4: GOOGLE NEWS SCRAPING")
    
    try:
        import requests
        from bs4 import BeautifulSoup
        import time
        import re
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        analyzer = SentimentIntensityAnalyzer()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        queries = [
            "American Airlines news",
            "Southwest Airlines news",
            "American Airlines reviews",
            "Southwest Airlines passenger"
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        saved = 0
        
        for query in queries:
            print(f"\n🔍 Searching: {query}")
            
            url = f"https://news.google.com/search?q={query.replace(' ', '%20')}&hl=en-US&gl=US&ceid=US:en"
            
            try:
                response = requests.get(url, headers=headers, timeout=30)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article titles
                articles = soup.find_all('article')[:10]
                
                for i, article in enumerate(articles):
                    title_elem = article.find('a', class_='JtKRv')
                    if not title_elem:
                        title_elem = article.find('h3')
                    
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        
                        # Detect airline
                        title_lower = title.lower()
                        if 'american' in title_lower:
                            airline = 'American Airlines'
                        elif 'southwest' in title_lower:
                            airline = 'Southwest Airlines'
                        else:
                            continue
                        
                        # Estimate sentiment
                        compound = analyzer.polarity_scores(title)['compound']
                        if compound >= 0.05:
                            sentiment = 'positive'
                        elif compound <= -0.05:
                            sentiment = 'negative'
                        else:
                            sentiment = 'neutral'
                        
                        post_id = f"gnews_{hash(title) % 10**8:08d}"
                        
                        cursor.execute("""
                            INSERT OR IGNORE INTO posts 
                            (post_id, source, author, title, body, full_text, score,
                             num_comments, created_utc, created_date, location, permalink)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            post_id, 'google_news', 'news', title, '', title,
                            0, 0, 0, datetime.now().strftime('%Y-%m-%d'), None, None
                        ))
                        
                        if cursor.rowcount > 0:
                            saved += 1
                            
                            cursor.execute("""
                                INSERT OR IGNORE INTO post_airlines (post_id, airline)
                                VALUES (?, ?)
                            """, (post_id, airline))
                            
                            cursor.execute("""
                                INSERT OR IGNORE INTO sentiment_labels 
                                (post_id, sentiment, confidence, negative_reason, aspect)
                                VALUES (?, ?, ?, ?, ?)
                            """, (post_id, sentiment, abs(compound), None, None))
                
                print(f"   Found {len(articles)} articles")
                time.sleep(2)
                
            except Exception as e:
                print(f"   ⚠ Error: {e}")
        
        cursor.execute("""
            INSERT OR REPLACE INTO data_sources (source_name, record_count, collected_at)
            SELECT 'google_news', COUNT(*), ?
            FROM posts WHERE source = 'google_news'
        """, (datetime.now().isoformat(),))
        
        conn.commit()
        print(f"\n✓ Saved {saved} Google News articles")
        conn.close()
        return True
        
    except Exception as e:
        print(f"\n⚠ Google News scraping error: {e}")
        return False


def build_derived_tables():
    """Build user and graph tables from collected data."""
    print_header("BUILDING DERIVED TABLES")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Build user table
    print("\n📊 Building user statistics...")
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
        WHERE p.author IS NOT NULL
        GROUP BY p.author
    """)
    
    # Build graph edges
    print("📊 Building graph edges...")
    cursor.execute("""
        INSERT OR REPLACE INTO graph_edges (source_id, target_id, edge_type, weight)
        SELECT author, post_id, 'authored', 1.0
        FROM posts
        WHERE author IS NOT NULL
    """)
    
    conn.commit()
    conn.close()
    print("✓ Derived tables built")


def print_final_summary():
    """Print final dataset summary."""
    print_header("FINAL DATASET SUMMARY")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total records
    cursor.execute("SELECT COUNT(*) FROM posts")
    total = cursor.fetchone()[0]
    print(f"\n📊 Total Records: {total:,}")
    
    # By source
    print("\n📊 By Data Source:")
    cursor.execute("""
        SELECT source, COUNT(*) FROM posts GROUP BY source ORDER BY COUNT(*) DESC
    """)
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]:,}")
    
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
        ORDER BY COUNT(*) DESC
    """)
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]:,} ({row[2]}%)")
    
    # Users
    cursor.execute("SELECT COUNT(*) FROM users WHERE post_count > 0")
    print(f"\n📊 Unique Authors: {cursor.fetchone()[0]:,}")
    
    # Graph edges
    cursor.execute("SELECT COUNT(*) FROM graph_edges")
    print(f"📊 Graph Edges: {cursor.fetchone()[0]:,}")
    
    conn.close()
    
    # Check if we have enough data
    if total < 1000:
        print("\n⚠ Warning: Dataset is small. Try to add more data sources.")
    elif total < 5000:
        print("\n✓ Dataset is usable for initial experiments.")
    else:
        print("\n✓ Dataset is good for comprehensive analysis!")


def export_for_modeling():
    """Export data for model training."""
    print_header("EXPORTING DATA FOR MODELING")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Main dataset
    df = pd.read_sql_query("""
        SELECT 
            p.post_id,
            p.source,
            p.full_text,
            p.score,
            p.created_date,
            pa.airline,
            s.sentiment,
            s.confidence
        FROM posts p
        JOIN post_airlines pa ON p.post_id = pa.post_id
        JOIN sentiment_labels s ON p.post_id = s.post_id
        WHERE p.full_text IS NOT NULL AND LENGTH(p.full_text) > 10
    """, conn)
    
    df.to_csv("posts_for_modeling.csv", index=False)
    print(f"✓ Exported {len(df):,} records to posts_for_modeling.csv")
    
    # Users
    users_df = pd.read_sql_query("SELECT * FROM users WHERE post_count > 0", conn)
    users_df.to_csv("users_for_graph.csv", index=False)
    print(f"✓ Exported {len(users_df):,} users to users_for_graph.csv")
    
    # Graph edges
    edges_df = pd.read_sql_query("SELECT * FROM graph_edges", conn)
    edges_df.to_csv("graph_edges.csv", index=False)
    print(f"✓ Exported {len(edges_df):,} edges to graph_edges.csv")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Collect all airline sentiment data')
    parser.add_argument('--newsapi_key', type=str, help='NewsAPI.org API key')
    parser.add_argument('--skip_kaggle', action='store_true', help='Skip Kaggle import')
    parser.add_argument('--skip_news', action='store_true', help='Skip news collection')
    parser.add_argument('--skip_reviews', action='store_true', help='Skip review scraping')
    args = parser.parse_args()
    
    print("="*70)
    print("   AIRLINE SENTIMENT DATA COLLECTION PIPELINE")
    print("   Multi-Source: Twitter + News + Reviews")
    print("="*70)
    
    # Step 1: Kaggle
    if not args.skip_kaggle:
        collect_kaggle_data()
    
    # Step 2: News API
    if not args.skip_news and args.newsapi_key:
        collect_news_data(args.newsapi_key)
    elif not args.skip_news:
        print("\n⚠ Skipping NewsAPI (no key provided)")
        print("   Add --newsapi_key YOUR_KEY to include news articles")
    
    # Step 3: Reviews
    if not args.skip_reviews:
        collect_review_data()
    
    # Step 4: Google News
    collect_google_news()
    
    # Build derived tables
    build_derived_tables()
    
    # Export
    export_for_modeling()
    
    # Summary
    print_final_summary()
    
    print("\n" + "="*70)
    print("   ✓ DATA COLLECTION COMPLETE")
    print("="*70)
    print(f"\nDatabase: {DB_PATH}")
    print("\nNext steps:")
    print("  python compare_models.py    # Run FTA-LSTM vs Bipartite GNN comparison")


if __name__ == "__main__":
    main()
