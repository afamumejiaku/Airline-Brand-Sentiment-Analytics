"""
News API Collector for Airline Sentiment
=========================================
Collects airline-related news articles from NewsAPI.org

Setup:
1. Create free account at https://newsapi.org/
2. Get your API key from the dashboard
3. Add it to config or pass as argument

Free tier limits:
- 100 requests/day
- 1 month historical data
- Headlines only (no full content)

Usage: python news_api_collector.py --api_key YOUR_API_KEY
"""

import requests
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import re
import argparse

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "airline_sentiment.db"
NEWS_API_BASE_URL = "https://newsapi.org/v2"

# Search queries for airlines
AIRLINE_QUERIES = {
    "American Airlines": [
        "American Airlines",
        "AA flight",
        "AAdvantage"
    ],
    "Southwest Airlines": [
        "Southwest Airlines", 
        "Southwest flight",
        "Southwest carrier"
    ]
}

# News categories to search
CATEGORIES = ["business", "general"]


# ============================================================
# NEWS API CLIENT
# ============================================================

class NewsAPIClient:
    """Client for NewsAPI.org"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = NEWS_API_BASE_URL
        self.requests_today = 0
        self.max_requests = 100  # Free tier limit
    
    def _make_request(self, endpoint: str, params: dict) -> Optional[dict]:
        """Make API request with rate limiting."""
        if self.requests_today >= self.max_requests:
            print(f"⚠ Daily request limit ({self.max_requests}) reached")
            return None
        
        params['apiKey'] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            self.requests_today += 1
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                print("❌ Invalid API key")
                return None
            elif response.status_code == 429:
                print("❌ Rate limit exceeded")
                return None
            else:
                print(f"⚠ API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"⚠ Request failed: {e}")
            return None
    
    def search_everything(
        self,
        query: str,
        from_date: str = None,
        to_date: str = None,
        language: str = "en",
        sort_by: str = "relevancy",
        page_size: int = 100
    ) -> List[dict]:
        """Search all articles matching query."""
        params = {
            'q': query,
            'language': language,
            'sortBy': sort_by,
            'pageSize': min(page_size, 100)
        }
        
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        result = self._make_request('everything', params)
        
        if result and result.get('status') == 'ok':
            return result.get('articles', [])
        return []
    
    def get_top_headlines(
        self,
        query: str = None,
        category: str = None,
        country: str = "us",
        page_size: int = 100
    ) -> List[dict]:
        """Get top headlines."""
        params = {
            'country': country,
            'pageSize': min(page_size, 100)
        }
        
        if query:
            params['q'] = query
        if category:
            params['category'] = category
        
        result = self._make_request('top-headlines', params)
        
        if result and result.get('status') == 'ok':
            return result.get('articles', [])
        return []


# ============================================================
# DATA PROCESSING
# ============================================================

def clean_article_text(text: str) -> str:
    """Clean article text."""
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove [+chars] truncation markers
    text = re.sub(r'\[\+\d+ chars\]', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def detect_airline(text: str) -> List[str]:
    """Detect which airlines are mentioned."""
    if not text:
        return []
    
    text_lower = text.lower()
    airlines = []
    
    for airline, keywords in AIRLINE_QUERIES.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                airlines.append(airline)
                break
    
    return airlines


def process_articles(articles: List[dict], source_query: str) -> pd.DataFrame:
    """Process raw articles into DataFrame."""
    records = []
    
    for article in articles:
        # Combine title and description
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        content = article.get('content', '') or ''
        
        full_text = f"{title} {description} {content}".strip()
        cleaned_text = clean_article_text(full_text)
        
        if len(cleaned_text) < 20:
            continue
        
        # Detect airlines
        airlines = detect_airline(full_text)
        if not airlines:
            # Use query as fallback
            for airline in AIRLINE_QUERIES.keys():
                if airline.lower() in source_query.lower():
                    airlines.append(airline)
                    break
        
        if not airlines:
            continue
        
        # Parse date
        published_at = article.get('publishedAt', '')
        try:
            created_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            created_utc = int(created_date.timestamp())
            created_date_str = created_date.strftime('%Y-%m-%d')
        except:
            created_utc = 0
            created_date_str = None
        
        # Generate unique ID
        url = article.get('url', '')
        post_id = f"news_{hash(url) % 10**8:08d}"
        
        records.append({
            'post_id': post_id,
            'source': 'newsapi',
            'author': article.get('author') or article.get('source', {}).get('name', 'Unknown'),
            'title': title,
            'body': description,
            'full_text': cleaned_text,
            'score': 0,
            'created_utc': created_utc,
            'created_date': created_date_str,
            'permalink': url,
            'airlines': airlines,
            'news_source': article.get('source', {}).get('name', 'Unknown')
        })
    
    return pd.DataFrame(records)


# ============================================================
# SENTIMENT ESTIMATION
# ============================================================

def estimate_sentiment(text: str) -> tuple:
    """
    Estimate sentiment using VADER.
    Returns (sentiment_label, confidence)
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'positive', abs(compound)
        elif compound <= -0.05:
            return 'negative', abs(compound)
        else:
            return 'neutral', 1 - abs(compound)
    except:
        return 'neutral', 0.5


# ============================================================
# DATABASE OPERATIONS
# ============================================================

def save_to_database(conn: sqlite3.Connection, df: pd.DataFrame):
    """Save articles to database."""
    if df.empty:
        print("   No articles to save")
        return
    
    cursor = conn.cursor()
    
    saved = 0
    for _, row in df.iterrows():
        try:
            # Insert post
            cursor.execute("""
                INSERT OR IGNORE INTO posts 
                (post_id, source, author, title, body, full_text, score,
                 num_comments, created_utc, created_date, location, permalink)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['post_id'], 'newsapi', row['author'], row['title'],
                row['body'], row['full_text'], 0, 0,
                row['created_utc'], row['created_date'], None, row['permalink']
            ))
            
            # Check if inserted (not duplicate)
            if cursor.rowcount > 0:
                saved += 1
                
                # Insert airlines
                for airline in row['airlines']:
                    cursor.execute("""
                        INSERT OR IGNORE INTO post_airlines (post_id, airline)
                        VALUES (?, ?)
                    """, (row['post_id'], airline))
                
                # Estimate and insert sentiment
                sentiment, confidence = estimate_sentiment(row['full_text'])
                cursor.execute("""
                    INSERT OR IGNORE INTO sentiment_labels 
                    (post_id, sentiment, confidence, negative_reason, aspect)
                    VALUES (?, ?, ?, ?, ?)
                """, (row['post_id'], sentiment, confidence, None, None))
        
        except Exception as e:
            print(f"   Error saving article: {e}")
    
    # Update data sources
    cursor.execute("""
        INSERT OR REPLACE INTO data_sources (source_name, record_count, collected_at)
        SELECT 'newsapi', COUNT(*), ?
        FROM posts WHERE source = 'newsapi'
    """, (datetime.now().isoformat(),))
    
    conn.commit()
    print(f"   ✓ Saved {saved} new articles")


# ============================================================
# MAIN COLLECTION
# ============================================================

def collect_news(api_key: str, days_back: int = 30):
    """Main collection function."""
    print("="*60)
    print("NEWS API COLLECTOR")
    print("="*60)
    
    client = NewsAPIClient(api_key)
    
    # Date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_back)
    
    print(f"\n📅 Date range: {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    all_articles = []
    
    # Search for each airline
    for airline, queries in AIRLINE_QUERIES.items():
        print(f"\n🔍 Searching for {airline}...")
        
        for query in queries[:1]:  # Use first query to conserve API calls
            print(f"   Query: '{query}'")
            
            # Search everything
            articles = client.search_everything(
                query=query,
                from_date=from_date.strftime('%Y-%m-%d'),
                to_date=to_date.strftime('%Y-%m-%d'),
                page_size=100
            )
            
            if articles:
                print(f"   Found {len(articles)} articles")
                df = process_articles(articles, query)
                all_articles.append(df)
                save_to_database(conn, df)
            else:
                print("   No articles found")
            
            time.sleep(1)  # Rate limiting
    
    # Get top headlines
    print("\n🔍 Fetching top headlines...")
    for query in ["airline", "aviation", "flight"]:
        headlines = client.get_top_headlines(query=query, page_size=50)
        
        if headlines:
            print(f"   Headlines for '{query}': {len(headlines)}")
            df = process_articles(headlines, query)
            save_to_database(conn, df)
        
        time.sleep(1)
    
    # Print summary
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM posts WHERE source = 'newsapi'")
    total = cursor.fetchone()[0]
    
    print(f"\n✓ Total news articles in database: {total}")
    print(f"✓ API requests used: {client.requests_today}/{client.max_requests}")
    
    conn.close()
    
    print("\n" + "="*60)
    print("✓ NEWS COLLECTION COMPLETE")
    print("="*60)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Collect airline news from NewsAPI')
    parser.add_argument('--api_key', type=str, required=True, help='NewsAPI.org API key')
    parser.add_argument('--days', type=int, default=30, help='Days of history to collect')
    args = parser.parse_args()
    
    collect_news(args.api_key, args.days)


if __name__ == "__main__":
    main()
