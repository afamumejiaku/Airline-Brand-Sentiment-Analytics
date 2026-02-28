"""
Airline Review Scraper - Skytrax & TripAdvisor
===============================================
Scrapes airline reviews from public review sites.

Sources:
- Skytrax (airlinequality.com) - Detailed airline reviews
- TripAdvisor - Airline reviews section

Note: Web scraping should be done responsibly:
- Respect robots.txt
- Add delays between requests
- Don't overload servers
- Check terms of service

Usage: python airline_review_scraper.py
"""

import requests
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd
from datetime import datetime
import time
import re
from typing import List, Dict, Optional
import random
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "airline_sentiment.db"

# Request settings
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

# Delay between requests (be respectful)
MIN_DELAY = 2
MAX_DELAY = 5

# Airline URLs
SKYTRAX_URLS = {
    "American Airlines": "https://www.airlinequality.com/airline-reviews/american-airlines/",
    "Southwest Airlines": "https://www.airlinequality.com/airline-reviews/southwest-airlines/"
}


# ============================================================
# SKYTRAX SCRAPER
# ============================================================

class SkytraxScraper:
    """Scrape reviews from Skytrax (airlinequality.com)"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
    
    def get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a page."""
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return BeautifulSoup(response.content, 'html.parser')
            else:
                print(f"   ⚠ HTTP {response.status_code} for {url}")
                return None
        except Exception as e:
            print(f"   ⚠ Error fetching {url}: {e}")
            return None
    
    def parse_review(self, review_div, airline: str) -> Optional[Dict]:
        """Parse a single review from Skytrax."""
        try:
            # Get review text
            text_div = review_div.find('div', class_='text_content')
            if not text_div:
                return None
            
            review_text = text_div.get_text(strip=True)
            
            # Get title
            title_elem = review_div.find('h2', class_='text_header')
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            # Get rating
            rating = None
            rating_div = review_div.find('div', class_='rating-10')
            if rating_div:
                filled_stars = rating_div.find_all('span', class_='fill')
                rating = len(filled_stars)
            
            # Get date
            date_elem = review_div.find('time')
            review_date = None
            if date_elem and date_elem.get('datetime'):
                try:
                    review_date = datetime.fromisoformat(date_elem['datetime'].split('T')[0])
                except:
                    pass
            
            # Get reviewer info
            author_div = review_div.find('h3', class_='text_sub_header')
            author = author_div.get_text(strip=True).replace('Review by', '').strip() if author_div else "Anonymous"
            
            # Get trip type / class
            header_div = review_div.find('div', class_='tc_sub')
            trip_info = header_div.get_text(strip=True) if header_div else ""
            
            # Determine sentiment from rating
            if rating:
                if rating >= 7:
                    sentiment = 'positive'
                elif rating <= 4:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                confidence = abs(rating - 5) / 5
            else:
                sentiment = 'neutral'
                confidence = 0.5
            
            # Generate unique ID
            post_id = f"skytrax_{hash(review_text[:100]) % 10**8:08d}"
            
            return {
                'post_id': post_id,
                'source': 'skytrax',
                'airline': airline,
                'author': author,
                'title': title,
                'body': review_text,
                'full_text': f"{title} {review_text}".strip(),
                'rating': rating,
                'sentiment': sentiment,
                'confidence': confidence,
                'trip_info': trip_info,
                'created_date': review_date.strftime('%Y-%m-%d') if review_date else None,
                'created_utc': int(review_date.timestamp()) if review_date else 0
            }
            
        except Exception as e:
            print(f"   ⚠ Error parsing review: {e}")
            return None
    
    def scrape_airline(self, airline: str, url: str, max_pages: int = 10) -> List[Dict]:
        """Scrape all reviews for an airline."""
        print(f"\n🔍 Scraping Skytrax reviews for {airline}...")
        
        reviews = []
        page = 1
        
        while page <= max_pages:
            page_url = f"{url}page/{page}/" if page > 1 else url
            print(f"   Page {page}...", end=" ")
            
            soup = self.get_page(page_url)
            if not soup:
                break
            
            # Find all review articles
            review_divs = soup.find_all('article', itemprop='review')
            
            if not review_divs:
                print("No reviews found")
                break
            
            page_reviews = 0
            for div in review_divs:
                review = self.parse_review(div, airline)
                if review:
                    reviews.append(review)
                    page_reviews += 1
            
            print(f"Found {page_reviews} reviews")
            
            # Check for next page
            pagination = soup.find('article', class_='pagination')
            if pagination:
                next_link = pagination.find('a', string=re.compile(r'>>|Next'))
                if not next_link:
                    break
            
            page += 1
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        
        print(f"   Total: {len(reviews)} reviews")
        return reviews


# ============================================================
# TRIPADVISOR SCRAPER
# ============================================================

class TripAdvisorScraper:
    """
    Scrape reviews from TripAdvisor.
    
    Note: TripAdvisor has strong anti-scraping measures.
    This scraper may need adjustments or may not work reliably.
    Consider using their official API if available.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
    
    def search_airline_reviews(self, airline: str) -> str:
        """Search for airline review page URL."""
        # TripAdvisor airline URLs (these may change)
        airline_slugs = {
            "American Airlines": "American_Airlines",
            "Southwest Airlines": "Southwest_Airlines"
        }
        
        slug = airline_slugs.get(airline)
        if slug:
            return f"https://www.tripadvisor.com/Airline_Review-d{slug}-Reviews.html"
        return None
    
    def scrape_reviews_page(self, url: str, airline: str) -> List[Dict]:
        """Scrape reviews from a TripAdvisor page."""
        reviews = []
        
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                print(f"   ⚠ HTTP {response.status_code}")
                return reviews
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find review containers (structure may vary)
            review_cards = soup.find_all('div', {'data-reviewid': True})
            
            for card in review_cards:
                try:
                    # Get review text
                    text_elem = card.find('q', class_='IRsGHoPm')
                    if not text_elem:
                        text_elem = card.find('span', class_='QewHA')
                    
                    if not text_elem:
                        continue
                    
                    review_text = text_elem.get_text(strip=True)
                    
                    # Get rating
                    rating_elem = card.find('span', class_=re.compile(r'bubble_\d+'))
                    rating = None
                    if rating_elem:
                        rating_class = rating_elem.get('class', [])
                        for c in rating_class:
                            match = re.search(r'bubble_(\d+)', c)
                            if match:
                                rating = int(match.group(1)) // 10
                                break
                    
                    # Get title
                    title_elem = card.find('a', class_='Qwuub')
                    title = title_elem.get_text(strip=True) if title_elem else ""
                    
                    # Determine sentiment
                    if rating:
                        if rating >= 4:
                            sentiment = 'positive'
                        elif rating <= 2:
                            sentiment = 'negative'
                        else:
                            sentiment = 'neutral'
                    else:
                        sentiment = 'neutral'
                    
                    post_id = f"tripadvisor_{hash(review_text[:100]) % 10**8:08d}"
                    
                    reviews.append({
                        'post_id': post_id,
                        'source': 'tripadvisor',
                        'airline': airline,
                        'author': 'TripAdvisor User',
                        'title': title,
                        'body': review_text,
                        'full_text': f"{title} {review_text}".strip(),
                        'rating': rating,
                        'sentiment': sentiment,
                        'confidence': 0.7,
                        'created_date': None,
                        'created_utc': 0
                    })
                    
                except Exception as e:
                    continue
            
        except Exception as e:
            print(f"   ⚠ Error: {e}")
        
        return reviews


# ============================================================
# ALTERNATIVE: KAGGLE REVIEW DATASETS
# ============================================================

def download_kaggle_reviews():
    """
    Alternative: Download pre-scraped review datasets from Kaggle.
    
    Available datasets:
    - Airline Reviews: kaggle.com/datasets/airlinequality/airline-reviews
    - Skytrax Reviews: kaggle.com/datasets/ryanwebert/airline-sentiment-analysis
    """
    print("\n📥 Attempting to download Kaggle airline reviews dataset...")
    
    try:
        import kaggle
        
        # Try different review datasets
        datasets = [
            "efehandanisman/skytrax-airline-reviews",
            "crowdflower/twitter-airline-sentiment",  # Backup - Twitter data
        ]
        
        for dataset in datasets:
            try:
                kaggle.api.dataset_download_files(
                    dataset,
                    path="./reviews_data",
                    unzip=True
                )
                print(f"✓ Downloaded: {dataset}")
                return True
            except:
                continue
        
        print("⚠ Could not download any review datasets")
        return False
        
    except ImportError:
        print("⚠ Kaggle package not installed")
        return False
    except Exception as e:
        print(f"⚠ Download failed: {e}")
        return False


# ============================================================
# DATABASE OPERATIONS
# ============================================================

def save_reviews_to_database(conn: sqlite3.Connection, reviews: List[Dict], source: str):
    """Save reviews to database."""
    if not reviews:
        print("   No reviews to save")
        return
    
    cursor = conn.cursor()
    saved = 0
    
    for review in reviews:
        try:
            # Insert post
            cursor.execute("""
                INSERT OR IGNORE INTO posts 
                (post_id, source, author, title, body, full_text, score,
                 num_comments, created_utc, created_date, location, permalink)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                review['post_id'],
                source,
                review['author'],
                review['title'],
                review['body'],
                review['full_text'],
                review.get('rating', 0) or 0,
                0,
                review.get('created_utc', 0),
                review.get('created_date'),
                None,
                None
            ))
            
            if cursor.rowcount > 0:
                saved += 1
                
                # Insert airline
                cursor.execute("""
                    INSERT OR IGNORE INTO post_airlines (post_id, airline)
                    VALUES (?, ?)
                """, (review['post_id'], review['airline']))
                
                # Insert sentiment
                cursor.execute("""
                    INSERT OR IGNORE INTO sentiment_labels 
                    (post_id, sentiment, confidence, negative_reason, aspect)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    review['post_id'],
                    review['sentiment'],
                    review.get('confidence', 0.5),
                    None,
                    None
                ))
        
        except Exception as e:
            print(f"   Error saving review: {e}")
    
    # Update data sources
    cursor.execute("""
        INSERT OR REPLACE INTO data_sources (source_name, record_count, collected_at)
        SELECT ?, COUNT(*), ?
        FROM posts WHERE source = ?
    """, (source, datetime.now().isoformat(), source))
    
    conn.commit()
    print(f"   ✓ Saved {saved} new reviews")


# ============================================================
# MANUAL CSV IMPORT
# ============================================================

def import_csv_reviews(csv_path: str, airline_column: str, text_column: str, 
                       rating_column: str = None, source_name: str = 'csv_reviews'):
    """
    Import reviews from any CSV file.
    
    Use this for manually downloaded datasets.
    """
    print(f"\n📂 Importing reviews from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df):,} rows")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        saved = 0
        for idx, row in df.iterrows():
            try:
                text = str(row[text_column])
                airline = str(row[airline_column])
                
                # Map airline names if needed
                airline_map = {
                    'american': 'American Airlines',
                    'southwest': 'Southwest Airlines',
                    'united': 'United Airlines',
                    'delta': 'Delta Air Lines'
                }
                airline_lower = airline.lower()
                for key, value in airline_map.items():
                    if key in airline_lower:
                        airline = value
                        break
                
                # Get rating and sentiment
                rating = None
                if rating_column and rating_column in df.columns:
                    rating = row[rating_column]
                    if pd.notna(rating):
                        rating = float(rating)
                        if rating >= 4:
                            sentiment = 'positive'
                        elif rating <= 2:
                            sentiment = 'negative'
                        else:
                            sentiment = 'neutral'
                    else:
                        sentiment = 'neutral'
                else:
                    # Use VADER to estimate
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    analyzer = SentimentIntensityAnalyzer()
                    compound = analyzer.polarity_scores(text)['compound']
                    if compound >= 0.05:
                        sentiment = 'positive'
                    elif compound <= -0.05:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                
                post_id = f"{source_name}_{idx:08d}"
                
                cursor.execute("""
                    INSERT OR IGNORE INTO posts 
                    (post_id, source, author, title, body, full_text, score,
                     num_comments, created_utc, created_date, location, permalink)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    post_id, source_name, 'reviewer', '', text, text,
                    int(rating) if rating else 0, 0, 0, None, None, None
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
                    """, (post_id, sentiment, 0.7, None, None))
            
            except Exception as e:
                continue
        
        cursor.execute("""
            INSERT OR REPLACE INTO data_sources (source_name, record_count, collected_at)
            VALUES (?, ?, ?)
        """, (source_name, saved, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        print(f"   ✓ Saved {saved} reviews")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("AIRLINE REVIEW SCRAPER")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Try Skytrax scraping
    print("\n" + "-"*60)
    print("SKYTRAX REVIEWS")
    print("-"*60)
    
    scraper = SkytraxScraper()
    all_reviews = []
    
    for airline, url in SKYTRAX_URLS.items():
        reviews = scraper.scrape_airline(airline, url, max_pages=5)
        all_reviews.extend(reviews)
    
    if all_reviews:
        save_reviews_to_database(conn, all_reviews, 'skytrax')
    
    # 2. Summary
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT source, COUNT(*) 
        FROM posts 
        GROUP BY source
    """)
    
    print("\n📊 Records by Source:")
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]:,}")
    
    cursor.execute("SELECT COUNT(*) FROM posts")
    print(f"\n📊 Total records: {cursor.fetchone()[0]:,}")
    
    conn.close()
    
    print("\n" + "="*60)
    print("✓ REVIEW COLLECTION COMPLETE")
    print("="*60)
    print("\nAlternative data sources if scraping blocked:")
    print("  1. Download Skytrax reviews from Kaggle:")
    print("     kaggle.com/datasets/efehandanisman/skytrax-airline-reviews")
    print("  2. Use import_csv_reviews() function for any CSV file")


if __name__ == "__main__":
    main()
