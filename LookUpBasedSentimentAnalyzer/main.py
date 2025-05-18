import json
import re
import datetime
import feedparser
from collections import defaultdict
import os
import argparse
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser as date_parser
import time

class SentimentAnalyzer:
    def __init__(self, ticker, keyword=None, learning_rate=0.05, polling_interval=60):
        self.ticker = ticker
        self.keyword = keyword
        self.rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'
        self.dictionary_file = f'sentiment_dictionary_{ticker}.json'
        self.log_file = f'sentiment_log_{ticker}.csv'
        self.learning_rate = learning_rate
        self.polling_interval = polling_interval
        self.positive_threshold = 0.05
        self.negative_threshold = -0.05
        self.seen_links = set()
        
        # Create directory for logs
        os.makedirs('logs', exist_ok=True)
        
        # Initialize sentiment dictionary and seen links
        self.sentiment_dict = self._load_dictionary()
        self.seen_links_file = f'logs/seen_links_{ticker}.json'
        self._load_seen_links()
        
        # Initialize log file with header if it doesn't exist
        if not os.path.exists(f'logs/{self.log_file}'):
            with open(f'logs/{self.log_file}', 'w') as log:
                log.write('timestamp,score,num_articles,sentiment,source\n')
    
    def _load_seen_links(self):
        try:
            with open(f'logs/{self.seen_links_file}', 'r') as f:
                self.seen_links = set(json.load(f))
                print(f"Loaded {len(self.seen_links)} previously seen links")
        except FileNotFoundError:
            self.seen_links = set()
            
    def _save_seen_links(self):
        with open(f'logs/{self.seen_links_file}', 'w') as f:
            json.dump(list(self.seen_links), f)
    
    def _load_dictionary(self):
        try:
            with open(f'logs/{self.dictionary_file}', 'r') as f:
                sentiment_dict = json.load(f)
                sentiment_dict = {k: float(v) for k, v in sentiment_dict.items()}
                print(f"Loaded dictionary with {len(sentiment_dict)} terms")
                return sentiment_dict
        except FileNotFoundError:
            print("Creating new sentiment dictionary")
            return {
                # Positive financial terms
                "gain": 1.0, "growth": 1.0, "increase": 1.0, "profit": 1.0, "positive": 1.0,
                "rise": 1.0, "soar": 1.0, "strong": 1.0, "record": 1.0, "surge": 1.0,
                "boost": 1.0, "improve": 1.0, "outperform": 1.0, "exceed": 1.0, "beat": 1.0,
                "bullish": 1.0, "upgrade": 1.0, "confident": 1.0, "recovery": 1.0, "opportunity": 1.0,
                
                # Negative financial terms
                "loss": -1.0, "fall": -1.0, "decline": -1.0, "negative": -1.0, "drop": -1.0,
                "plunge": -1.0, "weaken": -1.0, "concern": -1.0, "delay": -1.0, "down": -1.0,
                "miss": -1.0, "underperform": -1.0, "fear": -1.0, "crisis": -1.0, "lawsuit": -1.0,
                "bearish": -1.0, "downgrade": -1.0, "risk": -1.0, "warning": -1.0, "recall": -1.0
            }
    
    def score_with_dictionary(self, text):
        # Normalize text, handle negations
        text = text.lower()
        
        # Handle negations (e.g., "not good" becomes "not_good")
        negations = ["not", "no", "never", "without", "barely", "hardly", "doesn't", "isn't", "aren't", "wasn't", "weren't"]
        words = text.split()
        for i in range(len(words)-1):
            if words[i] in negations:
                j = 1
                while i+j < len(words) and words[i+j] in ['a', 'the', 'an', 'very', 'so', 'quite']:
                    j += 1
                if i+j < len(words):
                    words[i+j] = f"NOT_{words[i+j]}"
        
        processed_text = ' '.join(words)
        
        # Extract words and calculate score
        words = re.findall(r'\b\w+\b', processed_text)
        term_matches = [word for word in words if word in self.sentiment_dict]
        if term_matches:
            print(f"Matched sentiment terms: {', '.join(term_matches)}")
            
        word_scores = [(word, self.sentiment_dict.get(word, 0)) for word in words]
        score = sum(score for _, score in word_scores)
        
        # Normalize by text length
        if len(words) > 0:
            score = score / (len(words) ** 0.5)

        return score
    
    def update_dictionary(self, text, sentiment_score):
        # Don't update for neutral content
        if abs(sentiment_score) < 0.01:
            return
            
        # Extract unique words
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # Update dictionary
        for word in words:
            if len(word) >= 3:  # Ignore very short words
                self.sentiment_dict[word] = self.sentiment_dict.get(word, 0) + self.learning_rate * sentiment_score
    
    def save_dictionary(self):
        with open(f'logs/{self.dictionary_file}', 'w') as f:
            json.dump(self.sentiment_dict, f)
            
    def log_sentiment(self, timestamp, score, num_articles, source='live'):
        sentiment = "positive" if score >= self.positive_threshold else "negative" if score <= self.negative_threshold else "neutral"
        with open(f'logs/{self.log_file}', 'a') as log:
            log.write(f'{timestamp},{score:.4f},{num_articles},{sentiment},{source}\n')
    
    def analyze_sentiment(self):
        try:
            feed = feedparser.parse(self.rss_url)
            
            if hasattr(feed, 'bozo_exception'):
                print(f"Error parsing feed: {feed.bozo_exception}")
                return
                
            total_score = 0
            num_articles = 0
            
            print(f'\nChecking news for {self.ticker} (filter: "{self.keyword}")...')
            print(f'Found {len(feed.entries)} articles in feed')
            
            for entry in feed.entries:
                # Skip if article doesn't match keyword filter or already seen
                if (self.keyword and self.keyword.lower() not in entry.title.lower() and 
                    self.keyword.lower() not in entry.summary.lower()):
                    continue
                    
                if entry.link in self.seen_links:
                    continue
                    
                self.seen_links.add(entry.link)
                
                print(f'\n{"-" * 60}')
                print(f'Title: {entry.title}')
                print(f'Published: {entry.published}')
                print(f'Summary: {entry.summary}')
                
                # Score the article
                title_score = self.score_with_dictionary(entry.title) * 1.5  # Title has more weight
                summary_score = self.score_with_dictionary(entry.summary)
                score = (title_score + summary_score) / 2
                
                label = "Positive" if score > self.positive_threshold else "Negative" if score < self.negative_threshold else "Neutral"
                print(f'Sentiment: {label}, Raw Score: {score:.4f}')
                
                # Update sentiment dictionary
                self.update_dictionary(entry.title + " " + entry.summary, score)
                
                total_score += score
                num_articles += 1
            
            # Calculate overall sentiment
            if num_articles > 0:
                final_score = total_score / num_articles
                overall = "Positive" if final_score >= self.positive_threshold else "Negative" if final_score <= self.negative_threshold else "Neutral"
                print(f'\n>> Overall Sentiment: {overall} ({final_score:.4f}) from {num_articles} articles')
                
                timestamp = datetime.datetime.now().isoformat()
                self.log_sentiment(timestamp, final_score, num_articles, 'live')
                self.save_dictionary()
                self._save_seen_links()
                
                # Show top terms
                top_terms = sorted(self.sentiment_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                print("\nTop sentiment terms in dictionary:")
                for term, value in top_terms:
                    print(f"  {term}: {value:.4f}")
            else:
                print("No new relevant articles found.")
                
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
    
    def fetch_article_content(self, url):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text content
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            print(f"Error fetching article content: {e}")
            return ""
    
    def fetch_historical_news(self, days=30, max_articles=100):
        articles = []
        
        try:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            print(f"Fetching historical news from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            url = f"https://finance.yahoo.com/quote/{self.ticker}/news"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = soup.find_all('div', {'class': 'Ov(h)'})
            
            for item in news_items[:max_articles]:
                try:
                    title_elem = item.find('h3')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    
                    link_elem = item.find('a')
                    if not link_elem or not link_elem.has_attr('href'):
                        continue
                        
                    link = link_elem['href']
                    if not link.startswith('http'):
                        link = f"https://finance.yahoo.com{link}"
                        
                    summary_elem = item.find('p')
                    summary = summary_elem.text.strip() if summary_elem else title
                    
                    date_elem = item.find('span', {'class': 'C($tertiaryColor)'})
                    date_str = date_elem.text.strip() if date_elem else None
                    
                    # Parse date
                    article_date = datetime.datetime.now()
                    if date_str:
                        try:
                            if 'ago' in date_str.lower():
                                if 'minute' in date_str.lower():
                                    minutes = int(re.search(r'(\d+)', date_str).group(1))
                                    article_date = datetime.datetime.now() - datetime.timedelta(minutes=minutes)
                                elif 'hour' in date_str.lower():
                                    hours = int(re.search(r'(\d+)', date_str).group(1))
                                    article_date = datetime.datetime.now() - datetime.timedelta(hours=hours)
                                elif 'day' in date_str.lower():
                                    days = int(re.search(r'(\d+)', date_str).group(1))
                                    article_date = datetime.datetime.now() - datetime.timedelta(days=days)
                            else:
                                article_date = date_parser.parse(date_str)
                        except:
                            pass
                    
                    # Skip if outside date range or doesn't match keyword
                    if article_date < start_date or article_date > end_date:
                        continue
                        
                    if self.keyword and self.keyword.lower() not in title.lower() and self.keyword.lower() not in summary.lower():
                        continue
                        
                    if link in self.seen_links:
                        continue
                        
                    articles.append({
                        'title': title,
                        'summary': summary,
                        'link': link,
                        'date': article_date.isoformat()
                    })
                    
                except Exception as e:
                    print(f"Error processing news item: {e}")
                    continue
        
        except Exception as e:
            print(f"Error fetching historical news: {e}")
            
        return articles
    
    def analyze_historical_data(self, days=30, max_articles=100, fetch_full_content=False, mode='analyze'):
        articles = self.fetch_historical_news(days, max_articles)
        
        if not articles:
            print("No historical articles found")
            return
            
        print(f"Found {len(articles)} historical articles")
        
        # Group articles by date
        date_grouped = defaultdict(list)
        for article in articles:
            date_str = article['date'].split('T')[0]  # Get just the date part
            date_grouped[date_str].append(article)
            
        # Process each day's articles
        for date_str, day_articles in sorted(date_grouped.items()):
            total_score = 0
            num_articles = 0
            
            print(f"\nProcessing news from {date_str} ({len(day_articles)} articles)")
            
            for article in day_articles:
                print(f'\n{"-" * 60}')
                print(f'Title: {article["title"]}')
                print(f'Published: {article["date"]}')
                print(f'Summary: {article["summary"]}')
                
                content = ""
                if fetch_full_content:
                    print("Fetching full content...")
                    content = self.fetch_article_content(article['link'])
                    
                # Mark as seen
                self.seen_links.add(article['link'])
                
                # Score the article
                title_score = self.score_with_dictionary(article['title']) * 1.5
                summary_score = self.score_with_dictionary(article['summary'])
                
                if content:
                    content_score = self.score_with_dictionary(content) * 0.5
                    score = (title_score + summary_score + content_score) / 2.5
                else:
                    score = (title_score + summary_score) / 2
                
                label = "Positive" if score > self.positive_threshold else "Negative" if score < self.negative_threshold else "Neutral"
                print(f'Sentiment: {label}, Raw Score: {score:.4f}')
                
                # Update dictionary if in learning mode
                if mode == 'learn':
                    combined_text = article['title'] + " " + article['summary']
                    if content:
                        combined_text += " " + content
                    self.update_dictionary(combined_text, score)
                
                total_score += score
                num_articles += 1
            
            # Calculate and log daily sentiment
            if num_articles > 0:
                final_score = total_score / num_articles
                overall = "Positive" if final_score >= self.positive_threshold else "Negative" if final_score <= self.negative_threshold else "Neutral"
                print(f'\n>> {date_str} Overall Sentiment: {overall} ({final_score:.4f}) from {num_articles} articles')
                
                # Log to file with historical source
                timestamp = f"{date_str}T12:00:00"  # Use noon as default time
                self.log_sentiment(timestamp, final_score, num_articles, 'historical')
        
        # Save updated data
        if mode == 'learn':
            print("\nLearning mode: Saving updated dictionary")
            self.save_dictionary()
            
        self._save_seen_links()
        print("\nHistorical analysis complete")
    
    def plot_historical_sentiment(self):
        try:
            # Load the sentiment log
            data = pd.read_csv(f'logs/{self.log_file}')
            
            if len(data) == 0:
                print("No data to plot yet.")
                return
                
            # Convert timestamp to datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            plt.plot(data['timestamp'], data['score'], 'b-', label='Sentiment Score')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axhline(y=self.positive_threshold, color='g', linestyle='--', alpha=0.5, label='Positive Threshold')
            plt.axhline(y=self.negative_threshold, color='r', linestyle='--', alpha=0.5, label='Negative Threshold')
            
            plt.fill_between(data['timestamp'], data['score'], 0, where=(data['score'] >= 0), 
                           color='green', alpha=0.3, interpolate=True)
            plt.fill_between(data['timestamp'], data['score'], 0, where=(data['score'] <= 0), 
                           color='red', alpha=0.3, interpolate=True)
            
            plt.title(f'Sentiment Analysis for {self.ticker}')
            plt.ylabel('Sentiment Score')
            plt.xlabel('Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(f'logs/sentiment_plot_{self.ticker}.png')
            plt.close()
            
            print(f"Plot saved to logs/sentiment_plot_{self.ticker}.png")
            
        except Exception as e:
            print(f"Error plotting data: {e}")
    
    def run(self):
        """Run the sentiment analyzer in a loop."""
        print(f"Starting sentiment analysis for {self.ticker}")
        print(f"Filtering by keyword: {self.keyword if self.keyword else 'None'}")
        print(f"Checking for updates every {self.polling_interval} seconds")
        print(f"Press Ctrl+C to stop")
        
        try:
            while True:
                self.analyze_sentiment()
                time.sleep(self.polling_interval)
        except KeyboardInterrupt:
            print("\nStopped by user. Saving data...")
            self.save_dictionary()
            self._save_seen_links()
            print("Data saved.")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time news sentiment analyzer')
    parser.add_argument('--ticker', type=str, default='BA', help='Stock ticker symbol')
    parser.add_argument('--keyword', type=str, default=None, help='Keyword filter (optional)')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='Learning rate for dictionary updates')
    parser.add_argument('--interval', type=int, default=60, help='Polling interval in seconds')
    parser.add_argument('--plot', action='store_true', help='Plot historical sentiment data and exit')
    parser.add_argument('--historical', action='store_true', help='Analyze historical data')
    parser.add_argument('--days', type=int, default=30, help='Number of days to look back for historical analysis')
    parser.add_argument('--max-articles', type=int, default=100, help='Maximum articles to process for historical analysis')
    parser.add_argument('--full-content', action='store_true', help='Fetch full article content for historical analysis')
    parser.add_argument('--learning-mode', action='store_true', help='Update dictionary while processing historical data')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SentimentAnalyzer(
        ticker=args.ticker, 
        keyword=args.keyword,
        learning_rate=args.learning_rate,
        polling_interval=args.interval
    )
    
    # Determine what to do based on arguments
    if args.historical:
        print(f"Analyzing historical data for {args.ticker} over the past {args.days} days")
        mode = 'learn' if args.learning_mode else 'analyze'
        analyzer.analyze_historical_data(
            days=args.days,
            max_articles=args.max_articles,
            fetch_full_content=args.full_content,
            mode=mode
        )
    elif args.plot:
        analyzer.plot_historical_sentiment()
    else:
        analyzer.run()