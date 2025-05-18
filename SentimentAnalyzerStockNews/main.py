import pandas as pd
from datetime import datetime
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re
from finvizfinance.news import News
from finvizfinance.quote import finvizfinance

# Download VADER if not already available
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()

# Get general finviz news (fallback method)
def get_finviz_news():
    try:
        fnews = News()
        news_dict = fnews.get_news()

        if isinstance(news_dict, dict):
            print("News data is a dictionary with keys:", list(news_dict.keys()))
            if 'news' in news_dict:
                return pd.DataFrame(news_dict['news'])
            elif any(isinstance(v, list) for v in news_dict.values()):
                for key, value in news_dict.items():
                    if isinstance(value, list) and len(value) > 0:
                        return pd.DataFrame(value)
            else:
                return pd.DataFrame([news_dict])
        else:
            return news_dict
    except Exception as e:
        print(f"Error fetching news with finvizfinance: {e}")
        return None

# Get stock specific news from finvizfinance
def get_stock_news(ticker_symbols=None):
    if ticker_symbols is None:
        ticker_symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']

    if isinstance(ticker_symbols, str):
        ticker_symbols = [ticker_symbols]

    all_news = []

    for ticker in ticker_symbols:
        try:
            print(f"Fetching news for {ticker}...")
            stock = finvizfinance(ticker)
            stock_news = stock.ticker_news()

            if stock_news is not None and len(stock_news) > 0:
                if 'Ticker' not in stock_news.columns:
                    stock_news['Ticker'] = ticker
                all_news.append(stock_news)
                print(f"Found {len(stock_news)} news items for {ticker}")
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")

    if all_news:
        return pd.concat(all_news, ignore_index=True)
    else:
        print("No stock-specific news found. Fetching general market news...")
        return get_finviz_news()

# Process datetime strings from Finviz
def process_datetime(time_str):
    if not isinstance(time_str, str):
        return None

    current_date = datetime.now()

    try:
        if re.match(r'\d{1,2}:\d{2}(AM|PM)', time_str):
            dt = datetime.strptime(time_str, '%I:%M%p')
            return dt.replace(year=current_date.year, month=current_date.month, day=current_date.day)
        elif re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', time_str):
            return datetime.strptime(time_str, '%m/%d/%y')
        elif isinstance(time_str, datetime):
            return time_str
    except ValueError:
        return None

    return None

# Process news data
def process_news(news_data):
    if news_data is None:
        print("No news data available.")
        return pd.DataFrame()

    news_df = None
    if isinstance(news_data, dict):
        if 'news' in news_data:
            news_df = pd.DataFrame(news_data['news'])
        else:
            for key, value in news_data.items():
                if isinstance(value, list) and len(value) > 0:
                    news_df = pd.DataFrame(value)
                    break
    else:
        news_df = news_data if isinstance(news_data, pd.DataFrame) else None

    if news_df is None or news_df.empty:
        print("Could not convert news data to DataFrame.")
        return pd.DataFrame()

    time_col = next((col for col in ['Date', 'Time', 'date', 'time'] if col in news_df.columns), None)
    title_col = next((col for col in ['Title', 'Headline', 'title', 'headline'] if col in news_df.columns), None)
    ticker_col = next((col for col in ['Ticker', 'ticker', 'Symbol', 'symbol'] if col in news_df.columns), None)

    if not all([time_col, title_col]):
        print(f"Missing required columns. Available columns: {news_df.columns.tolist()}")
        return pd.DataFrame()

    data = []
    for _, row in news_df.iterrows():
        title = row.get(title_col, '')
        time_str = row.get(time_col, '')
        ticker = row.get(ticker_col, '') if ticker_col else ''

        if title and time_str:
            dt = process_datetime(time_str)
            if isinstance(ticker, str):
                ticker = re.sub(r'\s+', ' ', ticker).strip()

            sentiment_scores = vader.polarity_scores(title)
            compound = sentiment_scores['compound']
            sentiment = 'Positive' if compound > 0.05 else 'Negative' if compound < -0.05 else 'Neutral'
            data.append([ticker, dt, title, compound, sentiment])

    return pd.DataFrame(data, columns=['ticker', 'datetime', 'title', 'compound', 'sentiment'])

# Visualizations
def create_visualizations(df):
    if df.empty:
        return

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='sentiment', palette='coolwarm', order=['Positive', 'Neutral', 'Negative'])
    plt.title('Sentiment Distribution')

    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='compound', bins=20, kde=True)
    plt.title('Sentiment Score Distribution')

    plt.subplot(2, 2, 3)
    ticker_sentiment = df.groupby('ticker')['compound'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=ticker_sentiment.index, y=ticker_sentiment.values, palette='viridis')
    plt.title('Average Sentiment by Ticker')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 4)
    ticker_counts = df.groupby('ticker').size().sort_values(ascending=False).head(10)
    sns.barplot(x=ticker_counts.index, y=ticker_counts.values, palette='muted')
    plt.title('Number of News Items by Ticker')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('stock_news_sentiment.png')
    print("Visualizations saved to 'stock_news_sentiment.png'")
    plt.show()

# Main execution
def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
    print(f"Default stock tickers: {', '.join(tickers)}")
    user_input = input("Press Enter to use default tickers or enter your own (comma-separated): ")

    if user_input.strip():
        custom_tickers = [ticker.strip().upper() for ticker in user_input.split(',')]
        tickers = custom_tickers

    news_df = get_stock_news(tickers)

    if news_df is not None and hasattr(news_df, 'columns'):
        print("Sample news data:", news_df.head(2))

    df = process_news(news_df)

    if df.empty:
        print("No data found or error in fetching news.")
        return

    print("\nFinancial News Summary:")
    print(df.head())

    print("\nSentiment Distribution:")
    print(df.groupby('sentiment').size())

    print("\nAverage Sentiment by Ticker:")
    print(df.groupby('ticker')['compound'].mean().sort_values(ascending=False))

    print("\nNumber of News Items by Ticker:")
    print(df.groupby('ticker').size().sort_values(ascending=False))

    create_visualizations(df)

if __name__ == "__main__":
    main()
