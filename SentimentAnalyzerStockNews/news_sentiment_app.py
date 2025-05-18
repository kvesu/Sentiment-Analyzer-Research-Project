import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import time

# Set up the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Keep track of seen headlines to avoid duplication
seen_headlines = set()

def fetch_news():
    url = "https://finviz.com/news.ashx?v=3"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')
    rows = soup.select('table.fullview-news-outer tr')

    news_items = []
    for row in rows:
        time_tag = row.select_one('td:nth-of-type(1)')
        headline_tag = row.select_one('td:nth-of-type(2) a')

        if time_tag and headline_tag:
            time_text = time_tag.text.strip()
            headline = headline_tag.text.strip()
            link = 'https://finviz.com/' + headline_tag['href']

            if headline in seen_headlines:
                continue  # Skip already seen headlines
            seen_headlines.add(headline)

            sentiment = analyzer.polarity_scores(headline)
            score = sentiment['compound']
            label = 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral'

            news_items.append({
                'time': time_text,
                'headline': headline,
                'link': link,
                'sentiment': label,
                'score': round(score, 3)
            })

    return news_items

def main():
    st.set_page_config(page_title="Live News Sentiment Analyzer", layout="wide")
    st.title("📰 Real-Time Finviz News Sentiment Analyzer")

    news_placeholder = st.empty()

    while True:
        news_items = fetch_news()

        with news_placeholder.container():
            st.write("### Latest Headlines")
            for item in news_items:
                st.markdown(f"**[{item['headline']}]({item['link']})**")
                st.write(f"⏱️ {item['time']} | Sentiment: `{item['sentiment']}` | Score: `{item['score']}`")
                st.markdown("---")

        time.sleep(120)  # Refresh every 2 minutes

if __name__ == "__main__":
    main()
