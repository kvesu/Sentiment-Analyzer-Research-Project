import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

def fetch_news_from_api():
    api_key = os.getenv("NEWS_API_KEY")
    
    if not api_key:
        raise ValueError("API key not found. Make sure to set NEWS_API_KEY in your .env file.")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "apple",
        "from": "2025-05-06",
        "to": "2025-05-06",
        "sortBy": "popularity",
        "apiKey": api_key
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        headlines = [article["title"] for article in articles]
        return headlines
    else:
        print(f"Error fetching news: {response.status_code} - {response.text}")
        return []

# Example usage
if __name__ == "__main__":
    headlines = fetch_news_from_api()
    for i, headline in enumerate(headlines, 1):
        print(f"{i}. {headline}")
