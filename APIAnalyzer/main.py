from fetch_news import fetch_news_from_api
from preprocess import preprocess
from analyzer import load_sentiment_dict, analyze_sentiment, save_sentiment_dict
from updater import update_dictionary

def main():
    # Use real-time headlines
    news_list = fetch_news_from_api()

    sent_dict = load_sentiment_dict()

    for news in news_list:
        tokens = preprocess(news)
        score, unknown_words = analyze_sentiment(tokens, sent_dict)
        print(f"\nNews: {news}")
        print(f"Tokens: {tokens}")
        print(f"Sentiment Score: {score}")
        print(f"Unknown Words: {unknown_words}")

        # Update dictionary based on score
        sent_dict = update_dictionary(sent_dict, unknown_words, score)

    save_sentiment_dict(sent_dict)

if __name__ == "__main__":
    main()