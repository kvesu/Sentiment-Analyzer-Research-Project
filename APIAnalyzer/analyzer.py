import json
import os

SENTIMENT_DICT_FILE = "sentiment_dict.json"

def load_sentiment_dict():
    if os.path.exists(SENTIMENT_DICT_FILE):
        with open(SENTIMENT_DICT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_sentiment_dict(sent_dict):
    with open(SENTIMENT_DICT_FILE, "w") as f:
        json.dump(sent_dict, f, indent=4)

def analyze_sentiment(tokens, sent_dict):
    score = 0
    unknown_words = []

    for word in tokens:
        if word in sent_dict:
            score += sent_dict[word]
        else:
            unknown_words.append(word)

    return score, unknown_words