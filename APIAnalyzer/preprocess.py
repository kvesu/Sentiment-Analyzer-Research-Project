import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

def preprocess(text):
    # Lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return cleaned_tokens