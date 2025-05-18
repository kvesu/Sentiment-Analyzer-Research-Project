import pandas as pd
from langchain_community.document_loaders import NewsURLLoader
import os

urls = [
    "https://finance.yahoo.com/news/summer-travel-season-heats-up-with-lower-gas-prices-and-airfares-150006735.html",
    "https://finance.yahoo.com/news/walmart-should-eat-the-tariffs-trump-says-after-retailer-warns-of-looming-price-hikes-155126753.html",
    "https://finance.yahoo.com/news/jd-power-car-buyers-are-still-interested-in-evs-and-tesla-alternatives-144540737.html",
    "https://finance.yahoo.com/news/germany-does-not-expect-unicredit-191434131.html",
    "https://finance.yahoo.com/news/trump-speak-putin-zelenskyy-fresh-162953991.html"
]

loader = NewsURLLoader(urls=urls)
data = loader.load()

df = pd.DataFrame(
    [{"title": d.metadata["title"], "text":d.page_content} for d in data]
)

script_dir = os.path.dirname(os.path.abspath(__file__))  # folder where the script is
file_path = os.path.join(script_dir, "stopwords.txt")

with open(file_path, "r") as f:
    stopwords = f.read().split("\n")[:-1]

def preprocess_text(text):
    words = text.split()
    words = [w.lower() for w in words]
    words = [w for w in words if w not in stopwords]
    words = [w for w in words if w.isalpha()]
    return " ".join(words)

df["text_clean"] = df["text"].apply(preprocess_text)

lm_dict_path = os.path.join(script_dir, "Loughran-McDonald_MasterDictionary_1993-2024.csv")
lm_dict = pd.read_csv(lm_dict_path)

pos_words = lm_dict[lm_dict["Positive"] != 0]["Word"].str.lower().to_list()
neg_words = lm_dict[lm_dict["Negative"] != 0]["Word"].str.lower().to_list()

df["n"] = df["text_clean"].apply(lambda x: len(x.split()))
df["n_pos"] = df["text_clean"].apply(
    lambda x: len([w for w in x.split() if w in pos_words])
)
df["n_neg"] = df["text_clean"].apply(
    lambda x: len([w for w in x.split() if w in neg_words])
)

df["lm_level"] = df["n_pos"] - df["n_neg"]

df["lm_score1"] = (df["n_pos"] - df["n_neg"]) / df["n"]
df["lm_score2"] = (df["n_pos"] - df["n_neg"]) / (df["n_pos"] + df["n_neg"])

CUTOFF = 0.3
df["lm_sentiment"] = df["lm_score2"].apply(
    lambda x: "positive" if x > CUTOFF else "negative" if x < -CUTOFF else "neutral"
)

print(df)