import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_community.document_loaders import NewsURLLoader
import scipy
import torch

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

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def finbert_sentiment(text: str) -> tuple[float, float, float, str]:
    with torch.no_grad():
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        outputs = model(**inputs)
        logits = outputs.logits
        scores = {
            k: v
            for k, v in zip(
                model.config.id2label.values(),
                scipy.special.softmax(logits.numpy().squeeze()),
            )
        }
        return (
            scores["positive"],
            scores["negative"],
            scores["neutral"],
            max(scores, key=scores.get),
        )
    
# Notice that this is the raw text, no preprocessing
df[["finbert_pos", "finbert_neg", "finbert_neu", "finbert_sentiment"]] = (
    df["text"].apply(finbert_sentiment).apply(pd.Series)
)
df["finbert_score"] = df["finbert_pos"] - df["finbert_neg"]

df[
    [
        "title",
        "text",
        "finbert_pos",
        "finbert_neg",
        "finbert_neu",
        "finbert_sentiment",
        "finbert_score",
    ]
]

print(df)