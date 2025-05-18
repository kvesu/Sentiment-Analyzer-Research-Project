import pandas as pd
from langchain_community.document_loaders import NewsURLLoader
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from tenacity import retry, stop_after_attempt, RetryError

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

class SentimentClassification(BaseModel):
    sentiment: str = Field(
        ...,
        description="The sentiment of the text",
        enum=["positive", "negative", "neutral"],
    )
    score: float = Field(..., description="The score of the sentiment", ge=-1, le=1)
    justification: str = Field(..., description="The justification of the sentiment")
    main_entity: str = Field(..., description="The main entity discussed in the text")

@retry(stop=stop_after_attempt(5))

def run_chain(text: str, chain) -> dict:
    return chain.invoke({"news": text}).dict()


def llm_sentiment(text: str, llm) -> tuple[str, float, str, str]:
    parser = PydanticOutputParser(pydantic_object=SentimentClassification)

    prompt = PromptTemplate(
        template="Describe the sentiment of a text of financial news.\n{format_instructions}\n{news}\n",
        input_variables=["news"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    try:
        result = run_chain(text, chain)

        return (
            result["sentiment"],
            result["score"],
            result["justification"],
            result["main_entity"],
        )
    except RetryError as e:
        print(f"Error: {e}")
        return "error", 0, "", ""
    
# Replace with the correct model, or use ChatOpenAI if you want to use OpenAI
llama2 = ChatOllama(model="llama2", temperature=0.1)

df[
    ["llama2_sentiment", "llama2_score", "llama2_justification", "llama2_main_entity"]
] = (df["text"].apply(lambda x: llm_sentiment(x, llama2)).apply(pd.Series))

df[
    [
        "title",
        "text",
        "llama2_sentiment",
        "llama2_score",
        "llama2_justification",
        "llama2_main_entity",
    ]
]

print(df)