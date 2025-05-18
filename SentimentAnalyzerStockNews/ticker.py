from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

# FinViz news page URL
news_url = 'https://finviz.com/news.ashx?v=3'

# Send the request with a custom user-agent header
req = Request(url=news_url, headers={'user-agent': 'Mozilla/5.0'})
response = urlopen(req)

# Parse the HTML
html = BeautifulSoup(response, 'html.parser')

# Find all anchor tags (<a>) with href attributes
tickers = set()
for link in html.find_all('a', href=True):
    href = link['href']
    # Check if the href contains 'quote.ashx?t=' to find the ticker links
    if 'quote.ashx?t=' in href:
        ticker = href.split('quote.ashx?t=')[1].split('&')[0].split('#')[0]
        tickers.add(ticker.upper())

# Print the tickers found
print("Tickers found on FinViz news page:")
print(sorted(tickers))