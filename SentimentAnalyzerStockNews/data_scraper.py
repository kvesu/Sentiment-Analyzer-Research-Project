import requests
from bs4 import BeautifulSoup

# Define the URL
url = 'https://finviz.com/news.ashx?v=3'

# Set a User-Agent to mimic a real browser request (Finviz might block non-browser requests)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Send the GET request with the headers
response = requests.get(url, headers=headers)

# Parse the HTML content of the page
soup = BeautifulSoup(response.content, 'html.parser')

# Use a more specific selector for news headlines
news_items = soup.find_all('a', class_='tab-link-news')

# Loop through and print the headlines
if news_items:
    for item in news_items:
        headline = item.text.strip()  # Extract the text (headline)
        link = 'https://finviz.com/' + item['href']  # Construct the full URL
        print(f"Headline: {headline}\nLink: {link}\n")
else:
    print("No headlines found. The structure may have changed.")
