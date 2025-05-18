from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re

# Setup headless browser
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("user-agent=Mozilla/5.0")

# Start browser
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://finviz.com/news.ashx?v=3")

try:
    # Wait for the table to be present (max 10 seconds)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//table"))
    )

    soup = BeautifulSoup(driver.page_source, "html.parser")
    news_table = soup.find("table")

    records = []
    today = datetime.now().date()

    if news_table:
        for row in news_table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) != 2:
                continue
            time_str = cols[0].text.strip()
            link_tag = cols[1].find("a")
            if not link_tag:
                continue

            title = link_tag.text.strip()
            url = "https://finviz.com" + link_tag.get("href", "")

            # Extract tickers from href
            tickers = []
            for a in cols[1].find_all("a"):
                match = re.search(r"quote\.ashx\?t=([A-Z]+)", a.get("href", ""))
                if match:
                    tickers.append(match.group(1))

            records.append({
                "time": time_str,
                "title": title,
                "url": url,
                "tickers": ", ".join(set(tickers)) if tickers else None,
                "date": today.strftime("%Y-%m-%d")
            })

    df = pd.DataFrame(records)

    if not df.empty:
        print(df[['date', 'time', 'tickers', 'title']].to_string(index=False))
    else:
        print("No news articles found.")

finally:
    driver.quit()

