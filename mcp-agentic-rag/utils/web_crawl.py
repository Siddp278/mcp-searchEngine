# Work out the firecrawl API here

import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

url = os.getenv("FIRECRAWL_URL")
api_key = os.getenv('FIRECRAWL_API_KEY')

query = """
What is the meaning of life?
"""

payload = {"query": query, "timeout": 60000}
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    results = response.json().get("data", [])
except requests.exceptions.RequestException as e:
    print(f"Error connecting to Firecrawl API: {e}")
    results = []


def crawl_and_extract_text(target_url: str) -> str:
    try:
        r = requests.get(target_url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=' ', strip=True)
        return text[:5000] 
    except Exception as e:
        return f"Error fetching {target_url}: {e}"


for i, item in enumerate(results):
    page_url = item.get("url")
    page_title = item.get("title", "No Title")

    if not page_url:
        continue

    print(f"\nCrawling [{i+1}] {page_title}:\n{page_url}")
    extracted_text = crawl_and_extract_text(page_url)
    print(f"\nExtracted Text:\n{extracted_text[:1000]}...")

    time.sleep(1)

