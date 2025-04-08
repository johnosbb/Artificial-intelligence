import os
import sys
import requests
from bs4 import BeautifulSoup
from readability import Document
import ollama

# Get environment variable and command-line args
search_url = os.getenv("SEARCH_URL")
query = " ".join(sys.argv[1:])

print(f"Query: {query}")

def get_news_urls(query):
    SERPAPI_KEY = os.getenv("SERPAPI_KEY") or "your_api_key_here"

    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "num": 5,
    }

    response = requests.get("https://serpapi.com/search.json", params=params)
    response.raise_for_status()
    results = response.json()

    urls = [res["link"] for res in results.get("organic_results", [])][:5]
    skip_domains = ["amazon.com", "quora.com", "reddit.com", "bulbs.com"]
    urls = [u for u in urls if not any(skip in u for skip in skip_domains)]
    return urls


def get_cleaned_text(urls):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0.0.0 Safari/537.36"
    }

    texts = []
    for url in urls:
        print(f"Fetching {url}")
        res = requests.get(url, headers=headers)
        try:
            res.raise_for_status()
        except requests.HTTPError as e:
            print(f"Failed to fetch {url}: {e}")
            continue

        html = res.text
        text = html_to_text(html)
        texts.append(f"Source: {url}\n{text}\n\n")
        print(f"\n==== [CLEANED TEXT from {url}] ====\n{text[:800]}\n")
    return texts


def html_to_text(html):
    doc = Document(html)
    summary_html = doc.summary()
    soup = BeautifulSoup(summary_html, "html.parser")
    return soup.get_text(strip=True, separator=" ")

def answer_query(query, texts):
    prompt = (
        f"{query}. Summarize the information and provide an answer. "
        f"Use only the information in the following articles to answer the question:\n\n"
        + "\n\n".join(texts)
    )
    print("\n========== FINAL PROMPT ==========\n")
    print(prompt)
    print("\n==================================\n")

    stream = ollama.generate(
        model="llama3",  # or any valid one from `ollama list`
        prompt=prompt,
        stream=True,
        options={"num_ctx": 16000},
    )


    for chunk in stream:
        if chunk.get("response"):
            print(chunk["response"], end="", flush=True)

# Run the process
urls = get_news_urls(query)
alltexts = get_cleaned_text(urls)
answer_query(query, alltexts)
