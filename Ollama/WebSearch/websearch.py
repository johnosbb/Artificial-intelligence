import os
import sys
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from readability import Document
import ollama

# Get environment variable and command-line args
search_url = os.getenv("SEARCH_URL")  # Unused, can be removed if not used elsewhere
query = " ".join(sys.argv[1:])

print(f"Query: {query}")

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True)

test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/200px-PNG_transparency_demonstration_1.png"
response = requests.get(test_url)
with open(IMAGE_DIR / "test.png", "wb") as f:
    f.write(response.content)

print("Test image saved.")


OUTPUT_FILE = "output.txt"

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
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }

    texts = []
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n\n")
        for url in urls:
            print(f"Fetching {url}")
            try:
                res = requests.get(url, headers=headers)
                res.raise_for_status()
                html = res.text
                text = html_to_text(html)
                entry = f"Source: {url}\n{text}\n\n"
                texts.append(entry)
                f.write(entry)
                print(f"\n==== [CLEANED TEXT from {url}] ====\n{text[:800]}\n")
            except requests.HTTPError as e:
                print(f"Failed to fetch {url}: {e}")
    return texts


def extract_images_from_html(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    image_urls = []

    for img in soup.find_all("img"):
        src = img.get("src")
        if src:
            if src.startswith("//"):
                src = "https:" + src
            elif src.startswith("/"):
                src = base_url.rstrip("/") + src
            image_urls.append(src)

    return image_urls



def html_to_text(html):
    doc = Document(html)
    summary_html = doc.summary()
    soup = BeautifulSoup(summary_html, "html.parser")
    return soup.get_text(strip=True, separator=" ")

def answer_query(query, texts):
    prompt = (
        f"{query}. Summarize the information and provide a four paragraph summary of the story. "
        f"Use only the information in the following articles to answer the question:\n\n"
        + "\n\n".join(texts)
    )

    print("\n========== FINAL PROMPT ==========\n")
    print(prompt)
    print("\n==================================\n")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write("====== AI Generated Answer ======\n\n")
        stream = ollama.generate(
            model="llama3",  # Update if needed
            prompt=prompt,
            stream=True,
            options={"num_ctx": 16000},
        )

        for chunk in stream:
            if chunk.get("response"):
                response = chunk["response"]
                print(response, end="", flush=True)
                f.write(response)

# Run the process
urls = get_news_urls(query)
alltexts = get_cleaned_text(urls)
answer_query(query, alltexts)
