#!/usr/bin/env python3

from datetime import datetime
import sys
import ollama
import chromadb
from utilities import getconfig

def get_query(args):
    if not sys.stdin.isatty():  # Input is being piped or redirected
        return sys.stdin.read().strip()
    elif args:  # Arguments provided
        return " ".join(args)
    else:
        print('Usage: search.py "your question here"')
        print('       OR cat question.txt | search.py')
        sys.exit(1)

# Load config and connect to ChromaDB
embedmodel = getconfig()["embedmodel"]
mainmodel = getconfig()["mainmodel"]

try:
    chroma = chromadb.HttpClient(host="localhost", port=8000)
    collection = chroma.get_or_create_collection("buildragwithpython")
except Exception as e:
    print("\n❌ Unable to connect to ChromaDB at localhost:8000.")
    print("👉 Please make sure the Chroma server is running.")
    print("   You can usually start it with something like:\n\n   chroma run --path /mnt/500GB/ChromaDB\n")
    sys.exit(1)

# Parse arguments
args = sys.argv[1:]
save_docs = "--save-docs" in args
if save_docs:
    args.remove("--save-docs")

# Get input query
query = get_query(args)
queryembed = ollama.embeddings(model=embedmodel, prompt=query)['embedding']

# Perform vector search
results = collection.query(query_embeddings=[queryembed], n_results=5)
relevantdocs = results["documents"][0]
metadatas = results["metadatas"][0]

# Save documents and metadata to file
if save_docs:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/mnt/500GB/rag_output/retrieved_docs_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write(f"Query: {query}\n\n")
        for i, (doc, meta) in enumerate(zip(relevantdocs, metadatas), 1):
            f.write(f"[Document {i}]\n{doc}\n")
            f.write(f"Metadata: {meta}\n\n")
            f.write(f"---------------------------\n\n")
    print(f"\n📄 Retrieved documents saved to {filename}\n")

# Combine documents into a prompt
docs = "\n\n".join(relevantdocs)
modelquery = f"{query} - Answer that question using the following text as a resource: {docs}"

# Stream the answer from the model
stream = ollama.generate(model=mainmodel, prompt=modelquery, stream=True)

for chunk in stream:
    if chunk["response"]:
        print(chunk["response"], end="", flush=True)
