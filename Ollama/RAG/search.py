#!/usr/bin/env python3

import sys
import ollama
import chromadb
from utilities import getconfig

def get_query():
    if not sys.stdin.isatty():  # Input is being piped or redirected
        return sys.stdin.read().strip()
    elif len(sys.argv) > 1:  # Arguments provided
        return " ".join(sys.argv[1:])
    else:
        print('Usage: search.py "your question here"')
        print('       OR cat question.txt | search.py')
        sys.exit(1)

# Load config and connect to ChromaDB
embedmodel = getconfig()["embedmodel"]
mainmodel = getconfig()["mainmodel"]
chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection("buildragwithpython")

# Get input query
query = get_query()
queryembed = ollama.embeddings(model=embedmodel, prompt=query)['embedding']

# Perform search and build model prompt
relevantdocs = collection.query(query_embeddings=[queryembed], n_results=5)["documents"][0]
docs = "\n\n".join(relevantdocs)
modelquery = f"{query} - Answer that question using the following text as a resource: {docs}"

# Stream the answer
stream = ollama.generate(model=mainmodel, prompt=modelquery, stream=True)

for chunk in stream:
    if chunk["response"]:
        print(chunk["response"], end="", flush=True)
