#!/usr/bin/env python3
import json
import chromadb


OUTPUT_DIRECTORY="/mnt/500GB/rag_output"

# Connect to Chroma
chroma = chromadb.HttpClient(
    host="localhost",
    port=8000,
    headers={"X-Tenant-Id": "default_tenant"}
)

# Name of your collection
collection_name = "buildragwithpython"
collection = chroma.get_collection(collection_name)

# Retrieve all document metadata and IDs (no embeddings or distances to keep it fast/light)
documents = collection.get(include=["metadatas"])

# Save IDs to a file
with open(f"{OUTPUT_DIRECTORY}/chroma_doc_ids.json", "w") as f:
    json.dump(documents["ids"], f, indent=2)
print(f"✅ Saved {len(documents['ids'])} document IDs to 'chroma_doc_ids.json'")

# Save metadata to a file
with open("chroma_doc_metadatas.json", "w") as f:
    json.dump(documents["metadatas"], f, indent=2)
print(f"✅ Saved metadata for {len(documents['metadatas'])} documents to 'chroma_doc_metadatas.json'")
