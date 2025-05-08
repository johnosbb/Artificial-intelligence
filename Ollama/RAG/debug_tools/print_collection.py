#!/usr/bin/env python3
import chromadb

# Connect to Chroma
chroma = chromadb.HttpClient(
    host="localhost",
    port=8000,
    headers={"X-Tenant-Id": "default_tenant"}
)

# Retrieve the collection
collection_name = "buildragwithpython"
collection = chroma.get_collection(collection_name)

# Retrieve all entries (IDs are included by default)
documents = collection.get(include=["documents", "metadatas"])  # Removed 'ids'

# Count the documents
doc_count = len(documents["documents"])
print(f"📦 Total documents in '{collection_name}': {doc_count}")

# Show a few sample documents
for i in range(min(3, doc_count)):
    print(f"\n🧾 Document {i+1}:")
    print(f"ID: {documents['ids'][i]}")
    print(f"Metadata: {documents['metadatas'][i]}")
    print(f"Content:\n{documents['documents'][i][:100]}...")
