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

# Retrieve and print the documents
documents = collection.get()
print(documents)import chromadb

# Connect to Chroma
chroma = chromadb.HttpClient(
    host="localhost",
    port=8000,
    headers={"X-Tenant-Id": "default_tenant"}
)

# Retrieve the collection
collection_name = "buildragwithpython"
collection = chroma.get_collection(collection_name)

# Retrieve and print the documents
documents = collection.get()
print(documents)