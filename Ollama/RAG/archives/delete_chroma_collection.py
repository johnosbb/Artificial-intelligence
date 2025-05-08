import chromadb

# Initialize ChromaDB client
chroma = chromadb.HttpClient(
    host="localhost",
    port=8000,
    headers={"X-Tenant-Id": "default_tenant"}  # Update if you use a different tenant
)

# Step 1: Prompt the user for the collection name to delete
collection_name = input("Enter the name of the collection you want to delete: ")

# Step 2: Delete the specified collection
try:
    chroma.delete_collection(collection_name)
    print(f"Collection '{collection_name}' has been deleted successfully.")
except Exception as e:
    print(f"Error deleting collection '{collection_name}': {e}")
