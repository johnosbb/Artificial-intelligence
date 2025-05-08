import chromadb
import ollama  # Assuming you are using Ollama for generating embeddings
import time

# Initialize ChromaDB client
chroma = chromadb.HttpClient(
    host="localhost",
    port=8000,
    headers={"X-Tenant-Id": "default_tenant"}
)

collectionname = "simple_test_collection"

# Step 1: Try deleting the collection if it exists
try:
    chroma.delete_collection(collectionname)
    print(f"Collection '{collectionname}' deleted.")
except Exception as e:
    print(f"Error deleting collection: {e}")

# Step 2: Create a collection with metadata
try:
    metadata = {"type": "default"}  # Include type metadata to prevent KeyError
    collection = chroma.get_or_create_collection(name=collectionname, metadata=metadata)
    print(f"Collection '{collectionname}' created successfully.")
except Exception as e:
    print(f"Error creating collection: {e}")
    exit()

# Step 3: Use Ollama or another model to generate embeddings
embed_model = "your_embed_model_here"  # Use your actual embedding model name
document = "This is a simple test document."
embedding = ollama.embeddings(model=embed_model, prompt=document)["embedding"]

# Step 4: Add a document with valid embedding and metadata
try:
    doc_id = "test_doc_1"
    metadata = {"source": "test_document"}
    collection.add([doc_id], [embedding], documents=[document], metadatas=[metadata])
    print(f"Document added with ID '{doc_id}'.")
except Exception as e:
    print(f"Error adding document: {e}")
    exit()

# Step 5: Retrieve the document and verify
try:
    docs = collection.get()  # Retrieve documents
    print(f"Documents in collection '{collectionname}':", docs)
except Exception as e:
    print(f"Error retrieving documents: {e}")
