#!/usr/bin/env python3
import ollama, chromadb, time, os
from utilities import readtext, getconfig
import nltk
from mattsollamatools import chunker, chunk_text_by_sentences

# This script expects a folder called /mnt/500GB/docs with all of the target documentation
# It expects a chromadb server running on localhost:8000, for example:  chroma run --path /mnt/500GB/ChromaDB
# It will build a collection called 'buildragwithpython'

collectionname = "buildragwithpython"

nltk.download('punkt_tab')


# Set up ChromaDB
chroma = chromadb.HttpClient(
    host="localhost",
    port=8000,
    headers={"X-Tenant-Id": "default_tenant"}  # this is often required in newer Chroma versions
)
if any(collection.name == collectionname for collection in chroma.list_collections()):
    print('Deleting existing collection...')
    chroma.delete_collection(collectionname)

collection = chroma.get_or_create_collection(name=collectionname, metadata={"hnsw:space": "cosine"})

embedmodel = getconfig()["embedmodel"]
starttime = time.time()

# File tracking
skipped_files = []
processed_files = []

# Walk through all files in /docs, including nested folders
for root, dirs, files in os.walk("/mnt/500GB/docs"):
    for file in files:
        full_path = os.path.join(root, file)

        if not file.lower().endswith(".md"): # target files with the markdown extension
            skipped_files.append(full_path)
            continue

        try:
            text = readtext(full_path)
            chunks = chunk_text_by_sentences(source_text=text, sentences_per_chunk=7, overlap=0)
            print(f"{file}: {len(chunks)} chunks")

            for index, chunk in enumerate(chunks):
                embed = ollama.embeddings(model=embedmodel, prompt=chunk)['embedding']
                doc_id = f"{full_path}_chunk{index}"
                collection.add([doc_id], [embed], documents=[chunk], metadatas={"source": full_path})
                print(".", end="", flush=True)

            processed_files.append(full_path)

        except Exception as e:
            print(f"\nError processing {full_path}: {e}")
            skipped_files.append(full_path)

# Save processed and skipped file lists
with open("processed.txt", "w") as f:
    for item in processed_files:
        f.write(item + "\n")

with open("skipped.txt", "w") as f:
    for item in skipped_files:
        f.write(item + "\n")

print(f"\n--- {time.time() - starttime:.2f} seconds ---")
