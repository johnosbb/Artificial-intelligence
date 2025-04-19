#!/usr/bin/env python3

from datetime import datetime
import sys
import ollama
import chromadb
from utilities import getconfig
from sentence_transformers import CrossEncoder
import numpy as np
import rag_utilities as ru
from pprint import pprint
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # suppress CUDA warnings

DEBUGGING=False


# 1. Text to Vector (Embedding)
# When indexing a document chunk:
# You take the chunk (e.g. "search_document: release note: This release includes...) and pass it to ollama.embeddings(...).
# It returns a dense vector — a list of ~1000 floating point numbers (depends on model).
# This vector captures the semantic meaning of the text, not just keywords.
# You store that vector in ChromaDB.
# When querying:
# You take the query (e.g. "search_query: In what software release were new passwords introduced?")
# You embed it using the same model → get a query vector.
# 2. Vector Comparison in ChromaDB
# ChromaDB compares your query vector with all stored document vectors.
# It uses a similarity metric, usually cosine similarity (angle between vectors in high-dimensional space).
# It returns the top N documents whose embeddings are closest to the query vector.

reranker = CrossEncoder("BAAI/bge-reranker-large")

def rerank_results(query, documents):
    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in reranked]
    return reranked_docs

def build_prompt(model_id, docs, query):
    # Check for summary intent (simple keyword check, could be replaced with NLP classifier later)
    is_summary_request = "summarise" in query.lower() or "summarize" in query.lower()

    # Use tailored prompt for summary requests
    if is_summary_request:
        if "internlm2" in model_id:
            return (
                "<|im_start|>system\n"
                "You are a helpful assistant. ONLY use the provided documents to summarise the user's requested release. "
                "Provide a bullet-point list of the top 10 most important changes. If the answer is not in the documents, say so.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "=== DOCUMENTS ===\n"
                f"{docs}\n\n"
                "=== QUESTION ===\n"
                f"{query}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
                "- "
            )
        elif "gemma" in model_id:
            return (
                "You are a helpful assistant. Use the technical documents below to summarise the specified release.\n"
                "Return a bullet-point list of the top 10 changes. Do not include info from other releases.\n\n"
                "=== DOCUMENTS ===\n"
                f"{docs}\n\n"
                "=== QUESTION ===\n"
                f"{query}\n\n"
                "=== ANSWER ===\n"
                "- "
            )
        else:
            # Default summary prompt
            return (
                "Use the following technical documents to provide a summary of the requested release.\n"
                "Return 10 bullet points of the most important changes only from that release.\n\n"
                f"DOCS:\n{docs}\n\n"
                f"QUESTION:\n{query}\n\n"
                "ANSWER:\n- "
            )

    # Non-summary queries fall back to previous model-based logic
    if "gemma" in model_id:
        return (
            "You are a helpful assistant. Use the following technical documents to answer the question.\n\n"
            "=== DOCUMENTS ===\n"
            f"{docs}\n\n"
            "=== QUESTION ===\n"
            f"{query}\n\n"
            "=== ANSWER ==="
        )
    elif "internlm2" in model_id:
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant. Only use the provided documents to answer. If unsure, say so. Avoid guessing.'\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "=== DOCUMENTS ===\n"
            f"{docs}\n\n"
            "=== QUESTION ===\n"
            f"{query}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        return (
            "You are a helpful assistant. Here are some technical documents followed by a question. Please answer it accurately.\n\n"
            f"DOCS:\n{docs}\n\n"
            f"QUESTION:\n{query}\n\n"
            f"ANSWER:"
        )



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
save_docs = False
where_filter = {}
prefixed_query = ""
query=""
n_results=7

def load_collection():
    try:
        chroma = chromadb.HttpClient(host="localhost", port=8000)
        collection = chroma.get_or_create_collection("buildragwithpython")
        return collection
    except Exception as e:
        print("\n❌ Unable to connect to ChromaDB at localhost:8000.")
        print("👉 Please make sure the Chroma server is running.")
        print("   You can usually start it with something like:\n\n   chroma run --path /mnt/500GB/ChromaDB\n")
        sys.exit(1)


def usage():
    print(f"""
Usage: search.py [options] "your question here"
       OR cat question.txt | search.py [options]

Options:
  --save-docs               Save retrieved documents to a file
  --rerank                  Use bge-reranker-large to rerank results
  --release-notes-only      Filter to only documents with doctype 'release note'
  --release <version>       Filter to a specific release version (e.g., 1.4.0)
  --section <name>          Filter to a specific section (e.g., "new functionality")
  --n-results <number>      Number of documents to retrieve (default: 5)
  --help                    Show this help message and exit

Examples:
  search.py "How does the logging system work?"
  search.py --release-notes-only --release 1.4.0 --section "new functionality" "Summarize the changes"
  cat question.txt | search.py --rerank --save-docs
    """)
    sys.exit(0)


def parse_command_line():
    global save_docs, where_filter, prefixed_query, query, rerank, n_results, release_filter, section_filter

    args = sys.argv[1:]
    save_docs = "--save-docs" in args
    rerank = "--rerank" in args
    n_results = 5  # default
    where_filter = {}
    release_filter = None
    section_filter = None

    if "--help" in args:
        usage()

    # Clean up known flags
    for flag in ["--save-docs", "--rerank", "--help"]:
        if flag in args:
            args.remove(flag)

    if "--release-notes-only" in args:
        where_filter["doctype"] = "release note"
        args.remove("--release-notes-only")

    if "--release" in args:
        idx = args.index("--release")
        try:
            release_filter = args[idx + 1]
            del args[idx:idx + 2]
        except (IndexError, ValueError):
            print("❌ Error: --release requires a value (e.g., 1.4.0)")
            usage()

    if "--section" in args:
        idx = args.index("--section")
        try:
            section_filter = args[idx + 1]
            del args[idx:idx + 2]
        except (IndexError, ValueError):
            print("❌ Error: --section requires a value (e.g., \"new functionality\")")
            usage()

    if "--n-results" in args:
        index = args.index("--n-results")
        try:
            n_results = int(args[index + 1])
            del args[index:index + 2]
        except (IndexError, ValueError):
            print("❌ Error: --n-results requires a numeric value")
            usage()

    # Get actual user query
    query = get_query(args)
    prefixed_query = "search_query: " + query


def save_documents(relevantdocs,metadatas,query):
    # Save documents and metadata to file

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/mnt/500GB/rag_output/retrieved_docs_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write(f"Query: {query}\n\n")
        for i, (doc, meta) in enumerate(zip(relevantdocs, metadatas), 1):
            f.write(f"[Document {i}]\n{doc}\n")
            f.write(f"Metadata: {meta}\n\n")
            f.write(f"---------------------------\n\n")
    print(f"\n📄 Retrieved documents saved to {filename}\n")

def log_vector_scores(query, documents, metadatas, distances):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = f"/mnt/500GB/rag_output/vector_scores_{timestamp}.log"
    
    with open(logfile, "w") as f:
        f.write(f"Query: {query}\n\n")
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            sim_score = 1 - dist  # cosine similarity
            f.write(f"[Doc {i+1}]\nScore: {sim_score:.4f}\n")
            f.write(f"Metadata: {meta}\n")
            f.write(f"Content (first 300 chars):\n{doc[:300]}...\n")
            f.write("--------------------------------------------------\n\n")
    
    print(f"\n📝 Vector similarity scores logged to: {logfile}")

# ChromaDB, and their query filtering system is not a simple dictionary like many expect.
#  It requires one operator per query, such as $eq, $in, etc.
# For example: {"release": {"$eq": "1.4.0"}, "section": {"$eq": "new functionality"}}

def perform_vector_search(queryembed, collection, release=None):
    query_kwargs = {
        "query_embeddings": [queryembed],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"]
    }

    combined_filter = {}

    if where_filter:
        combined_filter.update({k: {"$eq": v} for k, v in where_filter.items()})

    if release:
        combined_filter["release"] = {"$eq": release}

    if release_filter:
        combined_filter["release"] = {"$eq": release_filter}

    if section_filter:
        combined_filter["section"] = {"$eq": section_filter}

    if combined_filter:
        query_kwargs["where"] = {
            "$and": [{k: v} for k, v in combined_filter.items()]
        }


 
    # Copy query_kwargs and replace embeddings with a placeholder
    debug_query = dict(query_kwargs)
    debug_query["query_embeddings"] = ["<embedding omitted for readability>"]

    print("\n🔍 Final query to ChromaDB:")
    pprint(debug_query)

    return collection.query(**query_kwargs)





def log_all_vector_scores(query_embed, collection, original_query, full_log_limit=200):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"/mnt/500GB/rag_output/all_vector_scores_{timestamp}.log"

    query_kwargs = {
        "query_embeddings": [query_embed],
        "n_results": full_log_limit,
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    full_results = collection.query(**query_kwargs)

    with open(log_file, "w") as f:
        f.write(f"Query: {original_query}\n\n")
        
        # If embeddings are not available in results, we need to embed the document chunks manually
        for i in range(len(full_results["documents"][0])):
            doc = full_results["documents"][0][i]
            meta = full_results["metadatas"][0][i]
            
            # Manually embed document chunk using Ollama
            doc_embed = ollama.embeddings(model=embedmodel, prompt=f"search_document: {doc}")['embedding']
            
            # Compute similarity score
            score = np.dot(query_embed, doc_embed) / (
                np.linalg.norm(query_embed) * np.linalg.norm(doc_embed)
            )
            
            f.write(f"[Document {i+1}]\n")
            f.write(f"Score: {score:.4f}\n")
            f.write(f"Metadata: {meta}\n")
            f.write(f"Text: {doc[:500]}...\n")  # Preview the first 500 chars of the doc
            f.write("-" * 40 + "\n\n")
    
    print(f"\n📊 All vector scores logged to {log_file}\n")

collection=load_collection()
parse_command_line()


print("Checking Embeddings.")
queryembed = ollama.embeddings(model=embedmodel, prompt=prefixed_query)['embedding']

# Perform vector search
release = ru.extract_release_version(prefixed_query)
results = perform_vector_search(queryembed, collection, release=release)

#results = perform_vector_search(queryembed,collection)
relevantdocs = results["documents"][0]

# Check if relevantdocs is empty
if not relevantdocs:
    print("⚠️ No relevant documents found. Abandoning search.")
    exit(0)  # Exit the function early or handle accordingly


metadatas = results["metadatas"][0]
distances = results["distances"][0]

# Save similarity scores to file
if (DEBUGGING):
    log_vector_scores(query, relevantdocs, metadatas, distances)

if save_docs:
    save_documents(relevantdocs,metadatas,query)

# Combine documents into a prompt
docs = "\n\n".join(relevantdocs)

# Combine documents into a prompt
docs = "\n\n".join(relevantdocs)

if(DEBUGGING):
    log_all_vector_scores(queryembed, collection, query)


if rerank:
    print("\n🔄 Reranking documents with bge-reranker-large...")
    relevantdocs = rerank_results(query, relevantdocs)
    docs = "\n\n".join(relevantdocs)



modelquery = build_prompt(mainmodel, docs, query)

print(f"Prompt: {modelquery}\n")
print("Sending Prompt to Model.")
# Stream the answer from the model
stream = ollama.generate(model=mainmodel, prompt=modelquery, stream=True)

for chunk in stream:
    if chunk["response"]:
        print(chunk["response"], end="", flush=True)
