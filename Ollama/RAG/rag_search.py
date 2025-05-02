import ollama
import chromadb
import numpy as np
from sentence_transformers import CrossEncoder
from pprint import pprint
from utilities import getconfig
from datetime import datetime
import json
import os
from config_loader import get_index_dir,get_output_dir
from prompt_builder.factory import get_prompt_builder

OUTPUT_DIRECTORY=get_output_dir()
# Reranking model
reranker = CrossEncoder("BAAI/bge-reranker-large")

def load_collection():
    try:
        chroma = chromadb.HttpClient(host="localhost", port=8000)
        collection = chroma.get_or_create_collection("buildragwithpython")
        return collection
    except Exception as e:
        print("\n❌ Unable to connect to ChromaDB at localhost:8000.")
        print("👉 Please make sure the Chroma server is running.")
        sys.exit(1)





def get_query_embedding(
    query,
    save_to_file=True,
    load_from_file=False,
    filename=f"{OUTPUT_DIRECTORY}/query_embedding.json"
):
    # Load embedding from file if requested
    if load_from_file and os.path.exists(filename):
        with open(filename, "r") as f:
            embedding = json.load(f)
        print(f"📥 Using previously saved embedding from {filename}")
        return embedding

    # Otherwise, generate embedding normally
    embedmodel = getconfig()["embedmodel"]
    embedding = ollama.embeddings(model=embedmodel, prompt=f"search_query: {query}")['embedding']

    # Optionally save the embedding
    if save_to_file:
        with open(filename, "w") as f:
            json.dump(embedding, f)
        print(f"📝 Saved embedding to {filename}")

    return embedding





def build_chroma_query_kwargs(query_embed, n_results=5, release=None, section=None, doc_types=None, doc_ids=None):
    query_kwargs = {
        "query_embeddings": [query_embed],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"]
    }

    filters = []

    if release:
        filters.append({"release": {"$eq": release}})
    if section:
        filters.append({"section": {"$eq": section}})
    if doc_types:
        normalized_doc_types = [dt.lower() for dt in doc_types]
        filters.append({"doctype": {"$in": normalized_doc_types}})
    if doc_ids:
        filters.append({"full_doc_id": {"$in": doc_ids}})


    if len(filters) == 1:
        query_kwargs["where"] = filters[0]
    elif len(filters) > 1:
        query_kwargs["where"] = {"$and": filters}

    return query_kwargs


def perform_vector_search(queryembed, collection, release=None, section=None, n_results=5, doc_types=None, doc_ids=None):
    query_kwargs = build_chroma_query_kwargs(
        query_embed=queryembed,
        n_results=n_results,
        release=release,
        section=section,
        doc_types=doc_types,
        doc_ids=doc_ids
    )

    # Debugging output
    debug_query = dict(query_kwargs)
    debug_query["query_embeddings"] = ["<embedding omitted>"]
    print("\n🔍 Final query to ChromaDB:")
    pprint(debug_query)

    return collection.query(**query_kwargs)

def rerank_results(query, documents, metadatas):
    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(documents, metadatas, scores), key=lambda x: x[2], reverse=True)
    reranked_docs = [doc for doc, _, _ in reranked]
    reranked_metas = [meta for _, meta, _ in reranked]
    return reranked_docs, reranked_metas

def build_prompt(model_id, docs, query,chat_history=None ):
    prompt_builder = get_prompt_builder(model_id)
    prompt = prompt_builder.build_prompt(
        model_id=model_id,
        docs=docs,             # your combined top retrieved documents
        query=query,           # user's question
        chat_history=chat_history      # optional for chat-based models
    )
    return prompt

# In rag_search.py, ensure that functions return outputs that can be easily displayed.
# this is required for the streamlit version
def perform_search(query, release, section, n_results, save_docs, rerank, where_filter):
    # The logic stays mostly the same, just ensure the return value is formatted for Streamlit.
    collection = load_collection()
    queryembed = get_query_embedding(query)

    # Perform the vector search
    results = perform_vector_search(queryembed, collection, release=release)

    relevantdocs = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not relevantdocs:
        return "⚠️ No relevant documents found. Abandoning search."

    # If reranking is enabled
    if rerank:
        relevantdocs, metadatas = rerank_results(query, relevantdocs, metadatas)

    docs = "\n\n".join(
        f"[Doc {i+1}] (release: {meta.get('release', '?')}, section: {meta.get('section', '?')})\n{doc}"
        for i, (doc, meta) in enumerate(zip(relevantdocs, metadatas))
    )

    # Return result
    return docs



def save_documents(relevantdocs, metadatas, query):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{OUTPUT_DIRECTORY}/retrieved_docs_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write(f"Query: {query}\n\n")
        for i, (doc, meta) in enumerate(zip(relevantdocs, metadatas), 1):
            f.write(f"[Document {i}]\n{doc}\n")
            f.write(f"Metadata: {meta}\n\n")
            f.write(f"---------------------------\n\n")
    print(f"\n📄 Retrieved documents saved to {filename}\n")

def log_vector_scores(query, documents, metadatas, distances):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = f"{OUTPUT_DIRECTORY}/vector_scores_{timestamp}.log"
    
    with open(logfile, "w") as f:
        f.write(f"Query: {query}\n\n")
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            sim_score = 1 - dist  # cosine similarity
            f.write(f"[Doc {i+1}]\nScore: {sim_score:.4f}\n")
            f.write(f"Metadata: {meta}\n")
            f.write(f"Content (first 300 chars):\n{doc[:300]}...\n")
            f.write("--------------------------------------------------\n\n")
    
    print(f"\n📝 Vector similarity scores logged to: {logfile}")

def log_all_vector_scores(query_embed, collection, original_query, full_log_limit=200):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{OUTPUT_DIRECTORY}/all_vector_scores_{timestamp}.log"

    query_kwargs = {
        "query_embeddings": [query_embed],
        "n_results": full_log_limit,
    }

    full_results = collection.query(**query_kwargs)

    with open(log_file, "w") as f:
        f.write(f"Query: {original_query}\n\n")
        
        for i in range(len(full_results["documents"][0])):
            doc = full_results["documents"][0][i]
            meta = full_results["metadatas"][0][i]
            
            doc_embed = ollama.embeddings(model=getconfig()["embedmodel"], prompt=f"search_document: {doc}")['embedding']
            
            score = np.dot(query_embed, doc_embed) / (
                np.linalg.norm(query_embed) * np.linalg.norm(doc_embed)
            )
            
            f.write(f"[Document {i+1}]\n")
            f.write(f"Score: {score:.4f}\n")
            f.write(f"Metadata: {meta}\n")
            f.write(f"Text: {doc[:500]}...\n")
            f.write("-" * 40 + "\n\n")
    
    print(f"\n📊 All vector scores logged to {log_file}\n")
