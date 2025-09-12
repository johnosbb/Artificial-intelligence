#!/usr/bin/env python3
import numpy as np
from ollama import embeddings
from utilities import getconfig

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Example query and chunk (you can customize this)
query_text = "In what release were new passwords introduced?"
chunk_text = "search_document: release note: Release 1.x.0 * Moved to New passwords"

embedmodel = getconfig()["embedmodel"]

# Get embeddings
query_vec = embeddings(model=embedmodel, prompt="search_query: " + query_text)['embedding']
chunk_vec = embeddings(model=embedmodel, prompt=chunk_text)['embedding']

# Compute similarity
similarity_score = cosine_similarity(query_vec, chunk_vec)

print(f"\n🧠 Cosine similarity between query and document chunk: {similarity_score:.4f}")
