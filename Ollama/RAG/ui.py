import streamlit as st
import ollama
import chromadb

from search import rerank_results, build_prompt, perform_vector_search, load_collection
from rag_utilities import extract_release_version

st.set_page_config(page_title="Semantic Search with RAG", layout="wide")

st.title("🔎 RAG Search over Technical Docs")

# Sidebar filters
query = st.text_input("Enter your question:", value="Summarize changes in version 1.4.0")
release_filter = st.text_input("Filter by Release (optional):")
section_filter = st.text_input("Filter by Section (optional):")
release_notes_only = st.checkbox("Release Notes Only")
rerank = st.checkbox("Rerank with bge-reranker-large")
n_results = st.slider("Number of results", 3, 20, 7)

if st.button("Search") and query:
    st.info("Running vector search...")

    # Set up collection and filters
    collection = load_collection()
    embedmodel = "nomic-embed-text"
    mainmodel = "llama3"

    prefixed_query = "search_query: " + query
    query_embed = ollama.embeddings(model=embedmodel, prompt=prefixed_query)['embedding']

    where_filter = {}
    if release_notes_only:
        where_filter["doctype"] = "release note"
    if release_filter:
        where_filter["release"] = {"$eq": release_filter}
    if section_filter:
        where_filter["section"] = {"$eq": section_filter}

    results = perform_vector_search(query_embed, collection, release=release_filter)
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    if rerank:
        docs, metadatas = rerank_results(query, docs, metadatas)

    # Display retrieved docs
    for i, (doc, meta) in enumerate(zip(docs, metadatas), 1):
        st.markdown(f"**Doc {i}** - Release: {meta.get('release')} | Section: {meta.get('section')}")
        with st.expander("Show content"):
            st.code(doc[:1000])

    # Build final prompt and get answer
    combined_docs = "\n\n".join(
        f"[Doc {i}] (release: {meta.get('release', '?')}, section: {meta.get('section', '?')})\n{doc}"
        for i, (doc, meta) in enumerate(zip(docs, metadatas))
    )
    prompt = build_prompt(mainmodel, combined_docs, query)

    st.info("Querying LLM...")
    full_response = ""
    stream = ollama.generate(model=mainmodel, prompt=prompt, stream=True)
    for chunk in stream:
        full_response += chunk["response"]
    st.markdown("### 💬 Final Answer")
    st.write(full_response)
