import streamlit as st
import rag_search as rs
import rag_utilities as ru
import keyword_search as ks
from utilities import getconfig
import ollama

# Predefined dropdown options
release_versions = ["", "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"]
sections = ["", "new functionality", "fixes", "improvements", "deprecations"]
all_doc_types = [
    "Release Notes",
    "Software Design Documents",
    "Architectural Documents",
    "Feature Description Documents",
    "Technical Documents",
    "Performance Analysis Documents",
    "Detailed Product Specifications"
]

# Function to process the query
def process_query(query, release, section, n_results, save_docs, rerank, doc_types, keyword_search_string):
    collection = rs.load_collection()

    # Prefix query
    prefixed_query = "search_query: " + query

    # Embedding
    query_embed = rs.get_query_embedding(prefixed_query)

    top_doc_ids = None

    # If user entered manual keywords, use them
    if keyword_search_string:
        keyword_string_to_use = keyword_search_string
    else:
        # Otherwise extract keywords automatically from query
        extracted_keywords = ru.extract_keywords(query)
        keyword_string_to_use = " ".join(extracted_keywords)

    if keyword_string_to_use:
        keyword_hits = ks.keyword_search_with_stemming(keyword_string_to_use)
        top_doc_ids = [hit["full_doc_id"] for hit in keyword_hits if hit.get("full_doc_id")]
        st.info(f"🔍 Found {len(top_doc_ids)} documents matching keywords: {keyword_string_to_use}")

        if not top_doc_ids:
            return "⚠️ No documents matched the keyword search. Abandoning search."

    # Perform vector search
    results = rs.perform_vector_search(
        query_embed,
        collection,
        release=release,
        section=section,
        n_results=n_results,
        doc_types=doc_types,
        doc_ids=top_doc_ids  # pass filtered IDs if keyword search used
    )

    relevant_docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not relevant_docs:
        return "⚠️ No relevant documents found after vector search."

    if rerank:
        relevant_docs, metadatas = rs.rerank_results(query, relevant_docs, metadatas)

    # Combine documents into a string for model input
    docs = "\n\n".join(
        f"[Doc {i+1}] (release: {meta.get('release', '?')}, section: {meta.get('section', '?')})\n{doc}"
        for i, (doc, meta) in enumerate(zip(relevant_docs, metadatas))
    )

    # Save documents if needed
    if save_docs:
        rs.save_documents(relevant_docs, metadatas, query)

    # Build prompt
    model_query = rs.build_prompt(getconfig()["mainmodel"], docs, query)
    st.text_area("Prompt Sent to Model", model_query, height=300)

    # Stream the model response
    stream = ollama.generate(model=getconfig()["mainmodel"], prompt=model_query, stream=True)

    answer = ""
    for chunk in stream:
        if chunk["response"]:
            answer += chunk["response"]

    return answer

# Streamlit UI
st.title("MkDocs Search Application")

query = st.text_input("🔍 Enter your question:")
keyword_search_string = st.text_input("🧹 Optional keyword search string (advanced):")

selected_doc_types = st.multiselect(
    "Select document types to include:",
    options=all_doc_types,
    default=all_doc_types
)

release = st.selectbox("📦 Filter by release version (optional):", release_versions)
section = st.selectbox("📑 Filter by section (optional):", sections)

n_results = st.slider("📄 Number of results to retrieve:", 1, 20, 5)
rerank = st.checkbox("🔁 Rerank with bge-reranker-large", value=True)
save_docs = st.checkbox("💾 Save retrieved documents")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = process_query(
                    query,
                    release if release else None,
                    section if section else None,
                    n_results,
                    save_docs,
                    rerank,
                    selected_doc_types,
                    keyword_search_string.strip() if keyword_search_string else None
                )
                st.success("✅ Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
