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
def process_query(query, release, section, n_results, save_docs, rerank, doc_types, keyword_search_string=None):
    collection = rs.load_collection()
    query_embed = rs.get_query_embedding(query)

    top_doc_ids = None
    if keyword_search_string:
        st.info(f"🔎 Using keywords for keyword search: {keyword_search_string}")
        keyword_hits = ks.keyword_search_with_stemming(keyword_search_string)
        top_doc_ids = [hit["full_doc_id"] for hit in keyword_hits if hit.get("full_doc_id")]

        if keyword_hits:
            with st.expander("🧪 Keyword Search Hits"):
                for hit in keyword_hits:
                    st.write(f"Doc ID: {hit['doc_id']} Metadata: {hit.get('metadata', '')}")
        else:
            st.warning("⚠️ No keyword search hits.")

    results = rs.perform_vector_search(
        query_embed,
        collection,
        release=release,
        section=section,
        doc_types=doc_types,
        n_results=n_results,
        doc_ids=top_doc_ids
    )

    relevant_docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not relevant_docs:
        return "⚠️ No relevant documents found. Abandoning search."

    if rerank:
        relevant_docs, metadatas = rs.rerank_results(query, relevant_docs, metadatas)

    docs = "\n\n".join(
        f"[Doc {i+1}] (release: {meta.get('release', '?')}, section: {meta.get('section', '?')}) "
        f"[View Document]({meta.get('url', '#')})\n{doc}"
        for i, (doc, meta) in enumerate(zip(relevant_docs, metadatas))
    )


    if save_docs:
        rs.save_documents(relevant_docs, metadatas, query)

    model_query = rs.build_prompt(getconfig()["mainmodel"], docs, query)
    st.text_area("📄 Model Query Sent to Model", model_query, height=300)

    stream = ollama.generate(model=getconfig()["mainmodel"], prompt=model_query, stream=True)

    answer = ""
    for chunk in stream:
        if chunk["response"]:
            answer += chunk["response"]

    return answer


# Streamlit UI
st.title("MkDocs Search Application")

query = st.text_input("🔍 Enter your question:")

auto_generate_keywords = st.checkbox("✨ Automatically generate keywords from query")
manual_keywords = st.text_input("🔑 Or provide specific keywords (comma-separated):")

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
                keyword_string = None
                if manual_keywords.strip():
                    keyword_string = manual_keywords
                    st.info(f"🔑 Using manually provided keywords: {keyword_string}")
                elif auto_generate_keywords:
                    extracted = ru.extract_keywords(query)
                    keyword_string = " ".join(extracted)
                    st.info(f"✨ Auto-generated keywords from query: {', '.join(extracted)}")

                answer = process_query(
                    query,
                    release if release else None,
                    section if section else None,
                    n_results,
                    save_docs,
                    rerank,
                    selected_doc_types,
                    keyword_search_string=keyword_string
                )
                st.success("✅ Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")


