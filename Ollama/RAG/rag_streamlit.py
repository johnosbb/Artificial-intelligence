import streamlit as st
import rag_search as rs
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




# Function to process the query using rag_search.py
def process_query(query, release, section, n_results, save_docs, rerank, doc_types):
    # Loading the collection
    collection = rs.load_collection()

    # Get query embedding
    query_embed = rs.get_query_embedding(query)

    # Perform vector search
    results = rs.perform_vector_search(query_embed, collection, release=release,doc_types=doc_types,n_results=n_results)

    relevant_docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not relevant_docs:
        return "⚠️ No relevant documents found. Abandoning search."

    if rerank:
        relevant_docs, metadatas = rs.rerank_results(query, relevant_docs, metadatas)

    # Combine documents into a string for display
    docs = "\n\n".join(
        f"[Doc {i+1}] (release: {meta.get('release', '?')}, section: {meta.get('section', '?')})\n{doc}"
        for i, (doc, meta) in enumerate(zip(relevant_docs, metadatas))
    )

    # Optionally save documents if selected
    if save_docs:
        rs.save_documents(relevant_docs, metadatas, query)

    # Build prompt for the model
    model_query = rs.build_prompt(getconfig()["mainmodel"], docs, query)
    st.text_area("Model Query", model_query)

    # Stream the answer from the model
    stream = ollama.generate(model=getconfig()["mainmodel"], prompt=model_query, stream=True)

    answer = ""
    for chunk in stream:
        if chunk["response"]:
            answer += chunk["response"]

    return answer


# Streamlit UI
st.title("MkDocs Search Application")

# Set default based on the list defined above
default_doc_types = ["Release Note"]  # You can also use all_doc_types if you want all pre-selected
query = st.text_input("🔍 Enter your question:")
selected_doc_types = st.multiselect(
    "Select document types to include:",
    options=all_doc_types,
    default=default_doc_types
)


release = st.selectbox("📦 Filter by release version (optional):", release_versions)
section = st.selectbox("📑 Filter by section (optional):", sections)

n_results = st.slider("📄 Number of results to retrieve:", 1, 20, 5)
rerank = st.checkbox("🔁 Rerank with bge-reranker-large", value=True)  # Set default value to True
save_docs = st.checkbox("💾 Save retrieved documents")


if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = process_query(
                    query, release if release else None,
                    section if section else None,
                    n_results, save_docs, rerank, selected_doc_types
                )
                st.success("✅ Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")