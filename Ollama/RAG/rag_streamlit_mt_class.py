import streamlit as st
# import rag_search as rs
# import rag_utilities as ru
from rag_search_class import RAGSearch
from rag_utilities_class import TextProcessingUtilities
from keyword_search_class import KeywordSearchEngine
from document_types import ALL_DOC_TYPES
# import keyword_search as ks
#from utilities import getconfig
import ollama

class MkDocsRAGChatAssistant:
    def __init__(self):
        # --- Session State Initialization ---
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # --- Dropdown and Filter Options ---
        self.release_versions = ["", "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"]
        self.sections = ["", "new functionality", "fixes", "improvements", "deprecations"]
        self.all_doc_types = ALL_DOC_TYPES

        # --- Streamlit UI ---
        st.title("🧠 MkDocs RAG Chat Assistant")

        self.query = st.text_input("💬 Enter your question:")

        clear = st.button("Clear Conversation")
        if clear:
            st.session_state.chat_history = []
            st.session_state["query"] = ""
            st.stop()

        self.auto_generate_keywords = st.checkbox("✨ Automatically generate keywords from query")
        self.manual_keywords = st.text_input("🔑 Or provide specific keywords (comma-separated):")

        self.selected_doc_types = st.multiselect(
            "📂 Select document types to include:",
            options=self.all_doc_types,
            default=self.all_doc_types
        )

        self.release = st.selectbox("📦 Filter by release version (optional):", self.release_versions)
        self.section = st.selectbox("📑 Filter by section (optional):", self.sections)
        self.n_results = st.slider("📄 Number of results to retrieve:", 1, 20, 5)
        self.rerank = st.checkbox("🔁 Rerank with bge-reranker-large", value=True)
        self.save_docs = st.checkbox("💾 Save retrieved documents")

    # --- Main Query Processing Function ---
    def process_query(self, query, ru, release, section, n_results, save_docs, rerank, doc_types, keyword_search_string=None):
        rs = RAGSearch()
        collection = rs.load_collection()
        query_embed = rs.get_query_embedding(query,ru)

        top_doc_ids = None
        if keyword_search_string:
            ks = KeywordSearchEngine()
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

        docs_list = []
        for i, (doc, meta) in enumerate(zip(relevant_docs, metadatas)):
            url = meta.get('url', '#').rstrip('} ')
            release = meta.get('release', '?')
            section = meta.get('section', '?')
            doc_entry = (
                f"[Doc {i+1}] (release: {release}, section: {section}) "
                f"[View Document]({url})\n{doc}"
            )
            docs_list.append(doc_entry)

        docs = "\n\n".join(docs_list)
        model_id = ru.get_config()["mainmodel"]
        if save_docs:
            rs.save_documents(relevant_docs, metadatas, query)

        model_query = rs.build_prompt(model_id, docs, query, st.session_state.chat_history)
        st.text_area("📄 Model Query Sent to Model", model_query, height=300)
        st.markdown("### 🔗 Retrieved Documents:")
        st.markdown(docs, unsafe_allow_html=True)

        stream = ollama.generate(model=model_id, prompt=model_query, stream=True)

        answer = ""
        for chunk in stream:
            if chunk["response"]:
                answer += chunk["response"]

        return answer

    def display_chat_history(self):
        st.markdown("### 🗨️ Conversation History")
        for turn in st.session_state.chat_history:
            if turn["role"] == "user":
                st.markdown(f"**👤 User**: {turn['content']}")
            else:
                st.markdown(f"**🤖 Assistant**: {turn['content']}")

    def run(self,ru):
        if st.button("Ask"):
            if not self.query.strip():
                st.warning("Please enter a question before submitting.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        keyword_string = None
                        if self.manual_keywords.strip():
                            keyword_string = self.manual_keywords
                            st.info(f"🔑 Using manually provided keywords: {keyword_string}")
                        elif self.auto_generate_keywords:
                            ru =  TextProcessingUtilities()
                            extracted = ru.extract_keywords(self.query)
                            keyword_string = " ".join(extracted)
                            st.info(f"✨ Auto-generated keywords from query: {', '.join(extracted)}")

                        # Add user query to chat history
                        st.session_state.chat_history.append({"role": "user", "content": self.query})

                        answer = self.process_query(
                            self.query,
                            ru,
                            self.release if self.release else None,
                            self.section if self.section else None,
                            self.n_results,
                            self.save_docs,
                            self.rerank,
                            self.selected_doc_types,
                            keyword_search_string=keyword_string
                        )

                        # Add model response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})

                        st.success("✅ Answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        self.display_chat_history()

if __name__ == "__main__":
    ru =  TextProcessingUtilities()
    app = MkDocsRAGChatAssistant()
    app.run(ru)
