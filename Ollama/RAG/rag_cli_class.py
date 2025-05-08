#!/usr/bin/env python3

import sys
from datetime import datetime
from rag_search_class import RAGSearch
from rag_utilities_class import TextProcessingUtilities
import keyword_search_class as ks
import ollama
from pprint import pprint

DEBUGGING=False

class RAGCLI:
    def __init__(self,rs):
        self.save_docs = False
        self.rerank = False
        self.n_results = 5  # Default number of results
        self.release_filter = None
        self.section_filter = None
        self.doc_types = []
        self.keyword_search = None
        self.auto_generate_keywords = False
        self.query = None
        self.prefixed_query = None
        self.collection = rs.load_collection()

    def get_query(self, args):
        if not sys.stdin.isatty():  # Input is being piped or redirected
            return sys.stdin.read().strip()
        elif args:  # Arguments provided
            return " ".join(args)
        else:
            print('Usage: search.py "your question here"')
            print('        OR cat question.txt | search.py')
            sys.exit(1)

    def usage(self):
        print(f"""
Usage: search.py [options] "your question here"
        OR cat question.txt | search.py [options]

Options:
  --save-docs                 Save retrieved documents to a file
  --rerank                    Use bge-reranker-large to rerank results
  --release-notes-only        Filter to only documents with doctype 'release note'
  --release <version>         Filter to a specific release version (e.g., 1.4.0)
  --section <name>            Filter to a specific section (e.g., "new functionality")
  --n-results <number>        Number of documents to retrieve (default: 5)
  --keyword-search <string>   Manually specify keywords for search
  --auto-keywords             Automatically extract keywords from query
  --help                      Show this help message and exit

Examples:
  search.py "How does the logging system work?"
  search.py --release-notes-only --release 1.4.0 --section "new functionality" "Summarize the changes"
  cat question.txt | search.py --rerank --save-docs
         """)
        sys.exit(0)

    def parse_command_line(self,rs):
        args = sys.argv[1:]

        self.save_docs = "--save-docs" in args
        self.rerank = "--rerank" in args

        if "--help" in args:
            self.usage()

        for flag in ["--save-docs", "--rerank", "--help"]:
            if flag in args:
                args.remove(flag)

        if "--doctype" in args:
            idx = args.index("--doctype")
            try:
                doctype_arg = args[idx + 1]
                self.doc_types = [dt.strip() for dt in doctype_arg.split(",")]
                del args[idx:idx + 2]
            except (IndexError, ValueError):
                print("❌ Error: --doctype requires a comma-separated list of types")
                self.usage()

        if "--release" in args:
            idx = args.index("--release")
            try:
                self.release_filter = args[idx + 1]
                del args[idx:idx + 2]
            except (IndexError, ValueError):
                print("❌ Error: --release requires a value (e.g., 1.4.0)")
                self.usage()

        if "--section" in args:
            idx = args.index("--section")
            try:
                self.section_filter = args[idx + 1]
                del args[idx:idx + 2]
            except (IndexError, ValueError):
                print("❌ Error: --section requires a value (e.g., \"new functionality\")")
                self.usage()

        if "--n-results" in args:
            idx = args.index("--n-results")
            try:
                self.n_results = int(args[idx + 1])
                del args[idx:idx + 2]
            except (IndexError, ValueError):
                print("❌ Error: --n-results requires a numeric value")
                self.usage()

        if "--keyword-search" in args:
            idx = args.index("--keyword-search")
            try:
                self.keyword_search = args[idx + 1]
                del args[idx:idx + 2]
            except (IndexError, ValueError):
                print("❌ Error: --keyword-search requires a string value")
                self.usage()

        self.auto_generate_keywords = "--auto-keywords" in args
        if self.auto_generate_keywords:
            args.remove("--auto-keywords")

        self.query = " ".join(args)
        self.prefixed_query = "search_query: " + self.query

        return self

    def process_query(self,rs,ru):
        print("Checking Embeddings.")
        queryembed = rs.get_query_embedding(self.prefixed_query,ru)

        release = self.release_filter
        if release is None:
            release = ru.extract_release_version(self.prefixed_query)

        top_doc_ids = None
        keyword_string_to_use = None

        if self.keyword_search:
            keyword_string_to_use = self.keyword_search
            print(f"\n🔑 Using manually specified keywords: {keyword_string_to_use}")
        elif self.auto_generate_keywords:
            extracted_keywords = ru.extract_keywords(self.query)
            keyword_string_to_use = " ".join(extracted_keywords)
            print(f"\n✨ Auto-generated keywords from query: {', '.join(extracted_keywords)}")
        else:
            print("\n🚫 No keyword search will be performed (vector search only).")

        if keyword_string_to_use:
            keyword_hits = ks.keyword_search_with_stemming(keyword_string_to_use)
            top_doc_ids = [hit["full_doc_id"] for hit in keyword_hits if hit.get("full_doc_id")]

            if keyword_hits:
                print("\n🧪 Keyword Search Hits:")
                for hit in keyword_hits:
                    print(f"Doc ID: {hit['doc_id']} Metadata: {hit.get('metadata', '')}")
            else:
                print("\n⚠️ No keyword search hits.")

        results = rs.perform_vector_search(
            queryembed,
            self.collection,
            release=release,
            section=self.section_filter,
            n_results=self.n_results,
            doc_types=self.doc_types,
            doc_ids=top_doc_ids
        )
        print(f"Number of documents returned: {len(results['documents'][0])}")

        relevantdocs = results["documents"][0]
        if not relevantdocs:
            print("⚠️ No relevant documents found. Abandoning search.")
            exit(0)

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if DEBUGGING:
            ru.log_vector_scores(self.query, relevantdocs, metadatas, distances)

        if self.save_docs:
            rs.save_documents(relevantdocs, metadatas, self.query)

        if self.rerank:
            print("\n🔄 Reranking documents with bge-reranker-large...")
            relevantdocs, metadatas = rs.rerank_results(self.query, relevantdocs, metadatas)

        docs = "\n\n".join(
            f"[Doc {i+1}] (release: {meta.get('release', '?')}, section: {meta.get('section', '?')})\n{doc}"
            for i, (doc, meta) in enumerate(zip(relevantdocs, metadatas))
        )
        model_id = ru.get_config()["mainmodel"]
        modelquery = rs.build_prompt(model_id, docs, self.query, None)
        print(f"\nPrompt: {modelquery}\n")

        print(f"Sending Prompt to Model: {model_id}.")

        stream = ollama.generate(model=model_id, prompt=modelquery, stream=True)

        for chunk in stream:
            if chunk.get("done"):  # if your API uses a 'done' field
                break
            if chunk.get("response"):
                print(chunk["response"], end="", flush=True)

def main():
    rs = RAGSearch()
    ru =  TextProcessingUtilities()
    rag_cli = RAGCLI(rs)
    rag_cli.parse_command_line(rs)
    rag_cli.process_query(rs,ru)

if __name__ == "__main__":
    main()