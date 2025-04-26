#!/usr/bin/env python3

import sys
from datetime import datetime
import rag_search as rs
from utilities import getconfig
import rag_utilities as ru
import keyword_search as ks
import ollama
from pprint import pprint

DEBUGGING=False

def get_query(args):
    if not sys.stdin.isatty():  # Input is being piped or redirected
        return sys.stdin.read().strip()
    elif args:  # Arguments provided
        return " ".join(args)
    else:
        print('Usage: search.py "your question here"')
        print('       OR cat question.txt | search.py')
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
    # Initialize the config variables
    save_docs = False
    rerank = False
    n_results = 5  # Default number of results
    release_filter = None
    section_filter = None
    doc_types = []
    keyword_search = None


    # Parse arguments
    args = sys.argv[1:]

    # Flags for --save-docs and --rerank
    save_docs = "--save-docs" in args
    rerank = "--rerank" in args

    # If --help flag is present, show usage
    if "--help" in args:
        usage()

    # Clean up known flags
    for flag in ["--save-docs", "--rerank", "--help"]:
        if flag in args:
            args.remove(flag)

    # Parse --doctype flag
    if "--doctype" in args:
        idx = args.index("--doctype")
        try:
            doctype_arg = args[idx + 1]
            doc_types = [dt.strip() for dt in doctype_arg.split(",")]
            del args[idx:idx + 2]
        except (IndexError, ValueError):
            print("❌ Error: --doctype requires a comma-separated list of types")
            usage()

    # Parse --release flag
    if "--release" in args:
        idx = args.index("--release")
        try:
            release_filter = args[idx + 1]
            del args[idx:idx + 2]
        except (IndexError, ValueError):
            print("❌ Error: --release requires a value (e.g., 1.4.0)")
            usage()

    # Parse --section flag
    if "--section" in args:
        idx = args.index("--section")
        try:
            section_filter = args[idx + 1]
            del args[idx:idx + 2]
        except (IndexError, ValueError):
            print("❌ Error: --section requires a value (e.g., \"new functionality\")")
            usage()

    # Parse --n-results flag
    if "--n-results" in args:
        idx = args.index("--n-results")
        try:
            n_results = int(args[idx + 1])
            del args[idx:idx + 2]
        except (IndexError, ValueError):
            print("❌ Error: --n-results requires a numeric value")
            usage()

    # Parse --keyword-search
    if "--keyword-search" in args:
        idx = args.index("--keyword-search")
        try:
            keyword_search = args[idx + 1]  # leave as string!
            del args[idx:idx + 2]
        except (IndexError, ValueError):
            print("❌ Error: --keyword-search requires a string value")
            usage()


    # Get the actual user query (the remaining arguments)
    query = " ".join(args)  # Combine remaining args as the query
    prefixed_query = "search_query: " + query

    # Return the config dictionary
    return {
        "query": query,
        "release": release_filter,
        "section": section_filter,
        "n_results": n_results,
        "save_docs": save_docs,
        "rerank": rerank,
        "doc_types": doc_types,  
        "prefixed_query": prefixed_query,
        "keyword_search": keyword_search
    }

def main():
    config = parse_command_line()
    collection = rs.load_collection()
    print("Checking Embeddings.")
    queryembed = rs.get_query_embedding(config["prefixed_query"])
    
    # Perform vector search
    release = config["release"]  # default from command line
    if release is None:
        release = ru.extract_release_version(config["prefixed_query"])
    top_doc_ids = None
    if config["keyword_search"]:
        keyword_hits = ks.keyword_search_with_stemming(config["keyword_search"])
        top_doc_ids = [hit["full_doc_id"] for hit in keyword_hits if hit.get("full_doc_id")]
        print("\n🧪 Keyword Search Hits:")
        for hit in keyword_hits:
            print(f"Doc ID: {hit['doc_id']} Metadata: {hit.get('metadata', '')}")

    
    results = rs.perform_vector_search(
    queryembed,
    collection,
    release=release,
    section=config["section"], 
    n_results=config["n_results"],
    doc_types=config["doc_types"],
    doc_ids=top_doc_ids
)
    print(f"Number of documents returned: {len(results['documents'][0])}")

    relevantdocs = results["documents"][0]
    # Check if relevantdocs is empty
    if not relevantdocs:
        print("⚠️ No relevant documents found. Abandoning search.")
        exit(0)  # Exit the function early or handle accordingly

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Save similarity scores to file
    if DEBUGGING:
        ru.log_vector_scores(config["query"], relevantdocs, metadatas, distances)

    if config["save_docs"]:
        rs.save_documents(relevantdocs, metadatas, config["query"])

    # If reranking is enabled, rerank the documents
    if config["rerank"]:
        print("\n🔄 Reranking documents with bge-reranker-large...")
        relevantdocs, metadatas = rs.rerank_results(config["query"], relevantdocs, metadatas)

    # Combine documents into a prompt
    docs = "\n\n".join(
        f"[Doc {i+1}] (release: {meta.get('release', '?')}, section: {meta.get('section', '?')})\n{doc}"
        for i, (doc, meta) in enumerate(zip(relevantdocs, metadatas))
    )

    modelquery = rs.build_prompt(getconfig()["mainmodel"], docs, config["query"])
    print(f"Prompt: {modelquery}\n")
    print("Sending Prompt to Model.")
    
    # Stream the answer from the model
    stream = ollama.generate(model=getconfig()["mainmodel"], prompt=modelquery, stream=True)

    for chunk in stream:
        if chunk["response"]:
            print(chunk["response"], end="", flush=True)


if __name__ == "__main__":
    main()
