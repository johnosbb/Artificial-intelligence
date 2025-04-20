#!/usr/bin/env python3

import sys
from datetime import datetime
from rag_search import load_collection, get_query_embedding, perform_vector_search, rerank_results, build_prompt, save_documents
from utilities import getconfig
import rag_utilities as ru
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


def main():
    parse_command_line()
    collection = load_collection()
    print("Checking Embeddings.")
    queryembed = get_query_embedding(prefixed_query)
    
    # Perform vector search
    release = ru.extract_release_version(prefixed_query)
    results = perform_vector_search(queryembed, collection, release=release,n_results=n_results)

    relevantdocs = results["documents"][0]
    # Check if relevantdocs is empty
    if not relevantdocs:
        print("⚠️ No relevant documents found. Abandoning search.")
        exit(0)  # Exit the function early or handle accordingly

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Save similarity scores to file
    if DEBUGGING:
        ru.log_vector_scores(query, relevantdocs, metadatas, distances)

    if save_docs:
        save_documents(relevantdocs, metadatas, query)

    # If reranking is enabled, rerank the documents
    if rerank:
        print("\n🔄 Reranking documents with bge-reranker-large...")
        relevantdocs, metadatas = rerank_results(query, relevantdocs, metadatas)

    # Combine documents into a prompt
    docs = "\n\n".join(
        f"[Doc {i+1}] (release: {meta.get('release', '?')}, section: {meta.get('section', '?')})\n{doc}"
        for i, (doc, meta) in enumerate(zip(relevantdocs, metadatas))
    )

    modelquery = build_prompt(getconfig()["mainmodel"], docs, query)
    print(f"Prompt: {modelquery}\n")
    print("Sending Prompt to Model.")
    
    # Stream the answer from the model
    stream = ollama.generate(model=getconfig()["mainmodel"], prompt=modelquery, stream=True)

    for chunk in stream:
        if chunk["response"]:
            print(chunk["response"], end="", flush=True)


if __name__ == "__main__":
    main()
