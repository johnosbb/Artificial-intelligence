import argparse
from rag_search import (
    get_query_embedding,
    load_collection,
    perform_vector_search,
    rerank_results,
    log_all_vector_scores,
    save_documents,
    getconfig
)

def main():
    parser = argparse.ArgumentParser(description="Debug RAG pipeline with detailed logging.")
    
    parser.add_argument("--query", type=str, required=True, help="Natural language query")
    parser.add_argument("--section", type=str, help="Section filter")
    parser.add_argument("--release", type=str, help="Release filter")
    parser.add_argument("--doctype", type=str, nargs="+", help="List of document types")
    parser.add_argument("--rerank", action="store_true", help="Enable reranking")
    parser.add_argument("--limit", type=int, default=200, help="Number of documents to compare")
    parser.add_argument("--use-embed-file", action="store_true", help="Use saved embedding")
    parser.add_argument("--embed-file", type=str, default="/mnt/500GB/rag_output/query_embedding.json", help="Path to saved embedding")

    args = parser.parse_args()

    # Load collection
    collection = load_collection()

    # Load or generate query embedding
    query_embed = get_query_embedding(
        query=args.query,
        save_to_file=True,
        load_from_file=args.use_embed_file,
        filename=args.embed_file
    )

    # Perform vector search with filters
    results = perform_vector_search(
        query_embed,
        collection,
        release=args.release,
        section=args.section,
        doc_types=args.doctype,
        n_results=args.limit,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        print("⚠️ No documents found in vector search.")
        return

    # Optional rerank
    if args.rerank:
        print("\n🔁 Reranking enabled.")
        documents, metadatas = rerank_results(args.query, documents, metadatas)

    # Save retrieved docs for inspection
    save_documents(documents, metadatas, args.query)

    # Log detailed vector scores
    log_all_vector_scores(
        query_embed=query_embed,
        collection=collection,
        original_query=args.query,
        full_log_limit=args.limit
    )

if __name__ == "__main__":
    main()
