#!/usr/bin/env python3
import chromadb
import sys

def main():
    show_summaries_only = "--summaries-only" in sys.argv
    show_full = "--full" in sys.argv

    try:
        chroma = chromadb.HttpClient(
            host="localhost",
            port=8000,
            headers={"X-Tenant-Id": "default_tenant"}
        )
        collection = chroma.get_or_create_collection("buildragwithpython")

        # Retrieve ALL docs (or summaries only)
        where_filter = {"is_summary": True} if show_summaries_only else {}
        # Retrieve documents, handling optional filter
        if show_summaries_only:
            results = collection.get(where={"is_summary": {"$eq": True}}, limit=50)
        else:
            results = collection.get(limit=8000)


        if not results["documents"]:
            print("ℹ️ No documents found in the collection.")
        else:
            for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
                print(f"\n=== Document {i+1} ===")
                print(doc if show_full else doc[:500])
                print("Metadata:", meta)

    except Exception as e:
        print("\n❌ An error occurred while inspecting the collection:")
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()