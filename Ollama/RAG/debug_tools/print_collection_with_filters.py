#!/usr/bin/env python3
import chromadb
import sys

def main():
    section = None
    doctype = None

    # Optionally pass section and doctype as CLI args like: --section "new functionality" --doctype "Release Notes"
    if "--section" in sys.argv:
        try:
            section_index = sys.argv.index("--section") + 1
            section = sys.argv[section_index]
        except IndexError:
            print("❌ You must provide a section name after '--section'")
            return

    if "--doctype" in sys.argv:
        try:
            doctype_index = sys.argv.index("--doctype") + 1
            doctype = sys.argv[doctype_index]
        except IndexError:
            print("❌ You must provide a document type after '--doctype'")
            return

    try:
        chroma = chromadb.HttpClient(
            host="localhost",
            port=8000,
            headers={"X-Tenant-Id": "default_tenant"}
        )
        collection = chroma.get_or_create_collection("buildragwithpython")

        query_filter = {}
        if section and doctype:
            query_filter = {"$and": [{"section": {"$eq": section}}, {"doctype": {"$eq": doctype}}]}
        elif section:
            query_filter = {"section": {"$eq": section}}
        elif doctype:
            query_filter = {"doctype": {"$eq": doctype}}

        # Get documents based on the filter
        docs = collection.get(where=query_filter, limit=100)
        print(f"🔎 Found {len(docs['documents'])} documents with section = '{section}' and doctype = '{doctype}'")
        print(f"📄 Document names in this section:")
        
        for meta in docs["metadatas"]:
            # Print all metadata to help with debugging
            print(f"Metadata: {meta}")

        # Retrieve and print available sections and document types
        sections = set()
        doctype_types = set()

        for meta in collection.get(limit=1000)["metadatas"]:
            if "section" in meta:
                sections.add(meta["section"])
            if "doctype" in meta:
                doctype_types.add(meta["doctype"])

        print("🔍 Available sections:", sorted(sections))
        print("🔍 Available document types:", sorted(doctype_types))

    except Exception as e:
        print("\n❌ An error occurred while inspecting the collection:")
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
