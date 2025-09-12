#!/usr/bin/env python3
import chromadb
import sys

def main():
    section = None

    # Optionally pass section as CLI arg like: --section "new functionality"
    if "--section" in sys.argv:
        try:
            section_index = sys.argv.index("--section") + 1
            section = sys.argv[section_index]
        except IndexError:
            print("❌ You must provide a section name after '--section'")
            return

    try:
        chroma = chromadb.HttpClient(
            host="localhost",
            port=8000,
            headers={"X-Tenant-Id": "default_tenant"}
        )
        collection = chroma.get_or_create_collection("buildragwithpython")

        if section:
            docs = collection.get(where={"section": {"$eq": section}}, limit=100)
            print(f"🔎 Found {len(docs['documents'])} documents with section = '{section}'")
        else:
            print("ℹ️ No --section provided. Use: --section \"your section name\"")

        sections = set()
        for meta in collection.get(limit=1000)["metadatas"]:
            if "section" in meta:
                sections.add(meta["section"])

        print("🔍 Available sections:", sorted(sections))
    

    except Exception as e:
        print("\n❌ An error occurred while inspecting the collection:")
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
