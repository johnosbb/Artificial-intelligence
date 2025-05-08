#!/usr/bin/env python3


import keyword_search as ks






if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: keyword_search_cli.py 'your search terms here'")
    else:
        query = " ".join(sys.argv[1:])
        results= ks.keyword_search(query)
        results=ks.keyword_search_with_stemming(query)
        print(f"\n🔍 Top results for: '{query}'\n")
        for i, hit in enumerate(results):
            print(f"[{i+1}] Doc ID: {hit['doc_id']}")
            print(f"Metadata: {hit.get('metadata', '')}")
            snippet = hit.highlights("content") or hit["content"][:200]
            print(f"Content: {snippet}...\n{'-'*40}")
