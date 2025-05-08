import os
from whoosh.index import create_in, open_dir
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import shutil
from config_loader_class import ConfigLoader

class KeywordSearchEngine:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.index_dir = self.config_loader.get_index_dir()
        self.schema = Schema(
            doc_id=ID(stored=True, unique=True),
            full_doc_id=ID(stored=True),
            content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            metadata=TEXT(stored=True)
        )

    def delete_existing_index(self):
        if os.path.exists(self.index_dir):
            shutil.rmtree(self.index_dir)
            print("✅ Existing index deleted.")

    def get_or_create_index(self):
        # If the index doesn't exist (or has been deleted), create a new one
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
            return create_in(self.index_dir, self.schema)
        else:
            return open_dir(self.index_dir)

    def keyword_search_with_stemming(self, query_string):
        ix = self.get_or_create_index()  # Ensure index exists
        with ix.searcher() as searcher:
            query = QueryParser("content", ix.schema).parse(query_string)
            results = searcher.search(query, limit=10)

            # Copy the fields we need before the searcher closes
            hits = []
            for r in results:
                hits.append({
                    "doc_id": r["doc_id"],
                    "full_doc_id": r.get("full_doc_id", ""),
                    "content": r["content"],
                    "metadata": r["metadata"]
                })

            return hits

    def keyword_search(self, query_str, top_k=10):
        ix = self.get_or_create_index()
        with ix.searcher() as searcher:
            parser = QueryParser("content", ix.schema)
            query = parser.parse(query_str)
            results = searcher.search(query, limit=top_k)

            hits = []
            for r in results:
                hits.append({
                    "doc_id": r["doc_id"],
                    "full_doc_id": r.get("full_doc_id", ""),
                    "content": r["content"],
                    "metadata": r["metadata"]
                })

            return hits

    @staticmethod
    def build_full_doc_id(full_path, source_label, chunk_index):
        label = source_label.replace(', ', '_').replace(' ', '_')
        return f"{full_path}_{label}_chunk{chunk_index}"

    def index_keyword_chunk(self, doc_id, content, metadata):
        ix = self.get_or_create_index()
        writer = ix.writer()

        # metadata is already a dict
        full_doc_id = metadata.get("full_doc_id", "")  # no need for eval/parse

        writer.update_document(
            doc_id=doc_id,
            full_doc_id=full_doc_id,
            content=content,
            metadata=str(metadata)  # still store it as string inside the index
        )
        writer.commit()

# Example usage:
if __name__ == "__main__":
    keyword_engine = KeywordSearchEngine()

    # Example of deleting the index
    # keyword_engine.delete_existing_index()

    # Example of building a doc ID
    full_id = keyword_engine.build_full_doc_id("/path/to/doc.txt", "Release Notes, 1.0", 5)
    print(f"Built Full Doc ID: {full_id}")

    # Example of indexing a chunk (you would typically do this in your data processing pipeline)
    # keyword_engine.index_keyword_chunk("doc1_chunk1", "This document talks about a new feature.", {"full_doc_id": "doc1_Release_Notes_chunk1", "release": "1.0"})

    # Example of performing a keyword search with stemming
    results_stemming = keyword_engine.keyword_search_with_stemming("new feature")
    print("\nKeyword Search with Stemming Results:")
    for hit in results_stemming:
        print(hit)

    # Example of performing a regular keyword search
    results_regular = keyword_engine.keyword_search("new feature")
    print("\nRegular Keyword Search Results:")
    for hit in results_regular:
        print(hit)
