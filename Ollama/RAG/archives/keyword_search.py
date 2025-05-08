import os
from whoosh.index import create_in, open_dir
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import shutil
from config_loader import get_index_dir,get_output_dir

INDEX_DIR = get_index_dir()

# ------------------- Indexing ---------------------





def delete_existing_index():
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
        print("✅ Existing index deleted.")

def keyword_search_with_stemming(query_string):
    ix = open_dir(INDEX_DIR)  # Your index directory
    with ix.searcher() as searcher:
        analyzer = StemmingAnalyzer()
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


def keyword_search(query_str, top_k=10):
    ix = get_or_create_index()
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


def build_full_doc_id(full_path, source_label, chunk_index):
    label = source_label.replace(', ', '_').replace(' ', '_')
    return f"{full_path}_{label}_chunk{chunk_index}"

def get_or_create_index():
    # Define the schema
    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        full_doc_id=ID(stored=True),
        content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        metadata=TEXT(stored=True)
    )

    # If the index doesn't exist (or has been deleted), create a new one
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
        return create_in(INDEX_DIR, schema)
    else:
        return open_dir(INDEX_DIR)

def index_keyword_chunk(doc_id, content, metadata):
    ix = get_or_create_index()
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



