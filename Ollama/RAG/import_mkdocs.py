#!/usr/bin/env python3

import os, sys, time
import re
import chromadb
import ollama
from utilities import readtext, getconfig
# from mattsollamatools import chunk_text_by_sentences
import nltk
import rag_utilities as ru

nltk.download('punkt_tab')

collectionname = "buildragwithpython"
document_folder_path="/mnt/500GB/docs"
# ------------------ Config & Globals ------------------

embedmodel = getconfig()["embedmodel"]
mainmodel = getconfig()["mainmodel"]
allowed_extensions = [".md", ".txt", ".html", ".py", ".c", ".h"]

# ------------------ Arg Parsing ------------------

def parse_args():
    return {
        "delete_existing": "--delete-existing-collection" in sys.argv,
        "summarize": "--summarize" in sys.argv,
    }

# ------------------ Chroma Setup ------------------

def setup_collection(delete_existing=False):
    chroma = chromadb.HttpClient(
        host="localhost",
        port=8000,
        headers={"X-Tenant-Id": "default_tenant"}
    )
    try:
        collection = chroma.get_collection(collectionname)
        if delete_existing:
            print(f"🗑️  Deleting existing collection '{collectionname}'...")
            chroma.delete_collection(collectionname)
            collection = chroma.get_or_create_collection(name=collectionname, metadata={"hnsw:space": "cosine"})
        else:
            print(f"📚 Using existing collection '{collectionname}'")
    except Exception:
        print(f"📦 Creating new collection '{collectionname}'")
        collection = chroma.get_or_create_collection(name=collectionname, metadata={"hnsw:space": "cosine"})
    return collection





# ------------------ Document Summarization ------------------

def generate_summary(text, doc_type):
    summary_prompt = (
        f"Summarize the following {doc_type.strip()} in clear, natural language. "
        "Include any notable features, versions, and product references:\n\n"
        f"{text}\n\nSummary:"
    )
    response = ollama.generate(model=mainmodel, prompt=summary_prompt)
    return response["response"].strip()

def index_summary(summary, full_path, doc_type, collection):
    chunk = f"search_document: {doc_type}(summary): {summary}"
    embed = ollama.embeddings(model=embedmodel, prompt=chunk)['embedding']
    doc_id = f"{full_path}_summary"
    collection.add(
        ids=[doc_id],
        embeddings=[embed],
        documents=[chunk],
        metadatas={
            "source": full_path,
            "doctype": doc_type.strip(": ").lower(),
            "is_summary": True
        }
    )
    print(f"✅ Summary added for {os.path.basename(full_path)}")

# ------------------ Chunking and Indexing ------------------





def index_chunks(text, full_path, doc_type, collection):
    dprefix = "search_document: "
    release_version = None

    doc_type_clean = doc_type.strip(": ").lower()
    section_text = text

    # Try to extract relevant section from release notes
    if doc_type_clean == "release note":
        release_version = ru.extract_release_version(text)
        new_func_text = ru.extract_section(text, "New functionality")
        if new_func_text:
            section_text = new_func_text

    # === Decide chunking strategy ===
    if ru.is_table_heavy(section_text):
        chunks = ru.chunk_by_tables(section_text)
    elif ru.is_bullet_heavy(section_text):
        chunks = ru.chunk_by_bullets(section_text)
    else:
        chunks = ru.chunk_by_sentences(source_text=section_text, sentences_per_chunk=7, overlap=0)

    base_filename = os.path.splitext(os.path.basename(full_path))[0]
    source_label = f"{base_filename}"
    if release_version:
        source_label += f" v{release_version}"

    # === Index chunks ===
    for index, chunk in enumerate(chunks):
        citation_tag = f"[{source_label}, chunk {index}]"
        full_chunk = f"{dprefix}{doc_type} {citation_tag}\n{chunk}"
        embed = ollama.embeddings(model=embedmodel, prompt=full_chunk)['embedding']
        doc_id = f"{full_path}_chunk{index}"

        metadata = {
            "source": full_path,
            "doctype": doc_type_clean,
            "source_label": f"{source_label}, chunk {index}"
        }

        if release_version:
            metadata["release"] = release_version
        if doc_type_clean == "release note" and section_text == new_func_text:
            metadata["section"] = "new functionality"

        if doc_type_clean == "source code":
            language = ru.detect_language(full_path)
            metadata["language"] = language
            metadata["component"] = ru.classify_code_component(full_path)  # optional


        collection.add(
            ids=[doc_id],
            embeddings=[embed],
            documents=[full_chunk],
            metadatas=metadata
        )
        print(".", end="", flush=True)



# ------------------ Main Ingestion Logic ------------------

def process_document(full_path, collection, flags, processed, skipped):
    try:
        text = readtext(full_path)
        is_code_file = full_path.endswith(('.py', '.c', '.h', '.cpp'))
        if is_code_file:
            doc_type = "source code"
        else:
            doc_type = ru.classify_doc_type(full_path, text)

        if flags["summarize"]:
            try:
                summary = generate_summary(text, doc_type)
                index_summary(summary, full_path, doc_type, collection)
            except Exception as e:
                print(f"⚠️  Summary failed for {os.path.basename(full_path)}: {e}")

        index_chunks(text, full_path, doc_type, collection)
        processed.append(full_path)

    except Exception as e:
        print(f"\n❌ Error processing {full_path}: {e}")
        skipped.append(full_path)



# ------------------ Entry Point ------------------

def main():
    flags = parse_args()
    collection = setup_collection(delete_existing=flags["delete_existing"])

    processed_files, skipped_files = [], []
    starttime = time.time()

    for root, dirs, files in os.walk(document_folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            if not any(file.lower().endswith(ext) for ext in allowed_extensions):
                skipped_files.append(full_path)
                continue
            print(f"\n📄 {file}")
            process_document(full_path, collection, flags, processed_files, skipped_files)

    # Save logs
    with open("processed.txt", "w") as f:
        f.writelines(p + "\n" for p in processed_files)
    with open("skipped.txt", "w") as f:
        f.writelines(s + "\n" for s in skipped_files)

    print(f"\n✅ Done! Processed {len(processed_files)} files in {time.time() - starttime:.2f} seconds.")

if __name__ == "__main__":
    main()
