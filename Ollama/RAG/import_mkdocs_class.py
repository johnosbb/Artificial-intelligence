#!/usr/bin/env python3
"""
Data Ingestion Pipeline for ChromaDB + Ollama-backed RAG

This script walks a document tree, classifies files, chunks content, optionally
creates a short summary, generates embeddings with Ollama, and indexes the data
into a ChromaDB collection. It also builds a keyword index via KeywordSearchEngine.

Prerequisites
------------
1) ChromaDB server running (default expected at http://127.0.0.1:8000).
   Example:
     chroma run --host 127.0.0.1 --port 8000 --path /path/to/ChromaDB

2) Ollama server running (default at http://127.0.0.1:11434) with the models
   configured in your config used by TextProcessingUtilities:
   - embedmodel
   - mainmodel

3) Python dependencies installed:
     pip install chromadb ollama nltk

4) Local modules available on PYTHONPATH:
   - rag_utilities_class
   - keyword_search_class
   - config_loader_class

Basic Usage
-----------
Run with defaults (uses existing collection if present):
  ./data_ingestion_pipeline.py

Force delete and recreate the target collection before ingest:
  ./data_ingestion_pipeline.py --delete-existing-collection

Generate a summary for each document and index it:
  ./data_ingestion_pipeline.py --summarize

Combine options:
  ./data_ingestion_pipeline.py --delete-existing-collection --summarize

Help:
  ./data_ingestion_pipeline.py -h
"""

import os
import sys
import time
import re
import argparse
import chromadb
import ollama
import nltk

from rag_utilities_class import TextProcessingUtilities
from keyword_search_class import KeywordSearchEngine
from config_loader_class import ConfigLoader


class DataIngestionPipeline:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.ru = TextProcessingUtilities()
        self.output_directory = self.config_loader.get_output_dir()
        self.index_dir = self.config_loader.get_index_dir()
        self.document_folder_path = self.config_loader.get_document_folder_path()
        self.collection_name = "buildragwithpython"
        self.embed_model = self.ru.get_config()["embedmodel"]
        self.main_model = self.ru.get_config()["mainmodel"]
        self.allowed_extensions = [".md", ".txt", ".html", ".py", ".c", ".h"]
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    def parse_args(self):
        parser = argparse.ArgumentParser(
            prog=os.path.basename(sys.argv[0]),
            description="Ingest documents into a ChromaDB collection with Ollama embeddings."
        )
        parser.add_argument(
            "--delete-existing-collection",
            "-d",
            dest="delete_existing",
            action="store_true",
            help="Delete and recreate the target collection before ingest."
        )
        parser.add_argument(
            "--summarize",
            "-s",
            action="store_true",
            help="Generate a short LLM summary for each document and index it."
        )
        return vars(parser.parse_args())

    def setup_collection(self, delete_existing=True):
        chroma = chromadb.HttpClient(
            host="localhost",
            port=8000,
            headers={"X-Tenant-Id": "default_tenant"}
        )
        try:
            collection = chroma.get_collection(self.collection_name)
            if delete_existing:
                ks = KeywordSearchEngine()
                print(f"Deleting existing collection and keyword index '{self.collection_name}'...")
                chroma.delete_collection(self.collection_name)
                collection = chroma.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                ks.delete_existing_index()
            else:
                print(f"Using existing collection '{self.collection_name}'.")
        except Exception:
            print(f"Creating new collection '{self.collection_name}'.")
            collection = chroma.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return collection

    def generate_summary(self, text, doc_type):
        summary_prompt = (
            f"Summarize the following {doc_type.strip()} in clear, natural language. "
            "Include any notable features, versions, and product references:\n\n"
            f"{text}\n\nSummary:"
        )
        response = ollama.generate(model=self.main_model, prompt=summary_prompt)
        return response["response"].strip()

    def index_summary(self, summary, full_path, doc_type, collection):
        chunk = f"search_document: {doc_type}(summary): {summary}"
        embed = ollama.embeddings(model=self.embed_model, prompt=chunk)["embedding"]
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
        print(f"Summary added for {os.path.basename(full_path)}")

    def index_chunks(self, text, full_path, doc_type, collection):
        ru = self.ru
        dprefix = "search_document: "
        release_version = None

        doc_type_clean = doc_type.strip(": ").lower()
        section_text = text
        chunking_strategy = None

        relative_path = os.path.relpath(full_path, self.document_folder_path)
        url_path = relative_path.replace("\\", "/")
        url_path = re.sub(r"\.md$|\.txt$|\.html$", "", url_path)
        url_path = url_path.rstrip("}/")
        full_url = f"http://10.211.129.241:8000/{url_path}/"

        if doc_type_clean == "release notes":
            release_version = ru.extract_release_version(text)
            new_func_text = ru.extract_section(text, "New functionality")
            if new_func_text:
                section_text = new_func_text

        if ru.is_table_heavy(section_text):
            chunks = ru.chunk_by_tables(section_text)
            chunking_strategy = "table"
        elif ru.is_bullet_heavy(section_text):
            chunks = ru.chunk_by_bullets(section_text)
            chunking_strategy = "bullet"
        else:
            chunks = ru.chunk_by_sentences(source_text=section_text, sentences_per_chunk=7, overlap=0)
            chunking_strategy = "sentence"

        base_filename = os.path.splitext(os.path.basename(full_path))[0]
        source_label = f"{base_filename}"
        if release_version:
            source_label += f" v{release_version}"

        for index, chunk in enumerate(chunks):
            ks = KeywordSearchEngine()
            citation_tag = f"[{source_label}, chunk {index}]"
            full_chunk = f"{dprefix}{doc_type} {citation_tag}\n{chunk}"
            embed = ollama.embeddings(model=self.embed_model, prompt=full_chunk)["embedding"]
            doc_id = f"{full_path}_chunk{index}"
            full_doc_id = ks.build_full_doc_id(full_path, source_label, index)
            metadata = {
                "source": full_path,
                "doctype": doc_type_clean,
                "source_label": f"{source_label}, chunk {index}",
                "chunking_strategy": chunking_strategy,
                "full_doc_id": full_doc_id,
                "url": full_url
            }

            if release_version:
                metadata["release"] = release_version
            if doc_type_clean == "release notes" and section_text == new_func_text:
                metadata["section"] = "new functionality"

            if doc_type_clean == "source code":
                language = ru.detect_language(full_path)
                metadata["language"] = language
                metadata["component"] = ru.classify_code_component(full_path)

            collection.add(
                ids=[doc_id],
                embeddings=[embed],
                documents=[full_chunk],
                metadatas=metadata
            )
            print(".", end="", flush=True)
            ks.index_keyword_chunk(doc_id, chunk, metadata)

    def process_document(self, full_path, collection, flags, processed, skipped):
        try:
            text = self.ru.readtext(full_path)
            ru = TextProcessingUtilities()
            is_code_file = full_path.endswith((".py", ".c", ".h", ".cpp"))
            if is_code_file:
                doc_type = "source code"
            else:
                doc_type = ru.classify_doc_type(full_path, text)

            if flags["summarize"]:
                try:
                    summary = self.generate_summary(text, doc_type)
                    self.index_summary(summary, full_path, doc_type, collection)
                except Exception as e:
                    print(f"Summary failed for {os.path.basename(full_path)}: {e}")

            self.index_chunks(text, full_path, doc_type, collection)
            processed.append(full_path)

        except Exception as e:
            print(f"\nError processing {full_path}: {e}")
            skipped.append(full_path)

    def main(self):
        flags = self.parse_args()
        collection = self.setup_collection(delete_existing=flags["delete_existing"])

        processed_files, skipped_files = [], []
        starttime = time.time()

        for root, dirs, files in os.walk(self.document_folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                if not any(file.lower().endswith(ext) for ext in self.allowed_extensions):
                    skipped_files.append(full_path)
                    continue
                print(f"\nIndexing {file}")
                self.process_document(full_path, collection, flags, processed_files, skipped_files)

        os.makedirs(self.output_directory, exist_ok=True)
        with open(f"{self.output_directory}/processed.txt", "w") as f:
            f.writelines(p + "\n" for p in processed_files)
        with open(f"{self.output_directory}/skipped.txt", "w") as f:
            f.writelines(s + "\n" for s in skipped_files)

        elapsed = time.time() - starttime
        print(f"\nDone. Processed {len(processed_files)} files in {elapsed:.2f} seconds.")
        if skipped_files:
            print(f"Skipped {len(skipped_files)} files (see {self.output_directory}/skipped.txt).")


if __name__ == "__main__":
    ingestion_pipeline = DataIngestionPipeline()
    ingestion_pipeline.main()
