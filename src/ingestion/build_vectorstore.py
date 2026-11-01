import os
import json
import glob
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHUNK_DIR = "data/chunked_docs"
DB_DIR = "data/chroma_db"

def build_vector_store():
    # 1. Initialize the local, free HuggingFace embedding model
    print("Loading HuggingFace embedding model (nomic-embed-text-v1.5)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"prompt": "search_document: ", "normalize_embeddings": True},
        query_encode_kwargs={"prompt": "search_query: ", "normalize_embeddings": True},
    )
    
    # 2. Initialize ChromaDB
    print(f"Initializing ChromaDB at {DB_DIR}...")
    vectorstore = Chroma(
        collection_name="sec_10k_filings",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    
    # 3. Read chunks and add them to the database
    json_files = glob.glob(os.path.join(CHUNK_DIR, "*.json"))
    if not json_files:
        print("No chunked JSON files found!")
        return

    for json_file in json_files:
        filename = os.path.basename(json_file)
        
        # We will use the original filename (e.g., MSFT) as metadata so we know where the answer came from
        source_doc = filename.replace("_chunks.json", ".pdf")
        
        print(f"Processing and embedding chunks from {filename}...")
        
        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        # Optional safety check if a document has no chunks
        if not chunks:
            continue
            
        # Attach metadata to every chunk so the AI can cite its sources later
        metadatas = [{"source": source_doc, "chunk_id": i} for i in range(len(chunks))]
        ids = [f"{source_doc}-{i}" for i in range(len(chunks))]
        
        # Add to ChromaDB (fixed ids -> re-running updates instead of duplicating)
        vectorstore.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
        print(f"    -> Embedded {len(chunks)} chunks into ChromaDB.")

    print(f"\n✅ Vector database built successfully at {DB_DIR}!")

if __name__ == "__main__":
    build_vector_store()