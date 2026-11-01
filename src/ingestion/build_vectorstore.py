import os
import json
import glob
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHUNK_DIR = "data/chunked_docs"
DB_DIR = "data/chroma_db"

def build_vector_store():
    # 1. Initialize the local, free HuggingFace embedding model
    print("Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
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
        metadatas = [{"source": source_doc} for _ in chunks]
        
        # Add to ChromaDB (this handles the embedding math automatically)
        vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        print(f"    -> Embedded {len(chunks)} chunks into ChromaDB.")

    print(f"\n✅ Vector database built successfully at {DB_DIR}!")

if __name__ == "__main__":
    build_vector_store()