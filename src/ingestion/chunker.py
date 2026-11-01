import os
import glob
import json
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_DIR = "data/parsed_docs"
OUTPUT_DIR = "data/chunked_docs"

def chunk_markdown_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    md_files = glob.glob(os.path.join(INPUT_DIR, "*.md"))
    if not md_files:
        print("No markdown files found! Make sure the parser finished running.")
        return

    # Set up the chunker to measure by tokens, using your exact specs
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", # Standard token encoding used by most modern LLMs
        chunk_size=1000,             # Max tokens per chunk
        chunk_overlap=100,           # 100 token overlap to prevent cutting context in half
        separators=[" "],            # word-level packing -> every chunk fills to ~1000 tokens (>= 800)
    )
    
    for md_path in md_files:
        filename = os.path.basename(md_path)
        json_filename = filename.replace(".md", "_chunks.json")
        output_path = os.path.join(OUTPUT_DIR, json_filename)
        
        if os.path.exists(output_path):
            print(f"Skipping {filename} (already chunked)")
            continue
            
        print(f"Chunking {filename}...")
        
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # Perform the actual chunking
        chunks = text_splitter.split_text(text)
        
        # Make the final chunk full-size if it came out shorter than 800 tokens
        enc = tiktoken.get_encoding("cl100k_base")
        if len(chunks) > 1 and len(enc.encode(chunks[-1])) < 800:
            chunks[-1] = enc.decode(enc.encode(text)[-1000:]).strip()
        
        # Save chunks to a JSON file for inspection and future database loading
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4)
            
        sizes = [len(enc.encode(c)) for c in chunks]
        print(f"    -> Created {len(chunks)} chunks (tokens min/max = {min(sizes)}/{max(sizes)}), saved to {output_path}")

if __name__ == "__main__":
    chunk_markdown_files()