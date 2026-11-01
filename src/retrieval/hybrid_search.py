import os
import json
import glob
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# New imports for the Re-ranking pipeline
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

CHUNK_DIR = "data/chunked_docs"
DB_DIR = "data/chroma_db"

def build_reranked_retriever(final_k=10, base_k=30):
    print("Loading embedding model and Vector Database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="sec_10k_filings",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    
    # Stage 1: Pull a larger pool of chunks (base_k = 30)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": base_k})
    
    print("Loading documents into BM25 Keyword Search index...")
    docs_for_bm25 = []
    json_files = glob.glob(os.path.join(CHUNK_DIR, "*.json"))
    
    for json_file in json_files:
        source_doc = os.path.basename(json_file).replace("_chunks.json", ".pdf")
        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            for chunk_text in chunks:
                doc = Document(page_content=chunk_text, metadata={"source": source_doc})
                docs_for_bm25.append(doc)

    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = base_k 

    print("Merging retrievers into Hybrid Ensemble...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    print("Loading Advanced Local Reranker (BAAI/bge-reranker-base)...")
    # This is a much stronger local cross-encoder than the default ms-marco
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=final_k)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )
    
    return compression_retriever

def retrieve_and_cite(query, final_k=10):
    retriever = build_reranked_retriever(final_k=final_k)
    
    print(f"\n🔍 Searching and Re-ranking for: '{query}'\n")
    results = retriever.invoke(query)
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown Source")
        # The CrossEncoder attaches its actual relevance score to the metadata!
        score = doc.metadata.get("relevance_score", "N/A")
        print(f"--- Result {i} | Source: {source} | Rerank Score: {score} ---")
        
        snippet = doc.page_content[:300].replace('\n', ' ')
        print(f"{snippet}...\n")

if __name__ == "__main__":
    test_query = "What are the primary risk factors related to supply chain disruptions?"
    retrieve_and_cite(test_query, final_k=10)