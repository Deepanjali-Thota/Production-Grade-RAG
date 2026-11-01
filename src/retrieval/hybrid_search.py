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

from dotenv import load_dotenv
load_dotenv()

RERANKER = os.getenv("RERANKER", "sbert").lower()   # "sbert" or "cohere"

CHUNK_DIR = "data/chunked_docs"
DB_DIR = "data/chroma_db"


class ScoredCrossEncoderReranker(CrossEncoderReranker):
    """Same as CrossEncoderReranker, but keeps the score in doc.metadata['relevance_score']."""
    def compress_documents(self, documents, query, callbacks=None):
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        results = []
        for doc, score in ranked[: self.top_n]:
            doc.metadata["relevance_score"] = round(float(score), 4)
            results.append(doc)
        return results


def build_reranked_retriever(final_k=10, base_k=30):
    print("Loading embedding model and Vector Database...")
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"prompt": "search_document: ", "normalize_embeddings": True},
        query_encode_kwargs={"prompt": "search_query: ", "normalize_embeddings": True},
    )
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
            for i, chunk_text in enumerate(chunks):
                doc = Document(page_content=chunk_text, metadata={"source": source_doc, "chunk_id": i})
                docs_for_bm25.append(doc)

    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = base_k 

    print("Merging retrievers into Hybrid Ensemble...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    if RERANKER == "cohere":
        print("Using Cohere Rerank API (rerank-v3.5)...")
        from langchain_cohere import CohereRerank
        compressor = CohereRerank(model="rerank-v3.5", top_n=final_k)   # reads COHERE_API_KEY from .env
    else:
        print("Loading Advanced Local Reranker (BAAI/bge-reranker-base)...")
        # This is a much stronger local cross-encoder than the default ms-marco
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = ScoredCrossEncoderReranker(model=model, top_n=final_k)
    
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
        # The reranker attaches its relevance score to the metadata
        score = doc.metadata.get("relevance_score", "N/A")
        print(f"--- Result {i} | Source: {source} | Chunk: {doc.metadata.get('chunk_id')} | Rerank Score: {score} ---")
        
        snippet = doc.page_content[:300].replace('\n', ' ')
        print(f"{snippet}...\n")

if __name__ == "__main__":
    test_query = "What are the primary risk factors related to supply chain disruptions?"
    retrieve_and_cite(test_query, final_k=10)