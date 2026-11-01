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


# Heavy objects (models, vector DB, BM25 corpus) are loaded ONCE and reused for every query
_shared = {}

def _load_shared():
    if _shared:
        return _shared

    print("Loading embedding model and Vector Database...")
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/modernbert-embed-base",
        encode_kwargs={"prompt": "search_document: ", "normalize_embeddings": True},
        query_encode_kwargs={"prompt": "search_query: ", "normalize_embeddings": True},
    )
    _shared["vectorstore"] = Chroma(
        collection_name="sec_10k_filings",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )

    print("Loading documents into BM25 Keyword Search index...")
    docs_by_source = {}   # source PDF -> its chunks (lets us search one company's chunks quickly)
    all_docs = []
    for json_file in glob.glob(os.path.join(CHUNK_DIR, "*.json")):
        source_doc = os.path.basename(json_file).replace("_chunks.json", ".pdf")
        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        docs = [Document(page_content=text, metadata={"source": source_doc, "chunk_id": i})
                for i, text in enumerate(chunks)]
        docs_by_source[source_doc] = docs
        all_docs.extend(docs)
    _shared["docs_by_source"] = docs_by_source
    _shared["bm25_all"] = BM25Retriever.from_documents(all_docs)

    if RERANKER != "cohere":
        print("Loading Advanced Local Reranker (BAAI/bge-reranker-base)...")
        _shared["cross_encoder"] = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    return _shared


def build_reranked_retriever(final_k=10, base_k=30, sources=None):
    """
    sources: optional list of PDF names to restrict the search to (e.g. one company's filings).
             None = search all documents.
    """
    shared = _load_shared()

    # Stage 1a: vector search (restricted to `sources` if given)
    search_kwargs = {"k": base_k}
    if sources:
        search_kwargs["filter"] = {"source": {"$in": list(sources)}}
    vector_retriever = shared["vectorstore"].as_retriever(search_kwargs=search_kwargs)

    # Stage 1b: BM25 keyword search over the same subset
    if sources:
        subset = [doc for s in sources for doc in shared["docs_by_source"].get(s, [])]
        bm25_retriever = BM25Retriever.from_documents(subset) if subset else None
    else:
        bm25_retriever = shared["bm25_all"]

    if bm25_retriever is not None:
        bm25_retriever.k = base_k
        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
    else:
        base_retriever = vector_retriever

    # Stage 2: cross-encoder re-ranking
    if RERANKER == "cohere":
        from langchain_cohere import CohereRerank
        compressor = CohereRerank(model="rerank-v3.5", top_n=final_k)   # reads COHERE_API_KEY from .env
    else:
        compressor = ScoredCrossEncoderReranker(model=shared["cross_encoder"], top_n=final_k)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

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