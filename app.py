import os
import re
import json
import unicodedata
from collections import Counter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import your highly optimized two-stage retriever
from src.retrieval.hybrid_search import build_reranked_retriever

FINAL_K = 10

# Company names extracted automatically from each document (src/ingestion/extract_companies.py)
COMPANIES_FILE = "data/companies.json"
# One page-map file per document (src/ingestion/map_pages.py)
PAGES_DIR = "data/chunk_pages"

# Legal-form words that people don't use when asking ("Cisco", not "Cisco Systems, Inc.")
LEGAL_WORDS = {"the", "inc", "incorporated", "corporation", "corp", "co", "company", "companies",
               "plc", "ltd", "limited", "llc", "holdings", "group", "sa", "nv", "ag"}

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

def load_companies():
    if os.path.exists(COMPANIES_FILE):
        with open(COMPANIES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"Warning: {COMPANIES_FILE} not found. Run src/ingestion/extract_companies.py first.")
    return {}

COMPANIES = load_companies()   # source PDF -> company name
_page_cache = {}               # only documents that appear in answers are loaded

def normalize_words(text):
    # "Estée Lauder's" -> ["estee", "lauder"];  "Cisco Systems, Inc." -> ["cisco", "systems"]
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"'s\b", "", text.replace("’", "'"))
    return [w for w in re.findall(r"[a-z0-9&]+", text) if w not in LEGAL_WORDS]

def build_company_index(companies):
    """Returns (company_key -> {"name", "sources"}, alias -> company_key)."""
    index = {}
    for source, name in companies.items():
        if name.startswith("UNKNOWN"):
            continue
        key = " ".join(normalize_words(name))
        if not key:
            continue
        entry = index.setdefault(key, {"name": name, "sources": []})
        entry["sources"].append(source)   # a company can have several filings (years)

    # Short form (first word, e.g. "cisco") is only used if no other company shares it
    first_words = Counter(key.split()[0] for key in index)
    aliases = {}
    for key in index:
        aliases[key] = key
        first = key.split()[0]
        if first_words[first] == 1 and len(first) >= 3:
            aliases[first] = key
    return index, aliases

COMPANY_INDEX, COMPANY_ALIASES = build_company_index(COMPANIES)

def detect_companies(question):
    q = " ".join(normalize_words(question))
    found = []
    for alias, key in COMPANY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q) and key not in found:
            found.append(key)
    return found

def retrieve(question):
    keys = detect_companies(question)
    if not keys:
        print("   (no company detected in the question -> searching all documents)")
        return build_reranked_retriever(final_k=FINAL_K).invoke(question)

    per_company = max(1, FINAL_K // len(keys))   # share the 10 slots fairly between companies
    docs = []
    for key in keys:
        entry = COMPANY_INDEX[key]
        print(f"   -> searching only {entry['name']} ({len(entry['sources'])} document(s)), top {per_company}")
        retriever = build_reranked_retriever(final_k=per_company, sources=entry["sources"])
        docs.extend(retriever.invoke(question))
    return docs

def company_of(source):
    return COMPANIES.get(source, "Unknown company")

def page_label(source, chunk_id):
    if source not in _page_cache:
        path = os.path.join(PAGES_DIR, os.path.splitext(source or "")[0] + ".json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _page_cache[source] = json.load(f)
        else:
            _page_cache[source] = {}
    pages = _page_cache[source].get(str(chunk_id))
    if not pages:
        return "page n/a"
    start, end = pages
    return f"p. {start}" if start == end else f"pp. {start}-{end}"

def source_label(doc):
    source = doc.metadata.get("source")
    return f"{source} | {company_of(source)} | {page_label(source, doc.metadata.get('chunk_id'))}"

def format_docs(docs):
    # Numbered excerpts: the LLM only has to copy a short number like [2]
    return "\n\n".join(f"[{i}] {source_label(doc)}\n{doc.page_content}" for i, doc in enumerate(docs, 1))

def extract_citations(answer):
    cited = set()
    for group in CITATION_RE.findall(answer):
        for n in group.split(","):
            cited.add(int(n.strip()))
    return sorted(cited)

def main():
    print("Initializing the Local Production RAG Application...")
    
    # 1. Load the retrieval models once (they are reused for every company search)
    build_reranked_retriever(final_k=FINAL_K)
    
    # 2. Initialize the local LLM via Ollama
    print("\nConnecting to local Llama 3.1 engine...")
    llm = ChatOllama(model="llama3.1", temperature=0, num_ctx=16384)
    
    # 3. Create the strict system prompt
    template = """You are a strict financial analyst. You will be provided with numbered excerpts from SEC 10-K filings.
    Each excerpt starts with its number in square brackets, followed by the file name, the company it belongs to, and the page(s).
    
    Your task:
    1. Answer the question using ONLY the provided excerpts. Every statement must come from an excerpt; never reuse wording from these instructions.
    2. If the question asks about multiple companies, create a separate section for each company.
    3. Under each company, use ONLY excerpts that belong to THAT company. Never attribute information from one company's excerpt to a different company. If no excerpt belongs to a company, say "No relevant information found in the provided excerpts."
    4. Write in the third person using the company name, not "we" or "our".
    5. End EVERY sentence or bullet with the number of the excerpt it came from, in square brackets, e.g. [3]. If it came from two excerpts, write [3][5]. Only use numbers that appear in the excerpts.
    
    Excerpts:
    {context}

    Question: {question}

    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    # 4. Execute the query
    question = "What are the primary risk factors related to supply chain disruptions for Cisco and FedEx?"
    print(f"\n🧠 Synthesizing answer for: '{question}'\n")
    print("-" * 50)

    docs = retrieve(question)
    
    # Stream the output directly to the console like ChatGPT
    answer_parts = []
    for chunk in chain.stream({"context": format_docs(docs), "question": question}):
        print(chunk, end="", flush=True)
        answer_parts.append(chunk)
    answer = "".join(answer_parts)

    # 5. Sources: the code (not the LLM) turns each cited number into file / company / pages
    print("\n" + "-" * 50)
    print("Sources:")
    cited = extract_citations(answer)
    for n in cited:
        if 1 <= n <= len(docs):
            print(f"  [{n}] {source_label(docs[n - 1])}")
        else:
            print(f"  [{n}] ⚠️ cited, but there is no excerpt with this number")
    if not cited:
        print("  (the answer contains no citations; treat it as unverified)")

if __name__ == "__main__":
    main()