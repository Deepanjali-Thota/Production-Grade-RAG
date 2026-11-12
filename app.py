import os
import sys
import re
import json
import unicodedata
from collections import Counter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import your highly optimized two-stage retriever
from src.retrieval.hybrid_search import build_reranked_retriever
# Settings and the versioned prompt live in config/settings.yaml and config/prompts.yaml
from src.config import SETTINGS, PROMPT_VERSION, PROMPT_TEMPLATE

FINAL_K = SETTINGS["retrieval"]["final_k"]
MIN_RELEVANCE = SETTINGS["enforcement"]["min_relevance"]
MAX_REMOVED_FRACTION = SETTINGS["enforcement"]["max_removed_fraction"]
REFUSAL_NO_EVIDENCE = SETTINGS["messages"]["refusal_no_evidence"]
REFUSAL_FAILED_CHECKS = SETTINGS["messages"]["refusal_failed_checks"]
NO_INFO = SETTINGS["messages"]["no_info"]
SECTION_EMPTY = SETTINGS["messages"]["section_empty"]

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

def question_for_company(question, key):
    """Rewrite the question so it names only ONE company (other names lower the reranker's scores)."""
    # Plain ASCII copy, so "Estée Lauder’s" matches the alias "estee lauder"
    q = unicodedata.normalize("NFKD", question.replace("’", "'")).encode("ascii", "ignore").decode()
    legal = "|".join(sorted((re.escape(w) for w in LEGAL_WORDS if w != "the"), key=len, reverse=True))
    # A mention is: optional "The", the alias, optional legal words ("Corporation", ", Inc."), optional 's
    tail = rf"(?:'s)?(?:[\s,]+(?:{legal})\b\.?)*(?:'s)?"
    for alias in sorted(COMPANY_ALIASES, key=len, reverse=True):   # longest first: "micron technology" before "micron"
        q = re.sub(rf"(?:\bthe\s+)?\b{re.escape(alias)}\b{tail}", "\x00", q, flags=re.IGNORECASE)
    # "Cisco, FedEx and P&G" -> one marker, then remove it
    q = re.sub(r"\x00(?:\s*(?:,|\band\b|\bor\b|&)\s*\x00)*", "\x00", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q.replace("\x00", "")).strip()
    # Drop connectors left dangling at the end, e.g. "...disruptions for and ?" -> "...disruptions"
    q = re.sub(r"(?:\s*(?:,|\band\b|\bor\b|\bfor\b|\bof\b|\bat\b|\bbetween\b))+\s*[?.!]*$", "", q, flags=re.IGNORECASE)
    q = q.rstrip(" ?.!")
    return f"{q} for {COMPANY_INDEX[key]['name']}?"

def retrieve(question):
    keys = detect_companies(question)
    if not keys:
        print("   (no company detected in the question -> searching all documents)")
        return build_reranked_retriever(final_k=FINAL_K).invoke(question)

    per_company = max(1, FINAL_K // len(keys))   # share the 10 slots fairly between companies
    docs = []
    for key in keys:
        entry = COMPANY_INDEX[key]
        company_question = question_for_company(question, key)
        print(f"   -> searching only {entry['name']} ({len(entry['sources'])} document(s)), top {per_company}")
        print(f"      query: {company_question}")
        retriever = build_reranked_retriever(final_k=per_company, sources=entry["sources"])
        docs.extend(retriever.invoke(company_question))
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

def apply_evidence_gate(docs):
    """Keep only chunks the reranker considers relevant enough to serve as evidence."""
    return [d for d in docs if (d.metadata.get("relevance_score") or 0) >= MIN_RELEVANCE]

def section_company(heading, docs):
    """Which retrieved company (if any) a section heading like '**Cisco Systems, Inc.**' names."""
    words = normalize_words(heading)
    for doc in docs:
        name = company_of(doc.metadata.get("source"))
        key_words = normalize_words(name)
        if key_words and key_words[0] in words:
            return name
    return None

def is_heading(text, plain):
    # "**Cisco Systems, Inc.**", "# FedEx", or an intro line ending in ":" -- structure, not a claim
    return text.startswith("#") or (text.startswith("**") and text.endswith("**")) or plain.endswith(":")

def claim_problem(text, docs, current_company):
    """Why a claim line fails its citation checks, or None if it passes."""
    numbers = [int(n) for group in CITATION_RE.findall(text) for n in group.split(",")]
    if not numbers:
        return "no citation"
    if any(not 1 <= n <= len(docs) for n in numbers):
        return "cites a non-existent excerpt"
    if current_company:
        wrong = [n for n in numbers if company_of(docs[n - 1].metadata.get("source")) != current_company]
        if wrong:
            return f"cites another company's excerpt {''.join(f'[{n}]' for n in wrong)}"
    return None

def check_claims(answer, docs):
    """
    Drop-and-keep: claim lines that fail their citation checks are removed, everything else is kept.
    Returns (kept_lines, removed, total_claims) where removed = [(claim, reason), ...].
    """
    kept, removed = [], []
    total_claims = 0
    current_company = None
    section_has_content = True   # becomes False at each company heading until a valid line follows

    def close_section():
        if not section_has_content:
            kept.extend([SECTION_EMPTY, ""])

    for line in answer.splitlines():
        text = line.strip()
        plain = text.strip("*#-• \t").strip()
        if not plain:
            kept.append(line)
            continue
        if is_heading(text, plain):
            company = section_company(plain, docs)
            if company and company != current_company:   # a new company section starts
                close_section()
                current_company = company
                section_has_content = False
            kept.append(line)
            continue
        if NO_INFO.lower() in plain.lower():
            kept.append(line)
            section_has_content = True
            continue
        total_claims += 1
        reason = claim_problem(text, docs, current_company)
        if reason:
            removed.append((plain, reason))
        else:
            kept.append(line)
            section_has_content = True
    close_section()
    return kept, removed, total_claims

def build_llm():
    """LLM chosen in config/settings.yaml: provider 'cohere' (hosted, fast) or 'ollama' (local)."""
    cfg = SETTINGS["llm"]
    provider = cfg.get("provider", "ollama").lower()
    if provider == "cohere":
        from langchain_cohere import ChatCohere   # reads COHERE_API_KEY from .env
        return ChatCohere(model=cfg["cohere_model"], temperature=cfg["temperature"])
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=cfg["model"], temperature=cfg["temperature"], num_ctx=cfg["num_ctx"])
    raise ValueError(f"Unknown llm.provider '{provider}' in config/settings.yaml. Use 'cohere' or 'ollama'.")

_chain = None

def get_chain():
    """Prompt + LLM, built once and reused for every question."""
    global _chain
    if _chain is None:
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        _chain = prompt | build_llm() | StrOutputParser()
    return _chain

def answer_question(question):
    """
    Runs the whole pipeline for one question and returns a dict:
      answer    - final text shown to the user (the refusal message if refused)
      refused   - True if the system refused to answer
      reason    - why it refused ("no_evidence" or "failed_checks"), else None
      contexts  - text of the chunks given to the LLM (used by the evaluation)
      sources   - [{"ref", "label", "relevance"}] for the excerpts the answer cites
      removed   - [(claim, reason)] claims dropped by the citation checks
    """
    result = {"question": question, "answer": "", "refused": False, "reason": None,
              "contexts": [], "sources": [], "removed": []}

    # Gate 1: evidence. Refuse without calling the LLM if no chunk is relevant enough.
    docs = apply_evidence_gate(retrieve(question))
    if not docs:
        result.update(answer=REFUSAL_NO_EVIDENCE, refused=True, reason="no_evidence")
        return result
    result["contexts"] = [doc.page_content for doc in docs]

    # Not streamed: the answer must pass the citation checks before it is shown
    raw_answer = get_chain().invoke({"context": format_docs(docs), "question": question})

    # Gate 2: citations. Drop claims that fail their checks; refuse if more than half fail.
    kept, removed, total_claims = check_claims(raw_answer, docs)
    result["removed"] = removed
    if total_claims and len(removed) > total_claims * MAX_REMOVED_FRACTION:
        result.update(answer=REFUSAL_FAILED_CHECKS, refused=True, reason="failed_checks")
        return result

        # The model found nothing to say (only "No relevant information..." lines): that is a refusal too
    if total_claims == len(removed):
        result.update(answer=REFUSAL_NO_EVIDENCE, refused=True, reason="no_answer_in_excerpts")
        return result

    answer = "\n".join(kept).strip()
    result["answer"] = answer
    # Sources: the code (not the LLM) turns each cited number into file / company / pages
    result["sources"] = [
        {"ref": n, "label": source_label(docs[n - 1]), "relevance": docs[n - 1].metadata.get("relevance_score")}
        for n in extract_citations(answer) if 1 <= n <= len(docs)
    ]
    return result

def main():
    # Usage: python app.py "your question"     (no argument -> the default question below)
    question = " ".join(sys.argv[1:]).strip() or \
        "What are the primary risk factors related to supply chain disruptions for Cisco and FedEx?"

    llm_cfg = SETTINGS["llm"]
    print(f"LLM: {llm_cfg.get('provider', 'ollama')} | prompt version: {PROMPT_VERSION}")
    print(f"\n🧠 Question: '{question}'\n")
    print("-" * 50)

    result = answer_question(question)
    print(result["answer"])

    if result["removed"]:
        print(f"\n{len(result['removed'])} claim(s) failed citation checks:")
        for claim, reason in result["removed"]:
            print(f"   - {reason}: {claim[:80]}")

    if not result["refused"]:
        print("\n" + "-" * 50)
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['ref']}] {s['label']} | relevance {s['relevance']}")

if __name__ == "__main__":
    main()