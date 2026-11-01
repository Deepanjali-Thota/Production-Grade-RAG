import os
import glob
import json
from langchain_ollama import ChatOllama

PARSED_DIR = "data/parsed_docs"
OUTPUT_FILE = "data/companies.json"

PROMPT = """Below is the beginning of a financial document (for example an SEC 10-K filing).
What is the full legal name of the company that this document belongs to?
Give the company name as written on the cover page (for example "Microsoft Corporation" or "Nike, Inc."),
NOT the stock ticker symbol.
Reply with ONLY the company name, nothing else.

Document:
{text}"""

def looks_like_ticker(name):
    # Tickers are one short word (ACN, MSFT, PANW); real names have spaces or are longer
    return " " not in name and len(name) <= 6

def ask_company(llm, md_path, num_chars):
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read(num_chars)
    response = llm.invoke(PROMPT.format(text=text))
    return response.content.strip().splitlines()[0].strip().strip('"').strip("*").strip()

def extract_companies():
    # Load existing results so we only process new documents
    companies = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            companies = json.load(f)

    md_files = sorted(glob.glob(os.path.join(PARSED_DIR, "*.md")))
    if not md_files:
        print("No markdown files found! Run batch_parse first.")
        return

    llm = ChatOllama(model="llama3.1", temperature=0, num_ctx=8192)

    for i, md_path in enumerate(md_files, 1):
        filename = os.path.basename(md_path)
        source_doc = filename.replace(".md", ".pdf")   # same "source" value stored in Chroma

        if source_doc in companies:
            print(f"[{i}/{len(md_files)}] Skipping {source_doc} -> {companies[source_doc]}")
            continue

        # The company name is on the cover page; start with the first 6000 characters
        name = ask_company(llm, md_path, 6000)

        # If the answer looks like a ticker, retry with more of the document
        if looks_like_ticker(name):
            print(f"    '{name}' looks like a ticker, retrying with more text...")
            name = ask_company(llm, md_path, 12000)
            if looks_like_ticker(name):
                name = f"UNKNOWN - fix manually (model said '{name}')"

        companies[source_doc] = name
        print(f"[{i}/{len(md_files)}] {source_doc} -> {name}")

        # Save after every document so progress is never lost
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(companies, f, indent=4)

    print(f"\n✅ Company names saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_companies()