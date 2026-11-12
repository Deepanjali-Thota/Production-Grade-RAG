"""
Step 3a: draft candidate question/answer pairs for the golden dataset.

For a sample of chunks from every document, the LLM writes ONE question that the chunk
answers, the answer, and the exact sentence it relied on. Everything is saved to
evaluation/golden_candidates.csv with an empty "verified" column.

These are only DRAFTS. A pair enters the golden dataset when YOU have checked it against
the PDF and typed "yes" in the verified column (see evaluation/build_golden.py).

Usage (from the project root):
    python evaluation/generate_candidates.py               # 5 chunks per document
    python evaluation/generate_candidates.py --per-doc 2   # quick trial
Re-running continues where it stopped; finished chunks are skipped.
"""
import os
import re
import csv
import glob
import json
import random
import argparse

import time
from dotenv import load_dotenv
from langchain_cohere import ChatCohere

load_dotenv()   # reads COHERE_API_KEY from .env
COHERE_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-a-03-2025")

CHUNK_DIR = "data/chunked_docs"
PAGES_DIR = "data/chunk_pages"
COMPANIES_FILE = "data/companies.json"
SETTINGS_FILE = "config/settings.yaml"
OUTPUT_FILE = "evaluation/golden_candidates.csv"

COLUMNS = ["id", "question", "ground_truth", "company", "source", "pages", "chunk_id",
           "evidence", "evidence_in_chunk", "verified", "notes"]

SKIP_FIRST_CHUNKS = 3   # cover page and table of contents make poor questions
SEED = 42               # same sample every run, so re-runs can resume

PROMPT = """Below is an excerpt from the SEC 10-K filing of {company}.

Write ONE question that a financial analyst might ask and that this excerpt answers.
Rules:
- The question must name the company ({company}) and make sense without seeing the excerpt.
- Prefer specific facts: figures, dates, named risks, business descriptions.
- The answer must be 1-3 sentences and use ONLY information in the excerpt.
- "evidence" must be one sentence copied word for word from the excerpt that contains the answer.

Reply with JSON only, in this form:
{{"question": "...", "answer": "...", "evidence": "..."}}

Excerpt:
{chunk}"""


def squash(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def page_label(source, chunk_id):
    pages = load_json(os.path.join(PAGES_DIR, os.path.splitext(source)[0] + ".json"), {}).get(str(chunk_id))
    if not pages:
        return ""
    return str(pages[0]) if pages[0] == pages[1] else f"{pages[0]}-{pages[1]}"


def read_done():
    """(source, chunk_id) pairs already in the CSV, and the number of rows."""
    if not os.path.exists(OUTPUT_FILE):
        return set(), 0
    with open(OUTPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {(r["source"], r["chunk_id"]) for r in rows}, len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-doc", type=int, default=5, help="chunks sampled per document")
    args = parser.parse_args()

    companies = load_json(COMPANIES_FILE, {})
    llm = ChatCohere(model=COHERE_MODEL, temperature=0)

    json_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "*_chunks.json")))
    if not json_files:
        print("No chunk files found. Run the ingestion steps first.")
        return

    # Pick the sample first, so the total is known and re-runs pick the same chunks
    rng = random.Random(SEED)
    sample = []
    for json_file in json_files:
        source = os.path.basename(json_file).replace("_chunks.json", ".pdf")
        chunks = load_json(json_file, [])
        ids = list(range(SKIP_FIRST_CHUNKS, len(chunks)))
        for chunk_id in sorted(rng.sample(ids, min(args.per_doc, len(ids)))):
            sample.append((source, chunk_id, chunks[chunk_id]))

    done, row_count = read_done()
    todo = [s for s in sample if (s[0], str(s[1])) not in done]
    print(f"{len(sample)} chunks sampled, {len(sample) - len(todo)} already done, {len(todo)} to do.")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    new_file = not os.path.exists(OUTPUT_FILE)
    # utf-8-sig so Excel opens the file with the right characters
    with open(OUTPUT_FILE, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()

        for n, (source, chunk_id, chunk) in enumerate(todo, 1):
            company = companies.get(source, os.path.splitext(source)[0])
            try:
                reply = llm.invoke(PROMPT.format(company=company, chunk=chunk)).content
                 # The model may wrap the JSON in ```json fences; take the {...} part
                data = json.loads(re.search(r"\{.*\}", reply, re.DOTALL).group(0))
                time.sleep(4)   # trial keys allow 20 chat calls per minute
                question = str(data["question"]).strip()
                answer = str(data["answer"]).strip()
                evidence = str(data.get("evidence", "")).strip()
            except Exception as e:
                print(f"[{n}/{len(todo)}] {source} chunk {chunk_id}: skipped ({e})")
                continue

            row_count += 1
            writer.writerow({
                "id": f"q{row_count:03d}",
                "question": question,
                "ground_truth": answer,
                "company": company,
                "source": source,
                "pages": page_label(source, chunk_id),
                "chunk_id": chunk_id,
                "evidence": evidence,
                # "no" means the model did not copy the sentence exactly: check this row extra carefully
                "evidence_in_chunk": "yes" if evidence and squash(evidence) in squash(chunk) else "no",
                "verified": "",
                "notes": "",
            })
            f.flush()   # progress is saved after every row
            print(f"[{n}/{len(todo)}] {source} chunk {chunk_id}: {question[:70]}")

    print(f"\nDone. Review {OUTPUT_FILE} and fill in the 'verified' column.")


if __name__ == "__main__":
    main()