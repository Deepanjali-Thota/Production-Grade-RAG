import os
import re
import glob
import json
from collections import Counter
from multiprocessing import Pool
import pymupdf

PDF_DIR = "data/production_baseline"
CHUNK_DIR = "data/chunked_docs"
PAGES_DIR = "data/chunk_pages"   # one small JSON per document -> scales to any number of PDFs

SHINGLE_SIZE = 5          # compare text in overlapping 5-word phrases
MAX_PAGES_PER_PHRASE = 3  # phrases found on more pages than this are boilerplate -> ignored
PROBES = 15               # distinctive phrases used to locate the start / the end of a chunk
MAX_SPAN = 4              # a ~1000-token chunk never spans more than a few pages

def words(text):
    # Lowercase letters/numbers only, so markdown symbols (|, #, **) and PDF line breaks don't matter
    return re.findall(r"[a-z0-9]+", text.lower())

def shingles(word_list):
    return {" ".join(word_list[i:i + SHINGLE_SIZE]) for i in range(len(word_list) - SHINGLE_SIZE + 1)}

def shingle_list(word_list):
    # Same as shingles(), but keeps the order (needed to know what is at the start / end of a chunk)
    return [" ".join(word_list[i:i + SHINGLE_SIZE]) for i in range(len(word_list) - SHINGLE_SIZE + 1)]

def build_page_index(pdf_path):
    """phrase -> set of PDF page numbers (1-based) that contain it"""
    index = {}
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            for phrase in shingles(words(page.get_text())):
                index.setdefault(phrase, set()).add(page_no)
    return index

def vote(ordered_phrases, index):
    """Let the first PROBES distinctive phrases vote for the page they appear on."""
    votes = Counter()
    used = 0
    for phrase in ordered_phrases:
        pages = index.get(phrase)
        if not pages or len(pages) > MAX_PAGES_PER_PHRASE:
            continue
        for p in pages:
            votes[p] += 1
        used += 1
        if used >= PROBES:
            break
    return votes

def find_pages(chunk_text, index):
    """Return [start_page, end_page] for a chunk, or None if it can't be located."""
    ordered = shingle_list(words(chunk_text))
    start_votes = vote(ordered, index)                    # where does the chunk begin?
    end_votes = vote(list(reversed(ordered)), index)      # where does the chunk end?

    if not start_votes and not end_votes:
        return None
    if not end_votes:
        s = start_votes.most_common(1)[0][0]
        return [s, s]
    if not start_votes:
        e = end_votes.most_common(1)[0][0]
        return [e, e]

    # Best (start, end) pair that is in order and close together.
    # This also resolves text that is repeated on several pages.
    best = None
    for s, s_votes in start_votes.items():
        for e, e_votes in end_votes.items():
            if 0 <= e - s <= MAX_SPAN:
                score = s_votes + e_votes
                if best is None or score > best[0]:
                    best = (score, s, e)
    if best:
        return [best[1], best[2]]

    s = start_votes.most_common(1)[0][0]
    return [s, s]

def output_path_for(source_doc):
    return os.path.join(PAGES_DIR, os.path.splitext(source_doc)[0] + ".json")

def needs_mapping(json_file):
    source_doc = os.path.basename(json_file).replace("_chunks.json", ".pdf")
    out = output_path_for(source_doc)
    # Re-map if never done, or if the chunks were regenerated after the last mapping
    return not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(json_file)

def map_one_document(json_file):
    """Runs in a worker process. Maps one document and writes its own output file."""
    source_doc = os.path.basename(json_file).replace("_chunks.json", ".pdf")  # same "source" as in Chroma
    pdf_path = os.path.join(PDF_DIR, source_doc)

    if not os.path.exists(pdf_path):
        return source_doc, None, 0

    with open(json_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = build_page_index(pdf_path)
    mapping = {str(chunk_id): find_pages(text, index) for chunk_id, text in enumerate(chunks)}

    # Atomic write: temp file then swap, so an interrupted run never leaves a broken file
    out = output_path_for(source_doc)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    os.replace(tmp, out)

    found = sum(1 for p in mapping.values() if p)
    return source_doc, found, len(chunks)

def map_chunks_to_pages():
    os.makedirs(PAGES_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "*_chunks.json")))
    todo = [f for f in all_files if needs_mapping(f)]
    print(f"{len(all_files)} documents total, {len(all_files) - len(todo)} already mapped, {len(todo)} to map.")
    if not todo:
        return

    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Mapping with {workers} parallel workers...")

    done = 0
    with Pool(processes=workers) as pool:
        for source_doc, found, total in pool.imap_unordered(map_one_document, todo):
            done += 1
            if found is None:
                print(f"[{done}/{len(todo)}] {source_doc}: PDF not found in {PDF_DIR}, skipped")
            else:
                print(f"[{done}/{len(todo)}] {source_doc}: located {found}/{total} chunks")

    print(f"\n✅ Page maps saved in {PAGES_DIR}/")

if __name__ == "__main__":
    map_chunks_to_pages()