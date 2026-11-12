"""
Step 3b: build the golden dataset from the rows you verified by hand.

Reads evaluation/golden_candidates.csv, keeps rows whose "verified" column is "yes",
and writes evaluation/golden_dataset.jsonl (one JSON object per line). That file is
committed to git and used by the evaluation script.

Usage (from the project root):
    python evaluation/build_golden.py
"""
import csv
import json
from collections import Counter

INPUT_FILE = "evaluation/golden_candidates.csv"
OUTPUT_FILE = "evaluation/golden_dataset.jsonl"
MIN_SIZE = 50


def read_rows(path):
    # Excel may save as UTF-8 (with a BOM) or as the Windows code page; accept both
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path}. Save it as 'CSV UTF-8' in Excel.")


def main():
    rows = read_rows(INPUT_FILE)
    status = Counter((r.get("verified") or "").strip().lower() or "(blank)" for r in rows)
    print("verified column:", dict(status))

    golden = []
    for r in rows:
        if (r.get("verified") or "").strip().lower() != "yes":
            continue
        question = (r.get("question") or "").strip()
        ground_truth = (r.get("ground_truth") or "").strip()
        if not question or not ground_truth:
            print(f"   {r.get('id')}: marked yes but question or ground_truth is empty, skipped")
            continue
        golden.append({
            "id": r["id"].strip(),
            "question": question,
            "ground_truth": ground_truth,
            "type": (r.get("type") or "single").strip().lower(),
            "company": (r.get("company") or "").strip(),
            "source": (r.get("source") or "").strip(),
            "pages": (r.get("pages") or "").strip(),
        })

    ids = Counter(g["id"] for g in golden)
    duplicates = [i for i, c in ids.items() if c > 1]
    if duplicates:
        raise ValueError(f"Duplicate ids in the CSV: {duplicates}. Every row needs its own id.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for g in golden:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print(f"\n{len(golden)} verified pairs written to {OUTPUT_FILE}")
    print("per company:", dict(Counter(g["company"] or "(none)" for g in golden)))
    if len(golden) < MIN_SIZE:
        print(f"Note: the target is at least {MIN_SIZE} pairs; {MIN_SIZE - len(golden)} more to go.")


if __name__ == "__main__":
    main()