"""
Step 4: offline evaluation on the golden dataset.

Stage 1 (answers): runs every golden question through the real pipeline (app.answer_question)
                   and saves the answer and the retrieved chunks.
Stage 2 (score):   - answerable questions: RAGAS faithfulness = share of the claims in the
                     generated answer that are supported by the retrieved chunks (0 to 1)
                   - unanswerable questions: correct if the system refused

The stages are separate so a failure while scoring never wastes the answers already generated.

Usage (from the project root):
    python evaluation/evaluate.py --limit 5         # small trial: first 5 questions
    python evaluation/evaluate.py                   # everything
    python evaluation/evaluate.py --stage score     # re-score saved answers only
    python evaluation/evaluate.py --fresh           # ignore saved answers and start again
"""
import os
import re
import sys
import json
import time
import asyncio
import argparse
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv()

from src.config import SETTINGS, PROMPT_VERSION

GOLDEN_FILE = "evaluation/golden_dataset.jsonl"
RESULTS_DIR = "evaluation/results"
ANSWERS_FILE = os.path.join(RESULTS_DIR, "answers.jsonl")
SCORES_FILE = os.path.join(RESULTS_DIR, "scores.jsonl")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "summary.json")

EVAL = SETTINGS["evaluation"]
CITATION_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- stage 1: answers
def generate_answers(golden, fresh):
    from app import answer_question   # imported here so "--stage score" needs no models

    done = {} if fresh else {row["id"]: row for row in read_jsonl(ANSWERS_FILE)}
    todo = [g for g in golden if g["id"] not in done]
    print(f"Answers: {len(golden) - len(todo)} already saved, {len(todo)} to generate.")

    with open(ANSWERS_FILE, "w" if fresh else "a", encoding="utf-8") as f:
        for n, g in enumerate(todo, 1):
            print(f"\n[{n}/{len(todo)}] {g['id']}: {g['question'][:80]}")
            try:
                result = answer_question(g["question"])
            except Exception as e:
                print(f"   failed, will be retried on the next run: {e}")
                time.sleep(EVAL["pause_seconds"])
                continue
            row = {
                "id": g["id"],
                "type": g.get("type", "single"),
                "question": g["question"],
                "ground_truth": g["ground_truth"],
                "answer": result["answer"],
                "refused": result["refused"],
                "reason": result["reason"],
                "contexts": result["contexts"],
                "removed_claims": len(result["removed"]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[g["id"]] = row
            time.sleep(EVAL["pause_seconds"])   # stay under the API's calls-per-minute limit

    return [done[g["id"]] for g in golden if g["id"] in done]


# ---------------------------------------------------------------- stage 2: score
def build_judge():
    """The LLM that RAGAS uses to judge faithfulness, reached through an OpenAI-compatible API."""
    import instructor
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    judge = EVAL["judge"]
    api_key = os.getenv(judge["api_key_env"])
    if not api_key:
        raise ValueError(f"{judge['api_key_env']} is not set in .env (needed for the evaluation judge).")
    client = AsyncOpenAI(api_key=api_key, base_url=judge["base_url"])
    llm = llm_factory(judge["model"], provider="openai", client=client, max_tokens=8000)
    return Faithfulness(llm=llm)


async def score_faithfulness(rows):
    metric = build_judge()
    for n, row in enumerate(rows, 1):
        response = CITATION_RE.sub("", row["answer"])   # the [n] markers are not part of the claims
        try:
            result = await metric.ascore(
                user_input=row["question"],
                response=response,
                retrieved_contexts=row["contexts"],
            )
            value = float(result.value)
            row["faithfulness"] = None if value != value else round(value, 4)   # NaN -> None
        except Exception as e:
            row["faithfulness"] = None
            row["score_error"] = str(e)[:300]
        print(f"   [{n}/{len(rows)}] {row['id']}: faithfulness = {row['faithfulness']}")
        await asyncio.sleep(EVAL["pause_seconds"])


def score(answers):
    answerable = [dict(a) for a in answers if a["type"] != "unanswerable"]
    unanswerable = [dict(a) for a in answers if a["type"] == "unanswerable"]

    # Answerable questions: only real answers can be judged for faithfulness
    answered = [a for a in answerable if not a["refused"]]
    # Reuse scores from the last run when the answer has not changed
    previous = {r["id"]: r for r in read_jsonl(SCORES_FILE) if r.get("faithfulness") is not None}
    for a in answered:
        old = previous.get(a["id"])
        if old and old.get("answer") == a["answer"]:
            a["faithfulness"] = old["faithfulness"]
    to_score = [a for a in answered if a.get("faithfulness") is None]
    print(f"\nScoring faithfulness with RAGAS: {len(answered) - len(to_score)} reused, {len(to_score)} to score...")
    if to_score:
        asyncio.run(score_faithfulness(to_score))

    # Unanswerable questions: the right behaviour is a refusal
    for u in unanswerable:
        u["refusal_correct"] = bool(u["refused"])

    scored = [a["faithfulness"] for a in answered if a.get("faithfulness") is not None]
    summary = {
        "prompt_version": PROMPT_VERSION,
        "generator": f"{SETTINGS['llm'].get('provider')}:{SETTINGS['llm'].get('cohere_model') if SETTINGS['llm'].get('provider') == 'cohere' else SETTINGS['llm'].get('model')}",
        "judge": EVAL["judge"]["model"],
        "questions": len(answers),
        "answerable": len(answerable),
        "answered": len(answered),
        "wrongly_refused": len(answerable) - len(answered),
        "answer_rate": round(len(answered) / len(answerable), 4) if answerable else None,
        "faithfulness_scored": len(scored),
        "faithfulness_failed_to_score": len(answered) - len(scored),
        "faithfulness": round(mean(scored), 4) if scored else None,
        "unanswerable": len(unanswerable),
        "refusal_accuracy": round(mean(u["refusal_correct"] for u in unanswerable), 4) if unanswerable else None,
    }

    refused_answerable = [a for a in answerable if a["refused"]]
    write_jsonl(SCORES_FILE, [{k: v for k, v in r.items() if k != "contexts"}
                              for r in answered + refused_answerable + unanswerable])
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary, answered, refused_answerable, unanswerable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only the first N golden questions")
    parser.add_argument("--stage", choices=["all", "answers", "score"], default="all")
    parser.add_argument("--fresh", action="store_true", help="ignore saved answers and regenerate them")
    args = parser.parse_args()

    golden = read_jsonl(GOLDEN_FILE)
    if not golden:
        print(f"{GOLDEN_FILE} not found or empty. Run evaluation/build_golden.py first.")
        sys.exit(1)
    if args.limit:
        golden = golden[: args.limit]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.stage in ("all", "answers"):
        answers = generate_answers(golden, args.fresh)
    else:
        wanted = {g["id"] for g in golden}
        answers = [a for a in read_jsonl(ANSWERS_FILE) if a["id"] in wanted]
        print(f"Loaded {len(answers)} saved answers.")
    if args.stage == "answers" or not answers:
        return

    summary, answered, refused_answerable, unanswerable = score(answers)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key:30} {value}")

    low = sorted((a for a in answered if a.get("faithfulness") is not None), key=lambda a: a["faithfulness"])[:5]
    if low:
        print("\nLowest faithfulness:")
        for a in low:
            print(f"  {a['faithfulness']:<6} {a['id']}: {a['question'][:70]}")
    if refused_answerable:
        print("\nAnswerable questions the system refused:")
        for a in refused_answerable:
            print(f"  {a['id']} ({a['reason']}): {a['question'][:70]}")
    wrong = [u for u in unanswerable if not u["refusal_correct"]]
    if wrong:
        print("\nUnanswerable questions the system answered anyway:")
        for u in wrong:
            print(f"  {u['id']}: {u['question'][:70]}")
    print(f"\nDetails: {SCORES_FILE}   Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()